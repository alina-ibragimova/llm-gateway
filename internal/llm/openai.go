package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"time"
)

// OpenAICompatible работает с любым API формата OpenAI Chat Completions:
// OpenAI, DeepSeek, OpenRouter, Groq, Gemini и т.д.
type OpenAICompatible struct {
	name         string
	baseURL      string
	apiKey       string
	defaultModel string
	client       *http.Client
}

// Проверка на этапе компиляции: *OpenAICompatible обязан реализовывать Provider.
var _ Provider = (*OpenAICompatible)(nil)

// NewOpenAICompatible создаёт клиент. Если client равен nil, используется клиент с короткими
// таймаутами на подключение. defaultModel подставляется, когда в запросе модель не указана.
//
// Общего http.Client.Timeout здесь нет намеренно: он оборвал бы долгие стримы.
// Время жизни запроса ограничивает context, который передаёт вызывающий.
func NewOpenAICompatible(name, baseURL, apiKey, defaultModel string, client *http.Client) *OpenAICompatible {
	if client == nil {
		client = &http.Client{
			Transport: &http.Transport{
				Proxy:                 http.ProxyFromEnvironment, // уважаем HTTPS_PROXY
				DialContext:           (&net.Dialer{Timeout: 5 * time.Second}).DialContext,
				TLSHandshakeTimeout:   5 * time.Second,
				ResponseHeaderTimeout: 30 * time.Second,
				ForceAttemptHTTP2:     true,
			},
		}
	}
	return &OpenAICompatible{
		name:         name,
		baseURL:      strings.TrimRight(baseURL, "/"),
		apiKey:       apiKey,
		defaultModel: defaultModel,
		client:       client,
	}
}

func (p *OpenAICompatible) Name() string { return p.name }

// Форматы запроса и ответа конкретно этого API. Наружу не выходят.
type chatCompletionRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
	Stream   bool      `json:"stream,omitempty"`
}

type chatCompletionResponse struct {
	Model   string `json:"model"`
	Choices []struct {
		Message Message `json:"message"`
	} `json:"choices"`
	Usage Usage `json:"usage"`
}

func (p *OpenAICompatible) Chat(ctx context.Context, req ChatRequest) (ChatResponse, error) {
	resp, err := p.post(ctx, chatCompletionRequest{Model: p.modelFor(req), Messages: req.Messages})
	if err != nil {
		return ChatResponse{}, err
	}
	defer resp.Body.Close()

	var parsed chatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return ChatResponse{}, fmt.Errorf("%s: decode response: %w", p.name, err)
	}
	if len(parsed.Choices) == 0 {
		return ChatResponse{}, fmt.Errorf("%s: empty choices in response", p.name)
	}

	return ChatResponse{
		Content:  parsed.Choices[0].Message.Content,
		Model:    parsed.Model,
		Provider: p.name,
		Usage:    parsed.Usage,
	}, nil
}

// modelFor возвращает модель из запроса или модель провайдера по умолчанию.
func (p *OpenAICompatible) modelFor(req ChatRequest) string {
	if req.Model != "" {
		return req.Model
	}
	return p.defaultModel
}

// post отправляет JSON-запрос. При статусе 200 возвращает ответ, и тело закрывает вызывающий.
// При любом другом статусе тело читается, закрывается, и возвращается *APIError.
func (p *OpenAICompatible) post(ctx context.Context, payload any) (*http.Response, error) {
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)

	resp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("%s: send request: %w", p.name, err)
	}

	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		// Читаем не больше 4 КБ тела, чтобы огромный ответ не забил память.
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<10))
		return nil, &APIError{Provider: p.name, StatusCode: resp.StatusCode, Body: string(msg)}
	}
	return resp, nil
}
