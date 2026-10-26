package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/alina-ibragimova/llm-gateway/internal/llm"
)

const (
	maxBodyBytes  = 1 << 20          // 1 МиБ
	chatTimeout   = 45 * time.Second // общий дедлайн обычного запроса
	streamTimeout = 3 * time.Minute  // общий дедлайн стрима
)

type chatRequest struct {
	Model    string        `json:"model"` // необязательное поле: если пусто, провайдер выберет свою модель
	Messages []llm.Message `json:"messages"`
	Stream   bool          `json:"stream"`
}

type chatResponse struct {
	Content   string    `json:"content"`
	Model     string    `json:"model"`
	Provider  string    `json:"provider"`
	Usage     llm.Usage `json:"usage"`
	LatencyMS int64     `json:"latency_ms"`
}

func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
	req, err := decodeChatRequest(w, r)
	if err != nil {
		var tooBig *http.MaxBytesError
		if errors.As(err, &tooBig) {
			writeError(w, http.StatusRequestEntityTooLarge, "request body too large")
			return
		}
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	if req.Stream {
		s.handleChatStream(w, r, req)
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), chatTimeout)
	defer cancel()

	start := time.Now()
	resp, err := s.provider.Chat(ctx, llm.ChatRequest{Model: req.Model, Messages: req.Messages})
	latency := time.Since(start)
	if err != nil {
		s.writeUpstreamError(w, err)
		return
	}

	// В лог идут только метаданные. Текст сообщений пользователей мы не логируем.
	s.logger.Info("chat completed",
		"provider", resp.Provider,
		"model", resp.Model,
		"prompt_tokens", resp.Usage.PromptTokens,
		"completion_tokens", resp.Usage.CompletionTokens,
		"latency_ms", latency.Milliseconds(),
	)

	writeJSON(w, http.StatusOK, chatResponse{
		Content:   resp.Content,
		Model:     resp.Model,
		Provider:  resp.Provider,
		Usage:     resp.Usage,
		LatencyMS: latency.Milliseconds(),
	})
}

// decodeChatRequest читает и проверяет тело запроса.
func decodeChatRequest(w http.ResponseWriter, r *http.Request) (chatRequest, error) {
	r.Body = http.MaxBytesReader(w, r.Body, maxBodyBytes)
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()

	var req chatRequest
	if err := dec.Decode(&req); err != nil {
		return chatRequest{}, fmt.Errorf("invalid JSON: %w", err)
	}
	if err := validateChat(req); err != nil {
		return chatRequest{}, err
	}
	return req, nil
}

func validateChat(req chatRequest) error {
	if len(req.Messages) == 0 {
		return errors.New("messages must not be empty")
	}
	for i, m := range req.Messages {
		switch m.Role {
		case "system", "user", "assistant":
		default:
			return fmt.Errorf("messages[%d]: unknown role %q", i, m.Role)
		}
		if strings.TrimSpace(m.Content) == "" {
			return fmt.Errorf("messages[%d]: content must not be empty", i)
		}
	}
	return nil
}

// writeUpstreamError превращает ошибку провайдеров в понятный клиенту HTTP-ответ.
func (s *Server) writeUpstreamError(w http.ResponseWriter, err error) {
	switch {
	case errors.Is(err, context.Canceled):
		// Клиент сам оборвал соединение, отвечать уже некому.
		s.logger.Debug("client cancelled request")
	case errors.Is(err, context.DeadlineExceeded):
		s.logger.Error("providers timed out", "err", err)
		writeError(w, http.StatusGatewayTimeout, "provider timed out")
	default:
		// Подробности (в том числе тела ответов провайдеров) только в лог, не клиенту.
		s.logger.Error("provider request failed", "err", err)
		writeError(w, http.StatusBadGateway, "provider request failed")
	}
}
