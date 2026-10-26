// Package llm описывает общий интерфейс для LLM-провайдеров и их реализации.
package llm

import "context"

// Message — одна реплика в диалоге. Role: "system", "user" или "assistant".
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatRequest — запрос к модели в нашем внутреннем формате.
// Если Model пуста, провайдер подставляет свою модель по умолчанию.
type ChatRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
}

// Usage — сколько токенов потратил запрос.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
}

// ChatResponse — ответ модели в нашем внутреннем формате.
type ChatResponse struct {
	Content  string
	Model    string
	Provider string
	Usage    Usage
}

// StreamSummary — итоги завершённого стрима. Usage может быть нулевым,
// если провайдер не прислал статистику токенов в потоке.
type StreamSummary struct {
	Model    string
	Provider string
	Usage    Usage
}

// Provider — любой LLM-провайдер, к которому умеет ходить шлюз.
type Provider interface {
	Name() string
	Chat(ctx context.Context, req ChatRequest) (ChatResponse, error)

	// ChatStream отдаёт ответ по частям: onChunk вызывается для каждого куска текста.
	// Если onChunk вернул ошибку, стрим прерывается и ChatStream возвращает эту ошибку.
	ChatStream(ctx context.Context, req ChatRequest, onChunk func(delta string) error) (StreamSummary, error)
}
