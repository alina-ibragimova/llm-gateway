package llm

import (
	"fmt"
	"net/http"
)

// APIError — провайдер ответил, но с кодом, отличным от 200.
type APIError struct {
	Provider   string
	StatusCode int
	Body       string
}

func (e *APIError) Error() string {
	return fmt.Sprintf("%s: unexpected status %d: %s", e.Provider, e.StatusCode, e.Body)
}

// Retryable сообщает, имеет ли смысл повторить запрос или попробовать другого провайдера.
// Ошибки 5xx и 429 (лимит) временные, а 400/401/404 повторять бессмысленно.
func (e *APIError) Retryable() bool {
	return e.StatusCode == http.StatusTooManyRequests || e.StatusCode >= http.StatusInternalServerError
}
