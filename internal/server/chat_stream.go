package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"time"

	"github.com/alina-ibragimova/llm-gateway/internal/llm"
)

// streamDelta — событие с очередным куском текста.
type streamDelta struct {
	Delta string `json:"delta"`
}

// streamDone — завершающее событие со статистикой.
type streamDone struct {
	Model     string    `json:"model"`
	Provider  string    `json:"provider"`
	Usage     llm.Usage `json:"usage"`
	LatencyMS int64     `json:"latency_ms"`
}

func (s *Server) handleChatStream(w http.ResponseWriter, r *http.Request, req chatRequest) {
	ctx, cancel := context.WithTimeout(r.Context(), streamTimeout)
	defer cancel()

	rc := http.NewResponseController(w)
	started := false

	// beginStream отправляет заголовки SSE. Вызывается перед первым куском: до этого момента
	// мы ещё можем ответить клиенту обычной JSON-ошибкой с нормальным статусом.
	beginStream := func() {
		if started {
			return
		}
		started = true
		h := w.Header()
		h.Set("Content-Type", "text/event-stream")
		h.Set("Cache-Control", "no-cache")
		h.Set("X-Accel-Buffering", "no") // просим nginx и подобные прокси не копить ответ
		w.WriteHeader(http.StatusOK)
	}

	start := time.Now()
	summary, err := s.provider.ChatStream(ctx, llm.ChatRequest{Model: req.Model, Messages: req.Messages}, func(delta string) error {
		beginStream()
		return writeSSE(w, rc, "", streamDelta{Delta: delta})
	})
	latency := time.Since(start)

	if err != nil {
		if !started {
			// Ничего не отправлено, можно ответить обычной ошибкой (502/504).
			s.writeUpstreamError(w, err)
			return
		}
		if r.Context().Err() != nil {
			s.logger.Debug("client disconnected during stream")
			return
		}
		// Статус 200 уже ушёл, поэтому ошибку сообщаем событием внутри потока.
		s.logger.Error("stream interrupted", "err", err)
		_ = writeSSE(w, rc, "error", map[string]string{"error": "stream interrupted"})
		return
	}

	beginStream() // на случай, если провайдер завершил ответ, не прислав ни одного куска
	s.logger.Info("chat stream completed",
		"provider", summary.Provider,
		"model", summary.Model,
		"prompt_tokens", summary.Usage.PromptTokens,
		"completion_tokens", summary.Usage.CompletionTokens,
		"latency_ms", latency.Milliseconds(),
	)
	_ = writeSSE(w, rc, "done", streamDone{
		Model:     summary.Model,
		Provider:  summary.Provider,
		Usage:     summary.Usage,
		LatencyMS: latency.Milliseconds(),
	})
}

// writeSSE пишет одно SSE-событие и сразу отправляет его клиенту.
func writeSSE(w http.ResponseWriter, rc *http.ResponseController, event string, data any) error {
	payload, err := json.Marshal(data)
	if err != nil {
		return err
	}
	if event != "" {
		if _, err := fmt.Fprintf(w, "event: %s\n", event); err != nil {
			return err
		}
	}
	if _, err := fmt.Fprintf(w, "data: %s\n\n", payload); err != nil {
		return err
	}
	// ErrNotSupported значит, что писатель не умеет сбрасывать буфер. Тогда стрим просто придёт пачкой.
	if err := rc.Flush(); err != nil && !errors.Is(err, http.ErrNotSupported) {
		return err
	}
	return nil
}
