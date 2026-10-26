package server_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/alina-ibragimova/llm-gateway/internal/llm"
	"github.com/alina-ibragimova/llm-gateway/internal/server"
)

// fakeProvider — заглушка, которая запоминает запрос и возвращает заготовленный ответ.
type fakeProvider struct {
	resp llm.ChatResponse
	err  error
	got  llm.ChatRequest

	chunks    []string // что отдаст ChatStream
	streamErr error    // ошибка, которой ChatStream завершится после кусков
}

func (f *fakeProvider) Name() string { return "fake" }

func (f *fakeProvider) Chat(_ context.Context, req llm.ChatRequest) (llm.ChatResponse, error) {
	f.got = req
	return f.resp, f.err
}

func (f *fakeProvider) ChatStream(_ context.Context, req llm.ChatRequest, onChunk func(string) error) (llm.StreamSummary, error) {
	f.got = req
	for _, c := range f.chunks {
		if err := onChunk(c); err != nil {
			return llm.StreamSummary{}, err
		}
	}
	if f.streamErr != nil {
		return llm.StreamSummary{}, f.streamErr
	}
	return llm.StreamSummary{
		Model:    "real-model",
		Provider: "fake",
		Usage:    llm.Usage{PromptTokens: 5, CompletionTokens: 2},
	}, nil
}

func newHandler(p llm.Provider) http.Handler {
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	return server.New(logger, p).Routes()
}

func post(h http.Handler, body string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(http.MethodPost, "/v1/chat", strings.NewReader(body))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec
}

func TestChat_OK(t *testing.T) {
	fp := &fakeProvider{resp: llm.ChatResponse{
		Content:  "Привет!",
		Model:    "real-model",
		Provider: "fake",
		Usage:    llm.Usage{PromptTokens: 5, CompletionTokens: 2},
	}}

	rec := post(newHandler(fp), `{"messages":[{"role":"user","content":"Привет"}]}`)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body: %s", rec.Code, rec.Body)
	}
	// Клиент модель не указал, значит, провайдеру уходит пустая строка
	// и он сам подставит свою модель по умолчанию.
	if fp.got.Model != "" {
		t.Errorf("provider got model %q, want empty", fp.got.Model)
	}

	var got struct {
		Content  string `json:"content"`
		Provider string `json:"provider"`
	}
	if err := json.NewDecoder(rec.Body).Decode(&got); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if got.Content != "Привет!" || got.Provider != "fake" {
		t.Errorf("response = %+v", got)
	}
}

func TestChat_BadRequest(t *testing.T) {
	tests := []struct {
		name string
		body string
	}{
		{"invalid json", `{`},
		{"no messages", `{"messages":[]}`},
		{"unknown role", `{"messages":[{"role":"bot","content":"hi"}]}`},
		{"empty content", `{"messages":[{"role":"user","content":"  "}]}`},
		{"unknown field", `{"messages":[{"role":"user","content":"hi"}],"foo":1}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rec := post(newHandler(&fakeProvider{}), tt.body)
			if rec.Code != http.StatusBadRequest {
				t.Errorf("status = %d, want 400", rec.Code)
			}
		})
	}
}

func TestChat_ProviderError(t *testing.T) {
	fp := &fakeProvider{err: &llm.APIError{Provider: "fake", StatusCode: 503, Body: "overloaded"}}

	rec := post(newHandler(fp), `{"messages":[{"role":"user","content":"hi"}]}`)

	if rec.Code != http.StatusBadGateway {
		t.Errorf("status = %d, want 502", rec.Code)
	}
	if strings.Contains(rec.Body.String(), "overloaded") {
		t.Error("provider error details must not leak to the client")
	}
}

const streamBody = `{"stream":true,"messages":[{"role":"user","content":"hi"}]}`

func TestChatStream_OK(t *testing.T) {
	fp := &fakeProvider{chunks: []string{"Hel", "lo\nworld"}}

	rec := post(newHandler(fp), streamBody)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body: %s", rec.Code, rec.Body)
	}
	if ct := rec.Header().Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("Content-Type = %q, want text/event-stream", ct)
	}

	body := rec.Body.String()
	for _, want := range []string{
		`data: {"delta":"Hel"}`,
		`data: {"delta":"lo\nworld"}`, // перенос строки экранирован внутри JSON и поток не ломает
		"event: done",
		`"provider":"fake"`,
	} {
		if !strings.Contains(body, want) {
			t.Errorf("body does not contain %q:\n%s", want, body)
		}
	}
}

func TestChatStream_ErrorBeforeFirstChunk(t *testing.T) {
	fp := &fakeProvider{streamErr: &llm.APIError{Provider: "fake", StatusCode: 503, Body: "overloaded"}}

	rec := post(newHandler(fp), streamBody)

	// Ничего не отправлено, поэтому клиент получает обычную JSON-ошибку со статусом.
	if rec.Code != http.StatusBadGateway {
		t.Errorf("status = %d, want 502", rec.Code)
	}
	if strings.Contains(rec.Body.String(), "overloaded") {
		t.Error("provider error details must not leak to the client")
	}
}

func TestChatStream_ErrorMidStream(t *testing.T) {
	fp := &fakeProvider{chunks: []string{"Hel"}, streamErr: errors.New("boom")}

	rec := post(newHandler(fp), streamBody)

	// Статус 200 уже отправлен, поэтому ошибка приходит событием внутри потока.
	if rec.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", rec.Code)
	}
	body := rec.Body.String()
	if !strings.Contains(body, `data: {"delta":"Hel"}`) {
		t.Errorf("first chunk missing:\n%s", body)
	}
	if !strings.Contains(body, "event: error") {
		t.Errorf("error event missing:\n%s", body)
	}
	if strings.Contains(body, "boom") {
		t.Error("internal error text must not leak to the client")
	}
}
