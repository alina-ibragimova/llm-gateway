package llm_test

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/alina-ibragimova/llm-gateway/internal/llm"
)

func TestOpenAICompatible_Chat(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Errorf("path = %q, want /chat/completions", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("Authorization = %q, want %q", got, "Bearer test-key")
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"model": "test-model",
			"choices": [{"message": {"role": "assistant", "content": "Привет!"}}],
			"usage": {"prompt_tokens": 9, "completion_tokens": 3}
		}`))
	}))
	defer srv.Close()

	p := llm.NewOpenAICompatible("test", srv.URL, "test-key", "test-model", nil)

	got, err := p.Chat(context.Background(), llm.ChatRequest{
		Model:    "test-model",
		Messages: []llm.Message{{Role: "user", Content: "Привет"}},
	})
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}
	if got.Content != "Привет!" {
		t.Errorf("Content = %q, want %q", got.Content, "Привет!")
	}
	if got.Usage.PromptTokens != 9 || got.Usage.CompletionTokens != 3 {
		t.Errorf("Usage = %+v, want {9 3}", got.Usage)
	}
}

func TestOpenAICompatible_Chat_APIError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "overloaded", http.StatusServiceUnavailable)
	}))
	defer srv.Close()

	p := llm.NewOpenAICompatible("test", srv.URL, "key", "test-model", nil)

	_, err := p.Chat(context.Background(), llm.ChatRequest{Model: "m"})

	var apiErr *llm.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("error = %v, want *llm.APIError", err)
	}
	if apiErr.StatusCode != http.StatusServiceUnavailable {
		t.Errorf("StatusCode = %d, want 503", apiErr.StatusCode)
	}
	if !apiErr.Retryable() {
		t.Error("503 must be retryable")
	}
}

func TestOpenAICompatible_ChatStream(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		if !strings.Contains(string(body), `"stream":true`) {
			t.Errorf("request body %s must contain stream:true", body)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, `data: {"model":"m","choices":[{"delta":{"content":"Hel"}}]}`+"\n\n")
		fmt.Fprint(w, `data: {"choices":[{"delta":{"content":"lo"},"finish_reason":"stop"}]}`+"\n\n")
		fmt.Fprint(w, `data: {"choices":[],"usage":{"prompt_tokens":4,"completion_tokens":2}}`+"\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	p := llm.NewOpenAICompatible("test", srv.URL, "key", "test-model", nil)

	var got []string
	summary, err := p.ChatStream(context.Background(), llm.ChatRequest{}, func(d string) error {
		got = append(got, d)
		return nil
	})
	if err != nil {
		t.Fatalf("ChatStream() error = %v", err)
	}
	if strings.Join(got, "") != "Hello" {
		t.Errorf("chunks = %v, want Hel + lo", got)
	}
	if summary.Model != "m" || summary.Provider != "test" {
		t.Errorf("summary = %+v", summary)
	}
	if summary.Usage.PromptTokens != 4 || summary.Usage.CompletionTokens != 2 {
		t.Errorf("usage = %+v, want {4 2}", summary.Usage)
	}
}

func TestOpenAICompatible_ChatStream_Truncated(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		// Один кусок и обрыв: ни finish_reason, ни [DONE].
		fmt.Fprint(w, `data: {"choices":[{"delta":{"content":"Hel"}}]}`+"\n\n")
	}))
	defer srv.Close()

	p := llm.NewOpenAICompatible("test", srv.URL, "key", "test-model", nil)

	_, err := p.ChatStream(context.Background(), llm.ChatRequest{}, func(string) error { return nil })
	if !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Errorf("error = %v, want io.ErrUnexpectedEOF", err)
	}
}
