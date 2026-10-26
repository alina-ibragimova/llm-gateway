package llm_test

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"slices"
	"testing"
	"time"

	"github.com/alina-ibragimova/llm-gateway/internal/llm"
)

type step struct {
	resp llm.ChatResponse
	err  error
}

// stubProvider отдаёт заготовленные результаты по очереди; последний повторяется.
type stubProvider struct {
	name  string
	steps []step
	calls int
}

func (s *stubProvider) Name() string { return s.name }

func (s *stubProvider) Chat(ctx context.Context, _ llm.ChatRequest) (llm.ChatResponse, error) {
	if err := ctx.Err(); err != nil {
		return llm.ChatResponse{}, err
	}
	i := min(s.calls, len(s.steps)-1)
	s.calls++
	return s.steps[i].resp, s.steps[i].err
}

// ChatStream переиспользует Chat: весь ответ отдаётся одним куском.
func (s *stubProvider) ChatStream(ctx context.Context, _ llm.ChatRequest, onChunk func(string) error) (llm.StreamSummary, error) {
	resp, err := s.Chat(ctx, llm.ChatRequest{})
	if err != nil {
		return llm.StreamSummary{}, err
	}
	if err := onChunk(resp.Content); err != nil {
		return llm.StreamSummary{}, err
	}
	return llm.StreamSummary{Provider: resp.Provider}, nil
}

func success(provider string) step {
	return step{resp: llm.ChatResponse{Content: "ok", Provider: provider}}
}

func failure(status int) step {
	return step{err: &llm.APIError{Provider: "stub", StatusCode: status}}
}

func newFallback(t *testing.T, ps ...llm.Provider) *llm.Fallback {
	t.Helper()
	f, err := llm.NewFallback(slog.New(slog.NewTextHandler(io.Discard, nil)), ps, llm.FallbackOptions{
		MaxAttempts:    2,
		BaseDelay:      time.Millisecond,
		AttemptTimeout: time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}
	return f
}

func TestFallback_RetriesSameProviderOnTransientError(t *testing.T) {
	a := &stubProvider{name: "a", steps: []step{failure(http.StatusServiceUnavailable), success("a")}}
	b := &stubProvider{name: "b", steps: []step{success("b")}}

	resp, err := newFallback(t, a, b).Chat(context.Background(), llm.ChatRequest{})
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}
	if resp.Provider != "a" {
		t.Errorf("answered by %q, want a", resp.Provider)
	}
	if a.calls != 2 || b.calls != 0 {
		t.Errorf("calls a=%d b=%d, want 2 and 0", a.calls, b.calls)
	}
}

func TestFallback_SwitchesProviderWithoutRetryOnPermanentError(t *testing.T) {
	a := &stubProvider{name: "a", steps: []step{failure(http.StatusUnauthorized)}}
	b := &stubProvider{name: "b", steps: []step{success("b")}}

	resp, err := newFallback(t, a, b).Chat(context.Background(), llm.ChatRequest{})
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}
	if resp.Provider != "b" {
		t.Errorf("answered by %q, want b", resp.Provider)
	}
	if a.calls != 1 {
		t.Errorf("a called %d times, want 1 (401 must not be retried)", a.calls)
	}
}

func TestFallback_SwitchesProviderAfterRetriesExhausted(t *testing.T) {
	a := &stubProvider{name: "a", steps: []step{failure(http.StatusTooManyRequests)}}
	b := &stubProvider{name: "b", steps: []step{success("b")}}

	resp, err := newFallback(t, a, b).Chat(context.Background(), llm.ChatRequest{})
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}
	if resp.Provider != "b" || a.calls != 2 {
		t.Errorf("provider = %q, a.calls = %d, want b and 2", resp.Provider, a.calls)
	}
}

func TestFallback_AllProvidersFailed(t *testing.T) {
	a := &stubProvider{name: "a", steps: []step{failure(http.StatusServiceUnavailable)}}
	b := &stubProvider{name: "b", steps: []step{failure(http.StatusUnauthorized)}}

	_, err := newFallback(t, a, b).Chat(context.Background(), llm.ChatRequest{})

	var apiErr *llm.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("error = %v, want it to wrap *llm.APIError", err)
	}
}

func TestFallback_StopsWhenCallerCancelled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	a := &stubProvider{name: "a", steps: []step{success("a")}}
	b := &stubProvider{name: "b", steps: []step{success("b")}}

	_, err := newFallback(t, a, b).Chat(ctx, llm.ChatRequest{})

	if !errors.Is(err, context.Canceled) {
		t.Errorf("error = %v, want context.Canceled", err)
	}
	if b.calls != 0 {
		t.Errorf("b called %d times after cancellation, want 0", b.calls)
	}
}

// --- Тесты стриминга ---

// midStreamProvider отдаёт один кусок и обрывается с ошибкой.
type midStreamProvider struct{ calls int }

func (m *midStreamProvider) Name() string { return "mid" }

func (m *midStreamProvider) Chat(context.Context, llm.ChatRequest) (llm.ChatResponse, error) {
	return llm.ChatResponse{}, errors.New("not used")
}

func (m *midStreamProvider) ChatStream(_ context.Context, _ llm.ChatRequest, onChunk func(string) error) (llm.StreamSummary, error) {
	m.calls++
	if err := onChunk("part"); err != nil {
		return llm.StreamSummary{}, err
	}
	return llm.StreamSummary{}, errors.New("connection reset")
}

// hangingProvider принимает запрос и молчит, пока его не отменят.
type hangingProvider struct{}

func (hangingProvider) Name() string { return "hanging" }

func (hangingProvider) Chat(context.Context, llm.ChatRequest) (llm.ChatResponse, error) {
	return llm.ChatResponse{}, errors.New("not used")
}

func (hangingProvider) ChatStream(ctx context.Context, _ llm.ChatRequest, _ func(string) error) (llm.StreamSummary, error) {
	<-ctx.Done()
	return llm.StreamSummary{}, ctx.Err()
}

func runStream(p llm.Provider) ([]string, error) {
	var got []string
	_, err := p.ChatStream(context.Background(), llm.ChatRequest{}, func(d string) error {
		got = append(got, d)
		return nil
	})
	return got, err
}

func TestFallbackStream_SwitchesProviderBeforeFirstChunk(t *testing.T) {
	a := &stubProvider{name: "a", steps: []step{failure(http.StatusServiceUnavailable)}}
	b := &stubProvider{name: "b", steps: []step{success("b")}}

	got, err := runStream(newFallback(t, a, b))
	if err != nil {
		t.Fatalf("ChatStream() error = %v", err)
	}
	if !slices.Equal(got, []string{"ok"}) {
		t.Errorf("chunks = %v, want [ok]", got)
	}
	if a.calls != 2 || b.calls != 1 {
		t.Errorf("calls a=%d b=%d, want 2 and 1", a.calls, b.calls)
	}
}

func TestFallbackStream_NoSwitchAfterFirstChunk(t *testing.T) {
	m := &midStreamProvider{}
	b := &stubProvider{name: "b", steps: []step{success("b")}}

	got, err := runStream(newFallback(t, m, b))
	if err == nil {
		t.Fatal("want error after interrupted stream")
	}
	if !slices.Equal(got, []string{"part"}) {
		t.Errorf("chunks = %v, want [part]", got)
	}
	if m.calls != 1 {
		t.Errorf("mid called %d times, want 1 (no retry after partial output)", m.calls)
	}
	if b.calls != 0 {
		t.Errorf("b called %d times, want 0 (no switch after partial output)", b.calls)
	}
}

func TestFallbackStream_FirstChunkTimeout(t *testing.T) {
	b := &stubProvider{name: "b", steps: []step{success("b")}}
	f, err := llm.NewFallback(slog.New(slog.NewTextHandler(io.Discard, nil)),
		[]llm.Provider{hangingProvider{}, b},
		llm.FallbackOptions{MaxAttempts: 1, BaseDelay: time.Millisecond, AttemptTimeout: 20 * time.Millisecond})
	if err != nil {
		t.Fatal(err)
	}

	got, err := runStream(f)
	if err != nil {
		t.Fatalf("ChatStream() error = %v", err)
	}
	if !slices.Equal(got, []string{"ok"}) || b.calls != 1 {
		t.Errorf("chunks = %v, b.calls = %d, want [ok] and 1", got, b.calls)
	}
}
