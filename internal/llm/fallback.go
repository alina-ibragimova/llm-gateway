package llm

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"math/rand/v2"
	"time"
)

// FallbackOptions задаёт политику повторов.
type FallbackOptions struct {
	MaxAttempts int           // попыток на одного провайдера (минимум 1)
	BaseDelay   time.Duration // пауза перед второй попыткой, дальше удваивается

	// AttemptTimeout — лимит на одну попытку. Для обычного запроса это время до полного
	// ответа, для стриминга — время до первого куска (дальше стрим идёт сколько нужно).
	AttemptTimeout time.Duration
}

// Fallback — тоже Provider: он обходит список провайдеров по порядку, пока кто-то не ответит.
type Fallback struct {
	logger    *slog.Logger
	providers []Provider
	opts      FallbackOptions
}

var _ Provider = (*Fallback)(nil)

func NewFallback(logger *slog.Logger, providers []Provider, opts FallbackOptions) (*Fallback, error) {
	if len(providers) == 0 {
		return nil, errors.New("fallback: no providers")
	}
	if opts.MaxAttempts < 1 {
		opts.MaxAttempts = 1
	}
	if opts.AttemptTimeout <= 0 {
		opts.AttemptTimeout = 30 * time.Second
	}
	return &Fallback{logger: logger, providers: providers, opts: opts}, nil
}

func (f *Fallback) Name() string { return "fallback" }

func (f *Fallback) Chat(ctx context.Context, req ChatRequest) (ChatResponse, error) {
	var errs []error
	for _, p := range f.providers {
		resp, err := f.tryProvider(ctx, p, req)
		if err == nil {
			return resp, nil
		}
		// Клиент ушёл или время вышло целиком — другим провайдерам просить уже незачем.
		if ctx.Err() != nil {
			return ChatResponse{}, ctx.Err()
		}
		f.logger.Warn("provider failed, trying next", "provider", p.Name(), "err", err)
		errs = append(errs, err)
	}
	return ChatResponse{}, fmt.Errorf("all providers failed: %w", errors.Join(errs...))
}

// ChatStream обходит провайдеров так же, как Chat, но с одним важным отличием:
// после первого отправленного куска переключаться и повторять нельзя,
// потому что клиент уже получил часть ответа.
func (f *Fallback) ChatStream(ctx context.Context, req ChatRequest, onChunk func(string) error) (StreamSummary, error) {
	var errs []error
	for _, p := range f.providers {
		summary, started, err := f.tryStream(ctx, p, req, onChunk)
		if err == nil {
			return summary, nil
		}
		if started {
			return StreamSummary{}, err
		}
		if ctx.Err() != nil {
			return StreamSummary{}, ctx.Err()
		}
		f.logger.Warn("provider stream failed, trying next", "provider", p.Name(), "err", err)
		errs = append(errs, err)
	}
	return StreamSummary{}, fmt.Errorf("all providers failed: %w", errors.Join(errs...))
}

// tryProvider делает до MaxAttempts попыток к одному провайдеру.
func (f *Fallback) tryProvider(ctx context.Context, p Provider, req ChatRequest) (ChatResponse, error) {
	var lastErr error
	for attempt := 1; attempt <= f.opts.MaxAttempts; attempt++ {
		attemptCtx, cancel := context.WithTimeout(ctx, f.opts.AttemptTimeout)
		resp, err := p.Chat(attemptCtx, req)
		cancel()
		if err == nil {
			return resp, nil
		}
		lastErr = err

		if ctx.Err() != nil || !retryable(err) || attempt == f.opts.MaxAttempts {
			return ChatResponse{}, lastErr
		}

		delay := backoff(f.opts.BaseDelay, attempt)
		f.logger.Warn("retrying provider", "provider", p.Name(), "attempt", attempt, "delay", delay.String(), "err", err)
		if err := sleep(ctx, delay); err != nil {
			return ChatResponse{}, err
		}
	}
	return ChatResponse{}, lastErr
}

// tryStream делает до MaxAttempts попыток стрима к одному провайдеру.
// Второй результат показывает, был ли уже отправлен хотя бы один кусок.
func (f *Fallback) tryStream(ctx context.Context, p Provider, req ChatRequest, onChunk func(string) error) (StreamSummary, bool, error) {
	started := false
	track := func(delta string) error {
		started = true
		return onChunk(delta)
	}

	var lastErr error
	for attempt := 1; attempt <= f.opts.MaxAttempts; attempt++ {
		summary, err := f.streamAttempt(ctx, p, req, track)
		if err == nil {
			return summary, started, nil
		}
		lastErr = err

		// После первого куска повторять нельзя: текст бы задвоился.
		if started || ctx.Err() != nil || !retryable(err) || attempt == f.opts.MaxAttempts {
			return StreamSummary{}, started, lastErr
		}

		delay := backoff(f.opts.BaseDelay, attempt)
		f.logger.Warn("retrying provider stream", "provider", p.Name(), "attempt", attempt, "delay", delay.String(), "err", err)
		if err := sleep(ctx, delay); err != nil {
			return StreamSummary{}, started, err
		}
	}
	return StreamSummary{}, started, lastErr
}

// streamAttempt — одна попытка стрима. Если за AttemptTimeout не пришёл ни один кусок,
// попытка отменяется. Таймер выключается, как только пришёл первый кусок.
func (f *Fallback) streamAttempt(ctx context.Context, p Provider, req ChatRequest, onChunk func(string) error) (StreamSummary, error) {
	attemptCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	timer := time.AfterFunc(f.opts.AttemptTimeout, cancel)
	defer timer.Stop()

	first := true
	summary, err := p.ChatStream(attemptCtx, req, func(delta string) error {
		if first {
			first = false
			timer.Stop()
		}
		return onChunk(delta)
	})

	// Родительский контекст жив, а наш отменён — значит, сработал именно наш таймер.
	if err != nil && ctx.Err() == nil && attemptCtx.Err() != nil {
		err = fmt.Errorf("%s: no data within %s: %w", p.Name(), f.opts.AttemptTimeout, context.DeadlineExceeded)
	}
	return summary, err
}

// retryable решает, имеет ли смысл повторять запрос к тому же провайдеру.
func retryable(err error) bool {
	var apiErr *APIError
	if errors.As(err, &apiErr) {
		return apiErr.Retryable()
	}
	return true // сетевая ошибка, таймаут попытки и т.п.
}

// backoff возвращает паузу base * 2^(attempt-1) плюс случайная добавка до +50%.
func backoff(base time.Duration, attempt int) time.Duration {
	d := base << (attempt - 1)
	return d + rand.N(d/2+1)
}

// sleep ждёт d, но просыпается сразу, если ctx отменили.
func sleep(ctx context.Context, d time.Duration) error {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-t.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}
