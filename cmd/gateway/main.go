package main

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/joho/godotenv"

	"github.com/alina-ibragimova/llm-gateway/internal/config"
	"github.com/alina-ibragimova/llm-gateway/internal/llm"
	"github.com/alina-ibragimova/llm-gateway/internal/server"
)

func main() {
	if err := run(); err != nil {
		slog.Error("fatal error", "err", err)
		os.Exit(1)
	}
}

func run() error {
	// В разработке читаем .env, если он есть. На сервере переменные задаёт окружение.
	if err := godotenv.Load(); err != nil && !errors.Is(err, fs.ErrNotExist) {
		return fmt.Errorf("load .env: %w", err)
	}

	cfg, err := config.Load()
	if err != nil {
		return err
	}

	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: cfg.LogLevel}))
	slog.SetDefault(logger)

	// Каждому настроенному провайдеру — свой клиент, а поверх них Fallback,
	// который перебирает их по порядку и повторяет запросы при временных сбоях.
	providers := make([]llm.Provider, 0, len(cfg.Providers))
	for _, pc := range cfg.Providers {
		providers = append(providers, llm.NewOpenAICompatible(pc.Name, pc.BaseURL, pc.APIKey, pc.DefaultModel, nil))
		logger.Info("provider configured", "name", pc.Name, "default_model", pc.DefaultModel)
	}
	provider, err := llm.NewFallback(logger, providers, llm.FallbackOptions{
		MaxAttempts:    2,
		BaseDelay:      300 * time.Millisecond,
		AttemptTimeout: 30 * time.Second,
	})
	if err != nil {
		return err
	}

	// ctx отменится, когда придёт Ctrl+C (SIGINT) или SIGTERM (его шлёт Docker при остановке).
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	srv := &http.Server{
		Addr:              cfg.Addr,
		Handler:           server.New(logger, provider).Routes(),
		ReadHeaderTimeout: 5 * time.Second,
	}

	serverErr := make(chan error, 1)
	go func() {
		logger.Info("server started", "addr", cfg.Addr)
		if err := srv.ListenAndServe(); !errors.Is(err, http.ErrServerClosed) {
			serverErr <- err
		}
	}()

	select {
	case err := <-serverErr:
		return err
	case <-ctx.Done():
		logger.Info("shutting down")
	}

	// Даём активным запросам до 10 секунд, чтобы спокойно завершиться.
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	return srv.Shutdown(shutdownCtx)
}
