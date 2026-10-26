// Package server содержит HTTP-обработчики шлюза.
package server

import (
	"encoding/json"
	"log/slog"
	"net/http"

	"github.com/alina-ibragimova/llm-gateway/internal/llm"
)

// Server хранит зависимости, которые нужны обработчикам.
type Server struct {
	logger       *slog.Logger
	provider     llm.Provider
	defaultModel string
}

func New(logger *slog.Logger, provider llm.Provider) *Server {
	return &Server{logger: logger, provider: provider}
}

// Routes собирает все маршруты приложения.
func (s *Server) Routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", s.handleHealth)
	mux.HandleFunc("POST /v1/chat", s.handleChat)
	return mux
}

func (s *Server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

// writeJSON сериализует v в JSON и отправляет клиенту с нужным статусом.
func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}
