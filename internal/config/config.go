// Package config загружает настройки приложения из переменных окружения.
package config

import (
	"errors"
	"fmt"
	"log/slog"
	"os"
)

// ProviderConfig описывает одного LLM-провайдера.
type ProviderConfig struct {
	Name         string
	BaseURL      string
	APIKey       string
	DefaultModel string
}

type Config struct {
	Addr      string
	LogLevel  slog.Level
	Providers []ProviderConfig // порядок = приоритет
}

// knownProviders — все провайдеры, о которых знает шлюз. Включается тот, для которого задан ключ.
// Названия моделей меняются, поэтому их можно переопределить через *_MODEL в .env.
var knownProviders = []struct {
	name         string
	baseURL      string
	keyEnv       string
	modelEnv     string
	defaultModel string
}{
	{"groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY", "GROQ_MODEL", "openai/gpt-oss-20b"},
	{"gemini", "https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY", "GEMINI_MODEL", "gemini-3.5-flash-lite"},
	{"openrouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "OPENROUTER_MODEL", "openrouter/free"},
	{"deepseek", "https://api.deepseek.com", "DEEPSEEK_API_KEY", "DEEPSEEK_MODEL", "deepseek-v4-flash"},
}

// Load читает конфигурацию из окружения, подставляя значения по умолчанию.
func Load() (Config, error) {
	var level slog.Level
	if err := level.UnmarshalText([]byte(getEnv("LOG_LEVEL", "info"))); err != nil {
		return Config{}, fmt.Errorf("parse LOG_LEVEL: %w", err)
	}

	var providers []ProviderConfig
	for _, kp := range knownProviders {
		key := getEnv(kp.keyEnv, "")
		if key == "" {
			continue // ключа нет — провайдер выключен
		}
		providers = append(providers, ProviderConfig{
			Name:         kp.name,
			BaseURL:      kp.baseURL,
			APIKey:       key,
			DefaultModel: getEnv(kp.modelEnv, kp.defaultModel),
		})
	}
	if len(providers) == 0 {
		return Config{}, errors.New("no LLM providers configured: set at least one of GROQ_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY, DEEPSEEK_API_KEY")
	}

	return Config{
		Addr:      getEnv("ADDR", ":8080"),
		LogLevel:  level,
		Providers: providers,
	}, nil
}

func getEnv(key, fallback string) string {
	if v, ok := os.LookupEnv(key); ok && v != "" {
		return v
	}
	return fallback
}
