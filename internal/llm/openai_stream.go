package llm

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

// streamChunk — один кусок потока в формате OpenAI. Наружу не выходит.
type streamChunk struct {
	Model   string `json:"model"`
	Choices []struct {
		Delta struct {
			Content string `json:"content"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
	Usage *Usage `json:"usage"`
}

// ChatStream запрашивает ответ в режиме stream и вызывает onChunk для каждого куска текста.
func (p *OpenAICompatible) ChatStream(ctx context.Context, req ChatRequest, onChunk func(string) error) (StreamSummary, error) {
	resp, err := p.post(ctx, chatCompletionRequest{Model: p.modelFor(req), Messages: req.Messages, Stream: true})
	if err != nil {
		return StreamSummary{}, err
	}
	defer resp.Body.Close()

	summary := StreamSummary{Provider: p.name}
	finished := false // видели ли признак нормального завершения

	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 0, 64<<10), 1<<20) // строки потока короткие, но запас не мешает

	for sc.Scan() {
		// SSE: события разделены пустой строкой, полезные строки начинаются с "data:".
		data, ok := strings.CutPrefix(sc.Text(), "data:")
		if !ok {
			continue // пустые строки, комментарии и другие служебные поля
		}
		data = strings.TrimSpace(data)
		if data == "[DONE]" {
			return summary, nil
		}

		var chunk streamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			return summary, fmt.Errorf("%s: decode stream chunk: %w", p.name, err)
		}
		if chunk.Model != "" {
			summary.Model = chunk.Model
		}
		if chunk.Usage != nil {
			summary.Usage = *chunk.Usage
		}
		for _, c := range chunk.Choices {
			if c.FinishReason != nil && *c.FinishReason != "" {
				finished = true
			}
			if c.Delta.Content == "" {
				continue
			}
			if err := onChunk(c.Delta.Content); err != nil {
				return summary, err
			}
		}
	}
	if err := sc.Err(); err != nil {
		return summary, fmt.Errorf("%s: read stream: %w", p.name, err)
	}
	if !finished {
		// Соединение закрылось, а признака завершения не было: ответ обрезан.
		return summary, fmt.Errorf("%s: stream ended unexpectedly: %w", p.name, io.ErrUnexpectedEOF)
	}
	return summary, nil
}
