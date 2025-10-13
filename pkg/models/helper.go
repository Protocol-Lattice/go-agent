package models

import (
	"context"
	"fmt"
)

func NewLLMProvider(ctx context.Context, provider string, model string, promptPrefix string) (Agent, error) {
	switch provider {
	case "openai":
		return NewOpenAILLM(model, promptPrefix), nil
	case "gemini", "google":
		return NewGeminiLLM(ctx, model, promptPrefix)
	case "ollama":
		return NewOllamaLLM(model, promptPrefix)
	case "anthropic", "claude":
		return NewAnthropicLLM(model, promptPrefix), nil
	default:
		return nil, fmt.Errorf("unknown provider: %s", provider)
	}
}
