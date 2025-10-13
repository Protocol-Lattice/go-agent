package models

import (
	"context"
	"fmt"
)

func NewLLMProvider(ctx context.Context, provider string, model string) (Agent, error) {
	switch provider {
	case "openai":
		return NewOpenAILLM(model), nil
	case "gemini", "google":
		return NewGeminiLLM(ctx, model)
	case "ollama":
		return NewOllamaLLM(model)
	case "anthropic", "claude":
		return NewAnthropicLLM(model), nil
	default:
		return nil, fmt.Errorf("unknown provider: %s", provider)
	}
}
