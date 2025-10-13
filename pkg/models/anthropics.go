package models

import (
	"context"
	"os"
	"strings"

	anthropic "github.com/anthropics/anthropic-sdk-go"
	anthropicopt "github.com/anthropics/anthropic-sdk-go/option"
)

// AnthropicLLM implements your Agent interface using Anthropic's Messages API.
type AnthropicLLM struct {
	Client    *anthropic.Client
	Model     string
	MaxTokens int
}

// NewAnthropicLLM constructs a client. It reads ANTHROPIC_API_KEY from the env.
func NewAnthropicLLM(model string) *AnthropicLLM {
	key := os.Getenv("ANTHROPIC_API_KEY")
	cl := anthropic.NewClient(
		anthropicopt.WithAPIKey(key), // SDK uses this header; set the env var in your runtime.
	)
	return &AnthropicLLM{
		Client:    &cl,
		Model:     model, // e.g. "claude-3-5-sonnet-20241022" or "claude-3-5-sonnet-latest"
		MaxTokens: 1024,  // tweak per your needs
	}
}

// Generate performs a single-turn completion and returns the concatenated text.
func (a *AnthropicLLM) Generate(ctx context.Context, prompt string) (any, error) {
	msg, err := a.Client.Messages.New(ctx, anthropic.MessageNewParams{
		Model:     anthropic.Model(a.Model),
		MaxTokens: int64(a.MaxTokens),
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock(prompt)),
		},
	})
	if err != nil {
		return nil, err
	}

	var b strings.Builder
	for _, cb := range msg.Content {
		if tb, ok := cb.AsAny().(anthropic.TextBlock); ok {
			b.WriteString(tb.Text)
		}
	}
	return b.String(), nil
}
