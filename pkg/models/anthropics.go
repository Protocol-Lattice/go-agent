package models

import (
	"context"
	"fmt"
	"os"
	"strings"

	anthropic "github.com/anthropics/anthropic-sdk-go"
	anthropicopt "github.com/anthropics/anthropic-sdk-go/option"
)

// AnthropicLLM implements your Agent interface using Anthropic's Messages API.
type AnthropicLLM struct {
	Client       *anthropic.Client
	Model        string
	MaxTokens    int
	PromptPrefix string
}

// NewAnthropicLLM constructs a client. It reads ANTHROPIC_API_KEY from the env.
func NewAnthropicLLM(model, promptPrefix string) *AnthropicLLM {
	key := os.Getenv("ANTHROPIC_API_KEY")
	cl := anthropic.NewClient(
		anthropicopt.WithAPIKey(key),
	)
	return &AnthropicLLM{
		Client:       &cl,
		Model:        model, // e.g. "claude-3-5-sonnet-latest"
		MaxTokens:    1024,
		PromptPrefix: promptPrefix,
	}
}

// Generate performs a single-turn completion and returns concatenated text.
func (a *AnthropicLLM) Generate(ctx context.Context, prompt string) (any, error) {
	fullPrompt := prompt
	if a.PromptPrefix != "" {
		fullPrompt = fmt.Sprintf("%s\n\n%s", a.PromptPrefix, prompt)
	}

	msg, err := a.Client.Messages.New(ctx, anthropic.MessageNewParams{
		Model:     anthropic.Model(a.Model),
		MaxTokens: int64(a.MaxTokens),
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock(fullPrompt)),
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

func (a *AnthropicLLM) UploadFiles(ctx context.Context, files []UploadFile) ([]UploadedFile, error) {
	uploads := make([]UploadedFile, 0, len(files))
	betas := anthropicUploadBetas()
	for _, file := range files {
		resolved, err := file.resolve()
		if err != nil {
			return nil, err
		}
		params := anthropic.BetaFileUploadParams{File: resolved.reader}
		if len(betas) > 0 {
			params.Betas = betas
		}
		meta, uploadErr := a.Client.Beta.Files.Upload(ctx, params)
		closeErr := resolved.Close()
		if uploadErr == nil {
			uploadErr = closeErr
		}
		if uploadErr != nil {
			return nil, uploadErr
		}
		uploads = append(uploads, UploadedFile{
			ID:        meta.ID,
			Name:      meta.Filename,
			SizeBytes: meta.SizeBytes,
			MIMEType:  meta.MimeType,
			Provider:  "anthropic",
			Purpose:   file.Purpose,
		})
	}
	return uploads, nil
}

func anthropicUploadBetas() []anthropic.AnthropicBeta {
	if raw := strings.TrimSpace(os.Getenv("ANTHROPIC_BETA_HEADERS")); raw != "" {
		parts := strings.Split(raw, ",")
		betas := make([]anthropic.AnthropicBeta, 0, len(parts))
		for _, part := range parts {
			trimmed := strings.TrimSpace(part)
			if trimmed == "" {
				continue
			}
			betas = append(betas, anthropic.AnthropicBeta(trimmed))
		}
		return betas
	}
	return []anthropic.AnthropicBeta{anthropic.AnthropicBetaMessageBatches2024_09_24}
}
