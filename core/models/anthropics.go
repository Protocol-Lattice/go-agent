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

func (a *AnthropicLLM) GenerateWithFiles(ctx context.Context, prompt string, files []File) (any, error) {
	// Normalize all file MIME types
	norm := make([]File, 0, len(files))
	for _, f := range files {
		norm = append(norm, File{
			Name: f.Name,
			MIME: normalizeMIME(f.Name, f.MIME),
			Data: f.Data,
		})
	}

	// Build message content with text and attachments
	var contentBlocks []anthropic.ContentBlockParamUnion

	// Add system prefix if present
	fullPrompt := prompt
	if a.PromptPrefix != "" {
		fullPrompt = fmt.Sprintf("%s\n\n%s", a.PromptPrefix, prompt)
	}

	// Add text prompt with inline text files
	textContent := combinePromptWithFiles(fullPrompt, norm)
	contentBlocks = append(contentBlocks, anthropic.NewTextBlock(textContent))

	// Attach images and videos as proper content blocks
	for _, f := range norm {
		if len(f.Data) == 0 {
			continue
		}

		mt := sanitizeForAnthropic(f.MIME)
		if mt == "" {
			continue // skip unsupported types
		}

		if strings.HasPrefix(mt, "image/") {
			// Anthropic expects base64-encoded images
			contentBlocks = append(contentBlocks, anthropic.NewImageBlockBase64(mt, string(f.Data)))
		}
		// Note: Anthropic doesn't support video in Messages API yet
		// Videos will be referenced in the text context only
	}

	msg, err := a.Client.Messages.New(ctx, anthropic.MessageNewParams{
		Model:     anthropic.Model(a.Model),
		MaxTokens: int64(a.MaxTokens),
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(contentBlocks...),
		},
	})
	if err != nil {
		return nil, fmt.Errorf("anthropic generateWithFiles: %w", err)
	}

	var b strings.Builder
	for _, cb := range msg.Content {
		if tb, ok := cb.AsAny().(anthropic.TextBlock); ok {
			b.WriteString(tb.Text)
		}
	}
	return b.String(), nil
}

// sanitizeForAnthropic filters MIME types to what Anthropic supports
func sanitizeForAnthropic(mt string) string {
	mt = strings.ToLower(strings.TrimSpace(mt))

	// Fix any double-prefix issues
	if strings.HasPrefix(mt, "image/image/") {
		mt = "image/" + strings.TrimPrefix(mt, "image/image/")
	}

	switch {
	case mt == "":
		return ""
	// Anthropic supports these image formats
	case mt == "image/jpeg" || mt == "image/jpg" || strings.HasPrefix(mt, "image/jpeg;"):
		return "image/jpeg"
	case mt == "image/png" || strings.HasPrefix(mt, "image/png;"):
		return "image/png"
	case mt == "image/gif" || strings.HasPrefix(mt, "image/gif;"):
		return "image/gif"
	case mt == "image/webp" || strings.HasPrefix(mt, "image/webp;"):
		return "image/webp"
	default:
		// Unsupported format
		return ""
	}
}
