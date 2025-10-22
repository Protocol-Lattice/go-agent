package models

import (
	"context"
	"fmt"
	"strings"
)

// NewLLMProvider returns a concrete Agent.
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

// ---------- Shared helpers for file-aware generation ----------

func isTextMIME(m string) bool {
	m = strings.ToLower(strings.TrimSpace(m))
	if m == "" {
		return false
	}
	if strings.HasPrefix(m, "text/") {
		return true
	}
	switch m {
	case "application/json",
		"application/xml",
		"application/x-yaml",
		"application/yaml",
		"text/markdown",
		"text/x-markdown":
		return true
	default:
		return false
	}
}

// combinePromptWithFiles renders a single prompt that includes file context.
func combinePromptWithFiles(base string, files []File) string {
	if len(files) == 0 {
		return base
	}

	var b strings.Builder
	b.WriteString(base)
	b.WriteString("\n\n---\nATTACHMENTS CONTEXT (inline for text files) — BEGIN\n")

	for i, f := range files {
		title := f.Name
		if strings.TrimSpace(title) == "" {
			title = fmt.Sprintf("file_%d", i+1)
		}
		if isTextMIME(f.MIME) && len(f.Data) > 0 {
			b.WriteString("\n<<<FILE ")
			b.WriteString(title)
			if f.MIME != "" {
				b.WriteString(" [")
				b.WriteString(f.MIME)
				b.WriteString("]")
			}
			b.WriteString(">>>:\n")
			b.Write(f.Data)
			b.WriteString("\n<<<END FILE ")
			b.WriteString(title)
			b.WriteString(">>>\n")
		} else {
			// Non-text or empty: reference only
			b.WriteString("\n[Non-text attachment] ")
			b.WriteString(title)
			if f.MIME != "" {
				b.WriteString(" (")
				b.WriteString(f.MIME)
				b.WriteString(")")
			}
		}
	}

	b.WriteString("\nATTACHMENTS CONTEXT — END\n---\n")
	return b.String()
}
