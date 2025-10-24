package models

import (
	"bytes"
	"context"
	"fmt"
	"mime"
	"os"
	"path/filepath"
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

// sanitizeForGemini coerces edge cases again and filters to what Gemini will accept.
// Return "" to skip attaching (fallback to text-only).
// sanitizeForGemini coerces edge cases again and filters to what Gemini will accept.
// Return "" to skip attaching (fallback to text-only).
func sanitizeForGemini(mt string) string {
	mt = strings.ToLower(strings.TrimSpace(mt))

	// Fix any double-prefix issues that might have slipped through
	if strings.HasPrefix(mt, "image/image/") {
		mt = "image/" + strings.TrimPrefix(mt, "image/image/")
	}
	if strings.HasPrefix(mt, "video/video/") {
		mt = "video/" + strings.TrimPrefix(mt, "video/video/")
	}

	switch {
	case mt == "":
		return ""
	// Use exact match or HasPrefix instead of Contains to avoid substring issues
	case mt == "image/png" || strings.HasPrefix(mt, "image/png;"):
		return "image/png"
	case mt == "image/jpeg" || mt == "image/jpg" || mt == "image/pjpeg" ||
		strings.HasPrefix(mt, "image/jpeg;") || strings.HasPrefix(mt, "image/jpg;"):
		return "image/jpeg"
	case mt == "image/webp" || strings.HasPrefix(mt, "image/webp;"):
		return "image/webp"
	case mt == "image/gif" || strings.HasPrefix(mt, "image/gif;"):
		return "image/gif"
	// videos (supported by Gemini 1.5 Pro/Flash)
	case mt == "video/mp4" || strings.HasPrefix(mt, "video/mp4;"):
		return "video/mp4"
	case mt == "video/quicktime" || mt == "video/mov" ||
		strings.HasPrefix(mt, "video/quicktime;"):
		return "video/quicktime"
	case mt == "video/webm" || strings.HasPrefix(mt, "video/webm;"):
		return "video/webm"
	default:
		// Unknown/unsupported -> skip attach
		return ""
	}
}

// normalizeMIME fixes messy/alias MIMEs and falls back to file extension.
func normalizeMIME(name, m string) string {
	strip := func(s string) string {
		if i := strings.IndexByte(s, ';'); i >= 0 {
			return strings.TrimSpace(s[:i])
		}
		return strings.TrimSpace(s)
	}
	fromExt := func() string {
		ext := strings.ToLower(filepath.Ext(name))
		if ext == "" {
			return ""
		}
		if mt := mime.TypeByExtension(ext); mt != "" {
			return strip(mt)
		}
		switch ext { // minimal fallbacks
		case ".jpg", ".jpeg":
			return "image/jpeg"
		case ".png":
			return "image/png"
		case ".gif":
			return "image/gif"
		case ".webp":
			return "image/webp"
		case ".bmp":
			return "image/bmp"
		case ".svg":
			return "image/svg+xml"
		case ".heic":
			return "image/heic"
		case ".mp4":
			return "video/mp4"
		case ".mov":
			return "video/quicktime"
		case ".webm":
			return "video/webm"
		case ".mkv":
			return "video/x-matroska"
		case ".avi":
			return "video/x-msvideo"
		case ".txt", ".log":
			return "text/plain"
		case ".md":
			return "text/markdown"
		case ".json":
			return "application/json"
		case ".yaml", ".yml":
			return "application/x-yaml"
		case ".xml":
			return "application/xml"
		}
		return ""
	}

	raw := strings.ToLower(strings.TrimSpace(m))
	if raw == "" {
		return fromExt()
	}
	raw = strip(raw)

	// FIX DUPLICATES FIRST - before any other logic
	original := raw
	for {
		fixed := false
		if strings.HasPrefix(raw, "image/image/") {
			raw = "image/" + strings.TrimPrefix(raw, "image/image/")
			fixed = true
		}
		if strings.HasPrefix(raw, "video/video/") {
			raw = "video/" + strings.TrimPrefix(raw, "video/video/")
			fixed = true
		}
		// Keep looping in case there are multiple duplications like "image/image/image/png"
		if !fixed {
			break
		}
	}
	// Debug logging - remove after fixing
	if original != raw {
		fmt.Fprintf(os.Stderr, "DEBUG: normalizeMIME fixed '%s' -> '%s' for file '%s'\n", original, raw, name)
	}

	// Now handle common aliases
	switch raw {
	case "image/jpg", "image/pjpeg":
		return "image/jpeg"
	case "image/x-png":
		return "image/png"
	case "video/mov":
		return "video/quicktime"
	}

	// malformed -> extension
	if !strings.Contains(raw, "/") || strings.HasSuffix(raw, "/") {
		if via := fromExt(); via != "" {
			return via
		}
	}
	return raw
}

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

// Only text files are inlined; everything else is referenced (and possibly attached by the provider path).
func combinePromptWithFiles(base string, files []File) string {
	if len(files) == 0 {
		return base
	}

	var b bytes.Buffer
	b.WriteString(base)
	b.WriteString("\n\n---\nATTACHMENTS CONTEXT (inline for text files) — BEGIN\n")

	for i, f := range files {
		title := strings.TrimSpace(f.Name)
		if title == "" {
			title = fmt.Sprintf("file_%d", i+1)
		}
		mt := normalizeMIME(f.Name, f.MIME)

		if isTextMIME(mt) && len(f.Data) > 0 {
			b.WriteString("\n<<<FILE ")
			b.WriteString(title)
			if mt != "" {
				b.WriteString(" [")
				b.WriteString(mt)
				b.WriteString("]")
			}
			b.WriteString(">>>:\n")
			b.Write(f.Data)
			b.WriteString("\n<<<END FILE ")
			b.WriteString(title)
			b.WriteString(">>>\n")
		} else {
			b.WriteString("\n[Non-text attachment] ")
			b.WriteString(title)
			if mt != "" {
				b.WriteString(" (")
				b.WriteString(mt)
				b.WriteString(")")
			}
		}
	}

	b.WriteString("\nATTACHMENTS CONTEXT — END\n---\n")
	return b.String()
}
