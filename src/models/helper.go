package models

import (
	"context"
	"fmt"
	"mime"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

// MIME type lookup tables for fast access
var (
	mimeExtMap = map[string]string{
		".jpg":  "image/jpeg",
		".jpeg": "image/jpeg",
		".png":  "image/png",
		".gif":  "image/gif",
		".webp": "image/webp",
		".bmp":  "image/bmp",
		".svg":  "image/svg+xml",
		".heic": "image/heic",
		".mp4":  "video/mp4",
		".mov":  "video/quicktime",
		".webm": "video/webm",
		".mkv":  "video/x-matroska",
		".avi":  "video/x-msvideo",
		".txt":  "text/plain",
		".log":  "text/plain",
		".md":   "text/markdown",
		".json": "application/json",
		".yaml": "application/x-yaml",
		".yml":  "application/x-yaml",
		".xml":  "application/xml",
	}

	mimeAliasMap = map[string]string{
		"image/jpg":   "image/jpeg",
		"image/pjpeg": "image/jpeg",
		"image/x-png": "image/png",
		"video/mov":   "video/quicktime",
	}

	// Cache for normalized MIME types
	mimeCache   = make(map[string]string, 100)
	mimeCacheMu sync.RWMutex
)

// NewLLMProvider returns a concrete Agent.
func NewLLMProvider(ctx context.Context, provider string, model string, promptPrefix string) (Agent, error) {
	return newLLMProvider(ctx, provider, model, promptPrefix, NewVertexLLM)
}

type vertexProviderFactory func(context.Context, string, string, string, string) (Agent, error)

func newLLMProvider(
	ctx context.Context,
	provider string,
	model string,
	promptPrefix string,
	newVertex vertexProviderFactory,
) (Agent, error) {
	var agent Agent
	var err error

	switch provider {
	case "openai":
		agent = NewOpenAILLM(model, promptPrefix)
	case "gemini", "google":
		agent, err = NewGeminiLLM(ctx, model, promptPrefix)
	case "vertex", "vertexai", "vertex-ai":
		project := strings.TrimSpace(os.Getenv("GOOGLE_CLOUD_PROJECT"))
		location := strings.TrimSpace(os.Getenv("GOOGLE_CLOUD_LOCATION"))
		if location == "" {
			location = strings.TrimSpace(os.Getenv("GOOGLE_CLOUD_REGION"))
		}
		agent, err = newVertex(ctx, model, promptPrefix, project, location)
	case "ollama":
		agent, err = NewOllamaLLM(model, promptPrefix)
	case "anthropic", "claude":
		agent = NewAnthropicLLM(model, promptPrefix)
	case "openrouter":
		agent = NewOpenRouterLLM(model, promptPrefix)
	default:
		return nil, fmt.Errorf("unknown provider: %s", provider)
	}

	if err != nil {
		return nil, err
	}

	return TryCreateCachedLLM(agent), nil
}

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

const mimeCacheLimit = 1000

// normalizeMIME fixes messy/alias MIMEs and falls back to file extension.
func normalizeMIME(name, m string) string {
	cacheKey := name + "|" + m
	mimeCacheMu.RLock()
	if cached, ok := mimeCache[cacheKey]; ok {
		mimeCacheMu.RUnlock()
		return cached
	}
	mimeCacheMu.RUnlock()

	result := normalizeMIMEUncached(name, m)
	mimeCacheMu.Lock()
	if len(mimeCache) < mimeCacheLimit {
		mimeCache[cacheKey] = result
	}
	mimeCacheMu.Unlock()
	return result
}

func normalizeMIMEUncached(name, m string) string {
	raw := strings.ToLower(strings.TrimSpace(m))
	if raw == "" {
		return mimeFromExtension(name)
	}

	raw = stripMIMEParameters(raw)

	for strings.HasPrefix(raw, "image/image/") || strings.HasPrefix(raw, "video/video/") {
		if strings.HasPrefix(raw, "image/image/") {
			raw = "image/" + strings.TrimPrefix(raw, "image/image/")
		}
		if strings.HasPrefix(raw, "video/video/") {
			raw = "video/" + strings.TrimPrefix(raw, "video/video/")
		}
	}

	if normalized, ok := mimeAliasMap[raw]; ok {
		return normalized
	}

	if !strings.Contains(raw, "/") || strings.HasSuffix(raw, "/") {
		if via := mimeFromExtension(name); via != "" {
			return via
		}
	}

	return raw
}

func stripMIMEParameters(value string) string {
	if index := strings.IndexByte(value, ';'); index >= 0 {
		return strings.TrimSpace(value[:index])
	}
	return strings.TrimSpace(value)
}

func mimeFromExtension(name string) string {
	ext := strings.ToLower(filepath.Ext(name))
	if ext == "" {
		return ""
	}
	if mediaType, ok := mimeExtMap[ext]; ok {
		return mediaType
	}
	return stripMIMEParameters(mime.TypeByExtension(ext))
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
// Optimized version with pre-allocated buffer.
func combinePromptWithFiles(base string, files []File) string {
	if len(files) == 0 {
		return base
	}

	// Pre-calculate approximate size to reduce allocations
	estimatedSize := len(base) + 200 // header/footer overhead
	for _, f := range files {
		estimatedSize += len(f.Name) + 100 // metadata overhead
		mt := normalizeMIME(f.Name, f.MIME)
		if isTextMIME(mt) {
			estimatedSize += len(f.Data)
		}
	}

	var b strings.Builder
	b.Grow(estimatedSize)

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

// isImageOrVideoMIME checks if the MIME type is an image or video
func isImageOrVideoMIME(m string) bool {
	m = strings.ToLower(strings.TrimSpace(m))
	if m == "" {
		return false
	}
	return strings.HasPrefix(m, "image/") || strings.HasPrefix(m, "video/")
}
