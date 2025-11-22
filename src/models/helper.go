package models

import (
	"context"
	"fmt"
	"mime"
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
// Optimized version with caching and lookup tables.
func normalizeMIME(name, m string) string {
	// Fast path: check cache first
	cacheKey := name + "|" + m
	mimeCacheMu.RLock()
	if cached, ok := mimeCache[cacheKey]; ok {
		mimeCacheMu.RUnlock()
		return cached
	}
	mimeCacheMu.RUnlock()

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

		// Fast lookup in our map
		if mt, ok := mimeExtMap[ext]; ok {
			return mt
		}

		// Fallback to mime package
		if mt := mime.TypeByExtension(ext); mt != "" {
			return strip(mt)
		}
		return ""
	}

	raw := strings.ToLower(strings.TrimSpace(m))
	if raw == "" {
		result := fromExt()
		// Cache the result
		mimeCacheMu.Lock()
		if len(mimeCache) < 1000 { // Limit cache size
			mimeCache[cacheKey] = result
		}
		mimeCacheMu.Unlock()
		return result
	}

	raw = strip(raw)

	// FIX DUPLICATES - optimized with strings.Count
	for strings.HasPrefix(raw, "image/image/") || strings.HasPrefix(raw, "video/video/") {
		if strings.HasPrefix(raw, "image/image/") {
			raw = "image/" + strings.TrimPrefix(raw, "image/image/")
		}
		if strings.HasPrefix(raw, "video/video/") {
			raw = "video/" + strings.TrimPrefix(raw, "video/video/")
		}
	}

	// Fast alias lookup
	if normalized, ok := mimeAliasMap[raw]; ok {
		mimeCacheMu.Lock()
		if len(mimeCache) < 1000 {
			mimeCache[cacheKey] = normalized
		}
		mimeCacheMu.Unlock()
		return normalized
	}

	// Malformed MIME -> use extension
	if !strings.Contains(raw, "/") || strings.HasSuffix(raw, "/") {
		if via := fromExt(); via != "" {
			mimeCacheMu.Lock()
			if len(mimeCache) < 1000 {
				mimeCache[cacheKey] = via
			}
			mimeCacheMu.Unlock()
			return via
		}
	}

	// Cache the result
	mimeCacheMu.Lock()
	if len(mimeCache) < 1000 {
		mimeCache[cacheKey] = raw
	}
	mimeCacheMu.Unlock()

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
