package uploads

import (
	"context"
	"regexp"
)

var (
	emailRegexp = regexp.MustCompile(`(?i)[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}`)
	phoneRegexp = regexp.MustCompile(`(?i)\b(?:\+?\d[\d -]{7,}\d)`)
)

// PIIRedactor removes common PII patterns before embedding content.
type PIIRedactor struct {
	Replacement string
}

func (r PIIRedactor) Process(_ context.Context, chunk *DocumentChunk) error {
	if chunk == nil {
		return nil
	}
	replacement := r.Replacement
	if replacement == "" {
		replacement = "[redacted]"
	}
	chunk.Content = emailRegexp.ReplaceAllString(chunk.Content, replacement)
	chunk.Content = phoneRegexp.ReplaceAllString(chunk.Content, replacement)
	if chunk.Metadata == nil {
		chunk.Metadata = map[string]any{}
	}
	chunk.Metadata["pii_redacted"] = true
	return nil
}
