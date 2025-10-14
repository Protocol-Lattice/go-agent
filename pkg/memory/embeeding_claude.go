package memory

import "context"

// Claude/Anthropic does not offer embeddings; keep a stub so config won't panic.
type ClaudeEmbedder struct {
	model string
}

func NewClaudeEmbedder(model string) (Embedder, error) {
	return &ClaudeEmbedder{model: model}, nil
}

func (c *ClaudeEmbedder) Embed(_ context.Context, _ string) ([]float32, error) {
	// Return a clear, actionable error. Callers use fallback.
	return nil, ErrNotSupported
}
