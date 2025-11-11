package embed

import (
	"context"
	"errors"
	"log"
	"os"
	"strings"
)

// Embedder is a pluggable text-embedding provider.
type Embedder interface {
	Embed(ctx context.Context, text string) ([]float32, error)
}

// ---------- Dummy (fallback) ----------
type DummyEmbedder struct{}

func (DummyEmbedder) Embed(_ context.Context, text string) ([]float32, error) {
	return DummyEmbedding(text), nil
}

// DummyEmbedding is kept for tests/back-compat.
func DummyEmbedding(text string) []float32 {
	vec := make([]float32, 768)
	for i, ch := range []byte(text) {
		vec[i%768] += float32(ch) / 255.0
	}
	return vec
}

// AutoEmbedder chooses a provider from env:
// ADK_EMBED_PROVIDER=openai|google|gemini|ollama|claude
// ADK_EMBED_MODEL=<model string>
// If not set, it infers from available API keys/OLLAMA_HOST, else dummy.
func AutoEmbedder() Embedder {
	provider := strings.ToLower(strings.TrimSpace(os.Getenv("ADK_EMBED_PROVIDER")))
	model := strings.TrimSpace(os.Getenv("ADK_EMBED_MODEL"))

	switch provider {
	case "openai":
		if e, err := NewOpenAIEmbedder(model); err == nil {
			return e
		}
	case "google", "gemini", "vertex", "vertexai":
		if e, err := NewVertexAIEmbedder(model); err == nil {
			return e
		}
	case "ollama":
		if e, err := NewOllamaEmbedder(model); err == nil {
			return e
		}
	case "claude", "anthropic":
		if e, err := NewClaudeEmbedder(model); err == nil {
			return e
		}
	case "fastembed":
		if opts := defaultFastEmbedOptions(); opts != nil {
			if e, err := NewFastEmbeed(context.Background(), opts); err == nil {
				return e
			}
		}
	}

	log.Printf("AutoEmbedder: falling back to DummyEmbedder")
	return DummyEmbedder{}
}

// safeEmbed is a helper that never fails (falls back to DummyEmbedding).
func safeEmbed(e Embedder, text string) []float32 {
	if e == nil {
		return DummyEmbedding(text)
	}
	v, err := e.Embed(context.Background(), text)
	if err != nil || len(v) == 0 {
		return DummyEmbedding(text)
	}
	return v
}

// ErrNotSupported is returned by providers that do not offer embeddings.
var ErrNotSupported = errors.New("embeddings not supported by this provider")
