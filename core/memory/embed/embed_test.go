package embed

import (
	"context"
	"errors"
	"testing"
)

type stubEmbedder struct {
	vec []float32
	err error
}

func (s stubEmbedder) Embed(_ context.Context, text string) ([]float32, error) {
	if s.vec != nil {
		return s.vec, nil
	}
	if s.err != nil {
		return nil, s.err
	}
	return []float32{float32(len(text))}, nil
}

func TestDummyEmbeddingLength(t *testing.T) {
	vec := DummyEmbedding("hello world")
	if len(vec) != 768 {
		t.Fatalf("expected dummy embedding to be length 768, got %d", len(vec))
	}
	if vec[0] == 0 {
		t.Fatalf("expected dummy embedding to have non-zero signal")
	}
}

func TestSafeEmbedFallbacks(t *testing.T) {
	dummy := DummyEmbedding("fallback")
	got := safeEmbed(nil, "fallback")
	if len(got) != len(dummy) {
		t.Fatalf("expected fallback embedding length %d, got %d", len(dummy), len(got))
	}

	failing := stubEmbedder{err: errors.New("boom")}
	got = safeEmbed(failing, "fallback")
	if len(got) != len(dummy) {
		t.Fatalf("expected fallback embedding length %d, got %d", len(dummy), len(got))
	}
}

func TestSafeEmbedUsesProvidedVector(t *testing.T) {
	expected := []float32{1, 2, 3}
	got := safeEmbed(stubEmbedder{vec: expected}, "irrelevant")
	if len(got) != len(expected) {
		t.Fatalf("expected %d values, got %d", len(expected), len(got))
	}
}

func TestAutoEmbedderSelection(t *testing.T) {
	t.Setenv("ADK_EMBED_PROVIDER", "openai")
	t.Setenv("ADK_EMBED_MODEL", "test-model")
	t.Setenv("OPENAI_API_KEY", "dummy-key")

	embedder := AutoEmbedder()
	if _, ok := embedder.(*OpenAIEmbedder); !ok {
		t.Fatalf("expected AutoEmbedder to return *OpenAIEmbedder, got %T", embedder)
	}
}

func TestAutoEmbedderFallback(t *testing.T) {
	t.Setenv("ADK_EMBED_PROVIDER", "")
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("OPENAI_KEY", "")
	t.Setenv("GOOGLE_API_KEY", "")
	t.Setenv("GEMINI_API_KEY", "")
	t.Setenv("OLLAMA_HOST", "")
	embedder := AutoEmbedder()
	if _, ok := embedder.(DummyEmbedder); !ok {
		t.Fatalf("expected AutoEmbedder to fall back to DummyEmbedder, got %T", embedder)
	}
}
