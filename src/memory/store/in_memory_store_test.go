package store

import (
	"context"
	"testing"

	"github.com/Protocol-Lattice/go-agent/src/memory/model"
)

func TestInMemoryStoreUsesMatrixVectorWhenEmbeddingMissing(t *testing.T) {
	store := NewInMemoryStore()
	metadata := map[string]any{
		model.EmbeddingMatrixKey: []any{
			[]any{0.1, 0.2, 0.3},
			[]any{},
		},
	}

	if err := store.StoreMemory(context.Background(), "session", "content", metadata, nil); err != nil {
		t.Fatalf("StoreMemory returned error: %v", err)
	}

	if len(store.records) != 1 {
		t.Fatalf("expected 1 record, got %d", len(store.records))
	}

	var rec model.MemoryRecord
	for _, r := range store.records {
		rec = r
		break
	}

	if len(rec.Embedding) == 0 {
		t.Fatalf("expected fallback embedding to be populated")
	}
	if len(rec.EmbeddingMatrix) != 1 {
		t.Fatalf("expected sanitized embedding matrix, got %#v", rec.EmbeddingMatrix)
	}
	if rec.Embedding[0] != 0.1 || rec.Embedding[1] != 0.2 || rec.Embedding[2] != 0.3 {
		t.Fatalf("unexpected fallback embedding: %#v", rec.Embedding)
	}
}
