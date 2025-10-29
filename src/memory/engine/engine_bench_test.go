package engine

import (
	"context"
	"fmt"
	"testing"
	"time"

	embedpkg "github.com/Protocol-Lattice/go-agent/src/memory/embed"
	storepkg "github.com/Protocol-Lattice/go-agent/src/memory/store"
)

func BenchmarkEngineRetrieve(b *testing.B) {
	store := storepkg.NewInMemoryStore()
	opts := Options{HalfLife: time.Hour, TTL: 720 * time.Hour}
	engine := NewEngine(store, opts).WithEmbedder(embedpkg.DummyEmbedder{})
	ctx := context.Background()

	for i := 0; i < 500; i++ {
		content := fmt.Sprintf("Document %d about system design", i)
		engine.Store(ctx, "bench", content, map[string]any{"source": "docs"})
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := engine.Retrieve(ctx, "system design", 5); err != nil {
			b.Fatalf("retrieve: %v", err)
		}
	}
}
