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
	benchmarkEngineRetrieve(b, 500)
}

func BenchmarkEngineRetrieveCardinality(b *testing.B) {
	for _, recordCount := range []int{1_000, 10_000} {
		b.Run(fmt.Sprintf("records=%d", recordCount), func(b *testing.B) {
			benchmarkEngineRetrieve(b, recordCount)
		})
	}
}

func benchmarkEngineRetrieve(b *testing.B, recordCount int) {
	store := storepkg.NewInMemoryStore()
	opts := Options{HalfLife: time.Hour, TTL: 720 * time.Hour}
	engine := NewEngine(store, opts).WithEmbedder(embedpkg.DummyEmbedder{})
	ctx := context.Background()

	for i := 0; i < recordCount; i++ {
		content := fmt.Sprintf("Document %d about system design", i)
		embedding := make([]float32, 64)
		embedding[i%len(embedding)] = 1
		embedding[(i/len(embedding))%len(embedding)] += 0.25
		if err := store.StoreMemory(ctx, "bench", content, map[string]any{"source": "docs"}, embedding); err != nil {
			b.Fatalf("seed memory: %v", err)
		}
	}
	if count, err := store.Count(ctx); err != nil {
		b.Fatalf("count seeded memories: %v", err)
	} else if count != recordCount {
		b.Fatalf("seeded %d memories, want %d", count, recordCount)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := engine.Retrieve(ctx, "bench", "system design", 5); err != nil {
			b.Fatalf("retrieve: %v", err)
		}
	}
}
