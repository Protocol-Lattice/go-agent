package store

import (
	"context"
	"sort"
	"strconv"
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

func TestInMemoryStoreSearchMemoryReturnsTopKInScoreOrder(t *testing.T) {
	store := NewInMemoryStore()
	ctx := context.Background()
	for _, record := range []struct {
		content   string
		embedding []float32
	}{
		{content: "least", embedding: []float32{0, 1}},
		{content: "second", embedding: []float32{0.8, 0.6}},
		{content: "most", embedding: []float32{1, 0}},
		{content: "negative", embedding: []float32{-1, 0}},
	} {
		if err := store.StoreMemory(ctx, "session", record.content, nil, record.embedding); err != nil {
			t.Fatalf("StoreMemory(%q) returned error: %v", record.content, err)
		}
	}

	results, err := store.SearchMemory(ctx, "session", []float32{1, 0}, 2)
	if err != nil {
		t.Fatalf("SearchMemory returned error: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("expected two results, got %d", len(results))
	}
	if results[0].Content != "most" || results[1].Content != "second" {
		t.Fatalf("unexpected top-k results: %#v", results)
	}
	if results[0].Score < results[1].Score {
		t.Fatalf("results are not descending by score: %#v", results)
	}
}

func TestInMemoryStoreSearchMemoryMatchesFullSortReference(t *testing.T) {
	store := NewInMemoryStore()
	ctx := context.Background()
	for i := 1; i <= 128; i++ {
		if err := store.StoreMemory(ctx, "session", "record-"+strconv.Itoa(i), nil, []float32{float32(i), 1}); err != nil {
			t.Fatalf("StoreMemory(%d) returned error: %v", i, err)
		}
	}
	query := []float32{1, 0}
	for _, limit := range []int{1, 3, 8, 128, 200} {
		got, err := store.SearchMemory(ctx, "session", query, limit)
		if err != nil {
			t.Fatalf("SearchMemory(limit=%d) returned error: %v", limit, err)
		}
		want := fullSortSearchMemory(store, "session", query, limit)
		if len(got) != len(want) {
			t.Fatalf("limit %d returned %d records, want %d", limit, len(got), len(want))
		}
		for i := range want {
			if got[i].ID != want[i].ID || got[i].Score != want[i].Score {
				t.Fatalf("limit %d result %d = %#v, want %#v", limit, i, got[i], want[i])
			}
		}
	}
}

func BenchmarkInMemoryStoreSearchMemory(b *testing.B) {
	const (
		recordCount = 10_000
		limit       = 8
	)
	store := NewInMemoryStore()
	ctx := context.Background()
	for i := 0; i < recordCount; i++ {
		embedding := make([]float32, 64)
		embedding[i%len(embedding)] = 1
		embedding[(i/len(embedding))%len(embedding)] += 0.25
		if err := store.StoreMemory(ctx, "session", "record", nil, embedding); err != nil {
			b.Fatalf("seed memory: %v", err)
		}
	}
	query := make([]float32, 64)
	query[0] = 1

	b.Run("top-k", func(b *testing.B) {
		var results []model.MemoryRecord
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			var err error
			results, err = store.SearchMemory(ctx, "session", query, limit)
			if err != nil {
				b.Fatal(err)
			}
		}
		if len(results) != limit {
			b.Fatalf("got %d results, want %d", len(results), limit)
		}
	})

	b.Run("full-sort-reference", func(b *testing.B) {
		var results []model.MemoryRecord
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			results = fullSortSearchMemory(store, "session", query, limit)
		}
		if len(results) != limit {
			b.Fatalf("got %d results, want %d", len(results), limit)
		}
	})
}

func fullSortSearchMemory(store *InMemoryStore, sessionID string, query []float32, limit int) []model.MemoryRecord {
	store.mu.RLock()
	defer store.mu.RUnlock()

	type scored struct {
		record model.MemoryRecord
		score  float64
	}
	queryVector := model.NewCosineQuery(query)
	scoredRecords := make([]scored, 0, len(store.records))
	for _, record := range store.records {
		if sessionID != "" && record.SessionID != sessionID {
			continue
		}
		score := queryVector.MaxSimilarity(record)
		record.Score = score
		scoredRecords = append(scoredRecords, scored{record: record, score: score})
	}
	sort.Slice(scoredRecords, func(i, j int) bool {
		return scoredRecords[i].score > scoredRecords[j].score
	})
	if len(scoredRecords) > limit {
		scoredRecords = scoredRecords[:limit]
	}
	results := make([]model.MemoryRecord, len(scoredRecords))
	for i, record := range scoredRecords {
		results[i] = record.record
	}
	return results
}
