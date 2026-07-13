package markdown

import (
	"context"
	"testing"
)

func TestStoreMemoryCanBeLoadedBySessionAfterReopen(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()

	store, err := NewStore(root)
	if err != nil {
		t.Fatalf("NewStore returned error: %v", err)
	}

	embedding := []float32{1, 0, 0}
	if err := store.StoreMemory(ctx, "session-1", "remember the launch checklist", nil, embedding); err != nil {
		t.Fatalf("StoreMemory returned error: %v", err)
	}

	reopened, err := NewStore(root)
	if err != nil {
		t.Fatalf("reopen NewStore returned error: %v", err)
	}

	records, err := reopened.SearchMemory(ctx, "session-1", embedding, 10)
	if err != nil {
		t.Fatalf("SearchMemory returned error: %v", err)
	}
	if len(records) != 1 {
		t.Fatalf("expected 1 loaded record, got %d", len(records))
	}
	if records[0].SessionID != "session-1" {
		t.Fatalf("expected session-1, got %q", records[0].SessionID)
	}
	if records[0].Content != "remember the launch checklist" {
		t.Fatalf("unexpected content: %q", records[0].Content)
	}
}

func TestSearchMemoryWithEmptySessionSearchesAllSessions(t *testing.T) {
	ctx := context.Background()

	store, err := NewStore(t.TempDir())
	if err != nil {
		t.Fatalf("NewStore returned error: %v", err)
	}

	embedding := []float32{0, 1, 0}
	if err := store.StoreMemory(ctx, "session-1", "shared research note", nil, embedding); err != nil {
		t.Fatalf("StoreMemory returned error: %v", err)
	}

	records, err := store.SearchMemory(ctx, "", embedding, 10)
	if err != nil {
		t.Fatalf("SearchMemory returned error: %v", err)
	}
	if len(records) != 1 {
		t.Fatalf("expected 1 record across all sessions, got %d", len(records))
	}
	if records[0].SessionID != "session-1" {
		t.Fatalf("expected session-1, got %q", records[0].SessionID)
	}
}

func TestSearchMemoryReturnsHighestSimilarityFirst(t *testing.T) {
	ctx := context.Background()
	store, err := NewStore(t.TempDir())
	if err != nil {
		t.Fatalf("NewStore returned error: %v", err)
	}

	for _, tc := range []struct {
		content   string
		embedding []float32
	}{
		{content: "orthogonal", embedding: []float32{0, 1}},
		{content: "second", embedding: []float32{0.8, 0.6}},
		{content: "best", embedding: []float32{1, 0}},
		{content: "opposite", embedding: []float32{-1, 0}},
	} {
		if err := store.StoreMemory(ctx, "session", tc.content, nil, tc.embedding); err != nil {
			t.Fatalf("StoreMemory(%q) returned error: %v", tc.content, err)
		}
	}

	records, err := store.SearchMemory(ctx, "session", []float32{1, 0}, 2)
	if err != nil {
		t.Fatalf("SearchMemory returned error: %v", err)
	}
	if len(records) != 2 {
		t.Fatalf("SearchMemory returned %d records, want 2", len(records))
	}
	if records[0].Content != "best" || records[1].Content != "second" {
		t.Fatalf("unexpected result order: %q, %q", records[0].Content, records[1].Content)
	}
}

func BenchmarkStoreSearchMemory(b *testing.B) {
	ctx := context.Background()
	store, err := NewStore(b.TempDir())
	if err != nil {
		b.Fatalf("NewStore returned error: %v", err)
	}

	const recordCount = 2_000
	for i := 0; i < recordCount; i++ {
		embedding := []float32{1, float32(i%200) / 200}
		if err := store.StoreMemory(ctx, "benchmark", "benchmark memory", nil, embedding); err != nil {
			b.Fatalf("StoreMemory returned error: %v", err)
		}
	}

	query := []float32{1, 0}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		records, err := store.SearchMemory(ctx, "benchmark", query, 8)
		if err != nil {
			b.Fatalf("SearchMemory returned error: %v", err)
		}
		if len(records) != 8 {
			b.Fatalf("SearchMemory returned %d records, want 8", len(records))
		}
	}
}
