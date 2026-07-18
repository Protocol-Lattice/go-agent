package markdown

import (
	"context"
	"os"
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
	if records[0].Score < records[1].Score || records[0].Score == 0 {
		t.Fatalf("search scores were not preserved: %#v", records)
	}
}

func TestStoreSearchCacheInvalidatesAfterSave(t *testing.T) {
	ctx := context.Background()
	store, err := NewStore(t.TempDir())
	if err != nil {
		t.Fatalf("NewStore returned error: %v", err)
	}
	if err := store.StoreMemory(ctx, "session", "first", nil, []float32{1, 0}); err != nil {
		t.Fatalf("StoreMemory(first) returned error: %v", err)
	}
	if _, err := store.SearchMemory(ctx, "session", []float32{1, 0}, 8); err != nil {
		t.Fatalf("initial SearchMemory returned error: %v", err)
	}
	if len(store.cache) == 0 {
		t.Fatal("expected initial search to populate the file cache")
	}

	if err := store.StoreMemory(ctx, "session", "second", nil, []float32{0.8, 0.6}); err != nil {
		t.Fatalf("StoreMemory(second) returned error: %v", err)
	}
	records, err := store.SearchMemory(ctx, "session", []float32{1, 0}, 8)
	if err != nil {
		t.Fatalf("SearchMemory after save returned error: %v", err)
	}
	if len(records) != 2 {
		t.Fatalf("SearchMemory returned %d records after save, want 2", len(records))
	}
}

func TestStoreSearchCacheDetectsExternalFileChanges(t *testing.T) {
	ctx := context.Background()
	store, err := NewStore(t.TempDir())
	if err != nil {
		t.Fatalf("NewStore returned error: %v", err)
	}
	if err := store.StoreMemory(ctx, "session", "first", nil, []float32{1, 0}); err != nil {
		t.Fatalf("StoreMemory returned error: %v", err)
	}
	if _, err := store.SearchMemory(ctx, "session", []float32{1, 0}, 8); err != nil {
		t.Fatalf("initial SearchMemory returned error: %v", err)
	}

	path, err := store.pathFor("sessions", "session")
	if err != nil {
		t.Fatalf("pathFor returned error: %v", err)
	}
	file, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		t.Fatalf("open cached markdown file: %v", err)
	}
	external := Record{SessionID: "session", Content: "external", Embedding: []float32{0.8, 0.6}}
	if _, err := file.WriteString(renderBlock(external)); err != nil {
		_ = file.Close()
		t.Fatalf("append external record: %v", err)
	}
	if err := file.Close(); err != nil {
		t.Fatalf("close cached markdown file: %v", err)
	}

	records, err := store.SearchMemory(ctx, "session", []float32{1, 0}, 8)
	if err != nil {
		t.Fatalf("SearchMemory after external change returned error: %v", err)
	}
	if len(records) != 2 {
		t.Fatalf("SearchMemory returned %d records after external change, want 2", len(records))
	}
}

func TestDeleteMemoryProcessesIDsInOnePass(t *testing.T) {
	ctx := context.Background()
	store, err := NewStore(t.TempDir())
	if err != nil {
		t.Fatalf("NewStore returned error: %v", err)
	}
	for _, content := range []string{"first", "second", "keep"} {
		if err := store.StoreMemory(ctx, "session", content, nil, []float32{1, 0}); err != nil {
			t.Fatalf("StoreMemory(%q) returned error: %v", content, err)
		}
	}
	records, err := store.List(ctx, "sessions", "session")
	if err != nil {
		t.Fatalf("List returned error: %v", err)
	}
	if len(records) != 3 {
		t.Fatalf("List returned %d records, want 3", len(records))
	}
	if err := store.DeleteMemory(ctx, []int64{records[0].NumID, records[1].NumID}); err != nil {
		t.Fatalf("DeleteMemory returned error: %v", err)
	}
	remaining, err := store.List(ctx, "sessions", "session")
	if err != nil {
		t.Fatalf("List after delete returned error: %v", err)
	}
	if len(remaining) != 1 || remaining[0].Content != "keep" {
		t.Fatalf("unexpected records after batch delete: %#v", remaining)
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
