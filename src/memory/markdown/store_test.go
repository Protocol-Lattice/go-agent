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
