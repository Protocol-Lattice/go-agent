package filestore

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/Protocol-Lattice/go-agent/src/memory/model"
)

func TestFileBackedStorePersistsAndMutatesRecords(t *testing.T) {
	t.Parallel()

	path := filepath.Join(t.TempDir(), "memory.json")
	store := NewFileBackedStore(path)
	ctx := context.Background()

	if err := store.StoreMemory(ctx, "first", "alpha", map[string]any{"source": "test"}, []float32{1, 0}); err != nil {
		t.Fatalf("store alpha: %v", err)
	}
	if err := store.StoreMemory(ctx, "second", "beta", nil, []float32{0, 1}); err != nil {
		t.Fatalf("store beta: %v", err)
	}

	got, err := store.SearchMemory(ctx, "first", []float32{1, 0}, 4)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(got) != 1 || got[0].Content != "alpha" {
		t.Fatalf("search returned %#v", got)
	}

	reloaded := NewFileBackedStore(path)
	count, err := reloaded.Count(ctx)
	if err != nil {
		t.Fatalf("count: %v", err)
	}
	if count != 2 {
		t.Fatalf("count = %d, want 2", count)
	}

	updatedAt := time.Now().UTC().Add(time.Minute)
	if err := reloaded.UpdateEmbedding(ctx, 1, []float32{0.5, 0.5}, updatedAt); err != nil {
		t.Fatalf("update embedding: %v", err)
	}
	if err := reloaded.DeleteMemory(ctx, []int64{2}); err != nil {
		t.Fatalf("delete: %v", err)
	}

	var records []model.MemoryRecord
	if err := reloaded.Iterate(ctx, func(record model.MemoryRecord) bool {
		records = append(records, record)
		return true
	}); err != nil {
		t.Fatalf("iterate: %v", err)
	}
	if len(records) != 1 {
		t.Fatalf("records = %d, want 1", len(records))
	}
	if records[0].ID != 1 || records[0].LastEmbedded != updatedAt {
		t.Fatalf("updated record = %#v", records[0])
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read persistence file: %v", err)
	}
	var persisted []model.MemoryRecord
	if err := json.Unmarshal(data, &persisted); err != nil {
		t.Fatalf("decode persistence file: %v", err)
	}
	if len(persisted) != 1 || persisted[0].ID != 1 {
		t.Fatalf("persisted records = %#v", persisted)
	}
}

func TestFileBackedStoreWritesRecordsInIDOrder(t *testing.T) {
	t.Parallel()

	path := filepath.Join(t.TempDir(), "memory.json")
	store := NewFileBackedStore(path)
	store.records[20] = model.MemoryRecord{ID: 20}
	store.records[3] = model.MemoryRecord{ID: 3}

	if err := store.save(); err != nil {
		t.Fatalf("save: %v", err)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read persistence file: %v", err)
	}
	var records []model.MemoryRecord
	if err := json.Unmarshal(data, &records); err != nil {
		t.Fatalf("decode persistence file: %v", err)
	}
	if len(records) != 2 || records[0].ID != 3 || records[1].ID != 20 {
		t.Fatalf("record order = %#v", records)
	}
}
