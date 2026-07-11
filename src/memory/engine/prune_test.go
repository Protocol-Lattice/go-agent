package engine

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/Protocol-Lattice/go-agent/src/memory/model"
	storepkg "github.com/Protocol-Lattice/go-agent/src/memory/store"
)

type recordingDeleteStore struct {
	*storepkg.InMemoryStore
	deleteErr             error
	batchSizes            []int
	spoolDir              string
	iterating             bool
	deleteDuringIteration bool
	spoolCreated          bool
	spoolObservationErr   error
}

func (s *recordingDeleteStore) DeleteMemory(ctx context.Context, ids []int64) error {
	s.batchSizes = append(s.batchSizes, len(ids))
	if s.iterating {
		s.deleteDuringIteration = true
		return errors.New("DeleteMemory called while Iterate is active")
	}
	if s.deleteErr != nil {
		return s.deleteErr
	}
	return s.InMemoryStore.DeleteMemory(ctx, ids)
}

func (s *recordingDeleteStore) Iterate(ctx context.Context, fn func(model.MemoryRecord) bool) error {
	s.iterating = true
	defer func() { s.iterating = false }()
	return s.InMemoryStore.Iterate(ctx, func(rec model.MemoryRecord) bool {
		keepGoing := fn(rec)
		s.observeSpool()
		return keepGoing
	})
}

func (s *recordingDeleteStore) observeSpool() {
	if s.spoolDir == "" || s.spoolObservationErr != nil {
		return
	}
	entries, err := os.ReadDir(s.spoolDir)
	if err != nil {
		s.spoolObservationErr = err
		return
	}
	for _, entry := range entries {
		if !strings.HasPrefix(entry.Name(), pruneDeletionSpoolPrefix) {
			continue
		}
		s.spoolCreated = true
	}
}

type iterateErrorStore struct {
	*storepkg.InMemoryStore
	err         error
	deleteCalls int
}

func (s *iterateErrorStore) Iterate(ctx context.Context, fn func(model.MemoryRecord) bool) error {
	called := false
	if err := s.InMemoryStore.Iterate(ctx, func(rec model.MemoryRecord) bool {
		called = true
		return fn(rec) && !called
	}); err != nil {
		return err
	}
	return s.err
}

func (s *iterateErrorStore) DeleteMemory(ctx context.Context, ids []int64) error {
	s.deleteCalls++
	return s.InMemoryStore.DeleteMemory(ctx, ids)
}

func TestPruneLargeDeletionBatchDoesNotDeadlock(t *testing.T) {
	store := &recordingDeleteStore{InMemoryStore: storepkg.NewInMemoryStore()}
	populateExpiredRecords(t, store, pruneDeleteBatchSize+1)

	now := time.Now().UTC().Add(2 * time.Hour)
	engine := NewEngine(store, Options{
		TTL:   time.Hour,
		Clock: func() time.Time { return now },
	})

	done := make(chan error, 1)
	go func() {
		done <- engine.Prune(context.Background())
	}()

	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("Prune returned error: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Prune deadlocked while deleting a full batch during iteration")
	}

	wantBatches := []int{pruneDeleteBatchSize, 1}
	if fmt.Sprint(store.batchSizes) != fmt.Sprint(wantBatches) {
		t.Fatalf("unexpected deletion batches: got %v, want %v", store.batchSizes, wantBatches)
	}
	remaining, err := store.Count(context.Background())
	if err != nil {
		t.Fatalf("Count returned error: %v", err)
	}
	if remaining != 0 {
		t.Fatalf("expected all expired records to be removed, got %d", remaining)
	}
	snapshot := engine.MetricsSnapshot()
	if snapshot.Pruned != pruneDeleteBatchSize+1 || snapshot.TTLExpired != pruneDeleteBatchSize+1 {
		t.Fatalf("unexpected prune metrics: %#v", snapshot)
	}
}

func TestPruneSpoolsLargeDeletionBatchAndCleansUp(t *testing.T) {
	spoolDir := t.TempDir()
	setPruneTempDir(t, spoolDir)
	store := &recordingDeleteStore{
		InMemoryStore: storepkg.NewInMemoryStore(),
		spoolDir:      spoolDir,
	}
	const extraRecords = 3
	count := 2*pruneDeleteBatchSize + extraRecords
	populateExpiredRecords(t, store, count)

	now := time.Now().UTC().Add(2 * time.Hour)
	engine := NewEngine(store, Options{
		TTL:   time.Hour,
		Clock: func() time.Time { return now },
	})

	if err := engine.Prune(context.Background()); err != nil {
		t.Fatalf("Prune returned error: %v", err)
	}
	if store.spoolObservationErr != nil {
		t.Fatalf("observing deletion spool: %v", store.spoolObservationErr)
	}
	if store.deleteDuringIteration {
		t.Fatal("DeleteMemory was called while Iterate was active")
	}
	if !store.spoolCreated {
		t.Fatal("expected a deletion spool to be created during iteration")
	}
	wantBatches := []int{pruneDeleteBatchSize, pruneDeleteBatchSize, extraRecords}
	if fmt.Sprint(store.batchSizes) != fmt.Sprint(wantBatches) {
		t.Fatalf("unexpected deletion batches: got %v, want %v", store.batchSizes, wantBatches)
	}
	assertPruneSpoolRemoved(t, spoolDir)
}

func TestPruneStopsAndReturnsDeleteError(t *testing.T) {
	wantErr := errors.New("delete failed")
	store := &recordingDeleteStore{
		InMemoryStore: storepkg.NewInMemoryStore(),
		deleteErr:     wantErr,
	}
	populateExpiredRecords(t, store, pruneDeleteBatchSize+1)

	now := time.Now().UTC().Add(2 * time.Hour)
	engine := NewEngine(store, Options{
		TTL:   time.Hour,
		Clock: func() time.Time { return now },
	})

	err := engine.Prune(context.Background())
	if !errors.Is(err, wantErr) {
		t.Fatalf("Prune error = %v, want %v", err, wantErr)
	}
	if len(store.batchSizes) != 1 || store.batchSizes[0] != pruneDeleteBatchSize {
		t.Fatalf("expected pruning to stop after the first failed batch, got %v", store.batchSizes)
	}
	snapshot := engine.MetricsSnapshot()
	if snapshot.Pruned != 0 || snapshot.TTLExpired != 0 {
		t.Fatalf("failed deletions must not update prune metrics: %#v", snapshot)
	}
	remaining, countErr := store.Count(context.Background())
	if countErr != nil {
		t.Fatalf("Count returned error: %v", countErr)
	}
	if remaining != pruneDeleteBatchSize+1 {
		t.Fatalf("failed deletion changed the store: got %d records", remaining)
	}
}

func TestPruneCleansDeletionSpoolAfterDeleteError(t *testing.T) {
	spoolDir := t.TempDir()
	setPruneTempDir(t, spoolDir)
	wantErr := errors.New("delete failed")
	store := &recordingDeleteStore{
		InMemoryStore: storepkg.NewInMemoryStore(),
		deleteErr:     wantErr,
	}
	populateExpiredRecords(t, store, pruneDeleteBatchSize+1)

	now := time.Now().UTC().Add(2 * time.Hour)
	engine := NewEngine(store, Options{
		TTL:   time.Hour,
		Clock: func() time.Time { return now },
	})

	err := engine.Prune(context.Background())
	if !errors.Is(err, wantErr) {
		t.Fatalf("Prune error = %v, want %v", err, wantErr)
	}
	assertPruneSpoolRemoved(t, spoolDir)
}

func TestPruneCleansDeletionSpoolAfterIterateError(t *testing.T) {
	spoolDir := t.TempDir()
	setPruneTempDir(t, spoolDir)
	wantErr := errors.New("iterate failed")
	store := &iterateErrorStore{
		InMemoryStore: storepkg.NewInMemoryStore(),
		err:           wantErr,
	}
	if err := store.StoreMemory(context.Background(), "session", "expired", nil, nil); err != nil {
		t.Fatalf("StoreMemory returned error: %v", err)
	}

	now := time.Now().UTC().Add(2 * time.Hour)
	engine := NewEngine(store, Options{
		TTL:   time.Hour,
		Clock: func() time.Time { return now },
	})

	err := engine.Prune(context.Background())
	if !errors.Is(err, wantErr) {
		t.Fatalf("Prune error = %v, want %v", err, wantErr)
	}
	if store.deleteCalls != 0 {
		t.Fatalf("DeleteMemory calls = %d, want 0", store.deleteCalls)
	}
	assertPruneSpoolRemoved(t, spoolDir)
}

func populateExpiredRecords(t *testing.T, store *recordingDeleteStore, count int) {
	t.Helper()
	ctx := context.Background()
	for i := 0; i < count; i++ {
		content := fmt.Sprintf("expired-record-%d", i)
		if err := store.StoreMemory(ctx, "session", content, nil, nil); err != nil {
			t.Fatalf("StoreMemory(%d) returned error: %v", i, err)
		}
	}
}

func assertPruneSpoolRemoved(t *testing.T, spoolDir string) {
	t.Helper()
	entries, err := os.ReadDir(spoolDir)
	if err != nil {
		t.Fatalf("ReadDir(%q): %v", spoolDir, err)
	}
	for _, entry := range entries {
		if strings.HasPrefix(entry.Name(), pruneDeletionSpoolPrefix) {
			t.Fatalf("prune deletion spool %q was not removed", entry.Name())
		}
	}
}

func setPruneTempDir(t *testing.T, spoolDir string) {
	t.Helper()
	// os.CreateTemp resolves different environment variables on Unix and Windows.
	t.Setenv("TMPDIR", spoolDir)
	t.Setenv("TMP", spoolDir)
	t.Setenv("TEMP", spoolDir)
}
