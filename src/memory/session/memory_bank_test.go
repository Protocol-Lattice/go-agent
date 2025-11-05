package session

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/Protocol-Lattice/go-agent/src/memory/embed"
	"github.com/Protocol-Lattice/go-agent/src/memory/model"
	"github.com/Protocol-Lattice/go-agent/src/memory/store"
	json "github.com/alpkeskin/gotoon"
)

type stubVectorStore struct {
	mu           sync.Mutex
	stored       []storedMemory
	searchResp   []model.MemoryRecord
	storeErr     error
	searchErr    error
	createSchema []string
	closed       bool
}

type storedMemory struct {
	sessionID string
	content   string
	metadata  map[string]any
	embedding []float32
}

func (s *stubVectorStore) StoreMemory(_ context.Context, sessionID, content string, metadata map[string]any, embedding []float32) error {
	if s.storeErr != nil {
		return s.storeErr
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	cp := make(map[string]any, len(metadata))
	for k, v := range metadata {
		cp[k] = v
	}
	s.stored = append(s.stored, storedMemory{
		sessionID: sessionID,
		content:   content,
		metadata:  cp,
		embedding: append([]float32(nil), embedding...),
	})
	return nil
}

func (s *stubVectorStore) SearchMemory(_ context.Context, _ []float32, _ int) ([]model.MemoryRecord, error) {
	if s.searchErr != nil {
		return nil, s.searchErr
	}
	return s.searchResp, nil
}

func (s *stubVectorStore) UpdateEmbedding(context.Context, int64, []float32, time.Time) error {
	return nil
}
func (s *stubVectorStore) DeleteMemory(context.Context, []int64) error                  { return nil }
func (s *stubVectorStore) Iterate(context.Context, func(model.MemoryRecord) bool) error { return nil }
func (s *stubVectorStore) Count(context.Context) (int, error)                           { return 0, nil }

func (s *stubVectorStore) CreateSchema(_ context.Context, schemaPath string) error {
	s.createSchema = append(s.createSchema, schemaPath)
	return nil
}

func (s *stubVectorStore) Close() error {
	s.closed = true
	return nil
}

var _ store.VectorStore = (*stubVectorStore)(nil)
var _ store.SchemaInitializer = (*stubVectorStore)(nil)

func TestMemoryBankStoreMemoryAddsSpace(t *testing.T) {
	svs := &stubVectorStore{}
	bank := NewMemoryBankWithStore(svs)
	err := bank.StoreMemory(context.Background(), "session-1", "content", "", []float32{1, 2})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(svs.stored) != 1 {
		t.Fatalf("expected one stored record, got %d", len(svs.stored))
	}
	stored := svs.stored[0]
	if stored.metadata["space"] != "session-1" {
		t.Fatalf("expected space metadata to default to session, got %v", stored.metadata["space"])
	}
}

func TestMemoryBankStoreMemoryRespectsProvidedSpace(t *testing.T) {
	svs := &stubVectorStore{}
	bank := NewMemoryBankWithStore(svs)
	meta := map[string]any{"space": "shared"}
	metaBytes, _ := json.Marshal(meta)
	err := bank.StoreMemory(context.Background(), "session-1", "content", string(metaBytes), []float32{1})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if svs.stored[0].metadata["space"] != "shared" {
		t.Fatalf("expected provided space to be preserved, got %v", svs.stored[0].metadata["space"])
	}
}

func TestMemoryBankSearchMemoryNilStore(t *testing.T) {
	var bank *MemoryBank
	if results, err := bank.SearchMemory(context.Background(), nil, 5); err != nil || results != nil {
		t.Fatalf("expected nil results and no error, got %v, %v", results, err)
	}
	bank = &MemoryBank{}
	if results, err := bank.SearchMemory(context.Background(), nil, 5); err != nil || results != nil {
		t.Fatalf("expected nil results and no error, got %v, %v", results, err)
	}
}

func TestMemoryBankCreateSchemaAndClose(t *testing.T) {
	svs := &stubVectorStore{}
	bank := NewMemoryBankWithStore(svs)
	if err := bank.CreateSchema(context.Background(), "schema.sql"); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(svs.createSchema) != 1 || svs.createSchema[0] != "schema.sql" {
		t.Fatalf("expected schema initializer to be called, got %v", svs.createSchema)
	}
	if err := bank.Close(); err != nil {
		t.Fatalf("unexpected error closing bank: %v", err)
	}
	if !svs.closed {
		t.Fatal("expected Close to be forwarded to store")
	}
}

type stubEmbedder struct {
	vec []float32
	err error
}

func (s stubEmbedder) Embed(_ context.Context, text string) ([]float32, error) {
	if s.err != nil {
		return nil, s.err
	}
	if s.vec != nil {
		return s.vec, nil
	}
	return []float32{float32(len(text))}, nil
}

func TestSessionMemoryAddAndFlush(t *testing.T) {
	svs := &stubVectorStore{}
	bank := NewMemoryBankWithStore(svs)
	sm := NewSessionMemory(bank, 2)
	sm.WithEmbedder(stubEmbedder{vec: []float32{1}})

	sm.AddShortTerm("s1", "first", "{}", []float32{1})
	sm.AddShortTerm("s1", "second", "{}", []float32{2})
	sm.AddShortTerm("s1", "third", "{}", []float32{3})
	if len(sm.shortTerm["s1"]) != 2 {
		t.Fatalf("expected short-term buffer to respect size, got %d", len(sm.shortTerm["s1"]))
	}
	if err := sm.FlushToLongTerm(context.Background(), "s1"); err != nil {
		t.Fatalf("unexpected flush error: %v", err)
	}
	if len(sm.shortTerm["s1"]) != 0 {
		t.Fatal("expected short-term cache to be cleared after flush")
	}
	if len(svs.stored) != 2 {
		t.Fatalf("expected two records to be stored, got %d", len(svs.stored))
	}
}

func TestSessionMemoryEmbedFallback(t *testing.T) {
	sm := NewSessionMemory(&MemoryBank{}, 1)
	sm.Embedder = stubEmbedder{err: assertErr{}}
	vec, err := sm.Embed(context.Background(), "text")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	dummy := embed.DummyEmbedding("text")
	if len(vec) != len(dummy) {
		t.Fatalf("expected dummy embedding length %d, got %d", len(dummy), len(vec))
	}
}

type assertErr struct{}

func (assertErr) Error() string { return "assert" }

func TestSessionMemoryRetrieveContextCombinesSources(t *testing.T) {
	svs := &stubVectorStore{searchResp: []model.MemoryRecord{{Content: "long"}}}
	bank := NewMemoryBankWithStore(svs)
	sm := NewSessionMemory(bank, 2)
	sm.WithEmbedder(stubEmbedder{vec: []float32{1}})

	sm.AddShortTerm("s1", "short", "{}", []float32{1})
	records, err := sm.RetrieveContext(context.Background(), "s1", "query", 5)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(records) != 2 {
		t.Fatalf("expected two combined records, got %d", len(records))
	}
	if records[0].Content != "short" || records[1].Content != "long" {
		t.Fatalf("unexpected record ordering: %#v", records)
	}
}
