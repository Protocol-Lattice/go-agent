package store

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/Protocol-Lattice/agent/core/memory/model"
)

type runCall struct {
	query  string
	params map[string]any
}

type fakeDriver struct {
	writeSession *fakeSession
	readSession  *fakeSession
	configs      []Neo4jSessionConfig
	closed       bool
	closeErr     error
}

func (d *fakeDriver) NewSession(_ context.Context, config Neo4jSessionConfig) (neo4jSession, error) {
	d.configs = append(d.configs, config)
	switch config.AccessMode {
	case AccessModeWrite:
		if d.writeSession == nil {
			d.writeSession = &fakeSession{}
		}
		return d.writeSession, nil
	case AccessModeRead:
		if d.readSession == nil {
			d.readSession = &fakeSession{}
		}
		return d.readSession, nil
	default:
		return nil, errors.New("unknown access mode")
	}
}

func (d *fakeDriver) Close(context.Context) error {
	d.closed = true
	return d.closeErr
}

type fakeSession struct {
	tx       *fakeTx
	runCalls []runCall
	runErr   error
	result   neo4jResult
	closed   bool
}

func (s *fakeSession) BeginTransaction(context.Context) (neo4jTransaction, error) {
	if s.tx == nil {
		s.tx = &fakeTx{}
	}
	return s.tx, nil
}

func (s *fakeSession) Run(_ context.Context, query string, params map[string]any) (neo4jResult, error) {
	s.runCalls = append(s.runCalls, runCall{query: query, params: params})
	if s.runErr != nil {
		return nil, s.runErr
	}
	if s.result != nil {
		return s.result, nil
	}
	return &fakeResult{}, nil
}

func (s *fakeSession) Close(context.Context) error {
	s.closed = true
	return nil
}

type fakeTx struct {
	runs        []runCall
	runErrs     []error
	commitErr   error
	rollbackErr error
	committed   bool
	rolledBack  bool
	closed      bool
}

func (tx *fakeTx) Run(_ context.Context, query string, params map[string]any) (neo4jResult, error) {
	tx.runs = append(tx.runs, runCall{query: query, params: params})
	if len(tx.runErrs) > 0 {
		err := tx.runErrs[0]
		tx.runErrs = tx.runErrs[1:]
		if err != nil {
			return nil, err
		}
	}
	return &fakeResult{}, nil
}

func (tx *fakeTx) Commit(context.Context) error {
	if tx.commitErr != nil {
		return tx.commitErr
	}
	tx.committed = true
	return nil
}

func (tx *fakeTx) Rollback(context.Context) error {
	tx.rolledBack = true
	return tx.rollbackErr
}

func (tx *fakeTx) Close(context.Context) error {
	tx.closed = true
	return nil
}

type fakeResult struct {
	records []map[string]any
	idx     int
	err     error
	closed  bool
}

func (r *fakeResult) Next(_ context.Context) bool {
	if r.idx >= len(r.records) {
		return false
	}
	r.idx++
	return true
}

func (r *fakeResult) Record() neo4jRecord {
	if r.idx == 0 || r.idx > len(r.records) {
		return fakeRecord(nil)
	}
	return fakeRecord(r.records[r.idx-1])
}

func (r *fakeResult) Err() error { return r.err }

func (r *fakeResult) Close(context.Context) error {
	r.closed = true
	return nil
}

type fakeRecord map[string]any

func (r fakeRecord) Get(key string) (any, bool) {
	if r == nil {
		return nil, false
	}
	v, ok := r[key]
	return v, ok
}

type schemaStore struct {
	*InMemoryStore
	schemaCalls int
}

func (s *schemaStore) CreateSchema(ctx context.Context, schemaPath string) error {
	s.schemaCalls++
	return nil
}

type closableInMemory struct {
	*InMemoryStore
	closed bool
}

func (c *closableInMemory) Close() error {
	c.closed = true
	return nil
}

func TestNewNeo4jStoreValidation(t *testing.T) {
	base := NewInMemoryStore()
	if _, err := NewNeo4jStore(nil, &fakeDriver{}, ""); err == nil {
		t.Fatal("expected error when base store is nil")
	}
	if _, err := NewNeo4jStore(base, nil, ""); err == nil {
		t.Fatal("expected error when driver is nil")
	}
}

func TestNeo4jStoreUpsertGraph(t *testing.T) {
	base := NewInMemoryStore()
	driver := &fakeDriver{writeSession: &fakeSession{tx: &fakeTx{}}}
	store, err := NewNeo4jStore(base, driver, "neo")
	if err != nil {
		t.Fatalf("new store: %v", err)
	}
	fixedNow := time.Date(2024, 5, 1, 12, 0, 0, 0, time.UTC)
	store.nowFn = func() time.Time { return fixedNow }
	record := model.MemoryRecord{
		ID:           42,
		SessionID:    "session-1",
		Content:      "memory",
		Metadata:     "",
		Importance:   0.75,
		Source:       "test",
		Summary:      "summary",
		CreatedAt:    fixedNow.Add(-time.Hour),
		LastEmbedded: fixedNow.Add(-30 * time.Minute),
	}
	edges := []model.GraphEdge{
		{Target: 7, Type: model.EdgeFollows},
		{Target: 0, Type: model.EdgeExplains}, // invalid and skipped
	}
	if err := store.UpsertGraph(context.Background(), record, edges); err != nil {
		t.Fatalf("upsert graph: %v", err)
	}
	if len(driver.configs) == 0 || driver.configs[0].AccessMode != AccessModeWrite {
		t.Fatalf("expected write session, configs=%v", driver.configs)
	}
	tx := driver.writeSession.tx
	if tx == nil {
		t.Fatalf("transaction not created")
	}
	if !tx.committed {
		t.Fatalf("transaction not committed")
	}
	if len(tx.runs) != 3 {
		t.Fatalf("expected 3 queries, got %d", len(tx.runs))
	}
	if tx.runs[0].query != neo4jUpsertNodeCypher {
		t.Fatalf("unexpected first query: %s", tx.runs[0].query)
	}
	nodeParams := tx.runs[0].params
	if got := nodeParams["metadata"].(string); got != "{}" {
		t.Fatalf("expected metadata fallback, got %q", got)
	}
	if got := nodeParams["space"].(string); got != "session-1" {
		t.Fatalf("expected space fallback to session, got %q", got)
	}
	if tx.runs[1].query != "MATCH (m:Memory {id: $id})-[r:RELATED_TO]->() DELETE r" {
		t.Fatalf("unexpected delete query: %s", tx.runs[1].query)
	}
	if tx.runs[2].query != neo4jUpsertEdgeCypher {
		t.Fatalf("unexpected edge query: %s", tx.runs[2].query)
	}
	if got := tx.runs[2].params["edge_type"].(string); got != string(model.EdgeFollows) {
		t.Fatalf("unexpected edge type: %s", got)
	}
	if driver.writeSession.tx.rolledBack {
		t.Fatalf("transaction should not roll back on success")
	}
}

func TestNeo4jStoreNeighborhood(t *testing.T) {
	base := NewInMemoryStore()
	driver := &fakeDriver{readSession: &fakeSession{}}
	store, err := NewNeo4jStore(base, driver, "neo")
	if err != nil {
		t.Fatalf("new store: %v", err)
	}
	now := time.Now().UTC().Format(time.RFC3339Nano)
	driver.readSession.result = &fakeResult{
		records: []map[string]any{
			{
				"id":            int64(7),
				"session_id":    "session-1",
				"space":         "",
				"content":       "context",
				"metadata":      `{"importance":0.9}`,
				"importance":    0.9,
				"source":        "system",
				"summary":       "sum",
				"created_at":    now,
				"last_embedded": now,
			},
		},
	}
	neighbors, err := store.Neighborhood(context.Background(), []int64{42}, 2, 5)
	if err != nil {
		t.Fatalf("neighborhood: %v", err)
	}
	if len(neighbors) != 1 {
		t.Fatalf("expected 1 neighbor, got %d", len(neighbors))
	}
	rec := neighbors[0]
	if rec.ID != 7 {
		t.Fatalf("unexpected neighbor id: %d", rec.ID)
	}
	if rec.Space != "session-1" {
		t.Fatalf("expected space fallback to session, got %q", rec.Space)
	}
	if rec.Importance == 0 {
		t.Fatalf("expected hydrated importance")
	}
	if rec.Content != "context" {
		t.Fatalf("unexpected content: %s", rec.Content)
	}
	if rec.Metadata == "" {
		t.Fatalf("expected metadata string")
	}
	if len(driver.configs) == 0 || driver.configs[0].AccessMode != AccessModeRead {
		t.Fatalf("expected read session config, got %#v", driver.configs)
	}
}

func TestNeo4jStoreCreateSchema(t *testing.T) {
	base := &schemaStore{InMemoryStore: NewInMemoryStore()}
	driver := &fakeDriver{writeSession: &fakeSession{}}
	store, err := NewNeo4jStore(base, driver, "neo")
	if err != nil {
		t.Fatalf("new store: %v", err)
	}
	if err := store.CreateSchema(context.Background(), "ignored"); err != nil {
		t.Fatalf("create schema: %v", err)
	}
	if base.schemaCalls != 1 {
		t.Fatalf("expected schema initializer to be called once")
	}
	if len(driver.writeSession.runCalls) != 3 {
		t.Fatalf("expected 3 schema queries, got %d", len(driver.writeSession.runCalls))
	}
	for i, call := range driver.writeSession.runCalls {
		if call.params != nil {
			t.Fatalf("expected schema query %d without params", i)
		}
	}
}

func TestNeo4jStoreClose(t *testing.T) {
	base := &closableInMemory{InMemoryStore: NewInMemoryStore()}
	driver := &fakeDriver{}
	store, err := NewNeo4jStore(base, driver, "neo")
	if err != nil {
		t.Fatalf("new store: %v", err)
	}
	if err := store.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	if !base.closed {
		t.Fatalf("expected base Close to be called")
	}
	if !driver.closed {
		t.Fatalf("expected driver Close to be called")
	}
}

func TestNeo4jUnavailable(t *testing.T) {
	base := NewInMemoryStore()
	store := &Neo4jStore{base: base}
	if err := store.UpsertGraph(context.Background(), model.MemoryRecord{ID: 1}, nil); !errors.Is(err, ErrNeo4jUnavailable) {
		t.Fatalf("expected ErrNeo4jUnavailable, got %v", err)
	}
	if _, err := store.Neighborhood(context.Background(), []int64{1}, 1, 1); !errors.Is(err, ErrNeo4jUnavailable) {
		t.Fatalf("expected ErrNeo4jUnavailable, got %v", err)
	}
}
