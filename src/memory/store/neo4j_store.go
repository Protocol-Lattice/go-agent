package store

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/Protocol-Lattice/go-agent/src/memory/model"
)

// Neo4jAccessMode controls whether a session is opened for read or write operations.
type Neo4jAccessMode string

const (
	// AccessModeWrite opens a session with write access.
	AccessModeWrite Neo4jAccessMode = "write"
	// AccessModeRead opens a session with read access.
	AccessModeRead Neo4jAccessMode = "read"
)

// Neo4jSessionConfig mirrors the minimal subset of Neo4j session configuration we require.
type Neo4jSessionConfig struct {
	AccessMode   Neo4jAccessMode
	DatabaseName string
}

// neo4jDriver abstracts the Neo4j driver capabilities used by the store. This allows tests to
// provide lightweight fakes without depending on the real driver package (which is guarded behind
// an optional build tag).
type neo4jDriver interface {
	NewSession(ctx context.Context, config Neo4jSessionConfig) (neo4jSession, error)
	Close(ctx context.Context) error
}

type neo4jSession interface {
	BeginTransaction(ctx context.Context) (neo4jTransaction, error)
	Run(ctx context.Context, query string, params map[string]any) (neo4jResult, error)
	Close(ctx context.Context) error
}

type neo4jTransaction interface {
	Run(ctx context.Context, query string, params map[string]any) (neo4jResult, error)
	Commit(ctx context.Context) error
	Rollback(ctx context.Context) error
	Close(ctx context.Context) error
}

type neo4jResult interface {
	Next(ctx context.Context) bool
	Record() neo4jRecord
	Err() error
	Close(ctx context.Context) error
}

type neo4jRecord interface {
	Get(key string) (any, bool)
}

// Neo4jStore composes an existing VectorStore with a Neo4j-backed knowledge graph implementation.
//
// Vector embeddings and similarity search remain delegated to the base store, while graph specific
// operations are persisted inside Neo4j.
type Neo4jStore struct {
	base     VectorStore
	driver   neo4jDriver
	database string
	nowFn    func() time.Time
}

var (
	_ VectorStore = (*Neo4jStore)(nil)
	_ GraphStore  = (*Neo4jStore)(nil)
)

// ErrNeo4jUnavailable is returned when graph operations are attempted without a configured driver.
var ErrNeo4jUnavailable = errors.New("neo4j driver not configured")

// NewNeo4jStore constructs a store that delegates vector operations to base and uses the provided
// Neo4j driver for graph persistence.
func NewNeo4jStore(base VectorStore, driver neo4jDriver, database string) (*Neo4jStore, error) {
	if base == nil {
		return nil, errors.New("base vector store is nil")
	}
	if driver == nil {
		return nil, errors.New("neo4j driver is nil")
	}
	return &Neo4jStore{base: base, driver: driver, database: database, nowFn: time.Now}, nil
}

// StoreMemory forwards the call to the underlying vector store.
func (s *Neo4jStore) StoreMemory(ctx context.Context, sessionID, content string, metadata map[string]any, embedding []float32) error {
	return s.base.StoreMemory(ctx, sessionID, content, metadata, embedding)
}

// SearchMemory forwards the call to the underlying vector store.
func (s *Neo4jStore) SearchMemory(ctx context.Context, queryEmbedding []float32, limit int) ([]model.MemoryRecord, error) {
	return s.base.SearchMemory(ctx, queryEmbedding, limit)
}

// UpdateEmbedding forwards the call to the underlying vector store.
func (s *Neo4jStore) UpdateEmbedding(ctx context.Context, id int64, embedding []float32, lastEmbedded time.Time) error {
	return s.base.UpdateEmbedding(ctx, id, embedding, lastEmbedded)
}

// DeleteMemory forwards the call to the underlying vector store.
func (s *Neo4jStore) DeleteMemory(ctx context.Context, ids []int64) error {
	return s.base.DeleteMemory(ctx, ids)
}

// Iterate forwards the call to the underlying vector store.
func (s *Neo4jStore) Iterate(ctx context.Context, fn func(model.MemoryRecord) bool) error {
	return s.base.Iterate(ctx, fn)
}

// Count forwards the call to the underlying vector store.
func (s *Neo4jStore) Count(ctx context.Context) (int, error) {
	return s.base.Count(ctx)
}

// CreateSchema delegates to the base store if it exposes SchemaInitializer and ensures Neo4j graph
// constraints are present.
func (s *Neo4jStore) CreateSchema(ctx context.Context, schemaPath string) error {
	if initializer, ok := s.base.(SchemaInitializer); ok {
		if err := initializer.CreateSchema(ctx, schemaPath); err != nil {
			return err
		}
	}
	if s.driver == nil {
		return nil
	}
	session, err := s.driver.NewSession(ctx, Neo4jSessionConfig{AccessMode: AccessModeWrite, DatabaseName: s.database})
	if err != nil {
		return fmt.Errorf("neo4j new session: %w", err)
	}
	defer session.Close(ctx)
	queries := []string{
		"CREATE CONSTRAINT IF NOT EXISTS FOR (m:Memory) REQUIRE m.id IS UNIQUE",
		"CREATE INDEX IF NOT EXISTS FOR (m:Memory) ON (m.space)",
		"CREATE INDEX IF NOT EXISTS FOR ()-[r:RELATED_TO]-() ON (r.target_id)",
	}
	for _, query := range queries {
		res, runErr := session.Run(ctx, query, nil)
		if runErr != nil {
			return fmt.Errorf("neo4j schema query: %w", runErr)
		}
		if res != nil {
			_ = res.Close(ctx)
		}
	}
	return nil
}

// Close releases both the base store (when it implements Close) and the Neo4j driver.
func (s *Neo4jStore) Close() error {
	var errs []string
	if closer, ok := s.base.(interface{ Close() error }); ok {
		if err := closer.Close(); err != nil {
			errs = append(errs, err.Error())
		}
	}
	if s.driver != nil {
		if err := s.driver.Close(context.Background()); err != nil {
			errs = append(errs, err.Error())
		}
	}
	if len(errs) > 0 {
		return errors.New(strings.Join(errs, "; "))
	}
	return nil
}

// UpsertGraph ensures the corresponding Memory node exists in Neo4j and refreshes its outgoing
// relationships.
func (s *Neo4jStore) UpsertGraph(ctx context.Context, record model.MemoryRecord, edges []model.GraphEdge) error {
	if s.driver == nil {
		return ErrNeo4jUnavailable
	}
	if record.ID == 0 {
		return nil
	}
	session, err := s.driver.NewSession(ctx, Neo4jSessionConfig{AccessMode: AccessModeWrite, DatabaseName: s.database})
	if err != nil {
		return fmt.Errorf("neo4j new session: %w", err)
	}
	defer session.Close(ctx)
	tx, err := session.BeginTransaction(ctx)
	if err != nil {
		return fmt.Errorf("neo4j begin tx: %w", err)
	}
	defer tx.Close(ctx)
	now := s.now()
	createdAt := record.CreatedAt
	if createdAt.IsZero() {
		createdAt = now
	}
	lastEmbedded := record.LastEmbedded
	if lastEmbedded.IsZero() {
		lastEmbedded = createdAt
	}
	space := normalizeSpace(record.Space, record.SessionID)
	params := map[string]any{
		"id":            record.ID,
		"session_id":    record.SessionID,
		"space":         space,
		"content":       record.Content,
		"metadata":      sanitizeMetadata(record.Metadata),
		"importance":    record.Importance,
		"source":        record.Source,
		"summary":       record.Summary,
		"created_at":    createdAt.UTC().Format(time.RFC3339Nano),
		"last_embedded": lastEmbedded.UTC().Format(time.RFC3339Nano),
		"updated_at":    now.UTC().Format(time.RFC3339Nano),
	}
	res, err := tx.Run(ctx, neo4jUpsertNodeCypher, params)
	if err != nil {
		tx.Rollback(ctx)
		return fmt.Errorf("neo4j upsert node: %w", err)
	}
	if res != nil {
		_ = res.Close(ctx)
	}
	res, err = tx.Run(ctx, "MATCH (m:Memory {id: $id})-[r:RELATED_TO]->() DELETE r", map[string]any{"id": record.ID})
	if err != nil {
		tx.Rollback(ctx)
		return fmt.Errorf("neo4j delete edges: %w", err)
	}
	if res != nil {
		_ = res.Close(ctx)
	}
	for _, edge := range edges {
		if err := edge.Validate(); err != nil {
			continue
		}
		edgeParams := map[string]any{
			"from":       record.ID,
			"target":     edge.Target,
			"edge_type":  string(edge.Type),
			"updated_at": now.UTC().Format(time.RFC3339Nano),
			"space":      space,
		}
		res, err = tx.Run(ctx, neo4jUpsertEdgeCypher, edgeParams)
		if err != nil {
			tx.Rollback(ctx)
			return fmt.Errorf("neo4j upsert edge: %w", err)
		}
		if res != nil {
			_ = res.Close(ctx)
		}
	}
	if err := tx.Commit(ctx); err != nil {
		tx.Rollback(ctx)
		return fmt.Errorf("neo4j commit: %w", err)
	}
	return nil
}

// Neighborhood returns the nodes within the requested number of hops from the provided seeds.
func (s *Neo4jStore) Neighborhood(ctx context.Context, seedIDs []int64, hops, limit int) ([]model.MemoryRecord, error) {
	if s.driver == nil {
		return nil, ErrNeo4jUnavailable
	}
	if len(seedIDs) == 0 || hops <= 0 || limit <= 0 {
		return nil, nil
	}
	session, err := s.driver.NewSession(ctx, Neo4jSessionConfig{AccessMode: AccessModeRead, DatabaseName: s.database})
	if err != nil {
		return nil, fmt.Errorf("neo4j new session: %w", err)
	}
	defer session.Close(ctx)
	params := map[string]any{
		"seed_ids": seedIDs,
		"hops":     hops,
		"limit":    limit,
	}
	result, err := session.Run(ctx, neo4jNeighborhoodQuery, params)
	if err != nil {
		return nil, fmt.Errorf("neo4j neighborhood: %w", err)
	}
	defer result.Close(ctx)
	records := make([]model.MemoryRecord, 0, limit)
	for result.Next(ctx) {
		rec, recErr := mapNeo4jRecord(result.Record())
		if recErr != nil {
			return nil, recErr
		}
		meta := model.DecodeMetadata(rec.Metadata)
		model.HydrateRecordFromMetadata(&rec, meta)
		if rec.Space == "" {
			rec.Space = normalizeSpace("", rec.SessionID)
		}
		records = append(records, rec)
	}
	if err := result.Err(); err != nil {
		return nil, err
	}
	return records, nil
}

func (s *Neo4jStore) now() time.Time {
	if s == nil || s.nowFn == nil {
		return time.Now().UTC()
	}
	return s.nowFn().UTC()
}

func sanitizeMetadata(meta string) string {
	if strings.TrimSpace(meta) == "" {
		return "{}"
	}
	return meta
}

func normalizeSpace(space, fallback string) string {
	if strings.TrimSpace(space) != "" {
		return strings.TrimSpace(space)
	}
	if strings.TrimSpace(fallback) != "" {
		return strings.TrimSpace(fallback)
	}
	return "_shared"
}

const (
	neo4jUpsertNodeCypher = `
MERGE (m:Memory {id: $id})
ON CREATE SET m.created_at = $created_at
SET m.session_id = $session_id,
    m.space = $space,
    m.content = $content,
    m.metadata = $metadata,
    m.importance = $importance,
    m.source = $source,
    m.summary = $summary,
    m.last_embedded = $last_embedded,
    m.updated_at = $updated_at
`
	neo4jUpsertEdgeCypher = `
MATCH (m:Memory {id: $from})
MERGE (target:Memory {id: $target})
ON CREATE SET target.space = COALESCE(target.space, $space)
MERGE (m)-[r:RELATED_TO {target_id: $target}]->(target)
SET r.edge_type = $edge_type,
    r.updated_at = $updated_at
`
	neo4jNeighborhoodQuery = `
UNWIND $seed_ids AS seed
MATCH (start:Memory {id: seed})
MATCH path=(start)-[:RELATED_TO*1..$hops]-(neighbor:Memory)
WHERE NOT neighbor.id IN $seed_ids
WITH neighbor, MIN(length(path)) AS depth
RETURN neighbor.id AS id,
       neighbor.session_id AS session_id,
       neighbor.space AS space,
       neighbor.content AS content,
       neighbor.metadata AS metadata,
       neighbor.importance AS importance,
       neighbor.source AS source,
       neighbor.summary AS summary,
       neighbor.created_at AS created_at,
       neighbor.last_embedded AS last_embedded
ORDER BY depth ASC, neighbor.updated_at DESC
LIMIT $limit
`
)

func mapNeo4jRecord(rec neo4jRecord) (model.MemoryRecord, error) {
	if rec == nil {
		return model.MemoryRecord{}, errors.New("neo4j record is nil")
	}
	var out model.MemoryRecord
	if v, ok := rec.Get("id"); ok {
		out.ID = toInt64(v)
	}
	if v, ok := rec.Get("session_id"); ok {
		out.SessionID = toString(v)
	}
	if v, ok := rec.Get("space"); ok {
		out.Space = strings.TrimSpace(toString(v))
	}
	if v, ok := rec.Get("content"); ok {
		out.Content = toString(v)
	}
	if v, ok := rec.Get("metadata"); ok {
		out.Metadata = toString(v)
	}
	if v, ok := rec.Get("importance"); ok {
		out.Importance = toFloat64(v)
	}
	if v, ok := rec.Get("source"); ok {
		out.Source = toString(v)
	}
	if v, ok := rec.Get("summary"); ok {
		out.Summary = toString(v)
	}
	if v, ok := rec.Get("created_at"); ok {
		out.CreatedAt = parseTime(toString(v))
	}
	if v, ok := rec.Get("last_embedded"); ok {
		out.LastEmbedded = parseTime(toString(v))
	}
	return out, nil
}

func toString(v any) string {
	switch t := v.(type) {
	case string:
		return t
	case fmt.Stringer:
		return t.String()
	}
	return fmt.Sprintf("%v", v)
}

func toInt64(v any) int64 {
	switch t := v.(type) {
	case int:
		return int64(t)
	case int32:
		return int64(t)
	case int64:
		return t
	case float32:
		return int64(t)
	case float64:
		return int64(t)
	case jsonNumber:
		if i, err := t.Int64(); err == nil {
			return i
		}
	}
	return 0
}

func toFloat64(v any) float64 {
	switch t := v.(type) {
	case float32:
		return float64(t)
	case float64:
		return t
	case int:
		return float64(t)
	case int64:
		return float64(t)
	case jsonNumber:
		if f, err := t.Float64(); err == nil {
			return f
		}
	}
	return 0
}

func parseTime(value string) time.Time {
	value = strings.TrimSpace(value)
	if value == "" {
		return time.Time{}
	}
	if ts, err := time.Parse(time.RFC3339Nano, value); err == nil {
		return ts
	}
	if ts, err := time.Parse(time.RFC3339, value); err == nil {
		return ts
	}
	return time.Time{}
}

type jsonNumber interface {
	Int64() (int64, error)
	Float64() (float64, error)
}
