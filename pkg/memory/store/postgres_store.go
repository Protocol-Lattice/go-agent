package store

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/Raezil/go-agent-development-kit/pkg/memory/model"
)

// PostgresStore implements VectorStore using Postgres + pgvector.
type PostgresStore struct {
	DB *pgxpool.Pool
}

// NewPostgresStore connects to Postgres and returns a Postgres-backed VectorStore implementation.
func NewPostgresStore(ctx context.Context, connStr string) (*PostgresStore, error) {
	db, err := pgxpool.New(ctx, connStr)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to Postgres: %w", err)
	}
	return &PostgresStore{DB: db}, nil
}

// StoreMemory inserts a long-term record into Postgres.
func (ps *PostgresStore) StoreMemory(ctx context.Context, sessionID, content string, metadata map[string]any, embedding []float32) error {
	if ps == nil || ps.DB == nil {
		return nil
	}
	if metadata == nil {
		metadata = map[string]any{}
	}
	if _, ok := metadata["space"]; !ok {
		metadata["space"] = sessionID
	}
	importance, source, summary, lastEmbedded, metadataJSON := model.NormalizeMetadata(metadata, time.Now().UTC())
	query := `
                INSERT INTO memory_bank (session_id, content, metadata, embedding, importance, source, summary, last_embedded)
                VALUES ($1, $2, $3::jsonb, $4::vector, $5, $6, $7, $8)
                RETURNING id;
        `
	jsonEmbed, _ := json.Marshal(embedding)
	var id int64
	if err := ps.DB.QueryRow(ctx, query, sessionID, content, metadataJSON, vectorFromJSON(jsonEmbed), importance, source, summary, lastEmbedded).Scan(&id); err != nil {
		return err
	}
	meta := model.DecodeMetadata(metadataJSON)
	rec := model.MemoryRecord{
		ID:              id,
		SessionID:       sessionID,
		Space:           model.StringFromAny(meta["space"]),
		Content:         content,
		Metadata:        metadataJSON,
		Embedding:       embedding,
		Importance:      importance,
		Source:          source,
		Summary:         summary,
		LastEmbedded:    lastEmbedded,
		GraphEdges:      model.ValidGraphEdges(meta),
		EmbeddingMatrix: model.ValidEmbeddingMatrix(meta),
	}
	if rec.Space == "" {
		rec.Space = sessionID
	}
	if err := ps.UpsertGraph(ctx, rec, rec.GraphEdges); err != nil {
		return err
	}
	return nil
}

// SearchMemory returns top-k similar memories from Postgres.
func (ps *PostgresStore) SearchMemory(ctx context.Context, queryEmbedding []float32, limit int) ([]model.MemoryRecord, error) {
	if ps == nil || ps.DB == nil {
		return nil, nil
	}
	jsonEmbed, _ := json.Marshal(queryEmbedding)
	rows, err := ps.DB.Query(ctx, `
        SELECT id, session_id, content, metadata::text, importance, source, summary, created_at, last_embedded, embedding::text, (embedding <-> $1::vector) AS score
        FROM memory_bank
        ORDER BY embedding <-> $1::vector
        LIMIT $2;
        `, vectorFromJSON(jsonEmbed), limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var records []model.MemoryRecord
	for rows.Next() {
		var rec model.MemoryRecord
		var embeddingText string
		if err := rows.Scan(&rec.ID, &rec.SessionID, &rec.Content, &rec.Metadata, &rec.Importance, &rec.Source, &rec.Summary, &rec.CreatedAt, &rec.LastEmbedded, &embeddingText, &rec.Score); err != nil {
			return nil, err
		}
		rec.Embedding = parseVector(embeddingText)
		model.HydrateRecordFromMetadata(&rec, model.DecodeMetadata(rec.Metadata))
		rec.Score = 1 - rec.Score
		if rec.Space == "" {
			rec.Space = rec.SessionID
		}
		records = append(records, rec)
	}
	return records, nil
}

func (ps *PostgresStore) UpdateEmbedding(ctx context.Context, id int64, embedding []float32, lastEmbedded time.Time) error {
	if ps == nil || ps.DB == nil {
		return nil
	}
	jsonEmbed, _ := json.Marshal(embedding)
	_, err := ps.DB.Exec(ctx, `
                UPDATE memory_bank
                SET embedding = $2::vector, last_embedded = $3
                WHERE id = $1
        `, id, vectorFromJSON(jsonEmbed), lastEmbedded)
	return err
}

func (ps *PostgresStore) DeleteMemory(ctx context.Context, ids []int64) error {
	if ps == nil || ps.DB == nil || len(ids) == 0 {
		return nil
	}
	_, err := ps.DB.Exec(ctx, `DELETE FROM memory_bank WHERE id = ANY($1)`, ids)
	return err
}

func (ps *PostgresStore) Iterate(ctx context.Context, fn func(model.MemoryRecord) bool) error {
	if ps == nil || ps.DB == nil {
		return nil
	}
	rows, err := ps.DB.Query(ctx, `
        SELECT id, session_id, content, metadata::text, importance, source, summary, created_at, last_embedded, embedding::text
        FROM memory_bank
        ORDER BY created_at ASC
        `)
	if err != nil {
		return err
	}
	defer rows.Close()
	for rows.Next() {
		var rec model.MemoryRecord
		var embeddingText string
		if err := rows.Scan(&rec.ID, &rec.SessionID, &rec.Content, &rec.Metadata, &rec.Importance, &rec.Source, &rec.Summary, &rec.CreatedAt, &rec.LastEmbedded, &embeddingText); err != nil {
			return err
		}
		rec.Embedding = parseVector(embeddingText)
		model.HydrateRecordFromMetadata(&rec, model.DecodeMetadata(rec.Metadata))
		if rec.Space == "" {
			rec.Space = rec.SessionID
		}
		cont := fn(rec)
		if !cont {
			break
		}
	}
	return rows.Err()
}

func (ps *PostgresStore) Count(ctx context.Context) (int, error) {
	if ps == nil || ps.DB == nil {
		return 0, nil
	}
	var count int
	err := ps.DB.QueryRow(ctx, `SELECT COUNT(*) FROM memory_bank`).Scan(&count)
	return count, err
}

// UpsertGraph ensures the knowledge graph stays aligned with stored memories.
func (ps *PostgresStore) UpsertGraph(ctx context.Context, record model.MemoryRecord, edges []model.GraphEdge) error {
	if ps == nil || ps.DB == nil || record.ID == 0 {
		return nil
	}
	tx, err := ps.DB.BeginTx(ctx, pgx.TxOptions{})
	if err != nil {
		return err
	}
	defer func() {
		if err != nil {
			_ = tx.Rollback(ctx)
		}
	}()
	fallback := record.SessionID
	if fallback == "" {
		fallback = record.Space
	}
	if err = ensureNodeTx(ctx, tx, record.ID, record.Space, fallback); err != nil {
		return err
	}
	if _, err = tx.Exec(ctx, `DELETE FROM memory_edges WHERE from_memory = $1`, record.ID); err != nil {
		return err
	}
	for _, edge := range edges {
		if err := edge.Validate(); err != nil {
			continue
		}
		if err = ensureNodeTx(ctx, tx, edge.Target, "", ""); err != nil {
			return err
		}
		if _, err = tx.Exec(ctx, `
                        INSERT INTO memory_edges (from_memory, to_memory, edge_type)
                        VALUES ($1, $2, $3)
                        ON CONFLICT (from_memory, to_memory, edge_type) DO NOTHING
                `, record.ID, edge.Target, string(edge.Type)); err != nil {
			return err
		}
	}
	if err = tx.Commit(ctx); err != nil {
		return err
	}
	return nil
}

// Neighborhood returns memories connected within the configured hop distance.
func (ps *PostgresStore) Neighborhood(ctx context.Context, seedIDs []int64, hops, limit int) ([]model.MemoryRecord, error) {
	if ps == nil || ps.DB == nil || len(seedIDs) == 0 || hops <= 0 || limit <= 0 {
		return nil, nil
	}
	rows, err := ps.DB.Query(ctx, `
WITH RECURSIVE walk AS (
        SELECT UNNEST($1::bigint[]) AS id, 0 AS depth
        UNION ALL
        SELECT CASE WHEN me.from_memory = walk.id THEN me.to_memory ELSE me.from_memory END AS id,
               walk.depth + 1 AS depth
        FROM memory_edges me
        JOIN walk ON me.from_memory = walk.id OR me.to_memory = walk.id
        WHERE walk.depth < $2
)
SELECT DISTINCT ON (mb.id)
        mb.id, mb.session_id, mb.content, mb.metadata::text, mb.importance, mb.source,
        mb.summary, mb.created_at, mb.last_embedded, mb.embedding::text,
        COALESCE(mn.space, mb.session_id) AS space,
        walk.depth
FROM walk
JOIN memory_bank mb ON mb.id = walk.id
LEFT JOIN memory_nodes mn ON mn.memory_id = mb.id
WHERE walk.depth > 0
ORDER BY mb.id, walk.depth ASC, mb.created_at DESC
LIMIT $3;
        `, seedIDs, hops, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	results := make([]model.MemoryRecord, 0)
	for rows.Next() {
		var rec model.MemoryRecord
		var embeddingText string
		var depth int
		if err := rows.Scan(&rec.ID, &rec.SessionID, &rec.Content, &rec.Metadata, &rec.Importance, &rec.Source, &rec.Summary, &rec.CreatedAt, &rec.LastEmbedded, &embeddingText, &rec.Space, &depth); err != nil {
			return nil, err
		}
		rec.Embedding = parseVector(embeddingText)
		meta := model.DecodeMetadata(rec.Metadata)
		if rec.Space == "" {
			rec.Space = model.StringFromAny(meta["space"])
		}
		model.HydrateRecordFromMetadata(&rec, meta)
		if rec.Space == "" {
			rec.Space = rec.SessionID
		}
		results = append(results, rec)
	}
	return results, rows.Err()
}

// CreateSchema ensures pgvector extension and memory table are available.
func (ps *PostgresStore) CreateSchema(ctx context.Context, schemaPath string) error {
	if ps == nil || ps.DB == nil {
		return nil
	}
	schema := defaultPostgresSchema
	if schemaPath != "" {
		data, err := os.ReadFile(schemaPath)
		if err != nil {
			return fmt.Errorf("failed to read schema file: %w", err)
		}
		schema = string(data)
	}

	_, err := ps.DB.Exec(ctx, schema)
	if err != nil {
		return fmt.Errorf("failed to execute schema: %w", err)
	}
	return nil
}

// Close releases the underlying Postgres connection pool.
func (ps *PostgresStore) Close() error {
	if ps == nil || ps.DB == nil {
		return nil
	}
	ps.DB.Close()
	return nil
}

func trimJSON(s string) string { return strings.Trim(s, "[]") }

func ensureNodeTx(ctx context.Context, tx pgx.Tx, memoryID int64, space, fallback string) error {
	if memoryID == 0 {
		return nil
	}
	space = strings.TrimSpace(space)
	if space == "" {
		var existing string
		if err := tx.QueryRow(ctx, `SELECT space FROM memory_nodes WHERE memory_id = $1`, memoryID).Scan(&existing); err == nil && strings.TrimSpace(existing) != "" {
			space = strings.TrimSpace(existing)
		}
	}
	if space == "" && fallback != "" {
		space = strings.TrimSpace(fallback)
	}
	if space == "" {
		var sessionID string
		if err := tx.QueryRow(ctx, `SELECT session_id FROM memory_bank WHERE id = $1`, memoryID).Scan(&sessionID); err == nil && strings.TrimSpace(sessionID) != "" {
			space = strings.TrimSpace(sessionID)
		}
	}
	if space == "" {
		space = "_shared"
	}
	_, err := tx.Exec(ctx, `
                INSERT INTO memory_nodes (memory_id, space, updated_at)
                VALUES ($1, $2, NOW())
                ON CONFLICT (memory_id) DO UPDATE SET space = EXCLUDED.space, updated_at = NOW()
        `, memoryID, space)
	return err
}

const defaultPostgresSchema = `
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS memory_bank (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB,
    embedding vector(768),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS memory_session_idx ON memory_bank (session_id);
CREATE INDEX IF NOT EXISTS memory_embedding_idx ON memory_bank USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

ALTER TABLE memory_bank ADD COLUMN IF NOT EXISTS importance DOUBLE PRECISION DEFAULT 0;
ALTER TABLE memory_bank ADD COLUMN IF NOT EXISTS source TEXT DEFAULT '';
ALTER TABLE memory_bank ADD COLUMN IF NOT EXISTS summary TEXT DEFAULT '';
ALTER TABLE memory_bank ADD COLUMN IF NOT EXISTS last_embedded TIMESTAMPTZ DEFAULT NOW();

CREATE TABLE IF NOT EXISTS memory_nodes (
    memory_id BIGINT PRIMARY KEY REFERENCES memory_bank(id) ON DELETE CASCADE,
    space TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS memory_edges (
    from_memory BIGINT NOT NULL REFERENCES memory_bank(id) ON DELETE CASCADE,
    to_memory BIGINT NOT NULL REFERENCES memory_bank(id) ON DELETE CASCADE,
    edge_type TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (from_memory, to_memory, edge_type)
);

CREATE INDEX IF NOT EXISTS memory_edges_to_idx ON memory_edges (to_memory);
`

func vectorFromJSON(jsonEmbed []byte) string {
	return fmt.Sprintf("[%s]", trimJSON(string(jsonEmbed)))
}

func parseVector(text string) []float32 {
	text = strings.Trim(text, "[]")
	if strings.TrimSpace(text) == "" {
		return nil
	}
	parts := strings.Split(text, ",")
	vec := make([]float32, 0, len(parts))
	for _, part := range parts {
		f, err := strconv.ParseFloat(strings.TrimSpace(part), 32)
		if err != nil {
			continue
		}
		vec = append(vec, float32(f))
	}
	return vec
}
