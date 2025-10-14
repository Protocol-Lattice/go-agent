package memory

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
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
	importance, source, summary, lastEmbedded, metadataJSON := normalizeMetadata(metadata, time.Now().UTC())
	query := `
                INSERT INTO memory_bank (session_id, content, metadata, embedding, importance, source, summary, last_embedded)
                VALUES ($1, $2, $3::jsonb, $4::vector, $5, $6, $7, $8)
                RETURNING id;
        `
	jsonEmbed, _ := json.Marshal(embedding)
	_, err := ps.DB.Exec(ctx, query, sessionID, content, metadataJSON, vectorFromJSON(jsonEmbed), importance, source, summary, lastEmbedded)
	return err
}

// SearchMemory returns top-k similar memories from Postgres.
func (ps *PostgresStore) SearchMemory(ctx context.Context, queryEmbedding []float32, limit int) ([]MemoryRecord, error) {
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

	var records []MemoryRecord
	for rows.Next() {
		var rec MemoryRecord
		var embeddingText string
		if err := rows.Scan(&rec.ID, &rec.SessionID, &rec.Content, &rec.Metadata, &rec.Importance, &rec.Source, &rec.Summary, &rec.CreatedAt, &rec.LastEmbedded, &embeddingText, &rec.Score); err != nil {
			return nil, err
		}
		rec.Embedding = parseVector(embeddingText)
		hydrateRecordFromMetadata(&rec, decodeMetadata(rec.Metadata))
		rec.Score = 1 - rec.Score
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

func (ps *PostgresStore) Iterate(ctx context.Context, fn func(MemoryRecord) bool) error {
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
		var rec MemoryRecord
		var embeddingText string
		if err := rows.Scan(&rec.ID, &rec.SessionID, &rec.Content, &rec.Metadata, &rec.Importance, &rec.Source, &rec.Summary, &rec.CreatedAt, &rec.LastEmbedded, &embeddingText); err != nil {
			return err
		}
		rec.Embedding = parseVector(embeddingText)
		hydrateRecordFromMetadata(&rec, decodeMetadata(rec.Metadata))
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
