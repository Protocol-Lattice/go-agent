package memory

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

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
func (ps *PostgresStore) StoreMemory(ctx context.Context, sessionID, content, metadata string, embedding []float32) error {
	if ps == nil || ps.DB == nil {
		return nil
	}
	jsonEmbed, _ := json.Marshal(embedding)
	query := `
                INSERT INTO memory_bank (session_id, content, metadata, embedding)
                VALUES ($1, $2, $3::jsonb, $4::vector);
        `
	_, err := ps.DB.Exec(ctx, query, sessionID, content, metadata, fmt.Sprintf("[%s]", trimJSON(string(jsonEmbed))))
	return err
}

// SearchMemory returns top-k similar memories from Postgres.
func (ps *PostgresStore) SearchMemory(ctx context.Context, queryEmbedding []float32, limit int) ([]MemoryRecord, error) {
	if ps == nil || ps.DB == nil {
		return nil, nil
	}
	jsonEmbed, _ := json.Marshal(queryEmbedding)
	rows, err := ps.DB.Query(ctx, `
        SELECT id, session_id, content, metadata, (embedding <-> $1::vector) AS score
        FROM memory_bank
        ORDER BY embedding <-> $1::vector
        LIMIT $2;
        `, fmt.Sprintf("[%s]", trimJSON(string(jsonEmbed))), limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var records []MemoryRecord
	for rows.Next() {
		var rec MemoryRecord
		if err := rows.Scan(&rec.ID, &rec.SessionID, &rec.Content, &rec.Metadata, &rec.Score); err != nil {
			return nil, err
		}
		records = append(records, rec)
	}
	return records, nil
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
`
