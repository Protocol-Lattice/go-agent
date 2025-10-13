package memory

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/jackc/pgx/v5/pgxpool"
)

type MemoryRecord struct {
	ID        int64     `json:"id"`
	SessionID string    `json:"session_id"`
	Content   string    `json:"content"`
	Metadata  string    `json:"metadata"`
	Embedding []float32 `json:"embedding"`
	Score     float64   `json:"score"`
}

type MemoryBank struct {
	DB *pgxpool.Pool
}

// NewMemoryBank creates a new connection to Postgres
func NewMemoryBank(ctx context.Context, connStr string) (*MemoryBank, error) {
	db, err := pgxpool.New(ctx, connStr)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to Postgres: %w", err)
	}
	return &MemoryBank{DB: db}, nil
}

// CreateSchema ensures pgvector and tables are ready
func (mb *MemoryBank) CreateSchema(ctx context.Context, schemaPath string) error {
	schema, err := os.ReadFile(schemaPath)
	if err != nil {
		return fmt.Errorf("failed to read schema file: %w", err)
	}
	_, err = mb.DB.Exec(ctx, string(schema))
	return err
}

// StoreMemory inserts a memory record
func (mb *MemoryBank) StoreMemory(ctx context.Context, sessionID, content, metadata string, embedding []float32) error {
	jsonEmbed, _ := json.Marshal(embedding)
	query := `
		INSERT INTO memory_bank (session_id, content, metadata, embedding)
		VALUES ($1, $2, $3::jsonb, $4::vector);
	`
	_, err := mb.DB.Exec(ctx, query, sessionID, content, metadata, fmt.Sprintf("[%s]", trimJSON(string(jsonEmbed))))
	return err
}

// SearchMemory returns top-k similar memories
// SearchMemory returns top-k similar memories
func (mb *MemoryBank) SearchMemory(ctx context.Context, queryEmbedding []float32, limit int) ([]MemoryRecord, error) {
	jsonEmbed, _ := json.Marshal(queryEmbedding)
	rows, err := mb.DB.Query(ctx, `
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

func trimJSON(s string) string {
	return strings.Trim(s, "[]")
}
