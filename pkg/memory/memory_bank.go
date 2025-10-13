package memory

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"

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

// SessionMemory wraps MemoryBank with short- and long-term layers
type SessionMemory struct {
	Bank          *MemoryBank
	shortTerm     map[string][]MemoryRecord
	mu            sync.RWMutex
	shortTermSize int
}

// NewMemoryBank creates a new connection to Postgres
func NewMemoryBank(ctx context.Context, connStr string) (*MemoryBank, error) {
	db, err := pgxpool.New(ctx, connStr)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to Postgres: %w", err)
	}
	return &MemoryBank{DB: db}, nil
}

// NewSessionMemory wraps MemoryBank with short-term cache
func NewSessionMemory(bank *MemoryBank, shortTermSize int) *SessionMemory {
	return &SessionMemory{
		Bank:          bank,
		shortTerm:     make(map[string][]MemoryRecord),
		shortTermSize: shortTermSize,
	}
}

// AddShortTerm stores in ephemeral session cache
func (sm *SessionMemory) AddShortTerm(sessionID, content, metadata string, embedding []float32) {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	record := MemoryRecord{SessionID: sessionID, Content: content, Metadata: metadata, Embedding: embedding}
	sm.shortTerm[sessionID] = append(sm.shortTerm[sessionID], record)

	if len(sm.shortTerm[sessionID]) > sm.shortTermSize {
		sm.shortTerm[sessionID] = sm.shortTerm[sessionID][len(sm.shortTerm[sessionID])-sm.shortTermSize:]
	}
}

// FlushToLongTerm writes short-term cache to Postgres
func (sm *SessionMemory) FlushToLongTerm(ctx context.Context, sessionID string) error {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	records := sm.shortTerm[sessionID]
	for _, r := range records {
		if err := sm.Bank.StoreMemory(ctx, sessionID, r.Content, r.Metadata, r.Embedding); err != nil {
			return err
		}
	}
	delete(sm.shortTerm, sessionID)
	return nil
}

// RetrieveContext returns combined short- and long-term memory
func (sm *SessionMemory) RetrieveContext(ctx context.Context, sessionID, query string, limit int) ([]MemoryRecord, error) {
	var longTerm []MemoryRecord
	if sm.Bank != nil {
		queryEmbedding := VertexAIEmbedding(query)
		records, err := sm.Bank.SearchMemory(ctx, queryEmbedding, limit)
		if err != nil {
			return nil, err
		}
		longTerm = records
	}

	sm.mu.RLock()
	shortTerm := sm.shortTerm[sessionID]
	sm.mu.RUnlock()

	return append(shortTerm, longTerm...), nil
}

// StoreMemory inserts a long-term record
func (mb *MemoryBank) StoreMemory(ctx context.Context, sessionID, content, metadata string, embedding []float32) error {
	if mb == nil || mb.DB == nil {
		return nil
	}
	jsonEmbed, _ := json.Marshal(embedding)
	query := `
                INSERT INTO memory_bank (session_id, content, metadata, embedding)
                VALUES ($1, $2, $3::jsonb, $4::vector);
        `
	_, err := mb.DB.Exec(ctx, query, sessionID, content, metadata, fmt.Sprintf("[%s]", trimJSON(string(jsonEmbed))))
	return err
}

// SearchMemory returns top-k similar memories
func (mb *MemoryBank) SearchMemory(ctx context.Context, queryEmbedding []float32, limit int) ([]MemoryRecord, error) {
	if mb == nil || mb.DB == nil {
		return nil, nil
	}
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

func trimJSON(s string) string { return strings.Trim(s, "[]") }

// CreateSchema ensures pgvector extension and memory table are available
func (mb *MemoryBank) CreateSchema(ctx context.Context, schemaPath string) error {
	if mb == nil || mb.DB == nil {
		return nil
	}
	data, err := os.ReadFile(schemaPath)
	if err != nil {
		return fmt.Errorf("failed to read schema file: %w", err)
	}

	// Execute schema SQL (CREATE EXTENSION, TABLE, INDEXES)
	_, err = mb.DB.Exec(ctx, string(data))
	if err != nil {
		return fmt.Errorf("failed to execute schema: %w", err)
	}
	return nil
}
