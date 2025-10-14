package memory

import (
	"context"
	"io"
	"sync"
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
	Store VectorStore
}

// SessionMemory wraps MemoryBank with short- and long-term layers
type SessionMemory struct {
	Bank          *MemoryBank
	shortTerm     map[string][]MemoryRecord
	mu            sync.RWMutex
	shortTermSize int
	Embedder      Embedder
}

// NewMemoryBank creates a new Postgres-backed memory bank.
func NewMemoryBank(ctx context.Context, connStr string) (*MemoryBank, error) {
	store, err := NewPostgresStore(ctx, connStr)
	if err != nil {
		return nil, err
	}
	return &MemoryBank{Store: store}, nil
}

// NewMemoryBankWithStore creates a memory bank backed by a custom vector store implementation.
func NewMemoryBankWithStore(store VectorStore) *MemoryBank {
	return &MemoryBank{Store: store}
}

// NewSessionMemory wraps MemoryBank with short-term cache
func NewSessionMemory(bank *MemoryBank, shortTermSize int) *SessionMemory {
	return &SessionMemory{
		Bank:          bank,
		shortTerm:     make(map[string][]MemoryRecord),
		shortTermSize: shortTermSize,
		Embedder:      AutoEmbedder(),
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

// FlushToLongTerm writes the short-term cache to the configured vector store.
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

func (sm *SessionMemory) Embed(ctx context.Context, text string) ([]float32, error) {
	if sm.Embedder == nil {
		sm.Embedder = AutoEmbedder()
	}
	vec, err := sm.Embedder.Embed(ctx, text)
	if err != nil || len(vec) == 0 {
		return DummyEmbedding(text), nil
	}
	return vec, nil
}

// RetrieveContext returns combined short- and long-term memory
func (sm *SessionMemory) RetrieveContext(ctx context.Context, sessionID, query string, limit int) ([]MemoryRecord, error) {
	var longTerm []MemoryRecord
	if sm.Bank != nil {

		queryEmbedding, err := sm.Embed(ctx, query)
		if err != nil {
			return nil, err
		}
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

func (sm *SessionMemory) WithEmbedder(e Embedder) *SessionMemory {
	if e != nil {
		sm.Embedder = e
	}
	return sm
}

// StoreMemory inserts a long-term record
func (mb *MemoryBank) StoreMemory(ctx context.Context, sessionID, content, metadata string, embedding []float32) error {
	if mb == nil || mb.Store == nil {
		return nil
	}
	return mb.Store.StoreMemory(ctx, sessionID, content, metadata, embedding)
}

// SearchMemory returns top-k similar memories
func (mb *MemoryBank) SearchMemory(ctx context.Context, queryEmbedding []float32, limit int) ([]MemoryRecord, error) {
	if mb == nil || mb.Store == nil {
		return nil, nil
	}
	return mb.Store.SearchMemory(ctx, queryEmbedding, limit)
}

// CreateSchema initialises the backing store if it supports schema management.
func (mb *MemoryBank) CreateSchema(ctx context.Context, schemaPath string) error {
	if mb == nil || mb.Store == nil {
		return nil
	}
	initializer, ok := mb.Store.(SchemaInitializer)
	if !ok {
		return nil
	}
	return initializer.CreateSchema(ctx, schemaPath)
}

// Close releases underlying resources if the store implements io.Closer.
func (mb *MemoryBank) Close() error {
	if mb == nil || mb.Store == nil {
		return nil
	}
	if closer, ok := mb.Store.(io.Closer); ok {
		return closer.Close()
	}
	return nil
}
