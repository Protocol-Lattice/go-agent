package memory

import (
	"context"
	"encoding/json"
	"io"
	"sync"
	"time"
)

type MemoryRecord struct {
	ID            int64       `json:"id"`
	SessionID     string      `json:"session_id"`
	Space         string      `json:"space"`
	Content       string      `json:"content"`
	Metadata      string      `json:"metadata"`
	Embedding     []float32   `json:"embedding"`
	Score         float64     `json:"score"`
	Importance    float64     `json:"importance"`
	Source        string      `json:"source"`
	Summary       string      `json:"summary"`
	CreatedAt     time.Time   `json:"created_at"`
	LastEmbedded  time.Time   `json:"last_embedded"`
	WeightedScore float64     `json:"weighted_score"`
	GraphEdges    []GraphEdge `json:"graph_edges"`
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
	Engine        *Engine
	Spaces        *SpaceRegistry
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
		Spaces:        NewSpaceRegistry(0),
	}
}

// AddShortTerm stores in ephemeral session cache
func (sm *SessionMemory) AddShortTerm(sessionID, content, metadata string, embedding []float32) {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	record := MemoryRecord{SessionID: sessionID, Space: sessionID, Content: content, Metadata: metadata, Embedding: embedding}
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
		if sm.Engine != nil {
			meta := decodeMetadata(r.Metadata)
			if _, err := sm.Engine.Store(ctx, sessionID, r.Content, meta); err != nil {
				return err
			}
		} else {
			if err := sm.Bank.StoreMemory(ctx, sessionID, r.Content, r.Metadata, r.Embedding); err != nil {
				return err
			}
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
	if sm.Engine != nil {
		records, err := sm.Engine.Retrieve(ctx, query, limit)
		if err != nil {
			return nil, err
		}
		longTerm = records
	} else if sm.Bank != nil {
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

// WithEngine attaches an advanced memory engine for long-term storage and retrieval.
func (sm *SessionMemory) WithEngine(engine *Engine) *SessionMemory {
	sm.Engine = engine
	return sm
}

// StoreMemory inserts a long-term record
func (mb *MemoryBank) StoreMemory(ctx context.Context, sessionID, content, metadata string, embedding []float32) error {
	if mb == nil || mb.Store == nil {
		return nil
	}
	meta := map[string]any{}
	if metadata != "" {
		_ = json.Unmarshal([]byte(metadata), &meta)
	}
	if _, ok := meta["space"]; !ok {
		meta["space"] = sessionID
	}
	return mb.Store.StoreMemory(ctx, sessionID, content, meta, embedding)
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
