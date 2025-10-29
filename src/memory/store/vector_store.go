package store

import (
	"context"
	"time"

	"github.com/Protocol-Lattice/go-agent/src/memory/model"
)

// VectorStore defines the contract for long-term memory backends.
type VectorStore interface {
	StoreMemory(ctx context.Context, sessionID, content string, metadata map[string]any, embedding []float32) error
	SearchMemory(ctx context.Context, queryEmbedding []float32, limit int) ([]model.MemoryRecord, error)
	UpdateEmbedding(ctx context.Context, id int64, embedding []float32, lastEmbedded time.Time) error
	DeleteMemory(ctx context.Context, ids []int64) error
	Iterate(ctx context.Context, fn func(model.MemoryRecord) bool) error
	Count(ctx context.Context) (int, error)
}

// SchemaInitializer allows stores to expose optional schema/bootstrap routines.
type SchemaInitializer interface {
	CreateSchema(ctx context.Context, schemaPath string) error
}

// GraphStore is implemented by vector stores that maintain graph neighborhoods for memories.
type GraphStore interface {
	UpsertGraph(ctx context.Context, record model.MemoryRecord, edges []model.GraphEdge) error
	Neighborhood(ctx context.Context, seedIDs []int64, hops, limit int) ([]model.MemoryRecord, error)
}
