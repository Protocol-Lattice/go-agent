package memory

import "context"

// VectorStore defines the contract for long-term memory backends.
type VectorStore interface {
	StoreMemory(ctx context.Context, sessionID, content, metadata string, embedding []float32) error
	SearchMemory(ctx context.Context, queryEmbedding []float32, limit int) ([]MemoryRecord, error)
}

// SchemaInitializer allows stores to expose optional schema/bootstrap routines.
type SchemaInitializer interface {
	CreateSchema(ctx context.Context, schemaPath string) error
}
