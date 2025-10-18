package model

import "time"

// MemoryRecord represents a persisted memory entry in the vector store.
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
