package model

import "time"

// MemoryRecord represents a persisted memory entry in the vector store.
type MemoryRecord struct {
	ID              int64       `json:"id"`
	SessionID       string      `json:"session_id"`
	Space           string      `json:"space"`
	Content         string      `json:"content"`
	Metadata        string      `json:"metadata"`
	Embedding       []float32   `json:"embedding"`
	MultiEmbeddings [][]float32 `json:"multi_embeddings,omitempty"`
	Score           float64     `json:"score"`
	Importance      float64     `json:"importance"`
	Source          string      `json:"source"`
	Summary         string      `json:"summary"`
	CreatedAt       time.Time   `json:"created_at"`
	LastEmbedded    time.Time   `json:"last_embedded"`
	WeightedScore   float64     `json:"weighted_score"`
	GraphEdges      []GraphEdge `json:"graph_edges"`
}

// AllEmbeddings returns the primary embedding and any additional vectors associated with the record.
func (r MemoryRecord) AllEmbeddings() [][]float32 {
	total := 0
	if len(r.Embedding) > 0 {
		total++
	}
	total += len(r.MultiEmbeddings)
	if total == 0 {
		return nil
	}
	vectors := make([][]float32, 0, total)
	if len(r.Embedding) > 0 {
		vectors = append(vectors, r.Embedding)
	}
	for _, extra := range r.MultiEmbeddings {
		if len(extra) == 0 {
			continue
		}
		vectors = append(vectors, extra)
	}
	return vectors
}
