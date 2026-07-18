package model

import "math"

// CosineQuery prepares the invariant portion of cosine similarity for reuse
// across many candidate vectors. The query slice must not be mutated while the
// prepared query is in use.
type CosineQuery struct {
	vector    []float32
	magnitude float64
}

// NewCosineQuery prepares a vector for repeated cosine-similarity comparisons.
func NewCosineQuery(vector []float32) CosineQuery {
	return CosineQuery{
		vector:    vector,
		magnitude: VectorMagnitude(vector),
	}
}

// VectorMagnitude computes the Euclidean magnitude of a vector. Callers that
// compare an immutable vector repeatedly can cache this value and pass it to
// SimilarityWithMagnitude.
func VectorMagnitude(vector []float32) float64 {
	var norm float64
	for _, value := range vector {
		norm += float64(value) * float64(value)
	}
	return math.Sqrt(norm)
}

// Similarity computes cosine similarity against a prepared query. Differing
// vector dimensions retain CosineSimilarity's truncated-vector semantics.
func (q CosineQuery) Similarity(vector []float32) float64 {
	if len(q.vector) == 0 || len(vector) == 0 {
		return 0
	}
	if len(q.vector) != len(vector) {
		return CosineSimilarity(q.vector, vector)
	}

	var dot, norm float64
	for i, value := range q.vector {
		dot += float64(value) * float64(vector[i])
		norm += float64(vector[i]) * float64(vector[i])
	}
	if q.magnitude == 0 || norm == 0 {
		return 0
	}
	return dot / (q.magnitude * math.Sqrt(norm))
}

// SimilarityWithMagnitude computes cosine similarity using a previously
// calculated candidate magnitude. Differing vector dimensions fall back to
// Similarity so its truncated-vector semantics remain unchanged.
func (q CosineQuery) SimilarityWithMagnitude(vector []float32, magnitude float64) float64 {
	if len(q.vector) == 0 || len(vector) == 0 {
		return 0
	}
	if len(q.vector) != len(vector) {
		return q.Similarity(vector)
	}
	if q.magnitude == 0 || magnitude == 0 {
		return 0
	}

	var dot float64
	for i, value := range q.vector {
		dot += float64(value) * float64(vector[i])
	}
	return dot / (q.magnitude * magnitude)
}

// CosineSimilarity computes the cosine similarity between two vectors.
func CosineSimilarity(a, b []float32) float64 {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}
	var dot, normA, normB float64
	length := len(a)
	if len(b) < length {
		length = len(b)
	}
	for i := 0; i < length; i++ {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

// CosineSimilarityMatrix computes the best cosine similarity between a query
// vector and any vector contained within the matrix.
func CosineSimilarityMatrix(query []float32, matrix [][]float32) float64 {
	prepared := NewCosineQuery(query)
	best := 0.0
	for _, vec := range matrix {
		if sim := prepared.Similarity(vec); sim > best {
			best = sim
		}
	}
	return best
}

// MaxCosineSimilarity returns the highest cosine similarity between the query
// vector and any embedding associated with the record.
func MaxCosineSimilarity(query []float32, rec MemoryRecord) float64 {
	return NewCosineQuery(query).MaxSimilarity(rec)
}

// MaxSimilarity returns the highest cosine similarity between a prepared query
// and any embedding associated with the record.
func (q CosineQuery) MaxSimilarity(rec MemoryRecord) float64 {
	var (
		best      float64
		hasVector bool
	)
	if len(rec.Embedding) > 0 {
		best = q.Similarity(rec.Embedding)
		hasVector = true
	}
	for _, vec := range rec.EmbeddingMatrix {
		if len(vec) == 0 {
			continue
		}
		sim := q.Similarity(vec)
		if !hasVector || sim > best {
			best = sim
			hasVector = true
		}
	}
	return best
}

// RecordSimilarity computes the maximum similarity between any pair of
// embeddings contained within the two records.
func RecordSimilarity(a, b MemoryRecord) float64 {
	var (
		best      float64
		hasVector bool
	)
	for ai := -1; ai < len(a.EmbeddingMatrix); ai++ {
		var av []float32
		if ai < 0 {
			av = a.Embedding
		} else {
			av = a.EmbeddingMatrix[ai]
		}
		if len(av) == 0 {
			continue
		}
		for bi := -1; bi < len(b.EmbeddingMatrix); bi++ {
			var bv []float32
			if bi < 0 {
				bv = b.Embedding
			} else {
				bv = b.EmbeddingMatrix[bi]
			}
			if len(bv) == 0 {
				continue
			}
			sim := CosineSimilarity(av, bv)
			if !hasVector || sim > best {
				best = sim
				hasVector = true
			}
		}
	}
	return best
}
