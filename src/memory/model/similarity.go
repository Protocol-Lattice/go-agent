package model

import "math"

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
	best := 0.0
	for _, vec := range matrix {
		if sim := CosineSimilarity(query, vec); sim > best {
			best = sim
		}
	}
	return best
}

// MaxCosineSimilarity returns the highest cosine similarity between the query
// vector and any embedding associated with the record.
func MaxCosineSimilarity(query []float32, rec MemoryRecord) float64 {
	var (
		best      float64
		hasVector bool
	)
	if len(rec.Embedding) > 0 {
		best = CosineSimilarity(query, rec.Embedding)
		hasVector = true
	}
	for _, vec := range rec.EmbeddingMatrix {
		if len(vec) == 0 {
			continue
		}
		sim := CosineSimilarity(query, vec)
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
