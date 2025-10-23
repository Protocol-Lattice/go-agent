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
	vectors := AllEmbeddings(rec)
	if len(vectors) == 0 {
		return 0
	}
	best := CosineSimilarity(query, vectors[0])
	for _, vec := range vectors[1:] {
		if sim := CosineSimilarity(query, vec); sim > best {
			best = sim
		}
	}
	return best
}

// RecordSimilarity computes the maximum similarity between any pair of
// embeddings contained within the two records.
func RecordSimilarity(a, b MemoryRecord) float64 {
	aVectors := AllEmbeddings(a)
	bVectors := AllEmbeddings(b)
	if len(aVectors) == 0 || len(bVectors) == 0 {
		return 0
	}
	best := CosineSimilarity(aVectors[0], bVectors[0])
	for _, va := range aVectors {
		for _, vb := range bVectors {
			if sim := CosineSimilarity(va, vb); sim > best {
				best = sim
			}
		}
	}
	return best
}
