package model

import (
	"encoding/json"
	"strconv"
)

// EmbeddingMatrixKey is the metadata field used for multi-vector support.
const EmbeddingMatrixKey = "embedding_matrix"

// SanitizeEmbeddingMatrix normalizes the embedding matrix stored in metadata.
func SanitizeEmbeddingMatrix(meta map[string]any) [][]float32 {
	if meta == nil {
		return nil
	}
	raw, ok := meta[EmbeddingMatrixKey]
	if !ok {
		return nil
	}
	matrix := DecodeEmbeddingMatrix(raw)
	if len(matrix) == 0 {
		delete(meta, EmbeddingMatrixKey)
		return nil
	}
	meta[EmbeddingMatrixKey] = matrix
	return matrix
}

// ValidEmbeddingMatrix extracts a validated embedding matrix from metadata.
func ValidEmbeddingMatrix(meta map[string]any) [][]float32 {
	if meta == nil {
		return nil
	}
	raw, ok := meta[EmbeddingMatrixKey]
	if !ok {
		return nil
	}
	matrix := DecodeEmbeddingMatrix(raw)
	if len(matrix) == 0 {
		return nil
	}
	sanitized := make([][]float32, 0, len(matrix))
	for _, vec := range matrix {
		if len(vec) == 0 {
			continue
		}
		sanitized = append(sanitized, append([]float32(nil), vec...))
	}
	if len(sanitized) == 0 {
		return nil
	}
	return sanitized
}

// DecodeEmbeddingMatrix coerces arbitrary inputs into a matrix of float32 vectors.
func DecodeEmbeddingMatrix(raw any) [][]float32 {
	switch v := raw.(type) {
	case nil:
		return nil
	case [][]float32:
		out := make([][]float32, len(v))
		for i, row := range v {
			out[i] = append([]float32(nil), row...)
		}
		return out
	case []float32:
		if len(v) == 0 {
			return nil
		}
		return [][]float32{append([]float32(nil), v...)}
	case [][]float64:
		out := make([][]float32, 0, len(v))
		for _, row := range v {
			out = append(out, float64SliceTo32(row))
		}
		return out
	case []any:
		out := make([][]float32, 0, len(v))
		for _, row := range v {
			vec := decodeVector(row)
			if len(vec) == 0 {
				continue
			}
			out = append(out, vec)
		}
		return out
	case json.RawMessage:
		return DecodeEmbeddingMatrix([]byte(v))
	case []byte:
		var intermediate any
		if err := json.Unmarshal(v, &intermediate); err != nil {
			return nil
		}
		return DecodeEmbeddingMatrix(intermediate)
	case string:
		if v == "" {
			return nil
		}
		var intermediate any
		if err := json.Unmarshal([]byte(v), &intermediate); err != nil {
			return nil
		}
		return DecodeEmbeddingMatrix(intermediate)
	}
	return nil
}

func decodeVector(raw any) []float32 {
	switch row := raw.(type) {
	case []float32:
		if len(row) == 0 {
			return nil
		}
		return append([]float32(nil), row...)
	case []float64:
		return float64SliceTo32(row)
	case []int:
		out := make([]float32, len(row))
		for i, val := range row {
			out[i] = float32(val)
		}
		return out
	case []any:
		out := make([]float32, 0, len(row))
		for _, item := range row {
			if f, ok := numericToFloat32(item); ok {
				out = append(out, f)
			}
		}
		if len(out) == 0 {
			return nil
		}
		return out
	case json.RawMessage:
		return decodeVector([]byte(row))
	case []byte:
		var intermediate any
		if err := json.Unmarshal(row, &intermediate); err != nil {
			return nil
		}
		return decodeVector(intermediate)
	case string:
		if row == "" {
			return nil
		}
		var intermediate any
		if err := json.Unmarshal([]byte(row), &intermediate); err != nil {
			return nil
		}
		return decodeVector(intermediate)
	}
	if f, ok := numericToFloat32(raw); ok {
		return []float32{f}
	}
	return nil
}

func float64SliceTo32(in []float64) []float32 {
	if len(in) == 0 {
		return nil
	}
	out := make([]float32, len(in))
	for i, val := range in {
		out[i] = float32(val)
	}
	return out
}

func numericToFloat32(v any) (float32, bool) {
	switch n := v.(type) {
	case float32:
		return n, true
	case float64:
		return float32(n), true
	case int:
		return float32(n), true
	case int8:
		return float32(n), true
	case int16:
		return float32(n), true
	case int32:
		return float32(n), true
	case int64:
		return float32(n), true
	case uint:
		return float32(n), true
	case uint8:
		return float32(n), true
	case uint16:
		return float32(n), true
	case uint32:
		return float32(n), true
	case uint64:
		return float32(n), true
	case json.Number:
		f, err := n.Float64()
		if err != nil {
			return 0, false
		}
		return float32(f), true
	case string:
		if f, err := strconv.ParseFloat(n, 32); err == nil {
			return float32(f), true
		}
	}
	return 0, false
}

// RepresentativeEmbedding returns the first non-empty embedding from the record.
func RepresentativeEmbedding(rec MemoryRecord) []float32 {
	if len(rec.Embedding) > 0 {
		return rec.Embedding
	}
	if len(rec.EmbeddingMatrix) > 0 {
		for _, vec := range rec.EmbeddingMatrix {
			if len(vec) > 0 {
				return vec
			}
		}
	}
	return nil
}

// AllEmbeddings returns every embedding vector associated with the record.
func AllEmbeddings(rec MemoryRecord) [][]float32 {
	vectors := make([][]float32, 0, 1+len(rec.EmbeddingMatrix))
	if len(rec.Embedding) > 0 {
		vectors = append(vectors, rec.Embedding)
	}
	for _, vec := range rec.EmbeddingMatrix {
		if len(vec) == 0 {
			continue
		}
		vectors = append(vectors, vec)
	}
	return vectors
}
