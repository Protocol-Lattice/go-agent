package model

import (
	"encoding/json"
	"time"
)

func NormalizeMetadata(meta map[string]any, fallback time.Time) (importance float64, source, summary string, lastEmbedded time.Time, jsonString string) {
	meta = CloneMetadata(meta)
	importance = FloatFromAny(meta["importance"])
	source = StringFromAny(meta["source"])
	summary = StringFromAny(meta["summary"])
	lastEmbedded = TimeFromAny(meta["last_embedded"])
	if mv := Float32MatrixFromAny(meta["multi_embeddings"]); len(mv) > 0 {
		meta["multi_embeddings"] = mv
	} else {
		delete(meta, "multi_embeddings")
	}
	if space := StringFromAny(meta["space"]); space != "" {
		meta["space"] = space
	}
	edges := SanitizeGraphEdges(meta)
	if lastEmbedded.IsZero() {
		if fallback.IsZero() {
			fallback = time.Now().UTC()
		}
		lastEmbedded = fallback
	}
	meta["importance"] = importance
	meta["source"] = source
	meta["summary"] = summary
	meta["last_embedded"] = lastEmbedded.UTC().Format(time.RFC3339Nano)
	if len(edges) > 0 {
		meta["graph_edges"] = edges
	}
	jsonBytes, _ := json.Marshal(meta)
	jsonString = string(jsonBytes)
	return
}

func CloneMetadata(meta map[string]any) map[string]any {
	if meta == nil {
		return map[string]any{}
	}
	cp := make(map[string]any, len(meta))
	for k, v := range meta {
		cp[k] = v
	}
	return cp
}

func FloatFromAny(v any) float64 {
	switch t := v.(type) {
	case float64:
		return t
	case float32:
		return float64(t)
	case int:
		return float64(t)
	case int64:
		return float64(t)
	case json.Number:
		f, _ := t.Float64()
		return f
	case string:
		var f float64
		if err := json.Unmarshal([]byte(t), &f); err == nil {
			return f
		}
	}
	return 0
}

func StringFromAny(v any) string {
	if v == nil {
		return ""
	}
	switch t := v.(type) {
	case string:
		return t
	}
	b, err := json.Marshal(v)
	if err != nil {
		return ""
	}
	return string(b)
}

func TimeFromAny(v any) time.Time {
	switch t := v.(type) {
	case time.Time:
		return t
	case string:
		ts, err := time.Parse(time.RFC3339Nano, t)
		if err == nil {
			return ts
		}
	}
	return time.Time{}
}

func DecodeMetadata(metadata string) map[string]any {
	if metadata == "" {
		return map[string]any{}
	}
	var meta map[string]any
	if err := json.Unmarshal([]byte(metadata), &meta); err != nil {
		return map[string]any{}
	}
	return meta
}

func HydrateRecordFromMetadata(rec *MemoryRecord, meta map[string]any) {
	if rec == nil {
		return
	}
	if rec.Importance == 0 {
		rec.Importance = FloatFromAny(meta["importance"])
	}
	if rec.Source == "" {
		rec.Source = StringFromAny(meta["source"])
	}
	if rec.Summary == "" {
		rec.Summary = StringFromAny(meta["summary"])
	}
	if rec.LastEmbedded.IsZero() {
		if ts := TimeFromAny(meta["last_embedded"]); !ts.IsZero() {
			rec.LastEmbedded = ts
		}
	}
	if rec.Space == "" {
		if space := StringFromAny(meta["space"]); space != "" {
			rec.Space = space
		}
	}
	if len(rec.GraphEdges) == 0 {
		rec.GraphEdges = ValidGraphEdges(meta)
	}
	if len(rec.MultiEmbeddings) == 0 {
		rec.MultiEmbeddings = Float32MatrixFromAny(meta["multi_embeddings"])
	}
}

func Float32SliceFromAny(v any) []float32 {
	switch t := v.(type) {
	case nil:
		return nil
	case []float32:
		out := make([]float32, len(t))
		copy(out, t)
		return out
	case []float64:
		out := make([]float32, len(t))
		for i, val := range t {
			out[i] = float32(val)
		}
		return out
	case []any:
		out := make([]float32, 0, len(t))
		for _, val := range t {
			out = append(out, float32(FloatFromAny(val)))
		}
		return out
	case json.RawMessage:
		var arr []float64
		if err := json.Unmarshal(t, &arr); err == nil {
			return Float32SliceFromAny(arr)
		}
	case string:
		if t == "" {
			return nil
		}
		var arr []float64
		if err := json.Unmarshal([]byte(t), &arr); err == nil {
			return Float32SliceFromAny(arr)
		}
	}
	return nil
}

func Float32MatrixFromAny(v any) [][]float32 {
	switch t := v.(type) {
	case nil:
		return nil
	case [][]float32:
		out := make([][]float32, len(t))
		for i, row := range t {
			if row == nil {
				continue
			}
			cp := make([]float32, len(row))
			copy(cp, row)
			out[i] = cp
		}
		return out
	case []any:
		out := make([][]float32, 0, len(t))
		for _, row := range t {
			if slice := Float32SliceFromAny(row); len(slice) > 0 {
				out = append(out, slice)
			}
		}
		return out
	case json.RawMessage:
		var arr []any
		if err := json.Unmarshal(t, &arr); err == nil {
			return Float32MatrixFromAny(arr)
		}
	case string:
		if t == "" {
			return nil
		}
		var arr []any
		if err := json.Unmarshal([]byte(t), &arr); err == nil {
			return Float32MatrixFromAny(arr)
		}
	}
	return nil
}
