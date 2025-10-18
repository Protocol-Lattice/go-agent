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
}
