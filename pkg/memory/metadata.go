package memory

import (
	"encoding/json"
	"time"
)

func normalizeMetadata(meta map[string]any, fallback time.Time) (importance float64, source, summary string, lastEmbedded time.Time, jsonString string) {
	meta = cloneMetadata(meta)
	importance = floatFromAny(meta["importance"])
	source = stringFromAny(meta["source"])
	summary = stringFromAny(meta["summary"])
	lastEmbedded = timeFromAny(meta["last_embedded"])
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
	jsonBytes, _ := json.Marshal(meta)
	jsonString = string(jsonBytes)
	return
}

func cloneMetadata(meta map[string]any) map[string]any {
	if meta == nil {
		return map[string]any{}
	}
	cp := make(map[string]any, len(meta))
	for k, v := range meta {
		cp[k] = v
	}
	return cp
}

func floatFromAny(v any) float64 {
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

func stringFromAny(v any) string {
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

func timeFromAny(v any) time.Time {
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

func decodeMetadata(metadata string) map[string]any {
	if metadata == "" {
		return map[string]any{}
	}
	var meta map[string]any
	if err := json.Unmarshal([]byte(metadata), &meta); err != nil {
		return map[string]any{}
	}
	return meta
}

func hydrateRecordFromMetadata(rec *MemoryRecord, meta map[string]any) {
	if rec == nil {
		return
	}
	if rec.Importance == 0 {
		rec.Importance = floatFromAny(meta["importance"])
	}
	if rec.Source == "" {
		rec.Source = stringFromAny(meta["source"])
	}
	if rec.Summary == "" {
		rec.Summary = stringFromAny(meta["summary"])
	}
	if rec.LastEmbedded.IsZero() {
		if ts := timeFromAny(meta["last_embedded"]); !ts.IsZero() {
			rec.LastEmbedded = ts
		}
	}
}
