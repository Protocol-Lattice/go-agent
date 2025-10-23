package store

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestNormalizeQdrantVectorsSingle(t *testing.T) {
	raw := json.RawMessage(`{"size": 1536}`)
	normalized, err := normalizeQdrantVectors(raw)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var obj map[string]any
	if err := json.Unmarshal(normalized, &obj); err != nil {
		t.Fatalf("unmarshal normalized: %v", err)
	}
	if obj["distance"] != string(DistanceCosine) {
		t.Fatalf("expected default cosine distance, got %v", obj["distance"])
	}
}

func TestNormalizeQdrantVectorsNamed(t *testing.T) {
	raw := json.RawMessage(`{"text": {"size": 768}, "image": {"size": 512, "distance": "Dot"}}`)
	normalized, err := normalizeQdrantVectors(raw)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var obj map[string]map[string]any
	if err := json.Unmarshal(normalized, &obj); err != nil {
		t.Fatalf("unmarshal normalized: %v", err)
	}
	if obj["text"]["distance"] != string(DistanceCosine) {
		t.Fatalf("expected cosine distance for text vector, got %v", obj["text"]["distance"])
	}
	if obj["image"]["distance"] != "Dot" {
		t.Fatalf("expected dot distance preserved for image vector, got %v", obj["image"]["distance"])
	}
}

func TestNormalizeQdrantVectorsMissingSize(t *testing.T) {
	raw := json.RawMessage(`{"text": {"distance": "Cosine"}}`)
	if _, err := normalizeQdrantVectors(raw); err == nil {
		t.Fatalf("expected error for missing size")
	}
}

func TestNormalizeQdrantVectorsInvalidJSON(t *testing.T) {
	raw := json.RawMessage(`[1,2,3]`)
	if _, err := normalizeQdrantVectors(raw); err == nil {
		t.Fatalf("expected error for non-object JSON")
	}
}

func TestIsPositiveNumber(t *testing.T) {
	cases := []struct {
		input any
		want  bool
	}{
		{input: 1, want: true},
		{input: 0, want: false},
		{input: -1, want: false},
		{input: "42", want: true},
		{input: "", want: false},
		{input: json.Number("5"), want: true},
		{input: json.Number("0"), want: false},
		{input: nil, want: false},
	}
	for _, tc := range cases {
		if got := isPositiveNumber(tc.input); got != tc.want {
			t.Fatalf("isPositiveNumber(%v) = %v, want %v", tc.input, got, tc.want)
		}
	}
}

func TestNormalizeQdrantVectorsIdempotent(t *testing.T) {
	raw := json.RawMessage(`{"text": {"size": 64, "distance": "Cosine"}}`)
	normalized, err := normalizeQdrantVectors(raw)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var original map[string]any
	if err := json.Unmarshal(raw, &original); err != nil {
		t.Fatalf("unmarshal original: %v", err)
	}
	var updated map[string]any
	if err := json.Unmarshal(normalized, &updated); err != nil {
		t.Fatalf("unmarshal normalized: %v", err)
	}
	if !reflect.DeepEqual(original, updated) {
		t.Fatalf("expected no change for already normalized config")
	}
}
