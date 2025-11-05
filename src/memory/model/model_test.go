package model

import (
	json "github.com/alpkeskin/gotoon"
	"math"
	"testing"
	"time"
)

func TestNormalizeMetadata(t *testing.T) {
	fallback := time.Date(2024, time.January, 2, 3, 4, 5, 0, time.UTC)
	meta := map[string]any{
		"importance": json.Number("0.75"),
		"source":     42,
		"summary":    map[string]string{"text": "hello"},
		"graph_edges": []any{
			map[string]any{"target": json.Number("12"), "type": string(EdgeExplains)},
			map[string]any{"target": 0, "type": "unknown"},
		},
		EmbeddingMatrixKey: []any{
			[]any{json.Number("0.5"), 1, "not-a-number"},
			[]any{},
		},
	}
	importance, source, summary, lastEmbedded, jsonStr := NormalizeMetadata(meta, fallback)
	if math.Abs(importance-0.75) > 1e-9 {
		t.Fatalf("unexpected importance: %v", importance)
	}
	if source != "42" {
		t.Fatalf("unexpected source: %q", source)
	}
	if summary != "{\"text\":\"hello\"}" {
		t.Fatalf("unexpected summary: %q", summary)
	}
	if lastEmbedded.IsZero() {
		t.Fatal("expected lastEmbedded to be set")
	}
	decoded := DecodeMetadata(jsonStr)
	if len(decoded) == 0 || decoded["importance"].(float64) != importance {
		t.Fatalf("metadata not serialized correctly: %v", jsonStr)
	}
	edges := ValidGraphEdges(decoded)
	if len(edges) != 1 || edges[0].Target != 12 || edges[0].Type != EdgeExplains {
		t.Fatalf("unexpected sanitized edges: %#v", edges)
	}
	matrix := ValidEmbeddingMatrix(decoded)
	if len(matrix) != 1 || len(matrix[0]) != 2 {
		t.Fatalf("unexpected sanitized matrix: %#v", matrix)
	}
}

func TestCloneMetadataReturnsCopy(t *testing.T) {
	original := map[string]any{"foo": "bar"}
	cloned := CloneMetadata(original)
	if &original == &cloned {
		t.Fatal("expected new map instance")
	}
	cloned["foo"] = "baz"
	if original["foo"].(string) != "bar" {
		t.Fatal("expected original to remain unchanged")
	}
}

func TestFloatFromAny(t *testing.T) {
	cases := []struct {
		name string
		in   any
		want float64
	}{
		{"float64", 1.5, 1.5},
		{"int", 2, 2},
		{"json", json.Number("3.25"), 3.25},
		{"string", "4.5", 4.5},
		{"invalid", struct{}{}, 0},
	}
	for _, tc := range cases {
		if got := FloatFromAny(tc.in); math.Abs(got-tc.want) > 1e-9 {
			t.Errorf("%s: expected %v, got %v", tc.name, tc.want, got)
		}
	}
}

func TestStringFromAny(t *testing.T) {
	if got := StringFromAny(nil); got != "" {
		t.Fatalf("expected empty string, got %q", got)
	}
	if got := StringFromAny("hello"); got != "hello" {
		t.Fatalf("expected \"hello\", got %q", got)
	}
	got := StringFromAny(map[string]int{"answer": 42})
	if got != "{\"answer\":42}" {
		t.Fatalf("unexpected serialization: %q", got)
	}
}

func TestTimeFromAny(t *testing.T) {
	now := time.Now().UTC().Truncate(time.Second)
	if got := TimeFromAny(now); !got.Equal(now) {
		t.Fatalf("expected exact time, got %v", got)
	}
	if got := TimeFromAny(now.Format(time.RFC3339Nano)); !got.Equal(now) {
		t.Fatalf("expected parsed time, got %v", got)
	}
	if got := TimeFromAny("invalid"); !got.IsZero() {
		t.Fatalf("expected zero time, got %v", got)
	}
}

func TestDecodeMetadata(t *testing.T) {
	if meta := DecodeMetadata(""); len(meta) != 0 {
		t.Fatalf("expected empty map, got %v", meta)
	}
	if meta := DecodeMetadata("not json"); len(meta) != 0 {
		t.Fatalf("expected empty map for invalid json, got %v", meta)
	}
	meta := DecodeMetadata("{\"foo\":\"bar\"}")
	if meta["foo"] != "bar" {
		t.Fatalf("unexpected metadata: %v", meta)
	}
}

func TestHydrateRecordFromMetadata(t *testing.T) {
	ts := time.Now().UTC()
	rec := &MemoryRecord{}
	meta := map[string]any{
		"importance":    0.8,
		"source":        "memory",
		"summary":       "short",
		"last_embedded": ts.Format(time.RFC3339Nano),
		"space":         "team",
		"graph_edges": []any{
			map[string]any{"target": 10, "type": string(EdgeFollows)},
		},
		EmbeddingMatrixKey: []any{
			[]any{0.1, 0.2, 0.3},
		},
	}
	HydrateRecordFromMetadata(rec, meta)
	if rec.Importance != 0.8 || rec.Source != "memory" || rec.Summary != "short" || rec.Space != "team" {
		t.Fatalf("unexpected record hydration: %#v", rec)
	}
	if rec.LastEmbedded.IsZero() {
		t.Fatal("expected last embedded time to be populated")
	}
	if len(rec.GraphEdges) != 1 || rec.GraphEdges[0].Target != 10 {
		t.Fatalf("expected graph edges to be hydrated, got %#v", rec.GraphEdges)
	}
	if len(rec.EmbeddingMatrix) != 1 || len(rec.EmbeddingMatrix[0]) != 3 {
		t.Fatalf("expected embedding matrix to hydrate, got %#v", rec.EmbeddingMatrix)
	}
}

func TestDecodeEmbeddingMatrixVariants(t *testing.T) {
	cases := []struct {
		name string
		raw  any
		want int
	}{
		{"nil", nil, 0},
		{"single", []float32{1, 2}, 1},
		{"nested_any", []any{[]any{1, "2", json.Number("3")}}, 1},
		{"json_string", `[[1.1,2.2],[3.3,4.4]]`, 2},
		{"bytes", []byte(`[[5,6]]`), 1},
	}
	for _, tc := range cases {
		if got := DecodeEmbeddingMatrix(tc.raw); len(got) != tc.want {
			t.Fatalf("%s: expected %d rows, got %d", tc.name, tc.want, len(got))
		}
	}
}

func TestGraphEdgeValidationAndSanitize(t *testing.T) {
	edge := GraphEdge{Target: 5, Type: EdgeContradicts}
	if err := edge.Validate(); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	bad := GraphEdge{}
	if err := bad.Validate(); err == nil {
		t.Fatal("expected validation error for zero target")
	}

	meta := map[string]any{
		"graph_edges": []any{
			map[string]any{"target": 5, "type": string(EdgeDerivedFrom)},
			map[string]any{"target": 0, "type": "unknown"},
		},
	}
	edges := SanitizeGraphEdges(meta)
	if len(edges) != 1 || edges[0].Type != EdgeDerivedFrom {
		t.Fatalf("unexpected sanitized edges: %#v", edges)
	}
	if _, ok := meta["graph_edges"].([]GraphEdge); !ok {
		t.Fatalf("expected metadata to contain normalized edges, got %#v", meta["graph_edges"])
	}
}

func TestDecodeGraphEdgesVariants(t *testing.T) {
	edge := GraphEdge{Target: 7, Type: EdgeFollows}
	cases := []any{
		[]GraphEdge{edge},
		[]any{map[string]any{"target": 7, "type": string(EdgeFollows)}},
		map[string]any{"target": 7, "type": string(EdgeFollows)},
	}
	for _, in := range cases {
		out := DecodeGraphEdges(in)
		if len(out) != 1 || out[0] != edge {
			t.Fatalf("failed to decode edges from %T", in)
		}
	}
}

func TestNumericToInt(t *testing.T) {
	cases := []struct {
		in   any
		want int64
	}{
		{float64(3), 3},
		{float32(4), 4},
		{int(5), 5},
		{json.Number("6"), 6},
		{"7", 7},
		{struct{}{}, 0},
	}
	for _, tc := range cases {
		if got := numericToInt(tc.in); got != tc.want {
			t.Fatalf("expected %d, got %d", tc.want, got)
		}
	}
}

func TestValidGraphEdges(t *testing.T) {
	meta := map[string]any{
		"graph_edges": []any{
			[]any{8, string(EdgeFollows)},
			[]any{0, "bad"},
		},
	}
	edges := ValidGraphEdges(meta)
	if len(edges) != 1 || edges[0].Target != 8 {
		t.Fatalf("expected a single valid edge, got %#v", edges)
	}
}

func TestCosineSimilarity(t *testing.T) {
	if sim := CosineSimilarity(nil, nil); sim != 0 {
		t.Fatalf("expected zero similarity for empty vectors, got %v", sim)
	}
	a := []float32{1, 0}
	b := []float32{1, 0}
	if sim := CosineSimilarity(a, b); math.Abs(sim-1) > 1e-9 {
		t.Fatalf("expected similarity of 1, got %v", sim)
	}
	c := []float32{1, 0}
	d := []float32{0, 1}
	if sim := CosineSimilarity(c, d); math.Abs(sim) > 1e-9 {
		t.Fatalf("expected orthogonal vectors to have zero similarity, got %v", sim)
	}
}
