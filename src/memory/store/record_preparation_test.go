package store

import (
	"context"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
	"time"

	"github.com/Protocol-Lattice/go-agent/src/memory/model"
)

func TestPrepareMemoryRecord(t *testing.T) {
	now := time.Date(2026, time.July, 11, 8, 30, 0, 0, time.UTC)
	earlier := now.Add(-time.Hour)
	validEdge := model.GraphEdge{Target: 42, Type: model.EdgeExplains}

	tests := []struct {
		name                       string
		metadata                   map[string]any
		embedding                  []float32
		persistDefaultSpace        bool
		wantSpace                  string
		wantMetadataSpace          string
		wantMetadataSpacePresent   bool
		wantImportance             float64
		wantSource                 string
		wantSummary                string
		wantLastEmbedded           time.Time
		wantEdges                  []model.GraphEdge
		wantMatrix                 [][]float32
		wantEmbedding              []float32
		wantDefaultSpaceSideEffect bool
	}{
		{
			name:             "nil metadata",
			metadata:         nil,
			embedding:        []float32{1, 2},
			wantSpace:        "session-1",
			wantLastEmbedded: now,
			wantEmbedding:    []float32{1, 2},
		},
		{
			name:                       "default space persisted",
			metadata:                   map[string]any{},
			embedding:                  []float32{1},
			persistDefaultSpace:        true,
			wantSpace:                  "session-1",
			wantMetadataSpace:          "session-1",
			wantMetadataSpacePresent:   true,
			wantLastEmbedded:           now,
			wantEmbedding:              []float32{1},
			wantDefaultSpaceSideEffect: true,
		},
		{
			name: "explicit space and normalized metadata",
			metadata: map[string]any{
				"space":         "shared",
				"importance":    "0.75",
				"source":        "user",
				"summary":       "context",
				"last_embedded": earlier,
			},
			embedding:                []float32{1},
			persistDefaultSpace:      true,
			wantSpace:                "shared",
			wantMetadataSpace:        "shared",
			wantMetadataSpacePresent: true,
			wantImportance:           0.75,
			wantSource:               "user",
			wantSummary:              "context",
			wantLastEmbedded:         earlier,
			wantEmbedding:            []float32{1},
		},
		{
			name: "graph edges",
			metadata: map[string]any{
				"graph_edges": []any{
					map[string]any{"target": 42, "type": "explains"},
					map[string]any{"target": 0, "type": "follows"},
					map[string]any{"target": 9, "type": "unsupported"},
				},
			},
			embedding:        []float32{1},
			wantSpace:        "session-1",
			wantLastEmbedded: now,
			wantEdges:        []model.GraphEdge{validEdge},
			wantEmbedding:    []float32{1},
		},
		{
			name: "embedding matrix",
			metadata: map[string]any{
				model.EmbeddingMatrixKey: []any{
					[]any{0.25, 0.5},
					[]any{0.75, 1.0},
				},
			},
			embedding:        []float32{9, 8},
			wantSpace:        "session-1",
			wantLastEmbedded: now,
			wantMatrix:       [][]float32{{0.25, 0.5}, {0.75, 1}},
			wantEmbedding:    []float32{9, 8},
		},
		{
			name: "empty primary embedding falls back to matrix",
			metadata: map[string]any{
				model.EmbeddingMatrixKey: []any{
					[]any{},
					[]any{0.25, 0.5},
					[]any{0.75, 1.0},
				},
			},
			wantSpace:        "session-1",
			wantLastEmbedded: now,
			wantMatrix:       [][]float32{{0.25, 0.5}, {0.75, 1}},
			wantEmbedding:    []float32{0.25, 0.5},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			record := prepareMemoryRecord(
				"session-1",
				"content",
				test.metadata,
				test.embedding,
				now,
				test.persistDefaultSpace,
			)

			if record.SessionID != "session-1" || record.Content != "content" {
				t.Fatalf("unexpected record identity: %+v", record)
			}
			if record.Space != test.wantSpace {
				t.Fatalf("Space = %q, want %q", record.Space, test.wantSpace)
			}
			if !record.CreatedAt.Equal(now) {
				t.Fatalf("CreatedAt = %v, want %v", record.CreatedAt, now)
			}
			if !record.LastEmbedded.Equal(test.wantLastEmbedded) {
				t.Fatalf("LastEmbedded = %v, want %v", record.LastEmbedded, test.wantLastEmbedded)
			}
			if record.Importance != test.wantImportance || record.Source != test.wantSource || record.Summary != test.wantSummary {
				t.Fatalf("unexpected normalized metadata fields: importance=%v source=%q summary=%q", record.Importance, record.Source, record.Summary)
			}
			if !reflect.DeepEqual(record.GraphEdges, test.wantEdges) {
				t.Fatalf("GraphEdges = %#v, want %#v", record.GraphEdges, test.wantEdges)
			}
			if !reflect.DeepEqual(record.EmbeddingMatrix, test.wantMatrix) {
				t.Fatalf("EmbeddingMatrix = %#v, want %#v", record.EmbeddingMatrix, test.wantMatrix)
			}
			if !reflect.DeepEqual(record.Embedding, test.wantEmbedding) {
				t.Fatalf("Embedding = %#v, want %#v", record.Embedding, test.wantEmbedding)
			}

			normalizedMetadata := model.DecodeMetadata(record.Metadata)
			metadataSpace, metadataSpacePresent := normalizedMetadata["space"]
			if metadataSpacePresent != test.wantMetadataSpacePresent {
				t.Fatalf("normalized metadata space presence = %v, want %v", metadataSpacePresent, test.wantMetadataSpacePresent)
			}
			if got := model.StringFromAny(metadataSpace); got != test.wantMetadataSpace {
				t.Fatalf("normalized metadata space = %q, want %q", got, test.wantMetadataSpace)
			}
			if got := model.ValidGraphEdges(normalizedMetadata); !reflect.DeepEqual(got, test.wantEdges) {
				t.Fatalf("normalized metadata graph edges = %#v, want %#v", got, test.wantEdges)
			}
			if got := model.ValidEmbeddingMatrix(normalizedMetadata); !reflect.DeepEqual(got, test.wantMatrix) {
				t.Fatalf("normalized metadata embedding matrix = %#v, want %#v", got, test.wantMatrix)
			}
			if got := model.TimeFromAny(normalizedMetadata["last_embedded"]); !got.Equal(test.wantLastEmbedded) {
				t.Fatalf("normalized metadata last_embedded = %v, want %v", got, test.wantLastEmbedded)
			}
			if test.metadata != nil {
				_, inputHasSpace := test.metadata["space"]
				if inputHasSpace != test.wantDefaultSpaceSideEffect && test.wantMetadataSpace == "session-1" {
					t.Fatalf("input metadata default-space side effect = %v, want %v", inputHasSpace, test.wantDefaultSpaceSideEffect)
				}
			}
		})
	}
}

func TestQdrantStoreMemoryPreservesMetadataSideEffects(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"status":"ok","result":{}}`))
	}))
	defer server.Close()

	validEdge := model.GraphEdge{Target: 42, Type: model.EdgeExplains}
	tests := []struct {
		name          string
		metadata      map[string]any
		wantEdges     []model.GraphEdge
		wantEdgesKept bool
	}{
		{
			name: "sanitizes mixed edges",
			metadata: map[string]any{
				"graph_edges": []any{
					map[string]any{"target": 42, "type": "explains"},
					map[string]any{"target": 0, "type": "follows"},
				},
			},
			wantEdges:     []model.GraphEdge{validEdge},
			wantEdgesKept: true,
		},
		{
			name: "removes invalid edges",
			metadata: map[string]any{
				"graph_edges": []any{
					map[string]any{"target": 0, "type": "follows"},
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			store := NewQdrantStore(server.URL, "memories", "")
			if err := store.StoreMemory(context.Background(), "session-1", "content", test.metadata, []float32{1}); err != nil {
				t.Fatalf("StoreMemory returned error: %v", err)
			}

			if got := model.StringFromAny(test.metadata["space"]); got != "session-1" {
				t.Fatalf("input metadata space = %q, want session-1", got)
			}
			rawEdges, edgesKept := test.metadata["graph_edges"]
			if edgesKept != test.wantEdgesKept {
				t.Fatalf("input metadata graph_edges presence = %v, want %v", edgesKept, test.wantEdgesKept)
			}
			if edgesKept {
				gotEdges, ok := rawEdges.([]model.GraphEdge)
				if !ok {
					t.Fatalf("input metadata graph_edges type = %T, want []model.GraphEdge", rawEdges)
				}
				if !reflect.DeepEqual(gotEdges, test.wantEdges) {
					t.Fatalf("input metadata graph_edges = %#v, want %#v", gotEdges, test.wantEdges)
				}
			}
		})
	}
}
