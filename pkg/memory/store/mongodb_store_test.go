package store

import (
	"context"
	"testing"
	"time"

	"github.com/Raezil/go-agent-development-kit/pkg/memory/model"
)

func TestFloatEmbeddingConversions(t *testing.T) {
	original := []float32{1.25, -2, 0, 3.5}
	converted := float64Embedding(original)
	if len(converted) != len(original) {
		t.Fatalf("unexpected converted length: got %d want %d", len(converted), len(original))
	}
	roundTrip := float32Embedding(converted)
	if len(roundTrip) != len(original) {
		t.Fatalf("unexpected round-trip length: got %d want %d", len(roundTrip), len(original))
	}
	for i := range original {
		if roundTrip[i] != original[i] {
			t.Fatalf("value mismatch at %d: got %f want %f", i, roundTrip[i], original[i])
		}
	}
}

func TestMongoMemoryDocumentToRecordHydratesMetadata(t *testing.T) {
	ts := time.Now().UTC().Truncate(time.Millisecond)
	metadata := `{"space":"custom","importance":0.8,"source":"user","summary":"context","last_embedded":"` + ts.Format(time.RFC3339Nano) + `","graph_edges":[{"target":7,"type":"explains"}],"multi_embeddings":[[0.1,0.2],[0.3,0.4]]}`

	doc := mongoMemoryDocument{
		ID:        42,
		SessionID: "session-123",
		Content:   "stored content",
		Metadata:  metadata,
		Embedding: []float64{0.1, 0.2, 0.3},
		CreatedAt: ts,
	}

	rec := doc.toRecord()

	if rec.ID != doc.ID {
		t.Fatalf("expected ID %d, got %d", doc.ID, rec.ID)
	}
	if rec.SessionID != doc.SessionID {
		t.Fatalf("expected session %q, got %q", doc.SessionID, rec.SessionID)
	}
	if rec.Space != "custom" {
		t.Fatalf("expected space %q, got %q", "custom", rec.Space)
	}
	if rec.Importance != 0.8 {
		t.Fatalf("expected importance 0.8, got %f", rec.Importance)
	}
	if rec.Source != "user" {
		t.Fatalf("expected source %q, got %q", "user", rec.Source)
	}
	if rec.Summary != "context" {
		t.Fatalf("expected summary %q, got %q", "context", rec.Summary)
	}
	if rec.LastEmbedded.IsZero() || !rec.LastEmbedded.Equal(ts) {
		t.Fatalf("expected last embedded %v, got %v", ts, rec.LastEmbedded)
	}
	if len(rec.GraphEdges) != 1 {
		t.Fatalf("expected 1 graph edge, got %d", len(rec.GraphEdges))
	}
	expectedEdge := model.GraphEdge{Target: 7, Type: model.EdgeExplains}
	if rec.GraphEdges[0] != expectedEdge {
		t.Fatalf("unexpected graph edge: got %+v want %+v", rec.GraphEdges[0], expectedEdge)
	}
	if len(rec.Embedding) != len(doc.Embedding) {
		t.Fatalf("expected embedding length %d, got %d", len(doc.Embedding), len(rec.Embedding))
	}
	if len(rec.MultiEmbeddings) != 2 {
		t.Fatalf("expected 2 multi embeddings, got %d", len(rec.MultiEmbeddings))
	}
}

func TestMongoStoreCloseNilClient(t *testing.T) {
	store := &MongoStore{}
	if err := store.Close(); err != nil {
		t.Fatalf("expected nil error closing nil client, got %v", err)
	}
}

func TestMongoStoreCreateSchemaOnNilStore(t *testing.T) {
	var store *MongoStore
	if err := store.CreateSchema(nil, ""); err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}
	store = &MongoStore{}
	if err := store.CreateSchema(nil, ""); err != nil {
		t.Fatalf("expected nil error when collection is nil, got %v", err)
	}
}

func TestMongoStoreStoreMemoryMultiMetadata(t *testing.T) {
	store := &MongoStore{}
	meta := map[string]any{}
	embeddings := [][]float32{{1, 0}, {0.5, 0.5}}
	if err := store.StoreMemoryMulti(context.Background(), "session", "content", meta, embeddings); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	extras, ok := meta["multi_embeddings"].([][]float32)
	if !ok {
		t.Fatalf("expected multi_embeddings stored in metadata")
	}
	if len(extras) != 1 {
		t.Fatalf("expected 1 auxiliary vector, got %d", len(extras))
	}
	embeddings[1][0] = 99
	if extras[0][0] == 99 {
		t.Fatalf("expected auxiliary vectors to be cloned, but saw mutation")
	}

	meta = map[string]any{"multi_embeddings": [][]float32{{42}}}
	if err := store.StoreMemoryMulti(context.Background(), "session", "content", meta, embeddings[:1]); err != nil {
		t.Fatalf("unexpected error for single vector: %v", err)
	}
	if _, exists := meta["multi_embeddings"]; exists {
		t.Fatalf("expected multi_embeddings removed when only primary vector provided")
	}
}

func TestMongoStoreSearchMemoryMultiNil(t *testing.T) {
	var store *MongoStore
	results, err := store.SearchMemoryMulti(context.Background(), [][]float32{{1, 2}}, 5)
	if err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}
	if results != nil {
		t.Fatalf("expected nil results, got %v", results)
	}
}
