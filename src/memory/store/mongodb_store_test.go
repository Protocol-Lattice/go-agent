package store

import (
	"context"
	"testing"
	"time"

	"github.com/Protocol-Lattice/go-agent/src/memory/model"
	"go.mongodb.org/mongo-driver/bson"
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

func TestMongoCosineSimilarityDenormalizesVectorSearchScore(t *testing.T) {
	for score, want := range map[float64]float64{1: 1, 0.5: 0, 0: -1, 0.9: 0.8} {
		if got := mongoCosineSimilarity(score); got != want {
			t.Fatalf("mongoCosineSimilarity(%v) = %v, want %v", score, got, want)
		}
	}
}

func TestMongoMemoryDocumentToRecordHydratesMetadata(t *testing.T) {
	ts := time.Now().UTC().Truncate(time.Millisecond)
	metadata := `{"space":"custom","importance":0.8,"source":"user","summary":"context","last_embedded":"` + ts.Format(time.RFC3339Nano) + `","graph_edges":[{"target":7,"type":"explains"}],"embedding_matrix":[[0.5,0.6]]}`

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
	if len(rec.EmbeddingMatrix) != 1 || len(rec.EmbeddingMatrix[0]) != 2 {
		t.Fatalf("expected embedding matrix hydrated, got %#v", rec.EmbeddingMatrix)
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
	if err := store.CreateSchema(context.TODO(), ""); err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}
	store = &MongoStore{}
	if err := store.CreateSchema(context.TODO(), ""); err != nil {
		t.Fatalf("expected nil error when collection is nil, got %v", err)
	}
}

func TestMongoVectorSearchPipelinePushesDownSessionFilter(t *testing.T) {
	pipeline := mongoVectorSearchPipeline("session-1", []float32{1, 0}, 8)
	if len(pipeline) != 2 {
		t.Fatalf("pipeline has %d stages, want 2", len(pipeline))
	}

	vectorSearch, ok := documentValue(pipeline[0], "$vectorSearch")
	if !ok {
		t.Fatalf("missing $vectorSearch stage: %#v", pipeline)
	}
	searchDoc, ok := vectorSearch.(bson.D)
	if !ok {
		t.Fatalf("$vectorSearch value has type %T, want bson.D", vectorSearch)
	}
	filter, ok := documentValue(searchDoc, "filter")
	if !ok {
		t.Fatalf("session filter was not pushed into $vectorSearch: %#v", searchDoc)
	}
	filterDoc, ok := filter.(bson.D)
	if !ok {
		t.Fatalf("filter has type %T, want bson.D", filter)
	}
	if got, ok := documentValue(filterDoc, "session_id"); !ok || got != "session-1" {
		t.Fatalf("session filter = %#v, want session-1", got)
	}
	for _, stage := range pipeline {
		if _, ok := documentValue(stage, "$match"); ok {
			t.Fatalf("session filter must not use a post-search $match: %#v", pipeline)
		}
	}
}

func documentValue(document bson.D, key string) (any, bool) {
	for _, element := range document {
		if element.Key == key {
			return element.Value, true
		}
	}
	return nil, false
}
