package store

import (
	"context"
	"errors"
	"sort"
	"time"

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"

	"github.com/Protocol-Lattice/agent/core/memory/model"
)

type MongoStore struct {
	client            *mongo.Client
	collection        *mongo.Collection
	counterCollection *mongo.Collection
}

const mongoCloseTimeout = 5 * time.Second

func NewMongoStore(ctx context.Context, uri, database, collection string) (*MongoStore, error) {
	if uri == "" {
		return nil, errors.New("mongo uri is required")
	}
	if database == "" {
		return nil, errors.New("mongo database name is required")
	}
	if collection == "" {
		return nil, errors.New("mongo collection name is required")
	}
	client, err := mongo.Connect(ctx, options.Client().ApplyURI(uri))
	if err != nil {
		return nil, err
	}
	if err := client.Ping(ctx, nil); err != nil {
		_ = client.Disconnect(ctx)
		return nil, err
	}
	db := client.Database(database)
	return &MongoStore{
		client:            client,
		collection:        db.Collection(collection),
		counterCollection: db.Collection("counters"),
	}, nil
}

func (ms *MongoStore) StoreMemory(ctx context.Context, sessionID, content string, metadata map[string]any, embedding []float32) error {
	if ms == nil || ms.collection == nil {
		return nil
	}
	if metadata == nil {
		metadata = map[string]any{}
	}
	if _, ok := metadata["space"]; !ok {
		metadata["space"] = sessionID
	}
	now := time.Now().UTC()
	importance, source, summary, lastEmbedded, metadataJSON := model.NormalizeMetadata(metadata, now)
	meta := model.DecodeMetadata(metadataJSON)
	space := model.StringFromAny(meta["space"])
	if space == "" {
		space = sessionID
	}
	edges := model.ValidGraphEdges(meta)
	matrix := model.ValidEmbeddingMatrix(meta)
	storedEmbedding := append([]float32(nil), embedding...)
	if len(storedEmbedding) == 0 {
		for _, vec := range matrix {
			if len(vec) == 0 {
				continue
			}
			storedEmbedding = append([]float32(nil), vec...)
			break
		}
	}

	id, err := ms.nextID(ctx)
	if err != nil {
		return err
	}

	doc := bson.M{
		"_id":           id,
		"session_id":    sessionID,
		"space":         space,
		"content":       content,
		"metadata":      metadataJSON,
		"embedding":     float64Embedding(storedEmbedding),
		"importance":    importance,
		"source":        source,
		"summary":       summary,
		"created_at":    now,
		"last_embedded": lastEmbedded,
	}
	if len(matrix) > 0 {
		doc["embedding_matrix"] = float64Matrix(matrix)
	}
	if len(edges) > 0 {
		doc["graph_edges"] = edges
	}
	_, err = ms.collection.InsertOne(ctx, doc)
	return err
}

func (ms *MongoStore) SearchMemory(ctx context.Context, queryEmbedding []float32, limit int) ([]model.MemoryRecord, error) {
	if ms == nil || ms.collection == nil || limit <= 0 {
		return nil, nil
	}
	cursor, err := ms.collection.Find(ctx, bson.M{})
	if err != nil {
		return nil, err
	}
	defer cursor.Close(ctx)

	type scored struct {
		record model.MemoryRecord
		score  float64
	}
	var scoredRecords []scored
	for cursor.Next(ctx) {
		var doc mongoMemoryDocument
		if err := cursor.Decode(&doc); err != nil {
			return nil, err
		}
		rec := doc.toRecord()
		rec.Score = model.MaxCosineSimilarity(queryEmbedding, rec)
		scoredRecords = append(scoredRecords, scored{record: rec, score: rec.Score})
	}
	if err := cursor.Err(); err != nil {
		return nil, err
	}
	sort.Slice(scoredRecords, func(i, j int) bool { return scoredRecords[i].score > scoredRecords[j].score })
	if len(scoredRecords) > limit {
		scoredRecords = scoredRecords[:limit]
	}
	records := make([]model.MemoryRecord, len(scoredRecords))
	for i, sc := range scoredRecords {
		records[i] = sc.record
	}
	return records, nil
}

func (ms *MongoStore) UpdateEmbedding(ctx context.Context, id int64, embedding []float32, lastEmbedded time.Time) error {
	if ms == nil || ms.collection == nil {
		return nil
	}
	_, err := ms.collection.UpdateByID(ctx, id, bson.M{
		"$set": bson.M{
			"embedding":     float64Embedding(embedding),
			"last_embedded": lastEmbedded,
		},
	})
	return err
}

func (ms *MongoStore) DeleteMemory(ctx context.Context, ids []int64) error {
	if ms == nil || ms.collection == nil || len(ids) == 0 {
		return nil
	}
	_, err := ms.collection.DeleteMany(ctx, bson.M{"_id": bson.M{"$in": ids}})
	return err
}

func (ms *MongoStore) Iterate(ctx context.Context, fn func(model.MemoryRecord) bool) error {
	if ms == nil || ms.collection == nil {
		return nil
	}
	opts := options.Find().SetSort(bson.D{{Key: "created_at", Value: 1}})
	cursor, err := ms.collection.Find(ctx, bson.M{}, opts)
	if err != nil {
		return err
	}
	defer cursor.Close(ctx)
	for cursor.Next(ctx) {
		var doc mongoMemoryDocument
		if err := cursor.Decode(&doc); err != nil {
			return err
		}
		if cont := fn(doc.toRecord()); !cont {
			break
		}
	}
	return cursor.Err()
}

func (ms *MongoStore) Count(ctx context.Context) (int, error) {
	if ms == nil || ms.collection == nil {
		return 0, nil
	}
	count, err := ms.collection.CountDocuments(ctx, bson.M{})
	return int(count), err
}

// CreateSchema ensures the primary collection has useful indexes and initializes the counter collection.
func (ms *MongoStore) CreateSchema(ctx context.Context, _ string) error {
	if ms == nil || ms.collection == nil {
		return nil
	}

	indexes := []mongo.IndexModel{
		{
			Keys:    bson.D{{Key: "session_id", Value: 1}, {Key: "created_at", Value: -1}},
			Options: options.Index().SetName("session_created_at"),
		},
		{
			Keys:    bson.D{{Key: "space", Value: 1}},
			Options: options.Index().SetName("space"),
		},
	}
	if _, err := ms.collection.Indexes().CreateMany(ctx, indexes); err != nil {
		return err
	}

	if ms.counterCollection != nil {
		_, err := ms.counterCollection.Indexes().CreateOne(ctx, mongo.IndexModel{
			Keys:    bson.D{{Key: "_id", Value: 1}},
			Options: options.Index().SetName("counter_id").SetUnique(true),
		})
		if err != nil {
			return err
		}
	}

	return nil
}

func (ms *MongoStore) nextID(ctx context.Context) (int64, error) {
	if ms.counterCollection == nil {
		return 0, errors.New("mongo counter collection is not configured")
	}
	opts := options.FindOneAndUpdate().SetUpsert(true).SetReturnDocument(options.After)
	res := ms.counterCollection.FindOneAndUpdate(ctx, bson.M{"_id": ms.collection.Name()}, bson.M{"$inc": bson.M{"seq": 1}}, opts)
	if res.Err() != nil {
		return 0, res.Err()
	}
	var doc struct {
		Seq int64 `bson:"seq"`
	}
	if err := res.Decode(&doc); err != nil {
		return 0, err
	}
	return doc.Seq, nil
}

type mongoMemoryDocument struct {
	ID           int64             `bson:"_id"`
	SessionID    string            `bson:"session_id"`
	Space        string            `bson:"space"`
	Content      string            `bson:"content"`
	Metadata     string            `bson:"metadata"`
	Embedding    []float64         `bson:"embedding"`
	EmbeddingMat [][]float64       `bson:"embedding_matrix,omitempty"`
	Importance   float64           `bson:"importance"`
	Source       string            `bson:"source"`
	Summary      string            `bson:"summary"`
	CreatedAt    time.Time         `bson:"created_at"`
	LastEmbedded time.Time         `bson:"last_embedded"`
	GraphEdges   []model.GraphEdge `bson:"graph_edges,omitempty"`
}

func (doc mongoMemoryDocument) toRecord() model.MemoryRecord {
	rec := model.MemoryRecord{
		ID:              doc.ID,
		SessionID:       doc.SessionID,
		Space:           doc.Space,
		Content:         doc.Content,
		Metadata:        doc.Metadata,
		Embedding:       float32Embedding(doc.Embedding),
		EmbeddingMatrix: float32Matrix(doc.EmbeddingMat),
		Importance:      doc.Importance,
		Source:          doc.Source,
		Summary:         doc.Summary,
		CreatedAt:       doc.CreatedAt,
		LastEmbedded:    doc.LastEmbedded,
		GraphEdges:      doc.GraphEdges,
	}
	meta := model.DecodeMetadata(rec.Metadata)
	model.HydrateRecordFromMetadata(&rec, meta)
	if len(rec.EmbeddingMatrix) == 0 {
		rec.EmbeddingMatrix = float32Matrix(doc.EmbeddingMat)
	}
	if rec.Space == "" {
		rec.Space = doc.Space
	}
	if rec.Space == "" {
		rec.Space = rec.SessionID
	}
	return rec
}

func float64Embedding(vec []float32) []float64 {
	if len(vec) == 0 {
		return nil
	}
	out := make([]float64, len(vec))
	for i, v := range vec {
		out[i] = float64(v)
	}
	return out
}

func float64Matrix(matrix [][]float32) [][]float64 {
	if len(matrix) == 0 {
		return nil
	}
	out := make([][]float64, len(matrix))
	for i, row := range matrix {
		if len(row) == 0 {
			continue
		}
		converted := make([]float64, len(row))
		for j, val := range row {
			converted[j] = float64(val)
		}
		out[i] = converted
	}
	return out
}

func float32Embedding(vec []float64) []float32 {
	if len(vec) == 0 {
		return nil
	}
	out := make([]float32, len(vec))
	for i, v := range vec {
		out[i] = float32(v)
	}
	return out
}

func float32Matrix(matrix [][]float64) [][]float32 {
	if len(matrix) == 0 {
		return nil
	}
	out := make([][]float32, 0, len(matrix))
	for _, row := range matrix {
		if len(row) == 0 {
			continue
		}
		converted := make([]float32, len(row))
		for i, val := range row {
			converted[i] = float32(val)
		}
		out = append(out, converted)
	}
	return out
}

// Close releases the underlying MongoDB client.
func (ms *MongoStore) Close() error {
	if ms == nil || ms.client == nil {
		return nil
	}
	ctx, cancel := context.WithTimeout(context.Background(), mongoCloseTimeout)
	defer cancel()
	return ms.client.Disconnect(ctx)
}
