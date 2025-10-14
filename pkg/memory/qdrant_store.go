package memory

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/google/uuid"
)

// QdrantStore implements VectorStore using the Qdrant HTTP API.
type QdrantStore struct {
	Endpoint   string
	APIKey     string
	Collection string
	Client     *http.Client
}

// NewQdrantStore configures a Qdrant-backed VectorStore implementation.
func NewQdrantStore(endpoint, apiKey, collection string) *QdrantStore {
	endpoint = strings.TrimSuffix(endpoint, "/")
	return &QdrantStore{
		Endpoint:   endpoint,
		APIKey:     apiKey,
		Collection: collection,
		Client: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// StoreMemory writes a single memory record to Qdrant via the upsert endpoint.
func (qs *QdrantStore) StoreMemory(ctx context.Context, sessionID, content, metadata string, embedding []float32) error {
	if qs == nil || qs.Client == nil {
		return nil
	}

	payload := map[string]any{
		"session_id": sessionID,
		"content":    content,
		"metadata":   metadata,
	}

	body := map[string]any{
		"points": []any{
			map[string]any{
				"id":      uuid.NewString(),
				"vector":  embedding,
				"payload": payload,
			},
		},
	}

	return qs.doRequest(ctx, http.MethodPut, fmt.Sprintf("%s/collections/%s/points", qs.Endpoint, qs.Collection), body, nil)
}

// SearchMemory queries Qdrant for the closest vectors.
func (qs *QdrantStore) SearchMemory(ctx context.Context, queryEmbedding []float32, limit int) ([]MemoryRecord, error) {
	if qs == nil || qs.Client == nil {
		return nil, nil
	}

	body := map[string]any{
		"vector":       queryEmbedding,
		"limit":        limit,
		"with_payload": true,
		"with_vectors": false,
	}

	var resp qdrantSearchResponse
	if err := qs.doRequest(ctx, http.MethodPost, fmt.Sprintf("%s/collections/%s/points/search", qs.Endpoint, qs.Collection), body, &resp); err != nil {
		return nil, err
	}

	records := make([]MemoryRecord, 0, len(resp.Result))
	for _, point := range resp.Result {
		record := MemoryRecord{Score: point.Score}

		switch v := point.ID.(type) {
		case float64:
			record.ID = int64(v)
		case json.Number:
			if i, err := v.Int64(); err == nil {
				record.ID = i
			}
		}

		if session, ok := point.Payload["session_id"].(string); ok {
			record.SessionID = session
		}
		if content, ok := point.Payload["content"].(string); ok {
			record.Content = content
		}
		if metadata := point.Payload["metadata"]; metadata != nil {
			record.Metadata = stringifyPayload(metadata)
		}

		records = append(records, record)
	}

	return records, nil
}

// CreateSchema initialises the Qdrant collection using the provided schema file when available.
func (qs *QdrantStore) CreateSchema(ctx context.Context, schemaPath string) error {
	if qs == nil || qs.Client == nil {
		return nil
	}

	schema := qdrantSchema{VectorSize: 768, Distance: "Cosine"}
	if schemaPath != "" {
		data, err := os.ReadFile(schemaPath)
		if err != nil {
			return fmt.Errorf("failed to read qdrant schema file: %w", err)
		}
		if err := json.Unmarshal(data, &schema); err != nil {
			return fmt.Errorf("failed to parse qdrant schema file: %w", err)
		}
	}

	if schema.VectorSize <= 0 {
		schema.VectorSize = 768
	}
	if schema.Distance == "" {
		schema.Distance = "Cosine"
	}

	return qs.EnsureCollection(ctx, schema.VectorSize, schema.Distance)
}

// EnsureCollection creates the collection in Qdrant if it does not exist.
func (qs *QdrantStore) EnsureCollection(ctx context.Context, vectorSize int, distance string) error {
	if qs == nil || qs.Client == nil {
		return nil
	}

	body := map[string]any{
		"vectors": map[string]any{
			"size":     vectorSize,
			"distance": distance,
		},
	}

	// Qdrant returns 200 if the collection exists and 201 if created.
	err := qs.doRequest(ctx, http.MethodPut, fmt.Sprintf("%s/collections/%s", qs.Endpoint, qs.Collection), body, nil)
	if err != nil && !isConflictError(err) {
		return err
	}
	return nil
}

func (qs *QdrantStore) doRequest(ctx context.Context, method, url string, body any, out any) error {
	if qs == nil || qs.Client == nil {
		return nil
	}

	var buf io.ReadWriter
	if body != nil {
		buf = &bytes.Buffer{}
		if err := json.NewEncoder(buf).Encode(body); err != nil {
			return fmt.Errorf("failed to encode qdrant request: %w", err)
		}
	}

	req, err := http.NewRequestWithContext(ctx, method, url, buf)
	if err != nil {
		return fmt.Errorf("failed to create qdrant request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if qs.APIKey != "" {
		req.Header.Set("api-key", qs.APIKey)
	}

	resp, err := qs.Client.Do(req)
	if err != nil {
		return fmt.Errorf("qdrant request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		data, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("qdrant request error: %s", strings.TrimSpace(string(data)))
	}

	if out != nil {
		decoder := json.NewDecoder(resp.Body)
		decoder.UseNumber()
		if err := decoder.Decode(out); err != nil {
			return fmt.Errorf("failed to decode qdrant response: %w", err)
		}
	}

	return nil
}

// stringifyPayload converts arbitrary payload values into a string representation for MemoryRecord.Metadata.
func stringifyPayload(value any) string {
	switch v := value.(type) {
	case string:
		return v
	case fmt.Stringer:
		return v.String()
	default:
		data, err := json.Marshal(v)
		if err != nil {
			return fmt.Sprint(v)
		}
		return string(data)
	}
}

func isConflictError(err error) bool {
	if err == nil {
		return false
	}
	return strings.Contains(err.Error(), "already exists") || strings.Contains(err.Error(), "409")
}

type qdrantSearchResponse struct {
	Result []qdrantPoint `json:"result"`
	Status string        `json:"status"`
}

type qdrantPoint struct {
	ID      any            `json:"id"`
	Score   float64        `json:"score"`
	Payload map[string]any `json:"payload"`
}

type qdrantSchema struct {
	VectorSize int    `json:"vector_size"`
	Distance   string `json:"distance"`
}
