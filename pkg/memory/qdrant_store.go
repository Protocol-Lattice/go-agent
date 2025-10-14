package memory

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
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
func (qs *QdrantStore) CreateSchema(ctx context.Context, baseURL, apiKey, collection string, req CreateCollectionRequest) error {
	if baseURL == "" {
		baseURL = "http://localhost:6333"
	}
	endpoint := fmt.Sprintf("%s/collections/%s", strings.TrimRight(baseURL, "/"), url.PathEscape(collection))

	body, err := json.Marshal(req)
	if err != nil {
		return fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPut, endpoint, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("new request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if apiKey != "" {
		httpReq.Header.Set("api-key", apiKey)                 // Qdrant supports this header
		httpReq.Header.Set("Authorization", "Bearer "+apiKey) // optional alternative
	}

	client := &http.Client{Timeout: 15 * time.Second}
	resp, err := client.Do(httpReq)
	if err != nil {
		return fmt.Errorf("do request: %w", err)
	}
	defer resp.Body.Close()

	data, _ := io.ReadAll(resp.Body)

	// Happy path
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		var ok qdrantOKResponse
		if err := json.Unmarshal(data, &ok); err == nil && strings.EqualFold(ok.Status, "ok") {
			return nil
		}
		// Some versions/situations might omit "ok"; consider a 2xx as success if JSON is parseable.
		return nil
	}

	// Try to parse the "status.error" form
	var qErr qdrantErrorResponse
	if err := json.Unmarshal(data, &qErr); err == nil && qErr.Status.Error != "" {
		// Make idempotent: ignore "already exists"
		if strings.Contains(strings.ToLower(qErr.Status.Error), "already exists") {
			return nil
		}
		return errors.New(qErr.Status.Error)
	}

	// Fallback
	return fmt.Errorf("qdrant error: http %d: %s", resp.StatusCode, string(data))
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

// ----- models -----

type Distance string

const (
	DistanceCosine Distance = "Cosine"
	DistanceDot    Distance = "Dot"
	DistanceEuclid Distance = "Euclid"
)

type VectorParams struct {
	Size     int      `json:"size"`
	Distance Distance `json:"distance"`
}

// Vectors can be either a single VectorParams or a map[string]VectorParams.
type CreateCollectionRequest struct {
	Vectors                any            `json:"vectors"` // VectorParams or map[string]VectorParams
	OptimizersConfig       map[string]any `json:"optimizers_config,omitempty"`
	HnswConfig             map[string]any `json:"hnsw_config,omitempty"`
	QuantizationConfig     map[string]any `json:"quantization_config,omitempty"`
	ShardNumber            *int           `json:"shard_number,omitempty"`
	WriteConsistencyFactor *int           `json:"write_consistency_factor,omitempty"`
}

// Success shape (typical):
// { "result": {...}, "status": "ok", "time": 0.0 }
type qdrantOKResponse struct {
	Status string `json:"status"`
	Time   any    `json:"time"`
	Result any    `json:"result"`
}

// Error shape sometimes differs, e.g.:
// { "status": { "error":"Not found: Collection ..."}, "time": 0.0 }
type qdrantErrorStatus struct {
	Error string `json:"error"`
}
type qdrantErrorResponse struct {
	Status qdrantErrorStatus `json:"status"`
	Time   any               `json:"time"`
}
