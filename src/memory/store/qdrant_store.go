package store

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"net/url"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/Protocol-Lattice/go-agent/src/memory/model"
)

// --- Qdrant types ---

type Distance string

const (
	DistanceCosine Distance = "Cosine"
	DistanceDot    Distance = "Dot"
	DistanceEuclid Distance = "Euclid"
)

// CreateCollectionRequest matches Qdrant's API; Vectors supports single or named vectors.
type CreateCollectionRequest struct {
	Vectors                json.RawMessage `json:"vectors"` // {"size":1536,"distance":"Cosine"} OR {"text":{"size":768,"distance":"Cosine"}}
	ShardNumber            *int            `json:"shard_number,omitempty"`
	ReplicationFactor      *int            `json:"replication_factor,omitempty"`
	WriteConsistencyFactor *int            `json:"write_consistency_factor,omitempty"`
	OnDiskPayload          *bool           `json:"on_disk_payload,omitempty"`
}

// qdrantStatus supports both `status: "ok"` and `status: {"error":"..."}`.
type qdrantStatus struct {
	State string // "ok" or "error"
	Error string // non-empty if error
}

func (s *qdrantStatus) UnmarshalJSON(b []byte) error {
	if len(b) > 0 && b[0] == '"' {
		var v string
		if err := json.Unmarshal(b, &v); err != nil {
			return err
		}
		s.State = strings.ToLower(v)
		return nil
	}
	var obj struct {
		Error string `json:"error"`
	}
	if err := json.Unmarshal(b, &obj); err != nil {
		return err
	}
	if obj.Error != "" {
		s.State = "error"
		s.Error = obj.Error
	}
	return nil
}

type qdrantEnvelope[T any] struct {
	Status qdrantStatus `json:"status"`
	Time   float64      `json:"time"`
	Result T            `json:"result"`
}

type qdrantPointResult struct {
	ID      json.RawMessage `json:"id"`
	Score   float64         `json:"score"`
	Payload map[string]any  `json:"payload"`
	Vector  []float32       `json:"vector"`
}

type qdrantScrollResult struct {
	Points []qdrantPointResult `json:"points"`
	Offset json.RawMessage     `json:"next_page_offset"`
}

type qdrantCountResult struct {
	Count int `json:"count"`
}

type qdrantGetResult struct {
	Points []qdrantPointResult `json:"points"`
}

// schema file format expected at schemaPath (JSON)
type qdrantSchemaFile struct {
	BaseURL    string                  `json:"base_url"`   // e.g. "http://localhost:6333"
	APIKey     string                  `json:"api_key"`    // optional; falls back to env QDRANT_API_KEY
	Collection string                  `json:"collection"` // required
	Request    CreateCollectionRequest `json:"request"`    // required
}

// Your store type.
type QdrantStore struct {
	baseURL    string
	apiKey     string
	collection string
	client     *http.Client
	mu         sync.Mutex
}

// NewQdrantStore creates a Qdrant-backed VectorStore implementation.
func NewQdrantStore(baseURL, collection, apiKey string) *QdrantStore {
	if baseURL == "" {
		baseURL = "http://localhost:6333"
	}
	return &QdrantStore{
		baseURL:    strings.TrimRight(baseURL, "/"),
		apiKey:     apiKey,
		collection: collection,
		client:     &http.Client{Timeout: 15 * time.Second},
	}
}

// CreateSchema implements SchemaInitializer.
// schemaPath must point to a JSON file that matches qdrantSchemaFile.
func (qs *QdrantStore) CreateSchema(ctx context.Context, schemaPath string) error {
	if schemaPath == "" {
		return errors.New("schemaPath is empty")
	}

	f, err := os.Open(schemaPath)
	if err != nil {
		return fmt.Errorf("open schema file: %w", err)
	}
	defer f.Close()

	// Limit read to 1 MiB for safety.
	data, err := io.ReadAll(io.LimitReader(f, 1<<20))
	if err != nil {
		return fmt.Errorf("read schema file: %w", err)
	}

	var cfg qdrantSchemaFile
	if err := json.Unmarshal(data, &cfg); err != nil {
		return fmt.Errorf("unmarshal schema file (JSON): %w", err)
	}

	// Defaults & validation
	if cfg.BaseURL == "" {
		cfg.BaseURL = "http://localhost:6333"
	}
	if cfg.APIKey == "" {
		cfg.APIKey = os.Getenv("QDRANT_API_KEY")
	}
	if cfg.Collection == "" {
		return errors.New("schema file missing 'collection'")
	}
	if len(cfg.Request.Vectors) == 0 {
		return errors.New("schema file 'request.vectors' is required")
	}

	return qs.createCollection(ctx, cfg.BaseURL, cfg.APIKey, cfg.Collection, cfg.Request)
}

// --- Internal HTTP call with robust handling (idempotent, dual-status parsing) ---

func (qs *QdrantStore) createCollection(ctx context.Context, baseURL, apiKey, collection string, req CreateCollectionRequest) error {
	u, err := url.Parse(strings.TrimRight(baseURL, "/"))
	if err != nil {
		return fmt.Errorf("bad baseURL: %w", err)
	}
	u.Path = fmt.Sprintf("/collections/%s", url.PathEscape(collection))

	body, err := json.Marshal(req)
	if err != nil {
		return fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPut, u.String(), bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("new request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("User-Agent", "go-adk-qdrant/1.0")
	if apiKey != "" {
		// Either header works; sending both covers deployments with either check.
		httpReq.Header.Set("api-key", apiKey)
		httpReq.Header.Set("Authorization", "Bearer "+apiKey)
	}

	client := &http.Client{Timeout: 15 * time.Second}
	resp, err := client.Do(httpReq)
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled) {
			return err
		}
		return fmt.Errorf("do request: %w", err)
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))

	var env qdrantEnvelope[json.RawMessage]
	_ = json.Unmarshal(respBody, &env) // best-effort parse

	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		// 2xx is success even if status isn't present in some versions.
		if strings.EqualFold(env.Status.State, "ok") && env.Status.Error == "" {
			return nil
		}
		return nil
	}

	// Non-2xx: surface structured error and make idempotent for existing collections.
	if env.Status.Error != "" {
		low := strings.ToLower(env.Status.Error)
		if strings.Contains(low, "already exists") {
			return nil
		}
		return errors.New(env.Status.Error)
	}

	return fmt.Errorf("qdrant error: http %d: %s", resp.StatusCode, strings.TrimSpace(string(respBody)))
}

// StoreMemory upserts a memory point into Qdrant.
func (qs *QdrantStore) StoreMemory(ctx context.Context, sessionID, content string, metadata map[string]any, embedding []float32) error {
	if qs == nil {
		return errors.New("nil qdrant store")
	}
	if qs.collection == "" {
		return errors.New("qdrant collection is empty")
	}
	now := time.Now().UTC()
	if metadata == nil {
		metadata = map[string]any{}
	}
	if _, ok := metadata["space"]; !ok {
		metadata["space"] = sessionID
	}
	edges := model.SanitizeGraphEdges(metadata)
	importance, source, summary, lastEmbedded, normalized := model.NormalizeMetadata(metadata, now)
	metaMap := model.DecodeMetadata(normalized)
	space := model.StringFromAny(metaMap["space"])
	if space == "" {
		space = sessionID
	}
	matrix := model.ValidEmbeddingMatrix(metaMap)
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
	payload := map[string]any{
		"session_id":    sessionID,
		"content":       content,
		"metadata":      model.DecodeMetadata(normalized),
		"importance":    importance,
		"source":        source,
		"summary":       summary,
		"created_at":    now.Format(time.RFC3339Nano),
		"last_embedded": lastEmbedded.Format(time.RFC3339Nano),
		"space":         space,
	}
	if len(edges) == 0 {
		if metaEdges := model.ValidGraphEdges(metaMap); len(metaEdges) > 0 {
			payload["graph_edges"] = metaEdges
		}
	} else {
		payload["graph_edges"] = edges
	}
	if len(matrix) > 0 {
		payload[model.EmbeddingMatrixKey] = matrix
	}
	pointID := qs.generateID()
	req := map[string]any{
		"points": []map[string]any{{
			"id":      pointID,
			"vector":  storedEmbedding,
			"payload": payload,
		}},
	}
	var resp qdrantEnvelope[json.RawMessage]
	if err := qs.do(ctx, http.MethodPut, fmt.Sprintf("/collections/%s/points", url.PathEscape(qs.collection)), req, &resp); err != nil {
		return err
	}
	if !strings.EqualFold(resp.Status.State, "ok") && resp.Status.Error != "" {
		return errors.New(resp.Status.Error)
	}
	return nil
}

// SearchMemory performs a similarity search.
func (qs *QdrantStore) SearchMemory(ctx context.Context, queryEmbedding []float32, limit int) ([]model.MemoryRecord, error) {
	if qs == nil {
		return nil, errors.New("nil qdrant store")
	}
	if limit <= 0 {
		return nil, nil
	}
	reqBody := map[string]any{
		"vector":       queryEmbedding,
		"limit":        limit,
		"with_vector":  true,
		"with_payload": true,
	}
	var resp qdrantEnvelope[[]qdrantPointResult]
	if err := qs.do(ctx, http.MethodPost, fmt.Sprintf("/collections/%s/points/search", url.PathEscape(qs.collection)), reqBody, &resp); err != nil {
		return nil, err
	}
	results := make([]model.MemoryRecord, 0, len(resp.Result))
	for _, point := range resp.Result {
		id, _ := parseQdrantID(point.ID)
		meta := mapFromPayload(point.Payload)
		metaMap, _ := meta["metadata"].(map[string]any)
		if metaMap == nil {
			metaMap = model.DecodeMetadata(encodeMetadata(meta["metadata"]))
		}
		record := model.MemoryRecord{
			ID:           id,
			SessionID:    model.StringFromAny(meta["session_id"]),
			Content:      model.StringFromAny(meta["content"]),
			Metadata:     encodeMetadata(meta["metadata"]),
			Embedding:    point.Vector,
			Importance:   model.FloatFromAny(meta["importance"]),
			Source:       model.StringFromAny(meta["source"]),
			Summary:      model.StringFromAny(meta["summary"]),
			CreatedAt:    model.TimeFromAny(meta["created_at"]),
			LastEmbedded: model.TimeFromAny(meta["last_embedded"]),
		}
		model.HydrateRecordFromMetadata(&record, metaMap)
		if len(record.EmbeddingMatrix) == 0 {
			if matrix := model.DecodeEmbeddingMatrix(meta[model.EmbeddingMatrixKey]); len(matrix) > 0 {
				record.EmbeddingMatrix = matrix
			} else if matrix := model.DecodeEmbeddingMatrix(point.Payload[model.EmbeddingMatrixKey]); len(matrix) > 0 {
				record.EmbeddingMatrix = matrix
			}
		}
		if record.Space == "" {
			record.Space = model.StringFromAny(meta["space"])
			if record.Space == "" {
				record.Space = record.SessionID
			}
		}
		if len(record.GraphEdges) == 0 {
			record.GraphEdges = model.ValidGraphEdges(metaMap)
		}
		record.Score = model.MaxCosineSimilarity(queryEmbedding, record)
		results = append(results, record)
	}
	sort.SliceStable(results, func(i, j int) bool { return results[i].Score > results[j].Score })
	if limit > 0 && len(results) > limit {
		results = results[:limit]
	}
	return results, nil
}

// UpdateEmbedding updates the vector and last embedded timestamp.
func (qs *QdrantStore) UpdateEmbedding(ctx context.Context, id int64, embedding []float32, lastEmbedded time.Time) error {
	if qs == nil {
		return errors.New("nil qdrant store")
	}
	point, err := qs.getPoint(ctx, id)
	if err != nil {
		return err
	}
	if point.Payload == nil {
		point.Payload = map[string]any{}
	}
	point.Payload["last_embedded"] = lastEmbedded.Format(time.RFC3339Nano)
	req := map[string]any{
		"points": []map[string]any{{
			"id":      id,
			"vector":  embedding,
			"payload": point.Payload,
		}},
	}
	return qs.do(ctx, http.MethodPut, fmt.Sprintf("/collections/%s/points", url.PathEscape(qs.collection)), req, nil)
}

// DeleteMemory removes points by id.
func (qs *QdrantStore) DeleteMemory(ctx context.Context, ids []int64) error {
	if qs == nil || len(ids) == 0 {
		return nil
	}
	req := map[string]any{
		"points": ids,
	}
	return qs.do(ctx, http.MethodPost, fmt.Sprintf("/collections/%s/points/delete", url.PathEscape(qs.collection)), req, nil)
}

// Iterate streams through all points in created_at order.
func (qs *QdrantStore) Iterate(ctx context.Context, fn func(model.MemoryRecord) bool) error {
	if qs == nil {
		return nil
	}

	var offset any
	const (
		limit    = 128
		maxPages = 100000 // hard stop against server/pathological loops
	)

	prevOffsetRaw := ""

	for page := 0; page < maxPages; page++ {
		req := map[string]any{
			"limit":        limit,
			"with_payload": true,
			"with_vector":  true,
		}
		if offset != nil {
			req["offset"] = offset
		}

		// NOTE: qdrantEnvelope[qdrantScrollResult] and qs.do(...) already exist in your codebase.
		var resp qdrantEnvelope[qdrantScrollResult]
		if err := qs.do(
			ctx,
			"POST",
			fmt.Sprintf("/collections/%s/points/scroll", url.PathEscape(qs.collection)),
			req,
			&resp,
		); err != nil {
			return err
		}

		// Deliver points
		for _, point := range resp.Result.Points {
			id, _ := parseQdrantID(point.ID)
			meta := mapFromPayload(point.Payload)

			rec := model.MemoryRecord{
				ID:           id,
				SessionID:    model.StringFromAny(meta["session_id"]),
				Content:      model.StringFromAny(meta["content"]),
				Metadata:     encodeMetadata(meta["metadata"]),
				Embedding:    point.Vector,
				Importance:   model.FloatFromAny(meta["importance"]),
				Source:       model.StringFromAny(meta["source"]),
				Summary:      model.StringFromAny(meta["summary"]),
				CreatedAt:    model.TimeFromAny(meta["created_at"]),
				LastEmbedded: model.TimeFromAny(meta["last_embedded"]),
			}
			metaMap := model.DecodeMetadata(rec.Metadata)
			model.HydrateRecordFromMetadata(&rec, metaMap)
			if len(rec.EmbeddingMatrix) == 0 {
				if matrix := model.DecodeEmbeddingMatrix(metaMap[model.EmbeddingMatrixKey]); len(matrix) > 0 {
					rec.EmbeddingMatrix = matrix
				} else if matrix := model.DecodeEmbeddingMatrix(point.Payload[model.EmbeddingMatrixKey]); len(matrix) > 0 {
					rec.EmbeddingMatrix = matrix
				}
			}

			if cont := fn(rec); !cont {
				return nil
			}
		}

		// Handle end-of-scroll conditions safely.
		raw := jsonString(resp.Result.Offset) // tolerate RawMessage/any
		if len(resp.Result.Points) == 0 || raw == "" || strings.EqualFold(raw, "null") || raw == prevOffsetRaw {
			return nil
		}
		prevOffsetRaw = raw
		offset = resp.Result.Offset
	}

	return fmt.Errorf("qdrant iterate: hit page limit (%d) – possible offset loop", maxPages)
}

// jsonString returns a compact JSON representation of v ("" on marshal error or nil).
func jsonString(v any) string {
	if v == nil {
		return ""
	}
	b, err := json.Marshal(v)
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(b))
}

// Count returns the total number of points in the collection.
func (qs *QdrantStore) Count(ctx context.Context) (int, error) {
	if qs == nil {
		return 0, nil
	}
	req := map[string]any{"exact": true}
	var resp qdrantEnvelope[qdrantCountResult]
	if err := qs.do(ctx, http.MethodPost, fmt.Sprintf("/collections/%s/points/count", url.PathEscape(qs.collection)), req, &resp); err != nil {
		return 0, err
	}
	return resp.Result.Count, nil
}

// UpsertGraph updates the stored payload to reflect graph metadata.
func (qs *QdrantStore) UpsertGraph(ctx context.Context, record model.MemoryRecord, edges []model.GraphEdge) error {
	if qs == nil || record.ID == 0 {
		return nil
	}
	if qs.collection == "" {
		return errors.New("qdrant collection is empty")
	}

	point, err := qs.getPoint(ctx, record.ID)
	if err != nil {
		return err
	}
	if point.Payload == nil {
		point.Payload = map[string]any{}
	}

	if record.Space != "" {
		point.Payload["space"] = record.Space
	}
	if len(edges) > 0 {
		point.Payload["graph_edges"] = edges
	} else {
		delete(point.Payload, "graph_edges")
	}

	if meta, ok := point.Payload["metadata"].(map[string]any); ok {
		if record.Space != "" {
			meta["space"] = record.Space
		}
		if len(edges) > 0 {
			meta["graph_edges"] = edges
		} else {
			delete(meta, "graph_edges")
		}
		point.Payload["metadata"] = meta
	}

	// Use set-payload to merge fields without requiring vectors.
	req := map[string]any{
		"points":  []int64{record.ID},
		"payload": point.Payload,
	}

	var resp qdrantEnvelope[json.RawMessage]
	if err := qs.do(
		ctx,
		http.MethodPost,
		fmt.Sprintf("/collections/%s/points/payload?wait=true", url.PathEscape(qs.collection)),
		req,
		&resp,
	); err != nil {
		return err
	}
	if !strings.EqualFold(resp.Status.State, "ok") && resp.Status.Error != "" {
		return errors.New(resp.Status.Error)
	}
	return nil
}

// Neighborhood walks the edge payloads to return nearby memories.
func (qs *QdrantStore) Neighborhood(ctx context.Context, seedIDs []int64, hops, limit int) ([]model.MemoryRecord, error) {
	if qs == nil || len(seedIDs) == 0 || hops <= 0 || limit <= 0 {
		return nil, nil
	}
	seen := make(map[int64]struct{}, len(seedIDs))
	frontier := make([]int64, 0, len(seedIDs))
	for _, id := range seedIDs {
		if id == 0 {
			continue
		}
		seen[id] = struct{}{}
		frontier = append(frontier, id)
	}
	neighborSet := make(map[int64]struct{})
	for depth := 0; depth < hops && len(frontier) > 0; depth++ {
		next := make([]int64, 0)
		for _, id := range frontier {
			point, err := qs.getPoint(ctx, id)
			if err != nil {
				continue
			}
			edges := extractEdgesFromPayload(point.Payload)
			for _, edge := range edges {
				if err := edge.Validate(); err != nil {
					continue
				}
				if _, ok := seen[edge.Target]; ok {
					continue
				}
				seen[edge.Target] = struct{}{}
				neighborSet[edge.Target] = struct{}{}
				next = append(next, edge.Target)
			}
		}
		frontier = next
	}
	if len(neighborSet) == 0 {
		return nil, nil
	}
	ids := make([]int64, 0, len(neighborSet))
	for id := range neighborSet {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	if len(ids) > limit {
		ids = ids[:limit]
	}
	points, err := qs.getPoints(ctx, ids)
	if err != nil {
		return nil, err
	}
	results := make([]model.MemoryRecord, 0, len(points))
	for _, point := range points {
		id, _ := parseQdrantID(point.ID)
		meta := mapFromPayload(point.Payload)
		metaMap, _ := meta["metadata"].(map[string]any)
		if metaMap == nil {
			metaMap = model.DecodeMetadata(encodeMetadata(meta["metadata"]))
		}
		record := model.MemoryRecord{
			ID:           id,
			SessionID:    model.StringFromAny(meta["session_id"]),
			Content:      model.StringFromAny(meta["content"]),
			Metadata:     encodeMetadata(meta["metadata"]),
			Embedding:    point.Vector,
			Importance:   model.FloatFromAny(meta["importance"]),
			Source:       model.StringFromAny(meta["source"]),
			Summary:      model.StringFromAny(meta["summary"]),
			CreatedAt:    model.TimeFromAny(meta["created_at"]),
			LastEmbedded: model.TimeFromAny(meta["last_embedded"]),
		}
		model.HydrateRecordFromMetadata(&record, metaMap)
		if record.Space == "" {
			record.Space = model.StringFromAny(meta["space"])
			if record.Space == "" {
				record.Space = record.SessionID
			}
		}
		if len(record.GraphEdges) == 0 {
			record.GraphEdges = model.ValidGraphEdges(metaMap)
		}
		results = append(results, record)
	}
	return results, nil
}

func (qs *QdrantStore) do(ctx context.Context, method, path string, body any, out any) error {
	if qs == nil {
		return errors.New("nil qdrant store")
	}
	u := qs.baseURL + path

	var buf io.ReadWriter
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return err
		}
		buf = bytes.NewBuffer(b)
	} else {
		buf = bytes.NewBuffer(nil)
	}

	req, err := http.NewRequestWithContext(ctx, method, u, buf)
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	if qs.apiKey != "" {
		req.Header.Set("api-key", qs.apiKey)
	}
	resp, err := qs.client.Do(req)

	if err != nil {
		return err
	}
	defer resp.Body.Close()

	payload, _ := io.ReadAll(resp.Body)
	if resp.StatusCode >= 400 {
		return fmt.Errorf("qdrant %s %s -> http %d: %s",
			method, u, resp.StatusCode, strings.TrimSpace(string(payload)))
	}

	if out != nil && len(payload) > 0 {
		if err := json.Unmarshal(payload, out); err != nil {
			return err
		}
	}
	return nil
}
func (qs *QdrantStore) getPoint(ctx context.Context, id int64) (*qdrantPointResult, error) {
	if qs == nil {
		return nil, errors.New("nil qdrant store")
	}
	if qs.collection == "" {
		return nil, errors.New("qdrant collection is empty")
	}

	req := map[string]any{
		"ids":          []int64{id},
		"with_payload": true,
		"with_vector":  true,
	}

	var resp qdrantEnvelope[qdrantGetResult]
	if err := qs.do(
		ctx,
		http.MethodPost,
		fmt.Sprintf("/collections/%s/points", url.PathEscape(qs.collection)),
		req,
		&resp,
	); err != nil {
		// Optional fallback for older clients: GET single point by id
		// (shape differs, but we can map it back into qdrantPointResult).
		var single qdrantEnvelope[qdrantPointResult]
		err2 := qs.do(
			ctx,
			http.MethodGet,
			fmt.Sprintf("/collections/%s/points/%d?with_payload=true&with_vector=true",
				url.PathEscape(qs.collection), id),
			nil,
			&single,
		)
		if err2 != nil {
			return nil, err // return the original error
		}
		// Wrap single result into the expected type
		return &single.Result, nil
	}

	if len(resp.Result.Points) == 0 {
		return nil, fmt.Errorf("point %d not found", id)
	}
	return &resp.Result.Points[0], nil
}

func (qs *QdrantStore) getPoints(ctx context.Context, ids []int64) ([]qdrantPointResult, error) {
	if len(ids) == 0 {
		return nil, nil
	}
	req := map[string]any{
		"ids":          ids,
		"with_payload": true,
		"with_vector":  true,
	}
	var resp qdrantEnvelope[qdrantGetResult]
	if err := qs.do(ctx, http.MethodPost, fmt.Sprintf("/collections/%s/points/get", url.PathEscape(qs.collection)), req, &resp); err != nil {
		return nil, err
	}
	return resp.Result.Points, nil
}

func (qs *QdrantStore) generateID() int64 {
	qs.mu.Lock()
	defer qs.mu.Unlock()
	v := time.Now().UnixNano() ^ rand.Int63()
	if v < 0 {
		v = -v
	}
	return v
}

func parseQdrantID(raw json.RawMessage) (int64, error) {
	if len(raw) == 0 {
		return 0, nil
	}
	var idInt int64
	if err := json.Unmarshal(raw, &idInt); err == nil {
		return idInt, nil
	}
	var idStr string
	if err := json.Unmarshal(raw, &idStr); err == nil {
		if val, err := strconv.ParseInt(idStr, 10, 64); err == nil {
			return val, nil
		}
	}
	return 0, errors.New("unrecognised qdrant id")
}

func mapFromPayload(payload map[string]any) map[string]any {
	if payload == nil {
		return map[string]any{}
	}
	if md, ok := payload["metadata"]; ok {
		switch t := md.(type) {
		case string:
			payload["metadata"] = model.DecodeMetadata(t)
		case json.RawMessage:
			payload["metadata"] = model.DecodeMetadata(string(t))
		case map[string]any:
			// already a map
		default:
			payload["metadata"] = map[string]any{}
		}
	} else {
		payload["metadata"] = map[string]any{}
	}
	return payload
}

func extractEdgesFromPayload(payload map[string]any) []model.GraphEdge {
	if payload == nil {
		return nil
	}
	if raw, ok := payload["graph_edges"]; ok {
		return model.ValidGraphEdges(map[string]any{"graph_edges": raw})
	}
	if meta, ok := payload["metadata"].(map[string]any); ok {
		return model.ValidGraphEdges(meta)
	}
	return nil
}

func encodeMetadata(v any) string {
	switch m := v.(type) {
	case string:
		return m
	case json.RawMessage:
		return string(m)
	case map[string]any:
		b, _ := json.Marshal(m)
		return string(b)
	case nil:
		return "{}"
	default:
		b, _ := json.Marshal(m)
		return string(b)
	}
}
