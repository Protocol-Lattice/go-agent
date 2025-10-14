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
	"os"
	"strings"
	"time"
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

// schema file format expected at schemaPath (JSON)
type qdrantSchemaFile struct {
	BaseURL    string                  `json:"base_url"`   // e.g. "http://localhost:6333"
	APIKey     string                  `json:"api_key"`    // optional; falls back to env QDRANT_API_KEY
	Collection string                  `json:"collection"` // required
	Request    CreateCollectionRequest `json:"request"`    // required
}

// Your store type.
type QdrantStore struct {
	// add fields if needed
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
