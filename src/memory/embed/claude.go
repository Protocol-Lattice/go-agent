package embed

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"

	json "github.com/alpkeskin/gotoon"
)

// ClaudeEmbedder proxies to Anthropic-recommended Voyage AI embeddings.
// Requires VOYAGE_API_KEY.
// Defaults:
//   - model: "voyage-3.5" (override via ADK_EMBED_MODEL)
//   - input_type: "document" (override via ADK_EMBED_INPUT_TYPE)
//   - endpoint: "https://api.voyageai.com/v1/embeddings" (override via VOYAGE_API_BASE)
type ClaudeEmbedder struct {
	client    *http.Client
	apiKey    string
	model     string
	inputType string
	endpoint  string
}

func NewClaudeEmbedder(model string) (Embedder, error) {
	apiKey := os.Getenv("VOYAGE_API_KEY")
	if model == "" {
		model = "voyage-3.5"
	}
	inputType := os.Getenv("ADK_EMBED_INPUT_TYPE")
	if inputType == "" {
		inputType = "document" // "query" also valid; see Voyage docs
	}
	endpoint := os.Getenv("VOYAGE_API_BASE")
	if endpoint == "" {
		endpoint = "https://api.voyageai.com/v1/embeddings"
	}

	return &ClaudeEmbedder{
		client:    &http.Client{Timeout: 60 * time.Second},
		apiKey:    apiKey,
		model:     model,
		inputType: inputType,
		endpoint:  endpoint,
	}, nil
}

func (c *ClaudeEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	if c.apiKey == "" {
		// Keep this explicit so callers see *why* this provider isn't working.
		return nil, errors.New("ClaudeEmbedder: VOYAGE_API_KEY not set; Anthropic does not offer first-party embeddings")
	}

	// Request payload (single input for our interface)
	payload := map[string]any{
		"input":      []string{text},
		"model":      c.model,
		"input_type": c.inputType,
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode/100 != 2 {
		slurp, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, fmt.Errorf("voyage embeddings HTTP %d: %s", resp.StatusCode, string(slurp))
	}

	var out struct {
		Data []struct {
			Embedding []float64 `json:"embedding"`
			Index     int       `json:"index"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	if len(out.Data) == 0 || len(out.Data[0].Embedding) == 0 {
		return nil, ErrNotSupported
	}

	return f64toF32(out.Data[0].Embedding), nil
}

func f64toF32(v []float64) []float32 {
	r := make([]float32, len(v))
	for i, x := range v {
		r[i] = float32(x)
	}
	return r
}
