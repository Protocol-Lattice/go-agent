// pkg/adk/modules/filectx/module.go
package filectx

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	contextio "github.com/Raezil/go-agent-development-kit/pkg/upload"
)

type UploaderToolArgs struct {
	Space string            `json:"space"`           // e.g. "team:shared"
	Name  string            `json:"name"`            // filename hint
	MIME  string            `json:"mime,omitempty"`  // optional
	Scope string            `json:"scope,omitempty"` // "local"|"space"|"global"
	Tags  []string          `json:"tags,omitempty"`
	TTL   string            `json:"ttl,omitempty"` // e.g., "168h"
	Meta  map[string]string `json:"meta,omitempty"`
	// Content is base64 (works across transports: UTCP, HTTP, WS)
	ContentB64 string `json:"content_b64"`
}

// Adapter interfaces from your runtime:
//  - Embedder: wrap your models provider embeddings.
//  - MemoryWriter: wrap your memory engine.
//  - Store: optional FS/S3 store for originals.

type Module struct {
	Ingest *contextio.Ingestor
}

func (m *Module) Name() string { return "file_context" }

// RegisterTool returns a callable the agent runtime can map into UTCP.
func (m *Module) RegisterTool() (name string, handler func(ctx context.Context, rawArgs []byte) (string, error)) {
	return "file.add_context", func(ctx context.Context, rawArgs []byte) (string, error) {
		var args UploaderToolArgs
		if err := jsonUnmarshal(rawArgs, &args); err != nil {
			return "", err
		}
		if args.Space == "" || args.Name == "" || args.ContentB64 == "" {
			return "", errors.New("space, name, content_b64 are required")
		}
		scope := contextio.ScopeSpace
		switch args.Scope {
		case "local":
			scope = contextio.ScopeLocal
		case "global":
			scope = contextio.ScopeGlobal
		}
		var ttlPtr *time.Duration
		if args.TTL != "" {
			d, err := time.ParseDuration(args.TTL)
			if err != nil {
				return "", fmt.Errorf("invalid ttl: %w", err)
			}
			ttlPtr = &d
		}
		data, err := base64.StdEncoding.DecodeString(args.ContentB64)
		if err != nil {
			return "", err
		}
		ids, err := m.Ingest.IngestReader(args.Name, args.MIME, bytesReader(data), contextio.IngestOptions{
			Space: args.Space,
			Scope: scope,
			Tags:  args.Tags,
			TTL:   ttlPtr,
			Meta:  args.Meta,
		})
		if err != nil {
			return "", err
		}
		return fmt.Sprintf("indexed %d chunks", len(ids)), nil
	}
}

// --- helpers (no third-party) ---
func jsonUnmarshal(b []byte, v any) error {
	// Defer to encoding/json. Aliased for easy drop-in replacement if needed.
	return json.Unmarshal(b, v)
}

func bytesReader(b []byte) *bytes.Reader { return bytes.NewReader(b) }
