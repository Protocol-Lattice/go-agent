package uploads

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"time"
)

// DocumentChunk represents a single chunk of text with rich provenance metadata.
type DocumentChunk struct {
	ID       string
	Content  string
	Metadata map[string]any
}

// WithProvenance ensures provenance fields exist on the metadata map.
func (c DocumentChunk) WithProvenance(source, uri string, page int, version string) DocumentChunk {
	meta := make(map[string]any, len(c.Metadata)+5)
	for k, v := range c.Metadata {
		meta[k] = v
	}
	if source != "" {
		meta["source"] = source
	}
	if uri != "" {
		meta["uri"] = uri
	}
	if page > 0 {
		meta["page"] = page
	}
	if version != "" {
		meta["version"] = version
	}
	if _, ok := meta["ingested_at"]; !ok {
		meta["ingested_at"] = time.Now().UTC().Format(time.RFC3339Nano)
	}
	if _, ok := meta["checksum"]; !ok {
		meta["checksum"] = checksum(c.Content)
	}
	c.Metadata = meta
	return c
}

// checksum calculates a deterministic checksum for provenance tracking.
func checksum(text string) string {
	sum := sha256.Sum256([]byte(text))
	return hex.EncodeToString(sum[:])
}

// Source describes the logical origin of an upload.
type Source struct {
	Name       string
	URI        string
	Version    string
	Additional map[string]any
}

// ReaderWithName couples an io.Reader with an optional filename for chunkers.
type ReaderWithName struct {
	Name   string
	Reader io.Reader
}

// Chunker is implemented by upload chunkers (text, markdown, pdf, code, ...).
type Chunker interface {
	Chunk(reader ReaderWithName, src Source) ([]DocumentChunk, error)
}

// ErrUnsupported is returned when a chunker cannot handle a file.
var ErrUnsupported = fmt.Errorf("uploads: unsupported format")
