package upload

import (
	"io"
	"time"
)

type Scope string

const (
	ScopeLocal  Scope = "local"  // visible only to current session
	ScopeSpace  Scope = "space"  // visible to a shared space (swarm)
	ScopeGlobal Scope = "global" // visible to all (careful!)
)

type Document struct {
	Name      string // file name (sanitized)
	MIME      string // detected MIME
	SizeBytes int64
	Source    string // "upload://<uploader>" or "fs://", "url://", etc.
	Tags      []string
	Meta      map[string]string // arbitrary metadata
	Reader    io.ReadSeeker     // rewindable
}

type Chunk struct {
	DocName   string
	Index     int
	Text      string
	TokenHint int
	Meta      map[string]string // e.g., page, section, language, code-lang
}

type Extractor interface {
	// Return UTF-8 text blocks; implementors may split by natural segments (pages/sections).
	Extract(doc *Document) ([]string, error)
	Supports(mime string) bool
}

type Chunker interface {
	Chunk(blocks []string) ([]Chunk, error)
}

type Redactor interface {
	Redact(s string) (string, bool) // returns redacted string and whether any changes were made
}

type EmbeddingsProvider interface {
	// Implement with your models provider; dims/normalization handled internally.
	EmbedTexts(texts []string) ([][]float32, error)
}

type MemoryWriter interface {
	// Space is your SharedSession or other scope key.
	// Returns IDs of stored memory records.
	WriteChunks(space string, scope Scope, chunks []Chunk, vectors [][]float32, ttl *time.Duration, meta map[string]string) ([]string, error)
}

type ContentStore interface {
	// Persist the original file (optional but useful for traceability / re-ingest).
	Put(doc *Document) (storedAt string, err error)
}
