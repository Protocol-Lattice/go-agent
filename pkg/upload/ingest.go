package upload

import (
	"bytes"
	"errors"
	"io"
	"math"
	"strings"
	"time"
)

type Ingestor struct {
	Extractors       []Extractor
	Chunker          Chunker
	Redactor         Redactor
	Store            ContentStore
	Embedder         EmbeddingsProvider
	Writer           MemoryWriter
	MaxBytes         int64          // cap to avoid huge uploads (default: 10MiB)
	DefaultTTL       *time.Duration // optional
	AllowedExtsLower map[string]struct{}
}

func NewDefaultIngestor(embed EmbeddingsProvider, writer MemoryWriter, store ContentStore) *Ingestor {
	max := int64(10 << 20) // 10 MiB
	return &Ingestor{
		Extractors: defaultExtractors(),
		Chunker:    FixedChunker{MaxRunes: 1200, Overlap: 120},
		Redactor:   nil, // set to NewDefaultRedactor() to enable
		Store:      store,
		Embedder:   embed,
		Writer:     writer,
		MaxBytes:   max,
		AllowedExtsLower: map[string]struct{}{
			".txt": {}, ".md": {}, ".markdown": {}, ".json": {}, ".yaml": {}, ".yml": {}, ".pdf": {},
			".go": {}, ".py": {}, ".js": {}, ".ts": {}, ".java": {}, ".rs": {}, ".cpp": {}, ".c": {}, ".cs": {}, ".sql": {},
		},
	}
}

type IngestOptions struct {
	Space string
	Scope Scope
	Tags  []string
	TTL   *time.Duration
	Meta  map[string]string
}

func (ig *Ingestor) IngestReader(name string, mimeOpt string, r io.Reader, opts IngestOptions) ([]string, error) {
	if ig.Embedder == nil || ig.Writer == nil {
		return nil, errors.New("ingestor requires Embedder and Writer")
	}
	// Read with cap
	limited := io.LimitReader(r, ig.MaxBytes+1)
	buf, err := io.ReadAll(limited)
	if err != nil {
		return nil, err
	}
	if int64(len(buf)) > ig.MaxBytes {
		return nil, errors.New("file too large")
	}
	// MIME sniff
	mime := mimeOpt
	if mime == "" {
		mime = DetectMIME(name, buf[:int(math.Min(512, float64(len(buf))))])
	}
	doc := &Document{
		Name:      name,
		MIME:      mime,
		SizeBytes: int64(len(buf)),
		Source:    "upload://local",
		Tags:      opts.Tags,
		Meta:      cloneMap(opts.Meta),
		Reader:    bytes.NewReader(buf),
	}

	// Optional: extension allow-list
	if ig.AllowedExtsLower != nil {
		ext := strings.ToLower(extOf(name))
		if _, ok := ig.AllowedExtsLower[ext]; !ok && strings.HasPrefix(mime, "text/") == false {
			return nil, errors.New("file type not allowed; add an extractor or allow-list the extension")
		}
	}

	// Store original (optional)
	if ig.Store != nil {
		loc, err := ig.Store.Put(doc)
		if err != nil {
			return nil, err
		}
		if doc.Meta == nil {
			doc.Meta = map[string]string{}
		}
		doc.Meta["stored_at"] = loc
	}

	// Extract
	var extractor Extractor
	for _, ex := range ig.Extractors {
		if ex.Supports(doc.MIME) {
			extractor = ex
			break
		}
	}
	if extractor == nil {
		return nil, errors.New("no extractor for MIME: " + doc.MIME)
	}
	blocks, err := extractor.Extract(doc)
	if err != nil {
		return nil, err
	}

	// Chunk
	chunks, err := ig.Chunker.Chunk(blocks)
	if err != nil {
		return nil, err
	}
	for i := range chunks {
		chunks[i].DocName = doc.Name
		if chunks[i].Meta == nil {
			chunks[i].Meta = map[string]string{}
		}
		chunks[i].Meta["mime"] = doc.MIME
		chunks[i].Meta["source"] = doc.Source
	}

	// Redact (optional)
	if ig.Redactor != nil {
		for i := range chunks {
			if red, changed := ig.Redactor.Redact(chunks[i].Text); changed {
				chunks[i].Text = red
				chunks[i].Meta["redacted"] = "true"
			}
		}
	}

	// Embed
	texts := make([]string, len(chunks))
	for i := range chunks {
		texts[i] = chunks[i].Text
	}
	vecs, err := ig.Embedder.EmbedTexts(texts)
	if err != nil {
		return nil, err
	}
	if len(vecs) != len(chunks) {
		return nil, errors.New("embedding count mismatch")
	}

	// Write to memory
	ttl := ig.DefaultTTL
	if opts.TTL != nil {
		ttl = opts.TTL
	}
	ids, err := ig.Writer.WriteChunks(opts.Space, opts.Scope, chunks, vecs, ttl, doc.Meta)
	if err != nil {
		return nil, err
	}
	return ids, nil
}

func extOf(name string) string {
	i := strings.LastIndexByte(name, '.')
	if i < 0 {
		return ""
	}
	return name[i:]
}

func cloneMap(m map[string]string) map[string]string {
	if m == nil {
		return nil
	}
	out := make(map[string]string, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}
