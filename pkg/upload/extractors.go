package upload

import (
	"bufio"
	"bytes"
	"io"
	"mime"
	"net/http"
	"path/filepath"
	"strings"
)

type TextExtractor struct{}

func (TextExtractor) Supports(m string) bool {
	// Accept anything that looks texty
	return strings.HasPrefix(m, "text/") ||
		m == "application/json" ||
		m == "application/xml" ||
		m == "application/yaml" ||
		m == "application/x-yaml"
}

func (TextExtractor) Extract(doc *Document) ([]string, error) {
	// Read all safely without loading huge files (ingest guards will cap size)
	var buf bytes.Buffer
	if _, err := io.Copy(&buf, doc.Reader); err != nil {
		return nil, err
	}
	// Normalize line endings
	s := strings.ReplaceAll(buf.String(), "\r\n", "\n")
	return []string{s}, nil
}

// Helper to sniff MIME when not provided.
func DetectMIME(name string, head []byte) string {
	if m := http.DetectContentType(head); m != "application/octet-stream" {
		return m
	}
	ext := strings.ToLower(filepath.Ext(name))
	if ext != "" {
		if byExt := mime.TypeByExtension(ext); byExt != "" {
			return byExt
		}
	}
	// Fallback: treat unknowns as text if they look texty
	if isLikelyUTF8(head) {
		return "text/plain"
	}
	return "application/octet-stream"
}

func isLikelyUTF8(head []byte) bool {
	r := bufio.NewReader(bytes.NewReader(head))
	for i := 0; i < len(head) && i < 2048; i++ {
		if _, _, err := r.ReadRune(); err != nil {
			return false
		}
	}
	return true
}
