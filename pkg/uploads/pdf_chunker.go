package uploads

import (
	"bytes"
	"fmt"
	"io"
	"regexp"
)

var (
	textObjectPattern = regexp.MustCompile(`(?s)\((.*?)\)\s+(?:Tj|TJ)`)
	pageSplitPattern  = regexp.MustCompile(`(?i)\n\s*endstream`)
)

// PDFChunker extracts best-effort page-level chunks using a lightweight parser.
type PDFChunker struct{}

func (p PDFChunker) Chunk(reader ReaderWithName, src Source) ([]DocumentChunk, error) {
	buf := new(bytes.Buffer)
	if _, err := io.Copy(buf, reader.Reader); err != nil {
		return nil, err
	}
	if buf.Len() == 0 {
		return nil, nil
	}
	pages := pageSplitPattern.Split(buf.String(), -1)
	var chunks []DocumentChunk
	for idx, raw := range pages {
		text := extractText(raw)
		if text == "" {
			continue
		}
		chunk := DocumentChunk{
			ID:      chunkID(reader.Name, idx),
			Content: text,
			Metadata: map[string]any{
				"chunk_index": idx,
			},
		}
		chunk = chunk.WithProvenance(src.Name, src.URI, idx+1, src.Version)
		for k, v := range src.Additional {
			chunk.Metadata[k] = v
		}
		chunks = append(chunks, chunk)
	}
	if len(chunks) == 0 {
		return nil, fmt.Errorf("pdf chunker: unable to extract text")
	}
	return chunks, nil
}

func extractText(raw string) string {
	matches := textObjectPattern.FindAllStringSubmatch(raw, -1)
	if len(matches) == 0 {
		return ""
	}
	var buf bytes.Buffer
	for _, m := range matches {
		if len(m) < 2 {
			continue
		}
		text := decodePDFString(m[1])
		if text == "" {
			continue
		}
		if buf.Len() > 0 {
			buf.WriteByte(' ')
		}
		buf.WriteString(text)
	}
	return normalizeWhitespace(buf.String())
}

func decodePDFString(s string) string {
	var buf bytes.Buffer
	escaped := false
	for i := 0; i < len(s); i++ {
		ch := s[i]
		if escaped {
			switch ch {
			case 'n':
				buf.WriteByte('\n')
			case 'r':
				buf.WriteByte('\r')
			case 't':
				buf.WriteByte('\t')
			case 'b':
				buf.WriteByte('\b')
			case 'f':
				buf.WriteByte('\f')
			default:
				buf.WriteByte(ch)
			}
			escaped = false
			continue
		}
		if ch == '\\' {
			escaped = true
			continue
		}
		buf.WriteByte(ch)
	}
	return buf.String()
}
