package uploads

import (
	"bufio"
	"fmt"
	"io"
	"strconv"
	"strings"
	"unicode/utf8"
)

// TextChunker splits plain text into roughly-even chunks based on word boundaries.
type TextChunker struct {
	MaxTokens int
}

func (t TextChunker) Chunk(reader ReaderWithName, src Source) ([]DocumentChunk, error) {
	max := t.MaxTokens
	if max <= 0 {
		max = 512
	}
	buf := new(strings.Builder)
	if _, err := io.Copy(buf, reader.Reader); err != nil {
		return nil, err
	}
	text := strings.TrimSpace(buf.String())
	if text == "" {
		return nil, nil
	}
	return chunkByWords(text, max, reader.Name, src)
}

func chunkByWords(text string, maxTokens int, name string, src Source) ([]DocumentChunk, error) {
	scanner := bufio.NewScanner(strings.NewReader(text))
	scanner.Split(bufio.ScanWords)

	var (
		chunks  []DocumentChunk
		builder strings.Builder
		count   int
		idx     int
	)

	emit := func() {
		if builder.Len() == 0 {
			return
		}
		chunk := makeChunk(idx, builder.String(), name, src)
		chunks = append(chunks, chunk)
		idx++
		builder.Reset()
		count = 0
	}

	for scanner.Scan() {
		word := scanner.Text()
		wordTokens := estimateTokens(word)
		if count+wordTokens > maxTokens && builder.Len() > 0 {
			emit()
		}
		if builder.Len() > 0 {
			builder.WriteByte(' ')
		}
		builder.WriteString(word)
		count += wordTokens
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	emit()
	return chunks, nil
}

func estimateTokens(word string) int {
	if word == "" {
		return 0
	}
	runes := utf8.RuneCountInString(word)
	switch {
	case runes <= 4:
		return 1
	case runes <= 8:
		return 2
	case runes <= 16:
		return 3
	default:
		return 4
	}
}

func makeChunk(idx int, text, name string, src Source) DocumentChunk {
	chunk := DocumentChunk{
		ID:      chunkID(name, idx),
		Content: strings.TrimSpace(text),
		Metadata: map[string]any{
			"chunk_index": idx,
		},
	}
	chunk = chunk.WithProvenance(src.Name, src.URI, 0, src.Version)
	for k, v := range src.Additional {
		chunk.Metadata[k] = v
	}
	return chunk
}

func chunkID(name string, idx int) string {
	if name == "" {
		return fmt.Sprintf("chunk-%d", idx)
	}
	sanitized := strings.ReplaceAll(strings.TrimSpace(name), " ", "_")
	sanitized = strings.ReplaceAll(sanitized, "/", "_")
	if sanitized == "" {
		sanitized = "chunk"
	}
	return fmt.Sprintf("%s#%d", sanitized, idx)
}

// Sequence helper for deterministic chunk ordering.
func sequenceID(base string, idx int) string {
	return fmt.Sprintf("%s-%s", base, strconv.Itoa(idx))
}
