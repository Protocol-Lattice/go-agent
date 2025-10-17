package uploads

import (
	"bufio"
	"io"
	"regexp"
	"strings"
)

var headingRegexp = regexp.MustCompile(`^(#{1,6})\s+(.*)$`)

// MarkdownChunker groups text under headings while respecting the configured token budget.
type MarkdownChunker struct {
	MaxTokens int
}

func (m MarkdownChunker) Chunk(reader ReaderWithName, src Source) ([]DocumentChunk, error) {
	max := m.MaxTokens
	if max <= 0 {
		max = 400
	}
	buf := new(strings.Builder)
	if _, err := io.Copy(buf, reader.Reader); err != nil {
		return nil, err
	}
	scanner := bufio.NewScanner(strings.NewReader(buf.String()))

	var (
		chunks     []DocumentChunk
		builder    strings.Builder
		heading    string
		idx        int
		tokenCount int
	)

	emit := func() {
		if builder.Len() == 0 {
			return
		}
		chunk := DocumentChunk{
			ID:      chunkID(reader.Name, idx),
			Content: strings.TrimSpace(builder.String()),
			Metadata: map[string]any{
				"chunk_index": idx,
			},
		}
		if heading != "" {
			chunk.Metadata["section_heading"] = heading
		}
		chunk = chunk.WithProvenance(src.Name, src.URI, 0, src.Version)
		for k, v := range src.Additional {
			chunk.Metadata[k] = v
		}
		chunks = append(chunks, chunk)
		idx++
		builder.Reset()
		heading = ""
		tokenCount = 0
	}

	for scanner.Scan() {
		line := scanner.Text()
		if matches := headingRegexp.FindStringSubmatch(line); len(matches) == 3 {
			if builder.Len() > 0 {
				emit()
			}
			heading = strings.TrimSpace(matches[2])
			continue
		}
		if strings.TrimSpace(line) == "" {
			if builder.Len() > 0 {
				builder.WriteString("\n")
			}
			continue
		}
		estimated := estimateTokens(line)
		if tokenCount+estimated > max && builder.Len() > 0 {
			emit()
		}
		if builder.Len() > 0 {
			builder.WriteString("\n")
		}
		builder.WriteString(line)
		tokenCount += estimated
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	emit()
	return chunks, nil
}
