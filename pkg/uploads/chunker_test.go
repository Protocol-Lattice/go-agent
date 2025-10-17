package uploads

import (
	"strings"
	"testing"
)

func TestTextChunker(t *testing.T) {
	chunker := TextChunker{MaxTokens: 4}
	src := Source{Name: "notes", URI: "file://notes.txt"}
	chunks, err := chunker.Chunk(ReaderWithName{Name: "notes.txt", Reader: strings.NewReader("one two three four five six")}, src)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunks) != 2 {
		t.Fatalf("expected 2 chunks got %d", len(chunks))
	}
	if chunks[0].Metadata["source"] != "notes" {
		t.Fatalf("expected provenance metadata")
	}
}

func TestMarkdownChunkerSections(t *testing.T) {
	chunker := MarkdownChunker{MaxTokens: 10}
	src := Source{Name: "doc"}
	md := "# Heading\nBody line\n\n## Sub\nMore text"
	chunks, err := chunker.Chunk(ReaderWithName{Name: "doc.md", Reader: strings.NewReader(md)}, src)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunks) != 2 {
		t.Fatalf("expected 2 chunks got %d", len(chunks))
	}
	if _, ok := chunks[0].Metadata["section_heading"]; !ok {
		t.Fatalf("expected section heading metadata")
	}
}
