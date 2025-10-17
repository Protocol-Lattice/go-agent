package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/Raezil/go-agent-development-kit/pkg/memory"
	"github.com/Raezil/go-agent-development-kit/pkg/uploads"
)

func main() {
	var (
		filePath = flag.String("pdf", "", "Path to the PDF file to ingest")
		source   = flag.String("source", "pdf", "Logical source label")
	)
	flag.Parse()
	if *filePath == "" {
		log.Fatal("--pdf path is required")
	}
	f, err := os.Open(*filePath)
	if err != nil {
		log.Fatalf("open pdf: %v", err)
	}
	defer f.Close()

	chunker := uploads.PDFChunker{}
	chunks, err := chunker.Chunk(uploads.ReaderWithName{Name: filepath.Base(*filePath), Reader: f}, uploads.Source{Name: *source, URI: *filePath})
	if err != nil {
		log.Fatalf("chunk pdf: %v", err)
	}
	fmt.Printf("Extracted %d chunks\n", len(chunks))

	pipeline := uploads.Pipeline{
		Embedder:       memory.AutoEmbedder(),
		SimilarityMode: "cosine",
		Normalize:      true,
		Middlewares:    []uploads.Middleware{uploads.PIIRedactor{}},
	}
	embedded, err := pipeline.Process(context.Background(), chunks)
	if err != nil {
		log.Fatalf("embed chunks: %v", err)
	}
	successes := 0
	for _, res := range embedded {
		if res.Err != nil {
			fmt.Printf("chunk %s failed: %v\n", res.Chunk.ID, res.Err)
			continue
		}
		successes++
		fmt.Printf("chunk %s -> dim=%d\n", res.Chunk.ID, len(res.Vector))
	}
	fmt.Printf("Embedded %d/%d chunks\n", successes, len(embedded))
}
