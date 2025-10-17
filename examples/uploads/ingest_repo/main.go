package main

import (
	"context"
	"flag"
	"fmt"
	"log"

	"github.com/Raezil/go-agent-development-kit/pkg/memory"
	"github.com/Raezil/go-agent-development-kit/pkg/uploads"
)

func main() {
	var (
		repoPath = flag.String("repo", ".", "Path to the repository to ingest")
		source   = flag.String("source", "repo", "Logical source label")
	)
	flag.Parse()

	chunker := uploads.CodeChunker{Root: *repoPath}
	chunks, err := chunker.Chunk(uploads.ReaderWithName{}, uploads.Source{Name: *source, URI: *repoPath})
	if err != nil {
		log.Fatalf("chunk repo: %v", err)
	}
	fmt.Printf("Discovered %d chunks\n", len(chunks))

	pipeline := uploads.Pipeline{
		Embedder:       memory.AutoEmbedder(),
		SimilarityMode: "dot",
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
	}
	fmt.Printf("Embedded %d/%d chunks\n", successes, len(embedded))
}
