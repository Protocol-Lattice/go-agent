package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"

	"github.com/Raezil/go-agent-development-kit/pkg/memory"
	"github.com/Raezil/go-agent-development-kit/pkg/uploads"
)

func main() {
	var (
		pdfPath   = flag.String("pdf", "", "Optional PDF to ingest")
		repoPath  = flag.String("repo", "", "Optional repo to ingest")
		question  = flag.String("q", "", "Natural language question")
		topK      = flag.Int("k", 3, "Results to display")
		normalize = flag.Bool("normalize", true, "Normalize embeddings for cosine similarity")
	)
	flag.Parse()
	if *question == "" {
		log.Fatal("--q question prompt is required")
	}

	embedder := memory.AutoEmbedder()
	pipeline := uploads.Pipeline{
		Embedder:       embedder,
		SimilarityMode: "cosine",
		Normalize:      *normalize,
		Middlewares:    []uploads.Middleware{uploads.PIIRedactor{}},
	}

	var chunks []uploads.DocumentChunk
	if *pdfPath != "" {
		f, err := os.Open(*pdfPath)
		if err != nil {
			log.Fatalf("open pdf: %v", err)
		}
		defer f.Close()
		pdfChunks, err := (uploads.PDFChunker{}).Chunk(uploads.ReaderWithName{Name: filepath.Base(*pdfPath), Reader: f}, uploads.Source{Name: "pdf", URI: *pdfPath})
		if err != nil {
			log.Fatalf("chunk pdf: %v", err)
		}
		chunks = append(chunks, pdfChunks...)
	}
	if *repoPath != "" {
		repoChunks, err := (uploads.CodeChunker{Root: *repoPath}).Chunk(uploads.ReaderWithName{}, uploads.Source{Name: "repo", URI: *repoPath})
		if err != nil {
			log.Fatalf("chunk repo: %v", err)
		}
		chunks = append(chunks, repoChunks...)
	}
	if len(chunks) == 0 {
		log.Fatal("at least one source (--pdf or --repo) must be provided")
	}

	embedded, err := pipeline.Process(context.Background(), chunks)
	if err != nil {
		log.Fatalf("embed chunks: %v", err)
	}
	filtered := embedded[:0]
	for _, res := range embedded {
		if res.Err == nil {
			filtered = append(filtered, res)
		}
	}
	if len(filtered) == 0 {
		log.Fatal("no embeddings generated from sources")
	}

	questionVec, err := embedder.Embed(context.Background(), *question)
	if err != nil {
		log.Fatalf("embed question: %v", err)
	}
	if *normalize {
		uploads.NormalizeVector(questionVec)
	}

	sort.Slice(filtered, func(i, j int) bool {
		return similarity(questionVec, filtered[i].Vector) > similarity(questionVec, filtered[j].Vector)
	})

	limit := *topK
	if limit > len(filtered) {
		limit = len(filtered)
	}
	for i := 0; i < limit; i++ {
		res := filtered[i]
		fmt.Printf("%d. score=%.4f source=%v id=%s\n", i+1, similarity(questionVec, res.Vector), res.Chunk.Metadata["source"], res.Chunk.ID)
		fmt.Printf("   preview: %.160s\n", res.Chunk.Content)
	}
}

func similarity(a []float32, b []float32) float64 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var sum float64
	for i := 0; i < n; i++ {
		sum += float64(a[i] * b[i])
	}
	return sum
}
