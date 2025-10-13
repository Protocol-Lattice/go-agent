package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/Raezil/go-agent-development-kit/pkg/memory"
)

func main() {
	ctx := context.Background()

	connStr := os.Getenv("DATABASE_URL")
	if connStr == "" {
		connStr = "postgres://admin:admin@localhost:5432/ragdb?sslmode=disable"
	}

	// -----------------------------------
	// 🧠 Connect and prepare schema
	// -----------------------------------
	mb, err := memory.NewMemoryBank(ctx, connStr)
	if err != nil {
		log.Fatalf("DB connect error: %v", err)
	}
	defer mb.DB.Close()

	if err := mb.CreateSchema(ctx, "pkg/memory/schema.sql"); err != nil {
		log.Fatalf("Schema apply failed: %v", err)
	}

	// -----------------------------------
	// 💬 Store a few sample memories
	// -----------------------------------
	texts := []string{
		"The Go agent learns pgvector memory.",
		"Postgres vector search is efficient for RAG.",
		"Embeddings can be generated locally or via LLMs.",
	}

	for i, text := range texts {
		embed := memory.DummyEmbedding(text) // or memory.dummyEmbedding(text)
		meta := fmt.Sprintf(`{"index": %d}`, i)
		err := mb.StoreMemory(ctx, "session-001", text, meta, embed)
		if err != nil {
			log.Fatalf("Insert failed: %v", err)
		}
		fmt.Printf("✅ Stored: %s\n", text)
	}

	// -----------------------------------
	// 🧩 Build a contextual prompt
	// -----------------------------------
	query := "How does the agent use vector memory?"
	prompt, err := mb.BuildPrompt(ctx, query, 3)
	if err != nil {
		log.Fatalf("Prompt build failed: %v", err)
	}

	fmt.Println("\n🧩 Generated Prompt:")
	fmt.Println(strings.Repeat("-", 60))
	fmt.Println(prompt)
	fmt.Println(strings.Repeat("-", 60))
}
