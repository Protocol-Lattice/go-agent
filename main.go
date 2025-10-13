package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/Raezil/go-agent-development-kit/pkg/memory"
)

func main() {
	ctx := context.Background()

	// Connection string (adjust user/password as needed)
	connStr := "postgres://admin:admin@localhost:5432/ragdb?sslmode=disable"

	// 1️⃣ Initialize Postgres connection
	bank, err := memory.NewMemoryBank(ctx, connStr)
	if err != nil {
		log.Fatalf("❌ failed to init memory bank: %v", err)
	}
	defer bank.DB.Close()

	// 2️⃣ Ensure schema (vector + table) exists
	if err := bank.CreateSchema(ctx, "schema.sql"); err != nil {
		log.Fatalf("❌ failed to create schema: %v", err)
	}

	// 3️⃣ Create a session memory manager (short-term cache of 5 messages)
	sessionMemory := memory.NewSessionMemory(bank, 5)
	sessionID := fmt.Sprintf("session-%d", time.Now().Unix())

	// 4️⃣ Add short-term memories (simulate recent conversation)
	sessionMemory.AddShortTerm(sessionID, "User likes Go and AI agent development.", `{}`, memory.VertexAIEmbedding("Go AI agent"))
	sessionMemory.AddShortTerm(sessionID, "User is learning about pgvector for semantic search.", `{}`, memory.VertexAIEmbedding("pgvector semantic search"))
	sessionMemory.AddShortTerm(sessionID, "User asked how to connect Vertex AI with Postgres memory bank.", `{}`, memory.VertexAIEmbedding("Vertex AI Postgres memory bank"))

	// 5️⃣ Optionally flush short-term to long-term (simulate persistence)
	if err := sessionMemory.FlushToLongTerm(ctx, sessionID); err != nil {
		log.Printf("⚠️ flush warning: %v", err)
	} else {
		log.Printf("✅ Flushed short-term memory for session %s to long-term storage", sessionID)
	}

	// 6️⃣ Build a prompt combining both memory layers
	query := "How does pgvector improve retrieval for AI agents?"
	prompt, err := sessionMemory.BuildPrompt(ctx, sessionID, query, 5)
	if err != nil {
		log.Fatalf("❌ failed to build prompt: %v", err)
	}

	fmt.Println("----- Generated Prompt -----")
	fmt.Println(prompt)
	fmt.Println("----------------------------")
}
