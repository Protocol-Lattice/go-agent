package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/Protocol-Lattice/go-agent"
	"github.com/Protocol-Lattice/go-agent/src/memory"
	"github.com/Protocol-Lattice/go-agent/src/models"
)

// Simple mock model for the example
type mockModel struct{}

func (m *mockModel) Generate(ctx context.Context, prompt string) (any, error) {
	// In a real app, this would call an LLM.
	// Here we just return a dummy response.
	return "I remember what you said!", nil
}

func (m *mockModel) GenerateWithFiles(ctx context.Context, prompt string, files []models.File) (any, error) {
	return m.Generate(ctx, prompt)
}

func (m *mockModel) GenerateStream(ctx context.Context, prompt string) (<-chan models.StreamChunk, error) {
	ch := make(chan models.StreamChunk, 1)
	ch <- models.StreamChunk{Delta: "I remember what you said!", FullText: "I remember what you said!", Done: true}
	close(ch)
	return ch, nil
}

func main() {
	ctx := context.Background()

	// 1. Initialize the first agent instance
	fmt.Println("--- Agent 1: Running ---")
	mem1 := memory.NewSessionMemory(&memory.MemoryBank{}, 10)
	// Use DummyEmbedder for example speed/simplicity
	mem1 = mem1.WithEmbedder(memory.DummyEmbedder{})

	agent1, err := agent.New(agent.Options{
		Model:        &mockModel{},
		Memory:       mem1,
		SystemPrompt: "You are a helpful assistant.",
	})
	if err != nil {
		log.Fatal(err)
	}

	// Simulate a conversation
	sessionID := "session-123"
	fmt.Println("User: My favorite color is blue.")
	// We manually inject memory here to simulate a turn, since our mock model is static.
	// In a real app, agent.Generate() would store the interaction.
	// Accessing internal memory for setup:
	// (Note: In real usage, you'd just call agent.Generate(ctx, sessionID, "My favorite color is blue"))
	_, _ = agent1.Generate(ctx, sessionID, "My favorite color is blue")

	// 2. Checkpoint the agent
	fmt.Println("\n--- Checkpointing Agent 1 ---")
	checkpointData, err := agent1.Checkpoint()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Checkpoint size: %d bytes\n", len(checkpointData))

	// Simulate saving to disk
	if err := os.WriteFile("agent_checkpoint.json", checkpointData, 0644); err != nil {
		log.Fatal(err)
	}
	fmt.Println("Saved checkpoint to agent_checkpoint.json")

	// 3. Create a NEW agent instance (simulating a restart)
	fmt.Println("\n--- Agent 2: Restoring from Checkpoint ---")
	mem2 := memory.NewSessionMemory(&memory.MemoryBank{}, 10)
	mem2 = mem2.WithEmbedder(memory.DummyEmbedder{})

	agent2, err := agent.New(agent.Options{
		Model:        &mockModel{},
		Memory:       mem2,
		SystemPrompt: "Default prompt", // This will be overwritten by restore
	})
	if err != nil {
		log.Fatal(err)
	}

	// Load from disk
	data, err := os.ReadFile("agent_checkpoint.json")
	if err != nil {
		log.Fatal(err)
	}

	// Restore state
	if err := agent2.Restore(data); err != nil {
		log.Fatal(err)
	}
	fmt.Println("Agent 2 restored successfully.")

	// 4. Verify memory
	// We'll inspect the memory to prove it's there.
	// In a real app, you'd ask the agent "What is my favorite color?"
	records, err := agent2.SessionMemory().RetrieveContext(ctx, sessionID, "favorite color", 5)
	if err != nil {
		log.Fatal(err)
	}

	found := false
	for _, r := range records {
		// Our mock model returns "I remember what you said! | My favorite color is blue"
		// or similar depending on how Generate is implemented.
		// But we also stored the user input.
		fmt.Printf("Memory found: [%s] %s\n", r.Space, r.Content)
		if r.Content == "My favorite color is blue" {
			found = true
		}
	}

	if found {
		fmt.Println("\nSUCCESS: Agent 2 remembered the user's favorite color!")
	} else {
		fmt.Println("\nFAILURE: Memory not found.")
		os.Exit(1)
	}

	// Cleanup
	os.Remove("agent_checkpoint.json")
}
