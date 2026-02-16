package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/Protocol-Lattice/go-agent"
	"github.com/Protocol-Lattice/go-agent/src/memory"
	"github.com/Protocol-Lattice/go-agent/src/models"
	"github.com/universal-tool-calling-protocol/go-utcp"
)

// MockModel simulates an LLM that can decide to call tools.
type MockModel struct {
	Name string
}

func (m *MockModel) Generate(ctx context.Context, prompt string) (any, error) {
	// Simulate Manager logic: if asked about "facts", call the researcher.
	if m.Name == "Manager" && strings.Contains(prompt, "fact") {
		// In a real LLM, this would be a structured tool call request.
		// Here we just simulate the decision logic for the example.
		return "I need to ask the researcher about this.", nil
	}

	// Simulate Researcher logic
	if m.Name == "Researcher" {
		return "Research complete: The sky is blue because of Rayleigh scattering.", nil
	}

	return fmt.Sprintf("[%s] I received: %s", m.Name, prompt), nil
}

func (m *MockModel) GenerateWithFiles(ctx context.Context, prompt string, files []models.File) (any, error) {
	return m.Generate(ctx, prompt)
}

func (m *MockModel) GenerateStream(ctx context.Context, prompt string) (<-chan models.StreamChunk, error) {
	ch := make(chan models.StreamChunk, 1)
	val, err := m.Generate(ctx, prompt)
	if err != nil {
		ch <- models.StreamChunk{Err: err, Done: true}
	} else {
		str := fmt.Sprint(val)
		ch <- models.StreamChunk{Delta: str, FullText: str, Done: true}
	}
	close(ch)
	return ch, nil
}

func main() {
	ctx := context.Background()

	// 1. Setup UTCP Client
	// This client will act as the "tool bus" connecting agents.
	client, err := utcp.NewUTCPClient(ctx, &utcp.UtcpClientConfig{}, nil, nil)
	if err != nil {
		log.Fatalf("Failed to create UTCP client: %v", err)
	}

	// 2. Create the "Researcher" Agent
	researcher, err := agent.New(agent.Options{
		Model:        &MockModel{Name: "Researcher"},
		Memory:       memory.NewSessionMemory(memory.NewMemoryBankWithStore(memory.NewInMemoryStore()), 8),
		SystemPrompt: "You are a researcher. You find facts.",
	})
	if err != nil {
		log.Fatalf("Failed to create researcher: %v", err)
	}

	// 3. Expose Researcher as a UTCP Tool
	// This registers the researcher agent on the UTCP client as "agent.researcher".
	err = researcher.RegisterAsUTCPProvider(ctx, client, "agent.researcher", "A specialist agent that performs research.")
	if err != nil {
		log.Fatalf("Failed to register researcher: %v", err)
	}
	fmt.Println("✅ Researcher agent registered as UTCP tool: 'agent.researcher'")

	// 4. Verify the tool works directly
	fmt.Println("\n--- Direct Tool Call Test ---")
	result, err := client.CallTool(ctx, "agent.researcher", map[string]any{
		"instruction": "Find facts about the sky.",
	})
	if err != nil {
		log.Fatalf("Tool call failed: %v", err)
	}
	fmt.Printf("Tool Output: %v\n", result)

	// 5. Create the "Manager" Agent
	// In a real app, you would give the Manager the 'client' so it can call tools.
	// Here we just demonstrate the setup.
	manager, err := agent.New(agent.Options{
		Model:        &MockModel{Name: "Manager"},
		Memory:       memory.NewSessionMemory(memory.NewMemoryBankWithStore(memory.NewInMemoryStore()), 8),
		SystemPrompt: "You are a manager. Delegate tasks to the researcher.",
	})
	if err != nil {
		log.Fatalf("Failed to create manager: %v", err)
	}

	// 6. Simulate Manager Workflow
	fmt.Println("\n--- Manager Agent Workflow ---")
	prompt := "Tell me a fact about the sky."
	fmt.Printf("User: %s\n", prompt)

	resp, err := manager.Generate(ctx, "session-manager", prompt)
	if err != nil {
		log.Fatalf("Manager failed: %v", err)
	}
	fmt.Printf("Manager: %s\n", resp)

	// In a real scenario with a real LLM, the Manager would output a tool call request
	// for "agent.researcher", which the system would execute using 'client.CallTool'.
}
