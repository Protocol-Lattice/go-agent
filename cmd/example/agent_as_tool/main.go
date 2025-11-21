package main

import (
	"context"
	"fmt"
	"log"

	"github.com/Protocol-Lattice/go-agent"
	"github.com/Protocol-Lattice/go-agent/src/memory"
	"github.com/Protocol-Lattice/go-agent/src/models"
	"github.com/universal-tool-calling-protocol/go-utcp"
)

// DummyModel simulates an LLM.
type DummyModel struct {
	Name string
}

func (m *DummyModel) Generate(ctx context.Context, prompt string) (any, error) {
	// In a real agent, this would use an LLM to generate a response.
	return fmt.Sprintf("Hello from %s! I received your request: %s", m.Name, prompt), nil
}

func (m *DummyModel) GenerateWithFiles(ctx context.Context, prompt string, files []models.File) (any, error) {
	return m.Generate(ctx, prompt)
}

func main() {
	ctx := context.Background()

	// 1. Create the Agent
	// We use a dummy model here, but you would normally use models.NewLLMProvider(...)
	ag, err := agent.New(agent.Options{
		Model:        &DummyModel{Name: "InternalAgent"},
		Memory:       memory.NewSessionMemory(memory.NewMemoryBankWithStore(memory.NewInMemoryStore()), 8),
		SystemPrompt: "You are a helpful assistant.",
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	// 2. Create a UTCP Client
	// This client will be used to "call" the agent as a tool.
	// In a real scenario, this client might be connected to other remote tools as well.
	client, err := utcp.NewUTCPClient(ctx, &utcp.UtcpClientConfig{}, nil, nil)
	if err != nil {
		log.Fatalf("Failed to create UTCP client: %v", err)
	}

	// 3. Register the Agent as a UTCP Provider
	// This exposes the agent as a tool named "my-agent-tool" on the UTCP client.
	// The tool takes an "instruction" argument and returns the agent's response.
	toolName := "helper.ask"
	err = ag.RegisterAsUTCPProvider(ctx, client, toolName, "An agent that can answer questions.")
	if err != nil {
		log.Fatalf("Failed to register agent as provider: %v", err)
	}

	fmt.Printf("Agent registered as tool '%s'.\n", toolName)

	// 4. Call the Agent via the UTCP Client
	// This simulates another part of the system (or another agent) calling this agent as a tool.
	fmt.Println("Calling agent tool via UTCP...")

	args := map[string]any{
		"instruction": "What is the meaning of life?",
	}

	result, err := client.CallTool(ctx, toolName, args)
	if err != nil {
		log.Fatalf("CallTool failed: %v", err)
	}

	fmt.Printf("Tool Result: %v\n", result)
}
