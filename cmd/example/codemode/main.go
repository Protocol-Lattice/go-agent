package main

import (
	"context"
	"fmt"
	"log"

	"github.com/Protocol-Lattice/go-agent/src/adk"
	"github.com/Protocol-Lattice/go-agent/src/adk/modules"
	"github.com/Protocol-Lattice/go-agent/src/memory"
	"github.com/Protocol-Lattice/go-agent/src/memory/engine"
	"github.com/Protocol-Lattice/go-agent/src/models"
	"github.com/universal-tool-calling-protocol/go-utcp"
)

// DummyCodeModeModel simulates an LLM.
type DummyCodeModeModel struct{}

func (m *DummyCodeModeModel) Generate(ctx context.Context, prompt string) (any, error) {
	return "I am a dummy model configured with CodeMode.", nil
}

func (m *DummyCodeModeModel) GenerateWithFiles(ctx context.Context, prompt string, files []models.File) (any, error) {
	return m.Generate(ctx, prompt)
}

func main() {
	ctx := context.Background()

	// 1. Setup UTCP Client
	// In a real application, you would configure this with providers (e.g. via a JSON config file)
	// so the agent can access external tools.
	client, err := utcp.NewUTCPClient(ctx, &utcp.UtcpClientConfig{}, nil, nil)
	if err != nil {
		log.Fatalf("Failed to create UTCP client: %v", err)
	}

	// 2. Setup Model
	model := &DummyCodeModeModel{}

	// 3. Build ADK with CodeModeUtcp
	// This configures the agent to use the UTCP client for tool execution.
	// The 'CodeModeUtcp' option integrates the UTCP client with the agent's tool usage loop.
	memOpts := engine.DefaultOptions()
	kit, err := adk.New(ctx,
		adk.WithDefaultSystemPrompt("You are a helpful assistant."),
		adk.WithModules(
			modules.NewModelModule("model", func(_ context.Context) (models.Agent, error) {
				return model, nil
			}),
			modules.InMemoryMemoryModule(10000, memory.AutoEmbedder(), &memOpts),
		),
		// Enable CodeMode with the UTCP client
		adk.WithCodeModeUtcp(client, model),
	)
	if err != nil {
		log.Fatalf("Failed to initialise kit: %v", err)
	}

	// 4. Build Agent
	ag, err := kit.BuildAgent(ctx)
	if err != nil {
		log.Fatalf("Failed to build agent: %v", err)
	}

	fmt.Println("Agent with CodeMode created successfully.")

	// 5. Run Agent
	resp, err := ag.Generate(ctx, "session-1", "Hello CodeMode!")
	if err != nil {
		log.Fatalf("Generate failed: %v", err)
	}

	fmt.Printf("Agent response: %v\n", resp)
}
