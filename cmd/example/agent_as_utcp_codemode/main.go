package main

import (
	"context"
	"fmt"
	"log"

	"github.com/Protocol-Lattice/go-agent"
	"github.com/Protocol-Lattice/go-agent/src/adk"
	"github.com/Protocol-Lattice/go-agent/src/adk/modules"
	"github.com/Protocol-Lattice/go-agent/src/memory"
	"github.com/Protocol-Lattice/go-agent/src/memory/engine"
	"github.com/Protocol-Lattice/go-agent/src/models"
	"github.com/universal-tool-calling-protocol/go-utcp"
)

func main() {
	ctx := context.Background()

	// 1. Initialise a UTCP client – this is the shared tool bus.
	client, err := utcp.NewUTCPClient(ctx, &utcp.UtcpClientConfig{}, nil, nil)
	if err != nil {
		log.Fatalf("Failed to create UTCP client: %v", err)
	}

	// 2. Create a specialist agent that will be exposed as a UTCP tool.
	// Use a real Gemini model
	specialistModel, err := models.NewGeminiLLM(ctx, "gemini-3-pro-preview", "You are a specialist that can answer trivia questions.")
	if err != nil {
		log.Fatalf("Failed to create specialist model: %v", err)
	}

	specialist, err := agent.New(agent.Options{
		Model:        specialistModel,
		Memory:       memory.NewSessionMemory(memory.NewMemoryBankWithStore(memory.NewInMemoryStore()), 8),
		SystemPrompt: "You are a versatile specialist agent that can help with various tasks.",
	})
	if err != nil {
		log.Fatalf("Failed to create specialist agent: %v", err)
	}

	// 3. Register the specialist as a UTCP provider.
	err = specialist.RegisterAsUTCPProvider(ctx, client, "specialist", "Answers trivia via UTCP")
	if err != nil {
		log.Fatalf("Failed to register specialist: %v", err)
	}
	fmt.Println("✅ Registered specialist as UTCP tool 'specialist'")

	// 4. Build an orchestrator agent with CodeMode enabled.

	// Use a real Gemini model for the orchestrator
	orchestratorModel, err := models.NewGeminiLLM(ctx, "gemini-3-pro-preview", "You orchestrate workflows using CodeMode. Generate ONLY valid Go code. Do not include package main or imports. Use codemode.CallTool to invoke tools.")
	if err != nil {
		log.Fatalf("Failed to create orchestrator model: %v", err)
	}

	memOpts := engine.DefaultOptions()
	kit, err := adk.New(ctx,
		adk.WithDefaultSystemPrompt("You orchestrate workflows using CodeMode. Generate ONLY valid Go code. Do not include package main or imports. Use codemode.CallTool to invoke tools."),
		adk.WithModules(
			modules.NewModelModule("model", func(_ context.Context) (models.Agent, error) { return orchestratorModel, nil }),
			modules.InMemoryMemoryModule(10000, memory.AutoEmbedder(), &memOpts),
		),
		// Enable CodeMode – it will generate calls to codemode.CallTool which under the hood uses the UTCP client.
		adk.WithCodeModeUtcp(client, orchestratorModel),
	)
	if err != nil {
		log.Fatalf("Failed to initialise ADK: %v", err)
	}
	orchestrator, err := kit.BuildAgent(ctx)
	if err != nil {
		log.Fatalf("Failed to build orchestrator: %v", err)
	}
	fmt.Println("✅ Orchestrator with CodeMode ready")

	// Register the orchestrator itself as a UTCP provider so it can be called recursively or for summarization
	err = orchestrator.RegisterAsUTCPProvider(ctx, client, "orchestrator", "Orchestrates workflows and summarizes information")
	if err != nil {
		log.Fatalf("Failed to register orchestrator: %v", err)
	}
	fmt.Println("✅ Registered orchestrator as UTCP tool 'orchestrator'")

	// 5. Run a natural‑language request that the orchestrator will turn into a CodeMode workflow.
	// This prompt describes a multi-step process that the agent would convert into a Go script
	// calling the respective UTCP tools (e.g. http.echo, http.timestamp, etc.).
	userPrompt := `
Step 1: Ask for a fun fact about the Eiffel Tower using the specialist tool.
Step 2: Ask for a fun fact about the Great Wall of China using the specialist tool.
Step 3: Use the orchestrator tool to summarize the facts from Step 1 and Step 2.
`
	fmt.Printf("\nUser Prompt:\n%s\n", userPrompt)

	// Invoke the orchestrator.
	// In a real scenario, the Orchestrator (powered by an LLM) would generate Go code
	// to execute these steps sequentially, passing data between them.
	resp, err := orchestrator.Generate(ctx, "session-orchestrator", userPrompt)
	if err != nil {
		log.Fatalf("Orchestrator failed: %v", err)
	}
	fmt.Printf("Orchestrator response: %v\n", resp)
}
