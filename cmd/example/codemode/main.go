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

// DemoModel simulates an LLM that can generate Go code calling codemode.CallTool
type DemoModel struct {
	Name string
}

func (m *DemoModel) Generate(ctx context.Context, prompt string) (any, error) {
	// In a real LLM, it would analyze the prompt and generate Go code
	// For this demo, we simulate the response
	return fmt.Sprintf("[%s] Processed: %s", m.Name, prompt), nil
}

func (m *DemoModel) GenerateWithFiles(ctx context.Context, prompt string, files []models.File) (any, error) {
	return m.Generate(ctx, prompt)
}

func (m *DemoModel) GenerateStream(ctx context.Context, prompt string) (<-chan models.StreamChunk, error) {
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

	fmt.Println("=== CodeMode + Agent as UTCP Tool Example ===")
	fmt.Println("Demonstrates how CodeMode orchestrates UTCP tools")
	fmt.Println()

	// 1. Create UTCP Client
	client, err := utcp.NewUTCPClient(ctx, &utcp.UtcpClientConfig{}, nil, nil)
	if err != nil {
		log.Fatalf("Failed to create UTCP client: %v", err)
	}

	// 2. Create and register a specialist agent as a UTCP tool
	analyst, err := agent.New(agent.Options{
		Model:        &DemoModel{Name: "Analyst"},
		Memory:       memory.NewSessionMemory(memory.NewMemoryBankWithStore(memory.NewInMemoryStore()), 8),
		SystemPrompt: "You are a data analyst.",
	})
	if err != nil {
		log.Fatalf("Failed to create analyst: %v", err)
	}

	// Register as "local.analyst" (provider.toolname format)
	err = analyst.RegisterAsUTCPProvider(ctx, client, "local.analyst", "Analyzes data")
	if err != nil {
		log.Fatalf("Failed to register analyst: %v", err)
	}
	fmt.Println("✅ Registered agent as 'local.analyst'")

	// 3. Verify direct tool call works
	fmt.Println("\n--- Direct UTCP Tool Call ---")
	result, err := client.CallTool(ctx, "local.analyst", map[string]any{
		"instruction": "Analyze Q4 sales data",
	})
	if err != nil {
		log.Fatalf("Direct tool call failed: %v", err)
	}
	fmt.Printf("Result: %v\n", result)

	// 4. Create an orchestrator agent with CodeMode enabled
	fmt.Println("\n--- Building Orchestrator with CodeMode ---")

	orchestratorModel := &DemoModel{Name: "Orchestrator"}
	memOpts := engine.DefaultOptions()
	kit, err := adk.New(ctx,
		adk.WithDefaultSystemPrompt("You orchestrate workflows using UTCP tools."),
		adk.WithModules(
			modules.NewModelModule("model", func(_ context.Context) (models.Agent, error) {
				return orchestratorModel, nil
			}),
			modules.InMemoryMemoryModule(10000, memory.AutoEmbedder(), &memOpts),
		),
		// WithCodeModeUtcp enables the agent to execute Go code that calls tools
		adk.WithCodeModeUtcp(client, orchestratorModel),
	)
	if err != nil {
		log.Fatalf("Failed to initialize ADK: %v", err)
	}

	orchestrator, err := kit.BuildAgent(ctx)
	if err != nil {
		log.Fatalf("Failed to build orchestrator: %v", err)
	}
	fmt.Println("✅ Orchestrator with CodeMode created")

	// 5. Use the orchestrator with a natural language workflow
	fmt.Println("\n--- Workflow Execution ---")
	fmt.Println("User Input: \"Analyze Q4 sales and provide insights\"")
	fmt.Println()
	fmt.Println("With a real LLM, CodeMode would generate and execute:")
	fmt.Println("  result, _ := codemode.CallTool(\"local.analyst\", map[string]any{")
	fmt.Println("    \"instruction\": \"Analyze Q4 sales and provide insights\",")
	fmt.Println("  })")
	fmt.Println()

	// Simulate the orchestrator processing the request
	response, err := orchestrator.Generate(ctx, "workflow-session", "Analyze Q4 sales and provide insights")
	if err != nil {
		log.Fatalf("Orchestrator failed: %v", err)
	}
	fmt.Printf("Orchestrator Response: %v\n", response)

	// Summary
	fmt.Println("\n--- Summary ---")
	fmt.Println("✓ Agents exposed as UTCP tools using RegisterAsUTCPProvider()")
	fmt.Println("✓ Tools can be called directly: client.CallTool(ctx, \"local.analyst\", args)")
	fmt.Println("✓ CodeMode enables agents to orchestrate tools via generated Go code")
	fmt.Println("✓ User provides natural language → CodeMode generates tool calls")
	fmt.Println()
	fmt.Println("--- Key Pattern ---")
	fmt.Println("User Input (Natural Language)")
	fmt.Println("  ↓")
	fmt.Println("LLM with CodeMode generates Go code:")
	fmt.Println("  codemode.CallTool(\"local.analyst\", {\"instruction\": userInput})")
	fmt.Println("  ↓")
	fmt.Println("UTCP executes the agent tool")
	fmt.Println("  ↓")
	fmt.Println("Result returned to user")
}
