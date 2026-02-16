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

// SimpleModel simulates an LLM for demonstration
type SimpleModel struct {
	Name string
}

func (m *SimpleModel) Generate(ctx context.Context, prompt string) (any, error) {
	return fmt.Sprintf("[%s] Analyzed: %s", m.Name, prompt), nil
}

func (m *SimpleModel) GenerateWithFiles(ctx context.Context, prompt string, files []models.File) (any, error) {
	return m.Generate(ctx, prompt)
}

func (m *SimpleModel) GenerateStream(ctx context.Context, prompt string) (<-chan models.StreamChunk, error) {
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

	fmt.Println("=== CodeMode Workflow Orchestration Example ===")
	fmt.Println("Demonstrates how CodeMode orchestrates UTCP tools via natural language prompts")
	fmt.Println()

	// 1. Setup UTCP Client
	client, err := utcp.NewUTCPClient(ctx, &utcp.UtcpClientConfig{}, nil, nil)
	if err != nil {
		log.Fatalf("Failed to create UTCP client: %v", err)
	}

	// 2. Create and register specialist agents as UTCP tools

	// Data Analyst Agent
	analyst, err := agent.New(agent.Options{
		Model:        &SimpleModel{Name: "DataAnalyst"},
		Memory:       memory.NewSessionMemory(memory.NewMemoryBankWithStore(memory.NewInMemoryStore()), 8),
		SystemPrompt: "You are a data analyst. You analyze data and provide insights.",
	})
	if err != nil {
		log.Fatalf("Failed to create analyst: %v", err)
	}
	err = analyst.RegisterAsUTCPProvider(ctx, client, "analyst", "Analyzes data and provides insights")
	if err != nil {
		log.Fatalf("Failed to register analyst: %v", err)
	}
	fmt.Println("✅ Registered: analyst")

	// Report Writer Agent
	writer, err := agent.New(agent.Options{
		Model:        &SimpleModel{Name: "ReportWriter"},
		Memory:       memory.NewSessionMemory(memory.NewMemoryBankWithStore(memory.NewInMemoryStore()), 8),
		SystemPrompt: "You are a report writer. You create professional reports.",
	})
	if err != nil {
		log.Fatalf("Failed to create writer: %v", err)
	}
	err = writer.RegisterAsUTCPProvider(ctx, client, "writer", "Creates professional reports")
	if err != nil {
		log.Fatalf("Failed to register writer: %v", err)
	}
	fmt.Println("✅ Registered: writer")

	// Reviewer Agent
	reviewer, err := agent.New(agent.Options{
		Model:        &SimpleModel{Name: "Reviewer"},
		Memory:       memory.NewSessionMemory(memory.NewMemoryBankWithStore(memory.NewInMemoryStore()), 8),
		SystemPrompt: "You are a reviewer. You review and provide feedback.",
	})
	if err != nil {
		log.Fatalf("Failed to create reviewer: %v", err)
	}
	err = reviewer.RegisterAsUTCPProvider(ctx, client, "reviewer", "Reviews content and provides feedback")
	if err != nil {
		log.Fatalf("Failed to register reviewer: %v", err)
	}
	fmt.Println("✅ Registered: reviewer")

	// 3. Build Orchestrator Agent with CodeMode
	orchestratorModel := &SimpleModel{Name: "Orchestrator"}
	memOpts := engine.DefaultOptions()
	kit, err := adk.New(ctx,
		adk.WithDefaultSystemPrompt("You are an orchestrator that coordinates workflows using UTCP tools."),
		adk.WithModules(
			modules.NewModelModule("model", func(_ context.Context) (models.Agent, error) {
				return orchestratorModel, nil
			}),
			modules.InMemoryMemoryModule(10000, memory.AutoEmbedder(), &memOpts),
		),
		// CodeMode enables the agent to orchestrate UTCP tools via natural language
		adk.WithCodeModeUtcp(client, orchestratorModel),
	)
	if err != nil {
		log.Fatalf("Failed to initialize ADK: %v", err)
	}

	orchestrator, err := kit.BuildAgent(ctx)
	if err != nil {
		log.Fatalf("Failed to build orchestrator: %v", err)
	}
	fmt.Println("✅ Built orchestrator agent with CodeMode")
	fmt.Println()

	// 4. Demonstrate workflow orchestration via natural language prompt
	fmt.Println("--- Workflow Orchestration via CodeMode ---")
	fmt.Println("User provides a natural language workflow description:")
	fmt.Println()

	workflowPrompt := `
Create a quarterly business report with the following workflow:

Step 1: Use the analyst tool to analyze Q4 sales data
Step 2: Use the writer tool to create a report based on the analysis 
Step 3: Use the reviewer tool to review the report and provide feedback
Step 4: Use the writer tool again to finalize the report based on reviewer feedback

Please execute this workflow.
`

	fmt.Println("Workflow Prompt:")
	fmt.Println(workflowPrompt)
	fmt.Println("\n--- Executing Workflow ---")

	// CodeMode will interpret this natural language workflow and:
	// 1. Parse the steps
	// 2. Call the appropriate UTCP tools (analyst, writer, reviewer)
	// 3. Chain the results together
	// 4. Return the final output

	result, err := orchestrator.Generate(ctx, "workflow-session", workflowPrompt)
	if err != nil {
		log.Fatalf("Workflow execution failed: %v", err)
	}

	fmt.Printf("\n--- Workflow Result ---\n%v\n\n", result)

	// 5. Demonstrate individual tool verification
	fmt.Println("--- Direct Tool Verification ---")
	directResult, err := client.CallTool(ctx, "analyst", map[string]any{
		"instruction": "Verify Q4 sales analysis",
	})
	if err != nil {
		log.Fatalf("Direct tool call failed: %v", err)
	}
	fmt.Printf("Direct call to analyst: %v\n\n", directResult)

	// Summary
	fmt.Println("--- Summary ---")
	fmt.Println("✓ CodeMode orchestrates UTCP tool workflows via natural language prompts")
	fmt.Println("✓ Agents exposed as UTCP tools can be chained together in workflows")
	fmt.Println("✓ The orchestrator interprets workflow steps and executes tools in sequence")
	fmt.Println("✓ This enables complex multi-agent workflows with simple natural language")
	fmt.Println()
	fmt.Println("--- Key Pattern ---")
	fmt.Println("User Input (Natural Language) → CodeMode → UTCP Tool Orchestration → Result")
	fmt.Println("Example: 'Step 1: analyze data, Step 2: write report, Step 3: review'")
}
