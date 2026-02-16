package main

import (
	"context"
	"fmt"
	"log"

	"github.com/Protocol-Lattice/go-agent"
	"github.com/Protocol-Lattice/go-agent/src/memory"
	"github.com/Protocol-Lattice/go-agent/src/models"
)

// DummyModel is a simple model that returns a fixed response or echoes the prompt.
// In a real scenario, you would use models.NewLLMProvider(...)
type DummyModel struct {
	Name string
}

func (m *DummyModel) Generate(ctx context.Context, prompt string) (any, error) {
	log.Printf("[%s] Received prompt: %s", m.Name, prompt)
	if m.Name == "Researcher" {
		return "Research complete: The sky is blue because of Rayleigh scattering.", nil
	}
	// Manager logic: if prompt contains "Research complete", summarize it.
	// Otherwise, ask the researcher.
	// Note: In a real LLM, the model would decide to call the tool.
	// Here we simulate the tool call decision for demonstration if the prompt asks for research.
	return "I have received the research report.", nil
}

func (m *DummyModel) GenerateWithFiles(ctx context.Context, prompt string, files []models.File) (any, error) {
	return m.Generate(ctx, prompt)
}

func (m *DummyModel) GenerateStream(ctx context.Context, prompt string) (<-chan models.StreamChunk, error) {
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

	// 1. Create the Specialist Agent (Researcher)
	researcherMem := memory.NewSessionMemory(memory.NewMemoryBankWithStore(memory.NewInMemoryStore()), 8)
	researcher, err := agent.New(agent.Options{
		Model:        &DummyModel{Name: "Researcher"},
		Memory:       researcherMem,
		SystemPrompt: "You are a researcher. Answer questions with facts.",
	})
	if err != nil {
		log.Fatalf("Failed to create researcher: %v", err)
	}

	// 2. Create the Manager Agent
	// The manager has the researcher as a tool.
	managerMem := memory.NewSessionMemory(memory.NewMemoryBankWithStore(memory.NewInMemoryStore()), 8)
	manager, err := agent.New(agent.Options{
		Model:        &DummyModel{Name: "Manager"},
		Memory:       managerMem,
		SystemPrompt: "You are a manager. Delegate research tasks.",
		Tools: []agent.Tool{
			researcher.AsTool("researcher", "Delegates research tasks to the researcher agent."),
		},
	})
	if err != nil {
		log.Fatalf("Failed to create manager: %v", err)
	}

	// 3. Simulate a run
	// In a real scenario with an LLM, the Manager would output a tool call JSON.
	// Since we are using a DummyModel, we can't easily simulate the full loop without
	// mocking the LLM's JSON output for tool calling.
	// However, we can demonstrate that the tool *exists* and is callable.

	fmt.Println("--- Agent Composability Example ---")
	fmt.Println("Manager Agent created with 'researcher' tool.")

	// Manually invoke the tool to demonstrate it works
	var tool agent.Tool
	for _, t := range manager.Tools() {
		if t.Spec().Name == "researcher" {
			tool = t
			break
		}
	}
	if tool == nil {
		log.Fatal("Researcher tool not found on manager!")
	}

	fmt.Println("Invoking 'researcher' tool directly...")
	resp, err := tool.Invoke(ctx, agent.ToolRequest{
		SessionID: "demo_session",
		Arguments: map[string]any{
			"instruction": "Why is the sky blue?",
		},
	})
	if err != nil {
		log.Fatalf("Tool invocation failed: %v", err)
	}

	fmt.Printf("Researcher replied: %s\n", resp.Content)
}
