package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/Raezil/go-agent-development-kit/pkg/agent"
	"github.com/Raezil/go-agent-development-kit/pkg/memory"
	"github.com/Raezil/go-agent-development-kit/pkg/models"
	"github.com/Raezil/go-agent-development-kit/pkg/subagents"
	"github.com/Raezil/go-agent-development-kit/pkg/tools"
)

func main() {
	ctx := context.Background()

	connStr := "postgres://admin:admin@localhost:5432/ragdb?sslmode=disable"

	bank, err := memory.NewMemoryBank(ctx, connStr)
	if err != nil {
		log.Fatalf("failed to init memory bank: %v", err)
	}
	defer bank.DB.Close()

	if err := bank.CreateSchema(ctx, "schema.sql"); err != nil {
		log.Fatalf("failed to create schema: %v", err)
	}

	sessionMemory := memory.NewSessionMemory(bank, 8)
	sessionID := fmt.Sprintf("session-%d", time.Now().Unix())

	coordinatorModel := models.NewDummyLLM("Coordinator response:")
	researcherModel := models.NewDummyLLM("Research summary:")

	researcher := subagents.NewResearcher(researcherModel)

	toolset := []agent.Tool{&tools.EchoTool{}, &tools.CalculatorTool{}, &tools.TimeTool{}}
	subagentsList := []agent.SubAgent{researcher}

	primaryAgent, err := agent.New(agent.Options{
		Model:        coordinatorModel,
		Memory:       sessionMemory,
		SystemPrompt: "You orchestrate tooling and specialists to help the user build AI agents.",
		ContextLimit: 6,
		Tools:        toolset,
		SubAgents:    subagentsList,
	})
	if err != nil {
		log.Fatalf("failed to create agent: %v", err)
	}

	fmt.Println("--- Agent Development Kit Demo ---")

	userQuestions := []string{
		"I want to design an AI agent with both short term and long term memory. How should I start?",
		"tool:calculator 21 / 3",
		"subagent:researcher Provide a concise brief on pgvector usage for AI memory.",
		"How can I wire everything together after gathering research?",
	}

	for _, msg := range userQuestions {
		response, err := primaryAgent.Respond(ctx, sessionID, msg)
		if err != nil {
			fmt.Printf("Agent error: %v\n", err)
			continue
		}
		fmt.Printf("User: %s\nAgent: %s\n\n", msg, response)
	}

	if err := primaryAgent.Flush(ctx, sessionID); err != nil {
		log.Printf("flush warning: %v", err)
	} else {
		log.Printf("flushed short-term memory for %s to long-term storage", sessionID)
	}
}
