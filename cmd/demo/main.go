package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/Raezil/go-agent-development-kit/pkg/agent"
	"github.com/Raezil/go-agent-development-kit/pkg/models"
	"github.com/Raezil/go-agent-development-kit/pkg/runtime"
	"github.com/Raezil/go-agent-development-kit/pkg/subagents"
	"github.com/Raezil/go-agent-development-kit/pkg/tools"
)

func main() {
	var (
		dsn         = flag.String("dsn", "postgres://admin:admin@localhost:5432/ragdb?sslmode=disable", "Postgres connection string used for the memory bank")
		schemaPath  = flag.String("schema", "schema.sql", "Path to the SQL schema that bootstraps the memory store")
		modelName   = flag.String("model", "gemini-2.5-pro", "Model identifier for both coordinator and researcher agents")
		sessionID   = flag.String("session", "", "Optional fixed session identifier")
		promptLimit = flag.Int("context", 6, "Number of conversation turns to send to the model")
		windowSize  = flag.Int("window", 8, "Short-term memory window size per session")
	)
	flag.Parse()

	ctx := context.Background()

	researcherModel, err := models.NewGeminiLLM(ctx, *modelName, "Research summary:")
	if err != nil {
		log.Fatalf("failed to create researcher model: %v", err)
	}

	cfg := runtime.Config{
		DSN:           *dsn,
		SchemaPath:    *schemaPath,
		SessionWindow: *windowSize,
		ContextLimit:  *promptLimit,
		SystemPrompt:  "You orchestrate tooling and specialists to help the user build AI agents.",
		CoordinatorModel: func(ctx context.Context) (models.Agent, error) {
			return models.NewGeminiLLM(ctx, *modelName, "Coordinator response:")
		},
		Tools: []agent.Tool{
			&tools.EchoTool{},
			&tools.CalculatorTool{},
			&tools.TimeTool{},
		},
		SubAgents: []agent.SubAgent{
			subagents.NewResearcher(researcherModel),
		},
	}

	rt, err := runtime.New(ctx, cfg)
	if err != nil {
		log.Fatalf("failed to create runtime: %v", err)
	}
	defer func() {
		if err := rt.Close(); err != nil {
			log.Printf("runtime shutdown warning: %v", err)
		}
	}()

	session := rt.NewSession(*sessionID)
	defer session.CloseFlush(ctx, func(err error) {
		log.Printf("flush warning: %v", err)
	})

	prompts := flag.Args()
	if len(prompts) == 0 {
		prompts = []string{
			"I want to design an AI agent with both short term and long term memory. How should I start?",
			"tool:calculator 21 / 3",
			"subagent:researcher Provide a concise brief on pgvector usage for AI memory.",
			"How can I wire everything together after gathering research?",
		}
	}

	fmt.Println("--- Agent Development Kit Demo ---")
	fmt.Printf("Using session: %s\n", session.ID())
	fmt.Printf("Tools: %s\n", names(rt.Tools()))
	fmt.Printf("Sub-agents: %s\n\n", names(rt.SubAgents()))

	for _, prompt := range prompts {
		start := time.Now()
		reply, err := session.Ask(ctx, prompt)
		duration := time.Since(start)
		if err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			continue
		}
		fmt.Printf("User: %s\nAgent: %s\n(%.2fs)\n\n", prompt, reply, duration.Seconds())
	}
}

func names[T interface{ Name() string }](items []T) string {
	if len(items) == 0 {
		return "<none>"
	}
	values := make([]string, 0, len(items))
	for _, item := range items {
		values = append(values, item.Name())
	}
	return strings.Join(values, ", ")
}
