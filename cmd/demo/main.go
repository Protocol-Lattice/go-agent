package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/Raezil/go-agent-development-kit/pkg/agent"
	"github.com/Raezil/go-agent-development-kit/pkg/memory"
	"github.com/Raezil/go-agent-development-kit/pkg/models"
	"github.com/Raezil/go-agent-development-kit/pkg/runtime"
	"github.com/Raezil/go-agent-development-kit/pkg/subagents"
	"github.com/Raezil/go-agent-development-kit/pkg/tools"
)

func main() {
	var (
		dsn         = flag.String("dsn", "postgres://admin:admin@localhost:5432/ragdb?sslmode=disable", "Postgres DSN")
		schemaPath  = flag.String("schema", "schema.sql", "Path to schema file")
		modelName   = flag.String("model", "gemini-2.5-pro", "Gemini model ID")
		sessionID   = flag.String("session", "", "Optional fixed session ID (reuse memory)")
		promptLimit = flag.Int("context", 6, "Number of conversation turns to send to model")
		windowSize  = flag.Int("window", 8, "Short-term memory window size")
	)
	flag.Parse()

	ctx := context.Background()

	// --- 🧠 1. Initialize Persistent MemoryBank ---
	bank, err := memory.NewMemoryBank(ctx, *dsn)
	if err != nil {
		log.Fatalf("❌ failed to connect to Postgres: %v", err)
	}
	defer bank.DB.Close()

	if err := bank.CreateSchema(ctx, *schemaPath); err != nil {
		log.Fatalf("❌ failed to ensure schema: %v", err)
	}

	embedding := memory.VertexAIEmbedding("system check")
	fmt.Printf("✅ VertexAI embedding initialized: %d dims\n", len(embedding))

	// --- 🧩 2. Create LLMs ---
	researcherModel, err := models.NewGeminiLLM(ctx, *modelName, "Research summary:")
	if err != nil {
		log.Fatalf("failed to create researcher model: %v", err)
	}

	// --- ⚙️ 3. Configure Runtime with persistent memory ---
	cfg := runtime.Config{
		DSN:           *dsn,
		SchemaPath:    *schemaPath,
		SessionWindow: *windowSize,
		ContextLimit:  *promptLimit,
		SystemPrompt:  "You orchestrate tooling and specialists to help the user build AI agents.",
		CoordinatorModel: func(ctx context.Context) (models.Agent, error) {
			return models.NewGeminiLLM(ctx, *modelName, "Coordinator response:")
		},
		MemoryFactory: func(_ context.Context, _ string) (*memory.MemoryBank, error) {
			return bank, nil // reuse persistent connection
		},
		SessionMemoryBuilder: func(bank *memory.MemoryBank, window int) *memory.SessionMemory {
			return memory.NewSessionMemory(bank, window)
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
	defer rt.Close()

	// --- 🧩 4. Reuse session if provided ---
	session := rt.NewSession(*sessionID)
	fmt.Printf("🧠 Using session: %s\n", session.ID())

	defer session.CloseFlush(ctx, func(err error) {
		if err != nil {
			log.Printf("flush warning: %v", err)
		}
	})

	// --- 💬 5. Prompts ---
	prompts := flag.Args()
	if len(prompts) == 0 {
		prompts = []string{
			"Summarize what I asked in our previous session.",
			"I want to design an AI agent with memory. What’s the first step?",
			"tool:calculator 21 / 3",
			"subagent:researcher Briefly explain pgvector and its benefits for retrieval.",
		}
	}

	fmt.Println("--- Agent Development Kit Demo ---")
	fmt.Printf("Tools: %s\n", names(rt.Tools()))
	fmt.Printf("Sub-agents: %s\n\n", names(rt.SubAgents()))

	type result struct {
		idx      int
		prompt   string
		reply    string
		err      error
		duration time.Duration
	}

	results := make([]result, len(prompts))
	resultsCh := make(chan result, len(prompts))

	var wg sync.WaitGroup
	for i, prompt := range prompts {
		wg.Add(1)
		go func(i int, prompt string) {
			defer wg.Done()
			start := time.Now()
			reply, err := session.Ask(ctx, prompt)
			resultsCh <- result{i, prompt, reply, err, time.Since(start)}
		}(i, prompt)
	}

	go func() {
		wg.Wait()
		close(resultsCh)
	}()

	for res := range resultsCh {
		results[res.idx] = res
	}

	for _, res := range results {
		if res.err != nil {
			fmt.Fprintf(os.Stderr, "❌ %v\n", res.err)
			continue
		}
		fmt.Printf("User: %s\nAgent: %s\n(%.2fs)\n\n", res.prompt, res.reply, res.duration.Seconds())
	}

	fmt.Println("💾 All interactions flushed to long-term memory.")
}

func names[T interface{ Name() string }](items []T) string {
	if len(items) == 0 {
		return "<none>"
	}
	names := make([]string, len(items))
	for i, item := range items {
		names[i] = item.Name()
	}
	return strings.Join(names, ", ")
}
