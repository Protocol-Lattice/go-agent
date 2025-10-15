package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/Raezil/go-agent-development-kit/pkg/agent"
	"github.com/Raezil/go-agent-development-kit/pkg/memory"
	"github.com/Raezil/go-agent-development-kit/pkg/models"
	"github.com/Raezil/go-agent-development-kit/pkg/runtime"
	"github.com/Raezil/go-agent-development-kit/pkg/subagents"
	"github.com/Raezil/go-agent-development-kit/pkg/tools"
	"github.com/universal-tool-calling-protocol/go-utcp"
)

// discovered flags whether we've serviced the UTCP discovery call yet
var discovered bool

func startServer(addr string) {
	http.HandleFunc("/tools", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		raw, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, fmt.Sprintf("failed to read body: %v", err), http.StatusBadRequest)
			return
		}
		defer r.Body.Close()

		log.Printf("Received raw request: %s", string(raw))

		// Discovery: first empty-body => discovery
		if len(raw) == 0 && !discovered {
			discovered = true
			// Read discovery response from tools.json
			data, err := os.ReadFile("tools.json")
			if err != nil {
				log.Printf("Failed to read tools.json: %v", err)
				return
			}
			var discoveryResponse map[string]interface{}
			if err := json.Unmarshal(data, &discoveryResponse); err != nil {
				log.Printf("Failed to unmarshal tools.json: %v", err)
				return
			}

			w.Header().Set("Content-Type", "application/json")
			if err := json.NewEncoder(w).Encode(discoveryResponse); err != nil {
				log.Printf("Failed to encode discovery response: %v", err)
			}
			return
		}

		// Empty-body after discovery => timestamp call
		if len(raw) == 0 {
			log.Printf("Empty body – timestamp call")
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]any{"result": time.Now().Format(time.RFC3339)})
			return
		}

		// Try to parse the JSON
		var probe map[string]interface{}
		if err := json.Unmarshal(raw, &probe); err != nil {
			http.Error(w, fmt.Sprintf("invalid JSON: %v", err), http.StatusBadRequest)
			return
		}

		// Standard tool call (has "tool" field)
		if toolName, hasToolField := probe["tool"].(string); hasToolField && toolName != "" {
			var req struct {
				Tool string                 `json:"tool"`
				Args map[string]interface{} `json:"args"`
			}
			if err := json.Unmarshal(raw, &req); err != nil {
				http.Error(w, fmt.Sprintf("invalid JSON for tool call: %v", err), http.StatusBadRequest)
				return
			}

			log.Printf("Standard tool call: %s with args: %v", req.Tool, req.Args)
			w.Header().Set("Content-Type", "application/json")

			switch req.Tool {
			case "echo":
				msg, _ := req.Args["message"].(string)
				json.NewEncoder(w).Encode(map[string]any{"result": msg})
			case "timestamp":
				json.NewEncoder(w).Encode(map[string]any{"result": time.Now().Format(time.RFC3339)})
			default:
				http.Error(w, "unknown tool", http.StatusNotFound)
			}
			return
		}

		// Direct echo call (has "message" field)
		if _, hasMessage := probe["message"]; hasMessage {
			log.Printf("Direct echo call with args: %v", probe)
			w.Header().Set("Content-Type", "application/json")
			msg, _ := probe["message"].(string)
			json.NewEncoder(w).Encode(map[string]any{"result": msg})
			return
		}

		// Unknown request format
		log.Printf("Unknown request format: %v", probe)
		http.Error(w, "unknown request format", http.StatusBadRequest)
	})

	log.Printf("HTTP mock server on %s", addr)
	log.Fatal(http.ListenAndServe(addr, nil))
}

func main() {
	var (
		dsn                    = flag.String("dsn", "postgres://admin:admin@localhost:5432/ragdb?sslmode=disable", "Postgres DSN")
		schemaPath             = flag.String("schema", "schema.sql", "Path to schema file")
		modelName              = flag.String("model", "gemini-2.5-pro", "Gemini model ID")
		sessionID              = flag.String("session", "", "Optional fixed session ID (reuse memory)")
		promptLimit            = flag.Int("context", 6, "Number of conversation turns to send to model")
		windowSize             = flag.Int("window", 8, "Short-term memory window size")
		memorySimWeight        = flag.Float64("memory-sim-weight", 0.55, "Similarity weight for memory retrieval scoring")
		memoryImportanceWeight = flag.Float64("memory-importance-weight", 0.25, "Importance weight for memory retrieval scoring")
		memoryRecencyWeight    = flag.Float64("memory-recency-weight", 0.15, "Recency weight for memory retrieval scoring")
		memorySourceWeight     = flag.Float64("memory-source-weight", 0.05, "Source weight for memory retrieval scoring")
		memoryLambda           = flag.Float64("memory-mmr-lambda", 0.7, "Lambda parameter for maximal marginal relevance")
		memoryHalfLife         = flag.Duration("memory-half-life", 72*time.Hour, "Half-life used for recency decay")
		memoryClusterSim       = flag.Float64("memory-cluster-sim", 0.83, "Similarity threshold for cluster summaries")
		memoryDrift            = flag.Float64("memory-drift-threshold", 0.90, "Cosine similarity threshold triggering re-embedding")
		memoryDuplicate        = flag.Float64("memory-duplicate-sim", 0.97, "Similarity threshold for deduplication")
		memoryTTL              = flag.Duration("memory-ttl", 720*time.Hour, "Time-to-live for stored memories")
		memoryMaxSize          = flag.Int("memory-max-size", 200000, "Maximum number of memories to retain before pruning")
		memorySourceBoost      = flag.String("memory-source-boost", "", "Comma separated source=weight overrides (e.g. pagerduty=1.0,slack=0.6)")
		memoryDisableSummaries = flag.Bool("memory-disable-summaries", false, "Disable cluster-based summarisation")
		sharedSpacesFlag       = flag.String("shared-spaces", "", "Comma-separated shared space session IDs (e.g. team:core,incident:2025-10-14)")
	)
	flag.Parse()
	go startServer(":8080")
	time.Sleep(200 * time.Millisecond)
	ctx := context.Background()

	var smRef *memory.SessionMemory

	utcpClient, err := utcp.NewUTCPClient(ctx, &utcp.UtcpClientConfig{
		ProvidersFilePath: "provider.json",
	}, nil, nil)
	if err != nil {
		panic(err)
	}

	// --- 🧠 1. Initialize Persistent MemoryBank ---
	bank, err := memory.NewMemoryBank(ctx, *dsn)
	if err != nil {
		log.Fatalf("❌ failed to connect to Postgres: %v", err)
	}
	defer bank.Close()

	if err := bank.CreateSchema(ctx, *schemaPath); err != nil {
		log.Fatalf("❌ failed to ensure schema: %v", err)
	}

	memoryOpts := memory.Options{
		Weights: memory.ScoreWeights{
			Similarity: *memorySimWeight,
			Importance: *memoryImportanceWeight,
			Recency:    *memoryRecencyWeight,
			Source:     *memorySourceWeight,
		},
		LambdaMMR:           *memoryLambda,
		HalfLife:            *memoryHalfLife,
		ClusterSimilarity:   *memoryClusterSim,
		DriftThreshold:      *memoryDrift,
		DuplicateSimilarity: *memoryDuplicate,
		TTL:                 *memoryTTL,
		MaxSize:             *memoryMaxSize,
		SourceBoost:         parseSourceBoostFlag(*memorySourceBoost),
		EnableSummaries:     !*memoryDisableSummaries,
	}

	engineLogger := log.New(os.Stderr, "memory-engine: ", log.LstdFlags)
	memoryEngine := memory.NewEngine(bank.Store, memoryOpts).WithLogger(engineLogger)

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
			smRef = memory.NewSessionMemory(bank, window).WithEngine(memoryEngine)
			return smRef
		},
		Tools: []agent.Tool{
			&tools.EchoTool{},
			&tools.CalculatorTool{},
			&tools.TimeTool{},
		},
		SubAgents: []agent.SubAgent{
			subagents.NewResearcher(researcherModel),
		},
		UTCPClient: utcpClient,
	}

	rt, err := runtime.New(ctx, cfg)
	if err != nil {
		log.Fatalf("failed to create runtime: %v", err)
	}
	defer rt.Close()

	// --- 4. UTCP Search and Call Tool ---
	tools, err := rt.Agent().UTCPClient.SearchTools("", 10)
	if err != nil {
		panic(err)
	}
	fmt.Println("Tools: ", len(tools))
	for _, tool := range tools {
		fmt.Println(" - ", tool.Name, ":", tool.Description)
	}
	resp, err := rt.Agent().UTCPClient.CallTool(ctx, tools[0].Name, map[string]any{"message": "Hello UTCP"})
	if err != nil {
		panic(err)
	}
	fmt.Println(resp)
	// --- 🧩 5. Reuse session if provided ---
	session := rt.NewSession(*sessionID)
	fmt.Printf("🧠 Using session: %s\n", session.ID())

	spaces := parseCSVList(*sharedSpacesFlag)
	shared := memory.NewSharedSession(smRef, session.ID(), spaces...)
	if len(spaces) > 0 {
		// Optional: seed a short-term note into the first shared space
		shared.AddShortTo(spaces[0], "🔧 Shared session started", map[string]string{"role": "system"})
	}
	defer session.CloseFlush(ctx, func(err error) {
		if err != nil {
			log.Printf("flush warning: %v", err)
		}
		// Also flush SharedSession
		if err := shared.FlushLocal(ctx); err != nil {
			log.Printf("flush local shared-session warning: %v", err)
		}
		for _, sp := range spaces {
			if err := shared.FlushSpace(ctx, sp); err != nil {
				log.Printf("flush shared space %q warning: %v", sp, err)
			}
		}
	})

	// --- 💬 5. Prompts ---
	prompts := flag.Args()
	if len(prompts) == 0 {
		prompts = []string{
			"Summarize what I asked in our previous session.",
			"I want to design an AI agent with memory. What’s the first step?",
			"subagent:researcher Briefly explain pgvector and its benefits for retrieval.",
		}
	}

	fmt.Println("--- Agent Development Kit Demo ---")
	fmt.Printf("Tools: %s\n", toolNames(rt.Tools()))
	fmt.Printf("Sub-agents: %s\n\n", subAgentNames(rt.SubAgents()))

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
			reply, err := rt.Generate(ctx, *sessionID, prompt)
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

func parseSourceBoostFlag(raw string) map[string]float64 {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil
	}
	boosts := make(map[string]float64)
	pairs := strings.Split(raw, ",")
	for _, pair := range pairs {
		parts := strings.SplitN(pair, "=", 2)
		if len(parts) != 2 {
			continue
		}
		key := strings.ToLower(strings.TrimSpace(parts[0]))
		value, err := strconv.ParseFloat(strings.TrimSpace(parts[1]), 64)
		if err != nil {
			continue
		}
		boosts[key] = value
	}
	if len(boosts) == 0 {
		return nil
	}
	return boosts
}

func toolNames(tools []agent.Tool) string {
	if len(tools) == 0 {
		return "<none>"
	}
	names := make([]string, len(tools))
	for i, tool := range tools {
		names[i] = tool.Spec().Name
	}
	return strings.Join(names, ", ")
}

func subAgentNames(subagents []agent.SubAgent) string {
	if len(subagents) == 0 {
		return "<none>"
	}
	names := make([]string, len(subagents))
	for i, sa := range subagents {
		names[i] = sa.Name()
	}
	return strings.Join(names, ", ")
}

func parseCSVList(raw string) []string {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil
	}
	parts := strings.Split(raw, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		s := strings.TrimSpace(p)
		if s != "" {
			out = append(out, s)
		}
	}
	return out
}
