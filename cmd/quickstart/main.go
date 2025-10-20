package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/Raezil/go-agent-development-kit/pkg/adk"
	adkmodules "github.com/Raezil/go-agent-development-kit/pkg/adk/modules"
	"github.com/Raezil/go-agent-development-kit/pkg/agent"
	"github.com/Raezil/go-agent-development-kit/pkg/helpers"
	"github.com/Raezil/go-agent-development-kit/pkg/memory"
	"github.com/Raezil/go-agent-development-kit/pkg/models"
	"github.com/Raezil/go-agent-development-kit/pkg/subagents"
	"github.com/Raezil/go-agent-development-kit/pkg/tools"
)

func main() {
	qdrantURL := flag.String("qdrant-url", "http://localhost:6333", "Qdrant base URL")
	qdrantCollection := flag.String("qdrant-collection", "adk_memories", "Qdrant collection name")
	// Advanced memory engine weights & knobs
	memorySimWeight := flag.Float64("memory-sim-weight", 0.55, "Similarity weight for memory retrieval scoring")
	memoryImportanceWeight := flag.Float64("memory-importance-weight", 0.25, "Importance weight for memory retrieval scoring")
	memoryRecencyWeight := flag.Float64("memory-recency-weight", 0.15, "Recency weight for memory retrieval scoring")
	memorySourceWeight := flag.Float64("memory-source-weight", 0.05, "Source weight for memory retrieval scoring")
	memoryLambda := flag.Float64("memory-mmr-lambda", 0.7, "Lambda parameter for maximal marginal relevance")
	memoryHalfLife := flag.Duration("memory-half-life", 72*time.Hour, "Half-life used for recency decay")
	memoryClusterSim := flag.Float64("memory-cluster-sim", 0.83, "Similarity threshold for cluster summaries")
	memoryDrift := flag.Float64("memory-drift-threshold", 0.90, "Cosine similarity threshold triggering re-embedding")
	memoryDuplicate := flag.Float64("memory-duplicate-sim", 0.97, "Similarity threshold for deduplication")
	memoryTTL := flag.Duration("memory-ttl", 720*time.Hour, "Time-to-live for stored memories")
	memoryMaxSize := flag.Int("memory-max-size", 200000, "Maximum number of memories to retain before pruning")
	memorySourceBoost := flag.String("memory-source-boost", "", "Comma separated source=weight overrides (e.g. pagerduty=1.0,slack=0.6)")
	memoryDisableSummaries := flag.Bool("memory-disable-summaries", false, "Disable cluster-based summarisation")
	modelName := flag.String("model", "gemini-2.5-pro", "Gemini model ID")
	sessionID := flag.String("session-id", "cli:quickstart", "Session identifier used to store memories for this CLI")
	sharedSpacesFlag := flag.String("shared-spaces", "", "Comma separated shared memory spaces to collaborate in")

	flag.Parse()
	ctx := context.Background()

	// --- 🧩 LLMs ---
	researcherModel, err := models.NewGeminiLLM(ctx, *modelName, "Research summary:")
	if err != nil {
		log.Fatalf("failed to create researcher model: %v", err)
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
		SourceBoost:         helpers.ParseSourceBoostFlag(*memorySourceBoost),
		EnableSummaries:     !*memoryDisableSummaries,
	}
	adk, err := adk.New(ctx,
		adk.WithDefaultSystemPrompt("You orchestrate a helpful assistant team."),
		adk.WithSubAgents(subagents.NewResearcher(researcherModel)),
		adk.WithModules(
			adkmodules.NewModelModule("gemini-model", func(_ context.Context) (models.Agent, error) {
				researcherModel, err := models.NewGeminiLLM(ctx, "gemini-2.5-pro", "Research summary:")
				if err != nil {
					return nil, err
				}
				return researcherModel, nil
			}),
			adkmodules.InQdrantMemory(100000, *qdrantURL, *qdrantCollection, memory.AutoEmbedder(), &memoryOpts),
			adkmodules.NewToolModule("essentials", adkmodules.StaticToolProvider([]agent.Tool{&tools.EchoTool{}}, nil)),
		),
	)
	if err != nil {
		log.Fatalf("failed to initialise kit: %v", err)
	}

	sharedSpaces := helpers.ParseCSVList(*sharedSpacesFlag)
	shared, err := adk.SharedSession(ctx, *sessionID, sharedSpaces...)
	if err != nil {
		log.Fatalf("failed to attach shared session: %v", err)
	}

	if len(sharedSpaces) > 0 {
		fmt.Printf("Sharing memories in spaces: %s\n", strings.Join(sharedSpaces, ", "))
	}

	agent, err := adk.BuildAgent(ctx)
	if err != nil {
		log.Fatalf("failed to build agent: %v", err)
	}

	fmt.Println("Agent Development Kit quickstart. Type a message and press enter (empty line exits).")
	reader := bufio.NewReader(os.Stdin)

	for {
		fmt.Print("> ")
		line, err := reader.ReadString('\n')
		if err != nil {
			log.Fatalf("read input: %v", err)
		}
		line = strings.TrimSpace(line)
		if line == "" {
			fmt.Println("Goodbye!")
			return
		}

		helpers.RecordPrompt(ctx, shared, sharedSpaces, "user", line)

		response, err := agent.Respond(ctx, *sessionID, line)
		if err != nil {
			fmt.Printf("error: %v\n", err)
			continue
		}
		fmt.Printf("%s\n", response)
		helpers.RecordPrompt(ctx, shared, sharedSpaces, "agent", response)
	}
}
