package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"sort"
	"strings"

	"github.com/Raezil/go-agent-development-kit/pkg/adk"
	adkmodules "github.com/Raezil/go-agent-development-kit/pkg/adk/modules"
	"github.com/Raezil/go-agent-development-kit/pkg/agent"
	"github.com/Raezil/go-agent-development-kit/pkg/helpers"
	"github.com/Raezil/go-agent-development-kit/pkg/memory"
	"github.com/Raezil/go-agent-development-kit/pkg/memory/engine"
	"github.com/Raezil/go-agent-development-kit/pkg/models"
	"github.com/Raezil/go-agent-development-kit/pkg/subagents"
	"github.com/Raezil/go-agent-development-kit/pkg/tools"
)

func main() {
	qdrantURL := flag.String("qdrant-url", "http://localhost:6333", "Qdrant base URL")
	qdrantCollection := flag.String("qdrant-collection", "adk_memories", "Qdrant collection name")
	modelName := flag.String("model", "gemini-2.5-pro", "Gemini model ID")
	sessionID := flag.String("session-id", "cli:goroutines", "Session identifier used to store memories for this CLI")
	sharedSpacesFlag := flag.String("shared-spaces", "", "Comma separated shared memory spaces to collaborate in")

	flag.Parse()
	ctx := context.Background()

	// --- 🧩 LLMs ---
	researcherModel, err := models.NewGeminiLLM(ctx, *modelName, "Research summary:")
	if err != nil {
		log.Fatalf("failed to create researcher model: %v", err)
	}

	memoryOpts := engine.DefaultOptions()
	kit, err := adk.New(ctx,
		adk.WithDefaultSystemPrompt("You orchestrate a helpful assistant team."),
		adk.WithSubAgents(subagents.NewResearcher(researcherModel)),
		adk.WithModules(
			adkmodules.NewModelModule("gemini-model", func(_ context.Context) (models.Agent, error) {
				coordinator, err := models.NewGeminiLLM(ctx, *modelName, "Swarm orchestration:")
				if err != nil {
					return nil, err
				}
				return coordinator, nil
			}),
			adkmodules.InQdrantMemory(100000, *qdrantURL, *qdrantCollection, memory.AutoEmbedder(), &memoryOpts),
			adkmodules.NewToolModule("essentials", adkmodules.StaticToolProvider([]agent.Tool{&tools.EchoTool{}}, nil)),
		),
	)
	if err != nil {
		log.Fatalf("failed to initialise kit: %v", err)
	}

	sharedSpaces := helpers.ParseCSVList(*sharedSpacesFlag)
	shared, err := kit.NewSharedSession(ctx, *sessionID, sharedSpaces...)
	if err != nil {
		log.Fatalf("failed to attach shared session: %v", err)
	}

	if len(sharedSpaces) > 0 {
		fmt.Printf("Sharing memories in spaces: %s\n", strings.Join(sharedSpaces, ", "))
	} else {
		fmt.Println("No shared spaces configured. Launch multiple instances with --shared-spaces to collaborate.")
	}

	ag, err := kit.BuildAgent(ctx)
	if err != nil {
		log.Fatalf("failed to build agent: %v", err)
	}

	// Hook shared session to agent and ensure grants so joins succeed.
	ag.SetSharedSpaces(shared)
	ag.EnsureSpaceGrants(*sessionID, sharedSpaces)
	for _, s := range sharedSpaces {
		if err := shared.Join(s); err != nil {
			fmt.Printf("join %s: %v\n", s, err)
		}
	}

	defer persistSwarm(ctx, shared)

	fmt.Println("Agent Development Kit quickstart. Type a message and press enter (empty line exits).")
	fmt.Println("Use /swarm to inspect or manage collaborative memory spaces.")

	previewSwarm(ctx, shared)

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

		if handleSwarmCommand(ctx, ag, *sessionID, shared, line) {
			continue
		}

		ag.Save(ctx, "user", line)

		response, err := ag.Generate(ctx, *sessionID, line)
		if err != nil {
			fmt.Printf("error: %v\n", err)
			continue
		}
		fmt.Printf("%s\n", response)
		ag.Save(ctx, "agent", response)
	}
}

func previewSwarm(ctx context.Context, shared *memory.SharedSession) {
	if shared == nil {
		return
	}
	records, err := shared.Retrieve(ctx, "recent swarm updates", 5)
	if err != nil {
		fmt.Printf("Unable to read swarm intelligence: %v\n", err)
		return
	}
	if len(records) == 0 {
		fmt.Println("Swarm memory is empty. Start the conversation to seed collaborative knowledge.")
		return
	}
	fmt.Println("Recent swarm intelligence:")
	renderSwarmRecords(records)
}

func handleSwarmCommand(ctx context.Context, ag *agent.Agent, sessionID string, shared *memory.SharedSession, line string) bool {
	if !strings.HasPrefix(line, "/swarm") {
		return false
	}
	fields := strings.Fields(line)
	if len(fields) == 1 {
		fmt.Println("/swarm commands: peek, spaces, join <space>, leave <space>, flush")
		previewSwarm(ctx, shared)
		return true
	}

	if shared == nil {
		fmt.Println("Swarm intelligence is not configured for this session.")
		return true
	}

	switch fields[1] {
	case "peek":
		previewSwarm(ctx, shared)
	case "spaces":
		spaces := shared.Spaces()
		sort.Strings(spaces)
		if len(spaces) == 0 {
			fmt.Println("Not connected to any shared spaces.")
			return true
		}
		fmt.Println("Connected shared spaces:")
		for _, space := range spaces {
			fmt.Printf("- %s\n", space)
		}
	case "join":
		if len(fields) < 3 {
			fmt.Println("Usage: /swarm join <space>")
			return true
		}
		space := fields[2]
		// Ensure grant for this space, then join.
		ag.EnsureSpaceGrants(sessionID, []string{space})
		if err := shared.Join(space); err != nil {
			fmt.Printf("Unable to join %s: %v\n", space, err)
			return true
		}
		fmt.Printf("Joined swarm space %s\n", space)
	case "leave":
		if len(fields) < 3 {
			fmt.Println("Usage: /swarm leave <space>")
			return true
		}
		space := fields[2]
		shared.Leave(space)
		fmt.Printf("Left swarm space %s\n", space)
	case "flush":
		persistSwarm(ctx, shared)
		fmt.Println("Swarm memories flushed to long-term storage.")
	default:
		fmt.Printf("Unknown /swarm command: %s\n", fields[1])
	}
	return true
}

func persistSwarm(ctx context.Context, shared *memory.SharedSession) {
	if shared == nil {
		return
	}
	if err := shared.FlushLocal(ctx); err != nil {
		fmt.Printf("Failed to flush local swarm memory: %v\n", err)
	}
	for _, space := range shared.Spaces() {
		if err := shared.FlushSpace(ctx, space); err != nil {
			fmt.Printf("Failed to flush swarm space %s: %v\n", space, err)
		}
	}
}

func renderSwarmRecords(records []memory.MemoryRecord) {
	if len(records) == 0 {
		fmt.Println("No swarm intelligence available yet.")
		return
	}
	for _, rec := range records {
		content := strings.TrimSpace(rec.Content)
		if content == "" {
			continue
		}
		fmt.Printf("- [%s] %s\n", rec.SessionID, content)
	}
}
