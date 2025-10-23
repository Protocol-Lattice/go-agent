// main.go — guided swarm quickstart wiring ADK agents into the CLI.
//
// Examples:
//
//   export GOOGLE_API_KEY=...
//   go run ./cmd/quickstart \
//       --participants "researcher:cli:researcher,planner:cli:planner" \
//       --shared-spaces "team:demo" \
//       --qdrant-url http://localhost:6333 \
//       --qdrant-collection adk_memories
//
//   export QDRANT_API_KEY=...
//   go run ./cmd/quickstart \
//       --model gemini-2.5-flash \
//       --shared-spaces "product:beta" \
//       --qdrant-url https://YOUR-QDRANT:6333 \
//       --qdrant-collection swarm_memories
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

	"github.com/Raezil/go-agent-development-kit/pkg/swarm"
)

// ---- Adapter: concrete ADK *agent.Agent -> swarm.ConversationAgent ----

type coreAgent interface {
	EnsureSpaceGrants(sessionID string, spaces []string)
	SetSharedSpaces(shared *memory.SharedSession)
	Save(ctx context.Context, role, content string) // matches *agent.Agent
	Generate(ctx context.Context, sessionID, prompt string) (string, error)
}

type agentAdapter struct {
	inner coreAgent
}

func (a *agentAdapter) EnsureSpaceGrants(sessionID string, spaces []string) {
	a.inner.EnsureSpaceGrants(sessionID, spaces)
}

// Translate swarm.SharedSession -> *memory.SharedSession for the concrete agent API.
func (a *agentAdapter) SetSharedSpaces(shared swarm.SharedSession) {
	if ss, ok := shared.(*memory.SharedSession); ok {
		a.inner.SetSharedSpaces(ss)
		return
	}
	// Best effort: ignore if not the concrete type (keeps CLI resilient).
}

func (a *agentAdapter) Save(ctx context.Context, role, content string) {
	a.inner.Save(ctx, role, content)
}

func (a *agentAdapter) Generate(ctx context.Context, sessionID, prompt string) (string, error) {
	return a.inner.Generate(ctx, sessionID, prompt)
}

// ---- End adapter ----

/*
Flags:

--participants      Comma-separated "<alias>:<sessionID>" entries.

	Default: "researcher:cli:researcher,planner:cli:planner,summarizer:cli:summarizer"

--shared-spaces     Comma-separated shared spaces to auto-join on startup.
--model             LLM model id (default gemini-2.5-pro).
--qdrant-url        Qdrant base URL.
--qdrant-collection Qdrant collection name.
*/
func main() {
	qdrantURL := flag.String("qdrant-url", "http://localhost:6333", "Qdrant base URL")
	qdrantCollection := flag.String("qdrant-collection", "adk_memories", "Qdrant collection name")
	modelName := flag.String("model", "gemini-2.5-pro", "Gemini model ID")

	defaultParticipants := "researcher:cli:researcher,planner:cli:planner,summarizer:cli:summarizer"
	participantsFlag := flag.String("participants", defaultParticipants, "Comma-separated <alias>:<sessionID> entries")
	sharedSpacesFlag := flag.String("shared-spaces", "", "Comma-separated shared spaces to collaborate in")
	flag.Parse()

	ctx := context.Background()

	// --- Runtime (shared) ---
	researcherModel, err := models.NewGeminiLLM(ctx, *modelName, "Research summary:")
	if err != nil {
		log.Fatalf("failed to create researcher model: %v", err)
	}
	memOpts := engine.DefaultOptions()

	kit, err := adk.New(ctx,
		adk.WithDefaultSystemPrompt("You orchestrate a helpful assistant team."),
		adk.WithSubAgents(subagents.NewResearcher(researcherModel)),
		adk.WithModules(
			adkmodules.NewModelModule("gemini-model", func(_ context.Context) (models.Agent, error) {
				return models.NewGeminiLLM(ctx, *modelName, "Swarm orchestration:")
			}),
			adkmodules.InQdrantMemory(100000, *qdrantURL, *qdrantCollection, memory.AutoEmbedder(), &memOpts),
			adkmodules.NewToolModule("essentials", adkmodules.StaticToolProvider([]agent.Tool{&tools.EchoTool{}}, nil)),
		),
	)
	if err != nil {
		log.Fatalf("failed to initialise kit: %v", err)
	}

	sharedSpaces := helpers.ParseCSVList(*sharedSpacesFlag)
	if len(sharedSpaces) > 0 {
		fmt.Printf("Sharing memories in spaces: %s\n", strings.Join(sharedSpaces, ", "))
	} else {
		fmt.Println("No shared spaces configured. You can join at runtime via /swarm join <space>.")
	}

	// --- Build participants ---
	entries := parseParticipantsFlag(*participantsFlag)
	if len(entries) == 0 {
		log.Fatalf("no participants specified (flag --participants is empty)")
	}

	participants := swarm.Participants{}
	for _, e := range entries {
		agCore, err := kit.BuildAgent(ctx)
		if err != nil {
			log.Fatalf("failed to build agent for %s (%s): %v", e.Alias, e.SessionID, err)
		}
		shared, err := kit.NewSharedSession(ctx, e.SessionID, sharedSpaces...)
		if err != nil {
			log.Fatalf("failed to attach shared session for %s: %v", e.SessionID, err)
		}

		// Wrap concrete agent so it satisfies swarm.ConversationAgent.
		ag := &agentAdapter{inner: agCore}
		ag.SetSharedSpaces(shared)

		p := &swarm.Participant{
			Alias:     e.Alias,
			SessionID: e.SessionID,
			Agent:     ag,     // ConversationAgent
			Shared:    shared, // SharedSession
		}
		for _, s := range sharedSpaces {
			_ = p.Join(s) // best-effort join; Participant.Join ensures grants, prints on error
		}
		participants[e.SessionID] = p
	}

	cluster := swarm.NewSwarm(&participants)
	defer cluster.Save(ctx)

	// Active = first participant
	activeID := entries[0].SessionID
	active := cluster.GetParticipant(activeID)
	if active == nil {
		log.Fatalf("active participant %q not found", activeID)
	}

	fmt.Printf("Swarm ready with %d participants. Active: %s (%s)\n", len(participants), active.Alias, active.SessionID)
	fmt.Println("Type a message and press enter (empty line exits).")
	fmt.Println("Commands: /agents, /as <sessionID>, /swarm [peek|spaces|join <space>|leave <space>|flush|clear], /help")

	previewParticipantSwarm(ctx, active)

	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Printf("[%s] > ", active.Alias)
		line, err := reader.ReadString('\n')
		if err != nil {
			log.Fatalf("read input: %v", err)
		}
		line = strings.TrimSpace(line)
		if line == "" {
			fmt.Println("Goodbye!")
			return
		}

		// Commands
		if strings.HasPrefix(line, "/") {
			switch {
			case line == "/help":
				printHelp()
				continue

			case line == "/agents":
				listAgents(cluster, activeID)
				continue

			case strings.HasPrefix(line, "/as "):
				newID := strings.TrimSpace(strings.TrimPrefix(line, "/as "))
				if newID == "" {
					fmt.Println("Usage: /as <sessionID>")
					continue
				}
				if cluster.GetParticipant(newID) == nil {
					fmt.Printf("No participant with sessionID %q\n", newID)
					continue
				}
				activeID = newID
				active = cluster.GetParticipant(activeID)
				fmt.Printf("Switched active participant to %s (%s)\n", active.Alias, active.SessionID)
				previewParticipantSwarm(ctx, active)
				continue

			default:
				if handleSwarmCommand(ctx, kit, cluster, activeID, line) {
					continue
				}
				fmt.Printf("Unknown command: %s (try /help)\n", line)
				continue
			}
		}

		// Regular message → active participant
		active.Agent.Save(ctx, "user", line)
		resp, err := active.Agent.Generate(ctx, active.SessionID, line)
		if err != nil {
			fmt.Printf("error: %v\n", err)
			continue
		}
		fmt.Printf("%s\n", resp)
		active.Agent.Save(ctx, "agent", resp)
	}
}

func handleSwarmCommand(ctx context.Context, kit *adk.AgentDevelopmentKit, cluster *swarm.Swarm, participantID string, line string) bool {
	if !strings.HasPrefix(line, "/swarm") {
		return false
	}
	fields := strings.Fields(line)
	if len(fields) == 1 {
		fmt.Println("/swarm commands: peek, spaces, join <space>, leave <space>, flush, clear")
		previewParticipantSwarm(ctx, cluster.GetParticipant(participantID))
		return true
	}

	p := cluster.GetParticipant(participantID)
	if p == nil {
		fmt.Println("Swarm participant not found for this session.")
		return true
	}
	if p.Shared == nil {
		fmt.Println("Swarm intelligence is not configured for this session.")
		return true
	}

	switch fields[1] {
	case "peek":
		previewParticipantSwarm(ctx, p)

	case "spaces":
		spaces := p.Shared.Spaces()
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
		// Grants + join happen inside Participant.Join via interfaces.
		if failed := cluster.Join(participantID, space); failed {
			return true
		}
		fmt.Printf("Joined swarm space %s\n", space)

	case "leave":
		if len(fields) < 3 {
			fmt.Println("Usage: /swarm leave <space>")
			return true
		}
		space := fields[2]

		// 1) Leave the space.
		cluster.Leave(participantID, space)
		fmt.Printf("Left swarm space %s\n", space)

		// 2) Reset short-term memory by re-binding a fresh SharedSession with remaining spaces.
		remaining := p.Shared.Spaces()
		newShared, err := kit.NewSharedSession(ctx, p.SessionID, remaining...)
		if err != nil {
			fmt.Printf("Warning: left space, but failed to reset short-term memory: %v\n", err)
			return true
		}
		p.Shared = newShared
		p.Agent.SetSharedSpaces(newShared)
		fmt.Println("Short-term memory cleared for this participant after leaving the space.")

	case "flush":
		p.Save(ctx)
		fmt.Println("Swarm memories flushed to long-term storage.")

	case "clear":
		// Manual short-term wipe: rebuild with same spaces.
		current := p.Shared.Spaces()
		newShared, err := kit.NewSharedSession(ctx, p.SessionID, current...)
		if err != nil {
			fmt.Printf("Unable to clear short-term memory: %v\n", err)
			return true
		}
		p.Shared = newShared
		p.Agent.SetSharedSpaces(newShared)
		fmt.Println("Short-term memory cleared (local buffer reset).")

	default:
		fmt.Printf("Unknown /swarm command: %s\n", fields[1])
	}
	return true
}

func previewParticipantSwarm(ctx context.Context, p *swarm.Participant) {
	if p == nil || p.Shared == nil {
		return
	}
	records, err := p.Retrieve(ctx)
	if err != nil {
		fmt.Printf("Unable to read swarm intelligence: %v\n", err)
		return
	}
	if len(records) == 0 {
		fmt.Println("Swarm memory is empty. Start the conversation to seed collaborative knowledge.")
		return
	}
	fmt.Printf("Recent swarm intelligence for %s (%s):\n", p.Alias, p.SessionID)
	renderSwarmRecords(records)
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

type participantEntry struct {
	Alias     string
	SessionID string
}

func parseParticipantsFlag(raw string) []participantEntry {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil
	}
	parts := helpers.ParseCSVList(raw)
	out := make([]participantEntry, 0, len(parts))
	for _, item := range parts {
		chunks := strings.SplitN(strings.TrimSpace(item), ":", 2)
		if len(chunks) != 2 {
			fmt.Printf("Skipping participant %q (expected <alias>:<sessionID>)\n", item)
			continue
		}
		alias := strings.TrimSpace(chunks[0])
		sid := strings.TrimSpace(chunks[1])
		if alias == "" || sid == "" {
			fmt.Printf("Skipping participant %q (empty alias/sessionID)\n", item)
			continue
		}
		out = append(out, participantEntry{Alias: alias, SessionID: sid})
	}
	return out
}

func listAgents(cluster *swarm.Swarm, activeID string) {
	fmt.Println("Participants:")
	ids := make([]string, 0, len(*cluster.Participants))
	for id := range *cluster.Participants {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	for _, id := range ids {
		p := cluster.GetParticipant(id)
		marker := " "
		if id == activeID {
			marker = "*"
		}
		fmt.Printf(" %s %s (%s)\n", marker, p.Alias, p.SessionID)
	}
}

func printHelp() {
	fmt.Println("Commands:")
	fmt.Println("  /agents                    - list swarm participants")
	fmt.Println("  /as <sessionID>            - switch active participant")
	fmt.Println("  /swarm                     - show swarm commands + recent intelligence")
	fmt.Println("  /swarm peek                - show recent swarm intelligence for active participant")
	fmt.Println("  /swarm spaces              - list connected shared spaces for active participant")
	fmt.Println("  /swarm join <space>        - join a shared space for active participant")
	fmt.Println("  /swarm leave <space>       - leave a shared space and CLEAR short-term memory")
	fmt.Println("  /swarm flush               - flush active participant's memories to long-term store")
	fmt.Println("  /swarm clear               - clear short-term memory manually (without leaving)")
}
