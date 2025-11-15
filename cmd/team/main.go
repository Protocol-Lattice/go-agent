// main.go — multi-agent swarm orchestrator with shared memory spaces.
//
// Examples:
//
//	export GOOGLE_API_KEY=...
//	go run ./cmd/team \
//	    --agents "alice=agent:alice,bob=agent:bob" \
//	    --shared-spaces "team:design,team:research" \
//	    --qdrant-url http://localhost:6333 \
//	    --qdrant-collection adk_memories
//
//	export QDRANT_API_KEY=...
//	go run ./cmd/team \
//	    --model gemini-2.5-flash \
//	    --agents "pm=agent:pm,researcher=agent:res" \
//	    --shared-spaces "product:beta" \
//	    --qdrant-url https://YOUR-QDRANT:6333 \
//	    --qdrant-collection swarm_memories
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

	"github.com/Protocol-Lattice/go-agent/src/adk"
	adkmodules "github.com/Protocol-Lattice/go-agent/src/adk/modules"
	"github.com/Protocol-Lattice/go-agent/src/helpers"
	"github.com/Protocol-Lattice/go-agent/src/memory"
	"github.com/Protocol-Lattice/go-agent/src/memory/engine"
	"github.com/Protocol-Lattice/go-agent/src/models"
	"github.com/Protocol-Lattice/go-agent/src/subagents"

	// swarm API
	"github.com/Protocol-Lattice/go-agent/src/swarm"
)

//
// Adapter: *agent.Agent -> swarm.ConversationAgent
//

type coreAgent interface {
	EnsureSpaceGrants(sessionID string, spaces []string)
	SetSharedSpaces(shared *memory.SharedSession)
	Save(ctx context.Context, role, content string)
	Generate(ctx context.Context, sessionID, prompt string) (string, error)
}

type agentAdapter struct{ inner coreAgent }

func (a *agentAdapter) EnsureSpaceGrants(sessionID string, spaces []string) {
	a.inner.EnsureSpaceGrants(sessionID, spaces)
}

func (a *agentAdapter) SetSharedSpaces(shared swarm.SharedSession) {
	if ss, ok := shared.(*memory.SharedSession); ok {
		a.inner.SetSharedSpaces(ss)
	}
}

func (a *agentAdapter) Save(ctx context.Context, role, content string) {
	a.inner.Save(ctx, role, content)
}

func (a *agentAdapter) Generate(ctx context.Context, sessionID, prompt string) (string, error) {
	return a.inner.Generate(ctx, sessionID, prompt)
}

//
// Main
//

func main() {
	// --- Flags
	qdrantURL := flag.String("qdrant-url", "http://localhost:6333", "Qdrant base URL")
	qdrantCollection := flag.String("qdrant-collection", "adk_memories", "Qdrant collection name")
	modelName := flag.String("model", "gemini-2.5-pro", "Gemini model ID")

	// Examples:
	//   --agents="agent:alice,agent:bob"
	//   --agents="alice=agent:alice,bob=agent:bob"
	agentsFlag := flag.String("agents", "alice=agent:alice2,bob=agent:bob2", "Comma-separated agents (alias=sessionID or alias=sessionID)")
	sharedSpacesFlag := flag.String("shared-spaces", "team:kamil,team:lukasz", "Comma-separated shared memory spaces")
	flag.Parse()

	ctx := context.Background()

	// --- Shared runtime
	researcherModel, err := models.NewGeminiLLM(ctx, *modelName, "Research summary:")
	if err != nil {
		log.Fatalf("create researcher model: %v", err)
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
		),
	)
	if err != nil {
		log.Fatalf("init kit: %v", err)
	}

	// --- Parse agents & spaces
	sharedSpaces := helpers.ParseCSVList(*sharedSpacesFlag)
	if len(sharedSpaces) == 0 {
		fmt.Println("No shared spaces configured. Use --shared-spaces=team:core,team:shared to collaborate.")
	}
	aliasToID, order := parseAgents(*agentsFlag)
	if len(aliasToID) == 0 {
		log.Fatal("no agents parsed from --agents")
	}

	// --- Build swarm participants
	participants := swarm.Participants{}
	for _, alias := range order {
		sessionID := aliasToID[alias]

		agCore, err := kit.BuildAgent(ctx)
		if err != nil {
			log.Fatalf("build agent %q: %v", alias, err)
		}
		shared, err := kit.NewSharedSession(ctx, sessionID, sharedSpaces...)
		if err != nil {
			log.Fatalf("attach shared session (%s): %v", alias, err)
		}

		// Wrap concrete agent so it satisfies swarm.ConversationAgent.
		ag := &agentAdapter{inner: agCore}
		ag.SetSharedSpaces(shared)

		p := &swarm.Participant{
			Alias:     alias,
			SessionID: sessionID,
			Agent:     ag,     // ConversationAgent
			Shared:    shared, // SharedSession
		}
		participants[sessionID] = p
	}

	cluster := swarm.NewSwarm(&participants)
	defer cluster.Save(ctx)

	// Join spaces (best-effort) for each participant using swarm facade
	for _, alias := range order {
		id := aliasToID[alias]
		for _, s := range sharedSpaces {
			_ = cluster.Join(id, s) // prints user-friendly error internally on failure
		}
	}

	// --- UX banner
	fmt.Println("Agent Development Kit — multi-agent swarm. Empty line to exit.")
	fmt.Printf("Agents: %s\n", strings.Join(order, ", "))
	if len(sharedSpaces) > 0 {
		fmt.Printf("Shared spaces: %s\n", strings.Join(sharedSpaces, ", "))
	}

	// Default active = first
	activeAlias := order[0]
	activeID := aliasToID[activeAlias]
	showPrompt(activeAlias, activeID)

	// Initial peek
	previewSwarm(ctx, cluster, activeID)

	// --- Interactive loop
	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Print("> ")
		line, err := reader.ReadString('\n')
		if err != nil {
			log.Fatalf("read input: %v", err)
		}
		line = strings.TrimSpace(line)
		if line == "" {
			flushAll(ctx, cluster)
			fmt.Println("Goodbye!")
			return
		}

		// Commands
		if handled, newActive := handleGlobalCommand(line, order, aliasToID); handled {
			if newActive != "" {
				activeAlias = newActive
				activeID = aliasToID[activeAlias]
				fmt.Printf("Switched to [%s]\n", activeAlias)
				showPrompt(activeAlias, activeID)
			}
			continue
		}
		if handleSwarmCommand(ctx, kit, cluster, activeID, line) {
			continue
		}
		if handled := handleRoutedSay(ctx, cluster, aliasToID, line); handled {
			continue
		}
		if handled := handleBroadcast(ctx, cluster, participants, line); handled {
			continue
		}

		// Default: send to active participant
		p := cluster.GetParticipant(activeID)
		p.Agent.Save(ctx, "user", line)
		resp, err := p.Agent.Generate(ctx, p.SessionID, line)
		if err != nil {
			fmt.Printf("[%s] error: %v\n", p.Alias, err)
			continue
		}
		fmt.Printf("[%s] %s\n", p.Alias, resp)
		p.Agent.Save(ctx, "agent", resp)
	}
}

// --- Parsing helpers

func parseAgents(csv string) (map[string]string, []string) {
	items := helpers.ParseCSVList(csv)
	aliasToID := make(map[string]string, len(items))
	order := make([]string, 0, len(items))
	for _, it := range items {
		alias := ""
		session := it
		if strings.Contains(it, "=") {
			s := strings.SplitN(it, "=", 2)
			alias, session = strings.TrimSpace(s[0]), strings.TrimSpace(s[1])
		}
		if alias == "" {
			alias = inferAlias(session)
		}
		if alias == "" || session == "" {
			continue
		}
		aliasToID[alias] = session
		order = append(order, alias)
	}
	sort.Strings(order)
	return aliasToID, order
}

func inferAlias(session string) string {
	parts := strings.Split(session, ":")
	return strings.TrimSpace(parts[len(parts)-1])
}

// --- UX helpers

func showPrompt(activeAlias, activeID string) {
	fmt.Printf("Active: [%s] (session=%s). Use /who, /use <alias>, /say <alias> <msg>, /broadcast <msg>, /swarm ...\n",
		activeAlias, activeID)
}

// --- Command handlers

// handleGlobalCommand: /who, /use <alias>
func handleGlobalCommand(line string, order []string, aliasToID map[string]string) (handled bool, newActive string) {
	if !strings.HasPrefix(line, "/") {
		return false, ""
	}
	fields := strings.Fields(line)
	switch fields[0] {
	case "/who":
		for _, a := range order {
			fmt.Printf("- %s (session=%s)\n", a, aliasToID[a])
		}
		return true, ""
	case "/use":
		if len(fields) < 2 {
			fmt.Println("Usage: /use <alias>")
			return true, ""
		}
		target := strings.TrimSpace(fields[1])
		if _, ok := aliasToID[target]; !ok {
			fmt.Printf("No such agent alias: %s\n", target)
			return true, ""
		}
		return true, target
	default:
		return false, ""
	}
}

// handleRoutedSay: `/say <alias> <message...>`
func handleRoutedSay(ctx context.Context, cluster *swarm.Swarm, aliasToID map[string]string, line string) bool {
	if !strings.HasPrefix(line, "/say ") {
		return false
	}
	rest := strings.TrimSpace(strings.TrimPrefix(line, "/say"))
	if rest == "" {
		fmt.Println("Usage: /say <alias> <message>")
		return true
	}
	seg := strings.Fields(rest)
	if len(seg) < 2 {
		fmt.Println("Usage: /say <alias> <message>")
		return true
	}
	alias := seg[0]
	msg := strings.TrimSpace(strings.TrimPrefix(rest, alias))
	id, ok := aliasToID[alias]
	if !ok {
		fmt.Printf("No such agent alias: %s\n", alias)
		return true
	}

	p := cluster.GetParticipant(id)
	p.Agent.Save(ctx, "user", msg)
	resp, err := p.Agent.Generate(ctx, p.SessionID, msg)
	if err != nil {
		fmt.Printf("[%s] error: %v\n", p.Alias, err)
		return true
	}
	fmt.Printf("[%s] %s\n", p.Alias, resp)
	p.Agent.Save(ctx, "agent", resp)
	return true
}

// handleBroadcast: `/broadcast <message...>`
func handleBroadcast(ctx context.Context, cluster *swarm.Swarm, participants swarm.Participants, line string) bool {
	if !strings.HasPrefix(line, "/broadcast ") {
		return false
	}
	msg := strings.TrimSpace(strings.TrimPrefix(line, "/broadcast"))
	if msg == "" {
		fmt.Println("Usage: /broadcast <message>")
		return true
	}
	for _, p := range participants {
		p.Agent.Save(ctx, "user", msg)
		resp, err := p.Agent.Generate(ctx, p.SessionID, msg)
		if err != nil {
			fmt.Printf("[%s] error: %v\n", p.Alias, err)
			continue
		}
		fmt.Printf("[%s] %s\n", p.Alias, resp)
		p.Agent.Save(ctx, "agent", resp)
	}
	return true
}

// handleSwarmCommand operates on the active participant via GetParticipant(id)
// NOTE: we pass kit to allow short-term memory reset via re-binding SharedSession.
func handleSwarmCommand(ctx context.Context, kit *adk.AgentDevelopmentKit, cluster *swarm.Swarm, participantID string, line string) bool {
	if !strings.HasPrefix(line, "/swarm") {
		return false
	}
	fields := strings.Fields(line)
	p := cluster.GetParticipant(participantID)
	if p == nil {
		fmt.Println("Swarm participant not found for this session.")
		return true
	}
	if len(fields) == 1 {
		fmt.Println("/swarm commands: peek, spaces, join <space>, leave <space>, flush, clear")
		previewSwarm(ctx, cluster, participantID)
		return true
	}
	if p.Shared == nil {
		fmt.Println("Swarm not configured for this agent.")
		return true
	}

	switch fields[1] {
	case "peek":
		previewSwarm(ctx, cluster, participantID)

	case "spaces":
		spaces := p.Shared.Spaces()
		sort.Strings(spaces)
		if len(spaces) == 0 {
			fmt.Println("Not connected to any shared spaces.")
			return true
		}
		fmt.Println("Connected shared spaces:")
		for _, s := range spaces {
			fmt.Printf("- %s\n", s)
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

		// 1) Leave the space
		cluster.Leave(participantID, space)
		fmt.Printf("Left swarm space %s\n", space)

		// 2) Reset short-term memory by re-binding SharedSession with remaining spaces
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
		// Manual short-term wipe: rebuild with same spaces
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

// --- Swarm helpers

func previewSwarm(ctx context.Context, cluster *swarm.Swarm, participantID string) {
	p := cluster.GetParticipant(participantID)
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
	fmt.Println("Recent swarm intelligence:")
	renderSwarmRecords(records)
}

func renderSwarmRecords(records []memory.MemoryRecord) {
	for _, rec := range records {
		content := strings.TrimSpace(rec.Content)
		if content == "" {
			continue
		}
		fmt.Printf("- [%s] %s\n", rec.SessionID, content)
	}
}

func flushAll(ctx context.Context, cluster *swarm.Swarm) {
	cluster.Save(ctx)
}
