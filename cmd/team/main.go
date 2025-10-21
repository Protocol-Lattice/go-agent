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

// participant binds an Agent with its own session identity and shared-session handle.
type participant struct {
	Alias     string
	SessionID string
	Agent     *agent.Agent
	Shared    *memory.SharedSession
}

func main() {
	// --- Flags
	qdrantURL := flag.String("qdrant-url", "http://localhost:6333", "Qdrant base URL")
	qdrantCollection := flag.String("qdrant-collection", "adk_memories", "Qdrant collection name")
	modelName := flag.String("model", "gemini-2.5-pro", "Gemini model ID")

	// Comma-separated agents. Each item can be either:
	//   - "agent:alice"            (alias inferred from last path element => "alice")
	//   - "alice=agent:alice"      (explicit alias)
	agentsFlag := flag.String("agents", "alice=agent:alice,bob=agent:bob", "Comma-separated agents (alias=sessionID or sessionID)")

	// Comma-separated shared spaces to collaborate in, e.g. "team:core,team:shared"
	sharedSpacesFlag := flag.String("shared-spaces", "team:shared", "Comma-separated shared memory spaces")

	flag.Parse()
	ctx := context.Background()

	// --- Coordinator LLM & modules shared by all participants
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

	// --- Parse agents & spaces
	sharedSpaces := helpers.ParseCSVList(*sharedSpacesFlag)
	if len(sharedSpaces) == 0 {
		fmt.Println("No shared spaces configured. Use --shared-spaces=team:core,team:shared to collaborate.")
	}

	parts := parseAgents(*agentsFlag)
	if len(parts) == 0 {
		log.Fatal("no agents parsed from --agents")
	}

	// --- Build participants (each with its own Agent + SharedSession)
	for i := range parts {
		ag, err := kit.BuildAgent(ctx)
		if err != nil {
			log.Fatalf("build agent %q: %v", parts[i].Alias, err)
		}

		shared, err := kit.NewSharedSession(ctx, parts[i].SessionID, sharedSpaces...)
		if err != nil {
			log.Fatalf("attach shared session (%s): %v", parts[i].Alias, err)
		}

		// Wire shared spaces + grants for this principal
		ag.SetSharedSpaces(shared)
		ag.EnsureSpaceGrants(parts[i].SessionID, sharedSpaces)

		for _, s := range sharedSpaces {
			if err := shared.Join(s); err != nil {
				fmt.Printf("[%s] join %s: %v\n", parts[i].Alias, s, err)
			}
		}

		parts[i].Agent = ag
		parts[i].Shared = shared
	}

	// --- UX banner
	fmt.Println("Agent Development Kit — multi-agent swarm. Empty line to exit.")
	fmt.Printf("Agents: %s\n", strings.Join(listAliases(parts), ", "))
	if len(sharedSpaces) > 0 {
		fmt.Printf("Shared spaces: %s\n", strings.Join(sharedSpaces, ", "))
	}

	// Default active agent = first
	active := 0
	showPrompt(parts, active)

	// Initial swarm peek (from active)
	previewSwarm(ctx, parts[active].Shared)

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
			flushAll(ctx, parts)
			fmt.Println("Goodbye!")
			return
		}

		// Commands first
		if handleGlobalCommand(line, &active, parts) {
			showPrompt(parts, active)
			continue
		}
		if handleSwarmCommand(ctx, parts[active], line) {
			continue
		}
		if routed := handleRoutedSay(ctx, parts, line); routed {
			continue
		}
		if broadcast := handleBroadcast(ctx, parts, line); broadcast {
			continue
		}

		// Default: send to active agent
		p := parts[active]
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

// parseAgents supports "alias=sessionID" or plain "sessionID" (alias is last token after ':').
func parseAgents(csv string) []participant {
	items := helpers.ParseCSVList(csv)
	out := make([]participant, 0, len(items))
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
		if session == "" || alias == "" {
			continue
		}
		out = append(out, participant{Alias: alias, SessionID: session})
	}
	return out
}

func inferAlias(session string) string {
	parts := strings.Split(session, ":")
	return strings.TrimSpace(parts[len(parts)-1])
}

func listAliases(parts []participant) []string {
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		out = append(out, p.Alias)
	}
	sort.Strings(out)
	return out
}

func showPrompt(parts []participant, active int) {
	fmt.Printf("Active: [%s] (session=%s). Use /who, /use <alias>, /say <alias> <msg>, /broadcast <msg>, /swarm ...\n",
		parts[active].Alias, parts[active].SessionID)
}

// --- Command handlers

// handleGlobalCommand: /who, /use <alias>
func handleGlobalCommand(line string, active *int, parts []participant) bool {
	if !strings.HasPrefix(line, "/") {
		return false
	}
	fields := strings.Fields(line)
	switch fields[0] {
	case "/who":
		for _, p := range parts {
			fmt.Printf("- %s (session=%s)\n", p.Alias, p.SessionID)
		}
		return true
	case "/use":
		if len(fields) < 2 {
			fmt.Println("Usage: /use <alias>")
			return true
		}
		target := strings.TrimSpace(fields[1])
		for i, p := range parts {
			if p.Alias == target {
				*active = i
				fmt.Printf("Switched to [%s]\n", p.Alias)
				return true
			}
		}
		fmt.Printf("No such agent alias: %s\n", target)
		return true
	default:
		return false
	}
}

// handleRoutedSay: `/say <alias> <message...>`
func handleRoutedSay(ctx context.Context, parts []participant, line string) bool {
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
	for _, p := range parts {
		if p.Alias != alias {
			continue
		}
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
	fmt.Printf("No such agent alias: %s\n", alias)
	return true
}

// handleBroadcast: `/broadcast <message...>`
func handleBroadcast(ctx context.Context, parts []participant, line string) bool {
	if !strings.HasPrefix(line, "/broadcast ") {
		return false
	}
	msg := strings.TrimSpace(strings.TrimPrefix(line, "/broadcast"))
	if msg == "" {
		fmt.Println("Usage: /broadcast <message>")
		return true
	}
	for _, p := range parts {
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

// handleSwarmCommand operates on the selected participant's shared session.
func handleSwarmCommand(ctx context.Context, p participant, line string) bool {
	if !strings.HasPrefix(line, "/swarm") {
		return false
	}
	fields := strings.Fields(line)
	if len(fields) == 1 {
		fmt.Println("/swarm commands: peek, spaces, join <space>, leave <space>, flush")
		previewSwarm(ctx, p.Shared)
		return true
	}
	if p.Shared == nil {
		fmt.Println("Swarm not configured for this agent.")
		return true
	}
	switch fields[1] {
	case "peek":
		previewSwarm(ctx, p.Shared)
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
		// Ensure grants then join.
		p.Agent.EnsureSpaceGrants(p.SessionID, []string{space})
		if err := p.Shared.Join(space); err != nil {
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
		p.Shared.Leave(space)
		fmt.Printf("Left swarm space %s\n", space)
	case "flush":
		flushOne(ctx, p)
		fmt.Println("Swarm memories flushed to long-term storage.")
	default:
		fmt.Printf("Unknown /swarm command: %s\n", fields[1])
	}
	return true
}

// --- Swarm helpers

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

func renderSwarmRecords(records []memory.MemoryRecord) {
	for _, rec := range records {
		content := strings.TrimSpace(rec.Content)
		if content == "" {
			continue
		}
		fmt.Printf("- [%s] %s\n", rec.SessionID, content)
	}
}

func flushOne(ctx context.Context, p participant) {
	if p.Shared == nil {
		return
	}
	if err := p.Shared.FlushLocal(ctx); err != nil {
		fmt.Printf("[%s] flush local: %v\n", p.Alias, err)
	}
	for _, space := range p.Shared.Spaces() {
		if err := p.Shared.FlushSpace(ctx, space); err != nil {
			fmt.Printf("[%s] flush space %s: %v\n", p.Alias, space, err)
		}
	}
}

func flushAll(ctx context.Context, parts []participant) {
	for _, p := range parts {
		flushOne(ctx, p)
	}
}
