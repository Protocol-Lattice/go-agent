package main

import (
	"bufio"
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"

	agent "github.com/Protocol-Lattice/go-agent"
	"github.com/Protocol-Lattice/go-agent/src/adk"
	adkmodules "github.com/Protocol-Lattice/go-agent/src/adk/modules"
	"github.com/Protocol-Lattice/go-agent/src/memory"
	"github.com/Protocol-Lattice/go-agent/src/models"
	utcp "github.com/universal-tool-calling-protocol/go-utcp"
)

const (
	doneMarker     = "AUTONOMOUS_DONE"
	continueMarker = "AUTONOMOUS_CONTINUE"
)

type config struct {
	Provider      string
	Model         string
	SessionID     string
	ContextWindow int
	MaxSteps      int
	MaxRecoveries int
	ToolPrefix    string
}

type specialistSpec struct {
	ID           string
	ToolSuffix   string
	Description  string
	ModelPrefix  string
	SystemPrompt string
}

type runtime struct {
	cfg         config
	utcpClient  utcp.UtcpClientInterface
	coordinator *agent.Agent
	specialists map[string]*agent.Agent
	toolNames   []string
}

var specialistsCatalog = []specialistSpec{
	{
		ID:          "researcher",
		ToolSuffix:  "researcher",
		Description: "Research specialist for discovery, options, and assumptions.",
		ModelPrefix: "Research agent:",
		SystemPrompt: `You are a research specialist.
- Clarify unknowns and assumptions.
- Produce focused findings and concrete next steps.
- Keep output concise and actionable.`,
	},
	{
		ID:          "builder",
		ToolSuffix:  "builder",
		Description: "Implementation specialist for architecture and code-level plans.",
		ModelPrefix: "Builder agent:",
		SystemPrompt: `You are an implementation specialist.
- Turn requirements into concrete implementation steps.
- Prefer practical designs over abstract discussion.
- Highlight interfaces, data flow, and execution order.`,
	},
	{
		ID:          "reviewer",
		ToolSuffix:  "reviewer",
		Description: "Risk specialist for bugs, regressions, and test coverage gaps.",
		ModelPrefix: "Reviewer agent:",
		SystemPrompt: `You are a software reviewer.
- Prioritize real failure modes.
- Identify missing tests and rollout risks.
- Suggest minimal mitigations with high impact.`,
	},
}

const coordinatorSystemPrompt = `You are an autonomous coordinator.

Primary objective:
- Achieve the user goal in iterative steps.
- Use UTCP tools when useful.

Available UTCP tools are specialist agents for:
- research (facts, assumptions, alternatives)
- implementation (architecture and concrete plans)
- review (risks, regressions, and missing tests)

Rules:
- Prefer tool use over guessing when uncertainty exists.
- Keep each step goal-oriented and cumulative.
- Stop as soon as the goal is complete.
- End your final response with the marker AUTONOMOUS_DONE on its own line.
- If still in progress, include AUTONOMOUS_CONTINUE on its own line.
`

const loopPromptTemplate = `User goal:
%s

Iteration:
%d/%d

Current scratchpad:
%s

Instructions:
1) Decide what to do next.
2) Call UTCP specialist tools when needed.
3) If the scratchpad includes recovery context, change the approach that failed;
   do not repeat the failed action unchanged.
4) Return concise progress.
5) If complete, include AUTONOMOUS_DONE on a separate line.
6) Otherwise include AUTONOMOUS_CONTINUE on a separate line.
`

func main() {
	if err := run(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func run(args []string, out, errOut io.Writer) error {
	if len(args) == 0 {
		printRootHelp(out)
		return nil
	}

	switch args[0] {
	case "agent":
		return runAgentCommand(args[1:], out, errOut)
	case "loop":
		return runLoopCommand(args[1:], out, errOut)
	case "chat":
		return runChatCommand(args[1:], out, errOut)
	case "tools":
		return runToolsCommand(args[1:], out, errOut)
	case "doctor":
		return runDoctorCommand(args[1:], out, errOut)
	case "help", "-h", "--help":
		printRootHelp(out)
		return nil
	default:
		return fmt.Errorf("unknown command %q", args[0])
	}
}

func runAgentCommand(args []string, out, errOut io.Writer) error {
	cfg := defaultConfig()
	fs := flag.NewFlagSet("agent", flag.ContinueOnError)
	fs.SetOutput(errOut)
	bindCommonFlags(fs, &cfg)

	goalDefault := envOr("AGENT_GOAL", "")
	messageDefault := envOr("AGENT_MESSAGE", "")

	var message string
	var target string
	var goal string
	var thinking string
	var local bool
	var deliver bool

	fs.StringVar(&message, "message", messageDefault, "single-turn message for the selected agent")
	fs.StringVar(&target, "agent", "coordinator", "target agent (coordinator|researcher|builder|reviewer|<tool-name>)")
	fs.StringVar(&goal, "goal", goalDefault, "optional goal context injected into the message")
	fs.StringVar(&thinking, "thinking", "medium", "reasoning depth hint (compat flag)")
	fs.BoolVar(&local, "local", true, "run locally/embedded (compat flag)")
	fs.BoolVar(&deliver, "deliver", false, "delivery intent (compat flag)")

	if err := fs.Parse(args); err != nil {
		return err
	}

	if strings.TrimSpace(message) == "" && fs.NArg() > 0 {
		message = strings.Join(fs.Args(), " ")
	}
	if strings.TrimSpace(message) == "" {
		return errors.New("message is required; use --message or positional text")
	}

	_ = local
	_ = deliver

	ctx := context.Background()
	rt, err := bootstrapRuntime(ctx, cfg)
	if err != nil {
		return err
	}

	prompt := buildSingleTurnPrompt(goal, thinking, message)
	resp, resolved, err := rt.generate(ctx, target, cfg.SessionID, prompt)
	if err != nil {
		return err
	}

	fmt.Fprintf(out, "agent=%s session=%s\n", resolved, cfg.SessionID)
	fmt.Fprintln(out, resp)
	return nil
}

func runLoopCommand(args []string, out, errOut io.Writer) error {
	cfg := defaultConfig()
	fs := flag.NewFlagSet("loop", flag.ContinueOnError)
	fs.SetOutput(errOut)
	bindCommonFlags(fs, &cfg)

	goalDefault := envOr("AGENT_GOAL", "")
	var goal string

	fs.StringVar(&goal, "goal", goalDefault, "autonomous goal")
	fs.IntVar(&cfg.MaxSteps, "max-steps", cfg.MaxSteps, "maximum autonomous iterations")
	fs.IntVar(&cfg.MaxRecoveries, "max-recoveries", cfg.MaxRecoveries, "maximum self-healing recovery attempts across the loop")

	if err := fs.Parse(args); err != nil {
		return err
	}

	if strings.TrimSpace(goal) == "" && fs.NArg() > 0 {
		goal = strings.Join(fs.Args(), " ")
	}
	if strings.TrimSpace(goal) == "" {
		return errors.New("goal is required; use --goal or AGENT_GOAL")
	}
	if cfg.MaxSteps < 1 {
		cfg.MaxSteps = 1
	}
	if cfg.MaxRecoveries < 0 {
		cfg.MaxRecoveries = 0
	}

	ctx := context.Background()
	rt, err := bootstrapRuntime(ctx, cfg)
	if err != nil {
		return err
	}

	return runAutonomousLoop(ctx, rt, goal, out)
}

func runChatCommand(args []string, out, errOut io.Writer) error {
	cfg := defaultConfig()
	fs := flag.NewFlagSet("chat", flag.ContinueOnError)
	fs.SetOutput(errOut)
	bindCommonFlags(fs, &cfg)

	goalDefault := envOr("AGENT_GOAL", "")
	var goal string
	var activeAgent string
	var maxTurns int

	fs.StringVar(&goal, "goal", goalDefault, "persistent goal context for every message")
	fs.StringVar(&activeAgent, "agent", "coordinator", "default agent target")
	fs.IntVar(&maxTurns, "max-turns", 0, "stop after N turns (0 means unlimited)")

	if err := fs.Parse(args); err != nil {
		return err
	}

	ctx := context.Background()
	rt, err := bootstrapRuntime(ctx, cfg)
	if err != nil {
		return err
	}

	fmt.Fprintln(out, "OpenClaw-style chat started. Commands: /help, /tools, /agent <name>, /exit")
	fmt.Fprintf(out, "session=%s active-agent=%s\n", cfg.SessionID, activeAgent)

	scanner := bufio.NewScanner(os.Stdin)
	turns := 0
	for {
		fmt.Fprint(out, "\n> ")
		if !scanner.Scan() {
			return scanner.Err()
		}
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		switch {
		case line == "/exit" || line == "/quit":
			fmt.Fprintln(out, "bye")
			return nil
		case line == "/help":
			printChatHelp(out)
			continue
		case line == "/tools":
			names, err := rt.listLiveTools(ctx)
			if err != nil {
				return err
			}
			for _, name := range names {
				fmt.Fprintf(out, "- %s\n", name)
			}
			continue
		case strings.HasPrefix(line, "/agent "):
			activeAgent = strings.TrimSpace(strings.TrimPrefix(line, "/agent "))
			if activeAgent == "" {
				activeAgent = "coordinator"
			}
			fmt.Fprintf(out, "active-agent=%s\n", activeAgent)
			continue
		}

		prompt := buildSingleTurnPrompt(goal, "medium", line)
		resp, resolved, err := rt.generate(ctx, activeAgent, cfg.SessionID, prompt)
		if err != nil {
			return err
		}
		fmt.Fprintf(out, "[%s]\n%s\n", resolved, resp)

		turns++
		if maxTurns > 0 && turns >= maxTurns {
			fmt.Fprintln(out, "max turns reached")
			return nil
		}
	}
}

func runToolsCommand(args []string, out, errOut io.Writer) error {
	cfg := defaultConfig()
	fs := flag.NewFlagSet("tools", flag.ContinueOnError)
	fs.SetOutput(errOut)
	bindCommonFlags(fs, &cfg)

	var live bool
	fs.BoolVar(&live, "live", false, "query live registered tools from UTCP runtime")

	if err := fs.Parse(args); err != nil {
		return err
	}

	if !live {
		fmt.Fprintln(out, "Configured specialist tools:")
		for _, spec := range specialistsCatalog {
			fmt.Fprintf(out, "- %s: %s\n", specialistToolName(cfg.ToolPrefix, spec.ToolSuffix), spec.Description)
		}
		fmt.Fprintln(out, "\nUse --live to verify runtime registration.")
		return nil
	}

	ctx := context.Background()
	rt, err := bootstrapRuntime(ctx, cfg)
	if err != nil {
		return err
	}

	names, err := rt.listLiveTools(ctx)
	if err != nil {
		return err
	}

	fmt.Fprintln(out, "Live UTCP tools:")
	for _, name := range names {
		fmt.Fprintf(out, "- %s\n", name)
	}
	return nil
}

func runDoctorCommand(args []string, out, errOut io.Writer) error {
	cfg := defaultConfig()
	fs := flag.NewFlagSet("doctor", flag.ContinueOnError)
	fs.SetOutput(errOut)
	bindCommonFlags(fs, &cfg)

	if err := fs.Parse(args); err != nil {
		return err
	}

	fmt.Fprintln(out, "Doctor checks:")
	fmt.Fprintf(out, "- provider: %s\n", cfg.Provider)
	fmt.Fprintf(out, "- model: %s\n", cfg.Model)
	fmt.Fprintf(out, "- session-id: %s\n", cfg.SessionID)
	fmt.Fprintf(out, "- context-window: %d\n", cfg.ContextWindow)
	fmt.Fprintf(out, "- tool-prefix: %s\n", normalizeToolPrefix(cfg.ToolPrefix))

	keys := providerKeys(cfg.Provider)
	if len(keys) == 0 {
		fmt.Fprintln(out, "- api-keys: no provider-specific checks for this provider")
		return nil
	}

	missing := 0
	for _, key := range keys {
		if strings.TrimSpace(os.Getenv(key)) == "" {
			fmt.Fprintf(out, "- %s: missing\n", key)
			missing++
		} else {
			fmt.Fprintf(out, "- %s: set\n", key)
		}
	}
	if missing == len(keys) {
		return fmt.Errorf("no API key found for provider %s", cfg.Provider)
	}
	return nil
}

func bootstrapRuntime(ctx context.Context, cfg config) (*runtime, error) {
	cfg.ToolPrefix = normalizeToolPrefix(cfg.ToolPrefix)
	if err := ensureRuntimeEnv(cfg); err != nil {
		return nil, err
	}

	utcpClient, err := utcp.NewUTCPClient(ctx, nil, nil, nil)
	if err != nil {
		return nil, fmt.Errorf("create UTCP client: %w", err)
	}

	rt := &runtime{
		cfg:         cfg,
		utcpClient:  utcpClient,
		specialists: map[string]*agent.Agent{},
		toolNames:   make([]string, 0, len(specialistsCatalog)),
	}

	for _, spec := range specialistsCatalog {
		toolName := specialistToolName(cfg.ToolPrefix, spec.ToolSuffix)
		specialist, err := registerSpecialist(ctx, utcpClient, cfg, spec, toolName)
		if err != nil {
			return nil, fmt.Errorf("register specialist %s: %w", spec.ID, err)
		}

		rt.specialists[spec.ID] = specialist
		rt.specialists[toolName] = specialist
		// Backward-compatible alias for CLI targeting only (not registered in UTCP).
		rt.specialists[normalizeToolPrefix(cfg.ToolPrefix)+spec.ToolSuffix] = specialist
		rt.toolNames = append(rt.toolNames, toolName)
	}

	return rt, nil
}

func registerSpecialist(
	ctx context.Context,
	client utcp.UtcpClientInterface,
	cfg config,
	spec specialistSpec,
	toolName string,
) (*agent.Agent, error) {
	model, err := models.NewLLMProvider(ctx, cfg.Provider, cfg.Model, spec.ModelPrefix)
	if err != nil {
		return nil, fmt.Errorf("create model provider=%s model=%s: %w", cfg.Provider, cfg.Model, err)
	}

	sessionMemory := newSpecialistMemory(cfg.ContextWindow)

	a, err := agent.New(agent.Options{
		Model:        model,
		Memory:       sessionMemory,
		SystemPrompt: spec.SystemPrompt,
		ContextLimit: cfg.ContextWindow,
		UTCPClient:   client,
	})
	if err != nil {
		return nil, fmt.Errorf("create specialist agent tool=%s provider=%s model=%s: %w", toolName, cfg.Provider, cfg.Model, err)
	}

	if err := a.RegisterAsUTCPProvider(ctx, client, toolName, spec.Description); err != nil {
		return nil, fmt.Errorf("register as UTCP provider: %w", err)
	}

	return a, nil
}

func newSpecialistMemory(contextWindow int) *memory.SessionMemory {
	if contextWindow < 1 {
		contextWindow = 8
	}
	store := memory.NewInMemoryStore()
	bank := memory.NewMemoryBankWithStore(store)
	return memory.NewSessionMemory(bank, contextWindow).WithEmbedder(memory.AutoEmbedder())
}

func buildCoordinator(
	ctx context.Context,
	cfg config,
	client utcp.UtcpClientInterface,
) (*agent.Agent, error) {
	codeModeModel, err := models.NewLLMProvider(ctx, cfg.Provider, cfg.Model, "CodeMode executor:")
	if err != nil {
		return nil, fmt.Errorf("create codemode model: %w", err)
	}

	memOpts := memory.DefaultOptions()

	kit, err := adk.New(
		ctx,
		adk.WithDefaultSystemPrompt(coordinatorSystemPrompt),
		adk.WithUTCP(client),
		adk.WithCodeModeUtcp(client, codeModeModel),
		adk.WithModules(
			adkmodules.NewModelModule("coordinator-model", func(modelCtx context.Context) (models.Agent, error) {
				return models.NewLLMProvider(modelCtx, cfg.Provider, cfg.Model, "Coordinator:")
			}),
			adkmodules.InMemoryMemoryModule(cfg.ContextWindow, memory.AutoEmbedder(), &memOpts),
		),
	)
	if err != nil {
		return nil, fmt.Errorf("create adk kit: %w", err)
	}

	coordinator, err := kit.BuildAgent(ctx)
	if err != nil {
		return nil, fmt.Errorf("build coordinator agent: %w", err)
	}

	return coordinator, nil
}

func (r *runtime) generate(ctx context.Context, target, sessionID, prompt string) (string, string, error) {
	ag, resolved, err := r.resolveAgent(ctx, target)
	if err != nil {
		return "", "", err
	}

	rawResp, err := ag.Generate(ctx, sessionID, prompt)
	if err != nil {
		return "", "", err
	}

	resp := strings.TrimSpace(fmt.Sprint(rawResp))
	return resp, resolved, nil
}

func (r *runtime) resolveAgent(ctx context.Context, target string) (*agent.Agent, string, error) {
	normalized := strings.TrimSpace(strings.ToLower(target))
	if normalized == "" || normalized == "coordinator" || normalized == "coord" || normalized == "ops" {
		coordinator, err := r.ensureCoordinator(ctx)
		if err != nil {
			return nil, "", err
		}
		return coordinator, "coordinator", nil
	}

	if ag, ok := r.specialists[normalized]; ok {
		return ag, normalized, nil
	}

	prefixed := normalizeToolPrefix(r.cfg.ToolPrefix) + normalized
	if ag, ok := r.specialists[prefixed]; ok {
		return ag, prefixed, nil
	}

	return nil, "", fmt.Errorf("unknown agent %q", target)
}

func (r *runtime) ensureCoordinator(ctx context.Context) (*agent.Agent, error) {
	if r.coordinator != nil {
		return r.coordinator, nil
	}
	coordinator, err := buildCoordinator(ctx, r.cfg, r.utcpClient)
	if err != nil {
		return nil, fmt.Errorf("build coordinator: %w", err)
	}
	r.coordinator = coordinator
	return coordinator, nil
}

func (r *runtime) listLiveTools(ctx context.Context) ([]string, error) {
	tools, err := r.utcpClient.SearchTools("", 200)
	if err != nil {
		return nil, err
	}

	toolSet := make(map[string]struct{}, len(r.toolNames))
	for _, name := range r.toolNames {
		toolSet[name] = struct{}{}
	}

	names := make([]string, 0, len(r.toolNames))
	for _, t := range tools {
		if _, ok := toolSet[t.Name]; ok {
			names = append(names, t.Name)
		}
	}
	// Fallback so users still get expected names even if SearchTools is stale.
	if len(names) == 0 {
		names = append(names, r.toolNames...)
	}
	sort.Strings(names)
	return names, nil
}

type loopExecutor func(context.Context, string) (response, resolvedAgent string, err error)

func runAutonomousLoop(ctx context.Context, rt *runtime, goal string, out io.Writer) error {
	return runSelfHealingLoop(ctx, rt.cfg.MaxSteps, rt.cfg.MaxRecoveries, goal, out,
		func(ctx context.Context, prompt string) (string, string, error) {
			return rt.generate(ctx, "coordinator", rt.cfg.SessionID, prompt)
		},
	)
}

func runSelfHealingLoop(
	ctx context.Context,
	maxSteps, maxRecoveries int,
	goal string,
	out io.Writer,
	execute loopExecutor,
) error {
	if maxSteps < 1 {
		maxSteps = 1
	}
	if maxRecoveries < 0 {
		maxRecoveries = 0
	}

	scratchpad := "No prior progress yet."
	recoveries := 0

	for step := 1; step <= maxSteps; step++ {
		for {
			prompt := fmt.Sprintf(loopPromptTemplate, goal, step, maxSteps, scratchpad)
			resp, resolved, err := execute(ctx, prompt)
			if err == nil {
				fmt.Fprintf(out, "\n=== step %d/%d (%s) ===\n", step, maxSteps, resolved)
				fmt.Fprintln(out, resp)

				if strings.Contains(resp, doneMarker) {
					fmt.Fprintln(out, "\nGoal marked complete.")
					return nil
				}

				scratchpad = clip(resp, 8000)
				break
			}

			if ctx.Err() != nil || errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
				return fmt.Errorf("step %d (%s): %w", step, recoveryAgentName(resolved), err)
			}
			if recoveries >= maxRecoveries {
				return fmt.Errorf("step %d (%s) failed after %d recovery attempts: %w", step, recoveryAgentName(resolved), recoveries, err)
			}

			recoveries++
			failure := fmt.Sprintf("Execution failed at step %d for %s: %v", step, recoveryAgentName(resolved), err)
			scratchpad = recoveryScratchpad(scratchpad, failure)
			fmt.Fprintf(out, "\n=== recovery %d/%d after step %d (%s) ===\n", recoveries, maxRecoveries, step, recoveryAgentName(resolved))
			fmt.Fprintf(out, "execution error: %v\nretrying with recovery context.\n", err)
		}
	}

	return fmt.Errorf("max steps (%d) reached before %s", maxSteps, doneMarker)
}

func recoveryScratchpad(scratchpad, failure string) string {
	return clip(fmt.Sprintf("Previous progress:\n%s\n\nRecovery context:\n%s\n\nUse a different safe approach that addresses this failure.", scratchpad, failure), 8000)
}

func recoveryAgentName(resolved string) string {
	if strings.TrimSpace(resolved) == "" {
		return "coordinator"
	}
	return resolved
}

func buildSingleTurnPrompt(goal, thinking, message string) string {
	parts := []string{}
	if strings.TrimSpace(goal) != "" {
		parts = append(parts, "Goal:\n"+goal)
	}
	parts = append(parts, "Thinking hint:\n"+thinking)
	parts = append(parts, "User message:\n"+message)
	return strings.Join(parts, "\n\n")
}

func defaultConfig() config {
	return config{
		Provider:      envOr("LLM_PROVIDER", "gemini"),
		Model:         envOr("LLM_MODEL", "gemini-3-flash-preview"),
		SessionID:     envOr("AGENT_SESSION", "autonomous-session"),
		ContextWindow: 20,
		MaxSteps:      8,
		MaxRecoveries: 2,
		ToolPrefix:    envOr("UTCP_TOOL_PREFIX", "local."),
	}
}

func bindCommonFlags(fs *flag.FlagSet, cfg *config) {
	fs.StringVar(&cfg.Provider, "provider", cfg.Provider, "LLM provider (gemini|anthropic|openai|ollama)")
	fs.StringVar(&cfg.Model, "model", cfg.Model, "model name")
	fs.StringVar(&cfg.SessionID, "session-id", cfg.SessionID, "session id")
	fs.IntVar(&cfg.ContextWindow, "context-window", cfg.ContextWindow, "memory context window")
	fs.StringVar(&cfg.ToolPrefix, "tool-prefix", cfg.ToolPrefix, "UTCP tool prefix namespace")
}

func specialistToolName(prefix, suffix string) string {
	base := strings.TrimSuffix(normalizeToolPrefix(prefix), ".")
	if base == "" {
		base = "local"
	}
	provider := base + "_" + suffix
	return provider + ".run"
}

func providerKeys(provider string) []string {
	switch strings.ToLower(strings.TrimSpace(provider)) {
	case "gemini", "google":
		return []string{"GOOGLE_API_KEY", "GEMINI_API_KEY"}
	case "anthropic", "claude":
		return []string{"ANTHROPIC_API_KEY"}
	case "openai":
		return []string{"OPENAI_API_KEY"}
	case "groq":
		return []string{"GROQ_API_KEY"}
	case "deepseek":
		return []string{"DEEPSEEK_API_KEY"}
	case "ollama":
		return nil
	default:
		return nil
	}
}

func ensureRuntimeEnv(cfg config) error {
	if err := ensureProviderCredentials(cfg.Provider); err != nil {
		return err
	}
	if err := ensureEmbedProvider(cfg.Provider); err != nil {
		return err
	}
	return nil
}

func ensureProviderCredentials(provider string) error {
	keys := providerKeys(provider)
	if len(keys) == 0 {
		return nil
	}
	for _, key := range keys {
		if strings.TrimSpace(os.Getenv(key)) != "" {
			return nil
		}
	}
	return fmt.Errorf(
		"missing provider credentials for %q: set one of %s",
		provider,
		strings.Join(keys, ", "),
	)
}

func ensureEmbedProvider(provider string) error {
	if strings.TrimSpace(os.Getenv("ADK_EMBED_PROVIDER")) != "" {
		return nil
	}
	inferred := inferEmbedProvider(provider)
	if inferred == "" {
		return fmt.Errorf(
			"ADK_EMBED_PROVIDER is not set and could not be inferred for provider %q; set ADK_EMBED_PROVIDER explicitly",
			provider,
		)
	}
	if err := os.Setenv("ADK_EMBED_PROVIDER", inferred); err != nil {
		return fmt.Errorf("set ADK_EMBED_PROVIDER=%s: %w", inferred, err)
	}
	return nil
}

func inferEmbedProvider(provider string) string {
	switch strings.ToLower(strings.TrimSpace(provider)) {
	case "gemini", "google":
		return "gemini"
	case "openai":
		return "openai"
	case "ollama":
		return "ollama"
	default:
		return ""
	}
}

func printRootHelp(out io.Writer) {
	fmt.Fprintln(out, "autonomous-agent: OpenClaw-style UTCP CLI")
	fmt.Fprintln(out)
	fmt.Fprintln(out, "Commands:")
	fmt.Fprintln(out, "  agent   Single-turn run (like openclaw agent)")
	fmt.Fprintln(out, "  loop    Autonomous multi-step execution")
	fmt.Fprintln(out, "  chat    Interactive REPL with agent switching")
	fmt.Fprintln(out, "  tools   Show configured or live UTCP tools")
	fmt.Fprintln(out, "  doctor  Validate provider/model/env setup")
	fmt.Fprintln(out)
	fmt.Fprintln(out, "Examples:")
	fmt.Fprintln(out, "  autonomous-agent agent --message \"Summarize repo risks\"")
	fmt.Fprintln(out, "  autonomous-agent agent --agent reviewer --message \"Review deployment plan\"")
	fmt.Fprintln(out, "  autonomous-agent loop --goal \"Design test strategy for service X\" --max-steps 6")
	fmt.Fprintln(out, "  autonomous-agent chat --goal \"Ship release checklist\"")
	fmt.Fprintln(out, "  autonomous-agent tools --live")
}

func printChatHelp(out io.Writer) {
	fmt.Fprintln(out, "Chat commands:")
	fmt.Fprintln(out, "  /help            show this help")
	fmt.Fprintln(out, "  /tools           list live UTCP tools")
	fmt.Fprintln(out, "  /agent <name>    switch active agent")
	fmt.Fprintln(out, "  /exit            quit chat")
}

func normalizeToolPrefix(prefix string) string {
	p := strings.TrimSpace(prefix)
	if p == "" {
		return "local."
	}
	if strings.HasSuffix(p, ".") {
		return p
	}
	return p + "."
}

func envOr(key, fallback string) string {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		return v
	}
	return fallback
}

func clip(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[len(s)-max:]
}
