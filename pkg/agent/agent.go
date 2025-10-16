package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"

	"github.com/Raezil/go-agent-development-kit/pkg/memory"
	"github.com/Raezil/go-agent-development-kit/pkg/models"
	"github.com/universal-tool-calling-protocol/go-utcp"
)

const defaultSystemPrompt = "You are the primary coordinator for an AI agent team. Provide concise, accurate answers and explain when you call tools or delegate work to specialist sub-agents."

// Agent orchestrates model calls, memory, tools, and sub-agents.
type Agent struct {
	model        models.Agent
	memory       *memory.SessionMemory
	systemPrompt string
	contextLimit int

	toolCatalog       ToolCatalog
	subAgentDirectory SubAgentDirectory
	UTCPClient        utcp.UtcpClientInterface
	planner           Planner
	mu                sync.Mutex
}

// Options configure a new Agent.
type Options struct {
	Model             models.Agent
	Memory            *memory.SessionMemory
	SystemPrompt      string
	ContextLimit      int
	Tools             []Tool
	SubAgents         []SubAgent
	ToolCatalog       ToolCatalog
	SubAgentDirectory SubAgentDirectory
	UTCPClient        utcp.UtcpClientInterface
	Planner           Planner
}

// New creates an Agent with the provided options.
func New(opts Options) (*Agent, error) {
	if opts.Model == nil {
		return nil, errors.New("agent requires a language model")
	}
	if opts.Memory == nil {
		return nil, errors.New("agent requires session memory")
	}

	ctxLimit := opts.ContextLimit
	if ctxLimit <= 0 {
		ctxLimit = 8
	}

	systemPrompt := opts.SystemPrompt
	if strings.TrimSpace(systemPrompt) == "" {
		systemPrompt = defaultSystemPrompt
	}

	toolCatalog := opts.ToolCatalog
	tolerantTools := false
	if toolCatalog == nil {
		toolCatalog = NewStaticToolCatalog(nil)
		tolerantTools = true
	}
	for _, tool := range opts.Tools {
		if tool == nil {
			continue
		}
		if err := toolCatalog.Register(tool); err != nil {
			if tolerantTools {
				continue
			}
			return nil, err
		}
	}

	subAgentDirectory := opts.SubAgentDirectory
	tolerantSubAgents := false
	if subAgentDirectory == nil {
		subAgentDirectory = NewStaticSubAgentDirectory(nil)
		tolerantSubAgents = true
	}
	for _, sa := range opts.SubAgents {
		if sa == nil {
			continue
		}
		if err := subAgentDirectory.Register(sa); err != nil {
			if tolerantSubAgents {
				continue
			}
			return nil, err
		}
	}

	a := &Agent{
		model:             opts.Model,
		memory:            opts.Memory,
		systemPrompt:      systemPrompt,
		contextLimit:      ctxLimit,
		toolCatalog:       toolCatalog,
		subAgentDirectory: subAgentDirectory,
		UTCPClient:        opts.UTCPClient,
		planner:           opts.Planner,
	}

	return a, nil
}

// Respond processes a user message, optionally invoking tools or sub-agents.
func (a *Agent) Respond(ctx context.Context, sessionID, userInput string) (string, error) {
	if strings.TrimSpace(userInput) == "" {
		return "", errors.New("user input is empty")
	}

	a.storeMemory(sessionID, "user", userInput, nil) // 🧠 ← called here

	if handled, output, metadata, err := a.handleCommand(ctx, sessionID, userInput); handled {
		if err != nil {
			a.storeMemory(sessionID, "assistant", fmt.Sprintf("tool error: %v", err), map[string]string{"source": "tool"})
			return "", err
		}
		extra := map[string]string{"source": "tool"}
		for k, v := range metadata {
			if strings.TrimSpace(k) == "" || strings.TrimSpace(v) == "" {
				continue
			}
			extra[k] = v
		}
		a.storeMemory(sessionID, "assistant", output, extra)
		return output, nil
	}

	records, err := a.memory.RetrieveContext(ctx, sessionID, userInput, a.contextLimit)
	if err != nil {
		return "", err
	}

	plannerNotes, err := a.runPlanner(ctx, sessionID, userInput, records)
	if err != nil {
		return "", err
	}

	prompt, err := a.buildPrompt(userInput, records, plannerNotes)
	if err != nil {
		return "", err
	}

	completion, err := a.model.Generate(ctx, prompt)
	if err != nil {
		return "", err
	}

	response := fmt.Sprint(completion)
	a.storeMemory(sessionID, "assistant", response, nil)
	return response, nil
}

// Flush persists session memory into the long-term store.
func (a *Agent) Flush(ctx context.Context, sessionID string) error {
	return a.memory.FlushToLongTerm(ctx, sessionID)
}

func (a *Agent) buildPrompt(userInput string, records []memory.MemoryRecord, plannerNotes string) (string, error) {
	var sb strings.Builder
	sb.WriteString(a.systemPrompt)

	if specs := a.ToolSpecs(); len(specs) > 0 {
		sb.WriteString("\n\nAvailable tools:\n")
		for _, spec := range specs {
			sb.WriteString(fmt.Sprintf("- %s: %s\n", spec.Name, spec.Description))
			if len(spec.InputSchema) > 0 {
				schemaJSON, _ := json.MarshalIndent(spec.InputSchema, "  ", "  ")
				sb.WriteString("  Input schema: ")
				sb.Write(schemaJSON)
				sb.WriteString("\n")
			}
			if len(spec.Examples) > 0 {
				sb.WriteString("  Examples:\n")
				for _, ex := range spec.Examples {
					exJSON, _ := json.MarshalIndent(ex, "    ", "  ")
					sb.Write(exJSON)
					sb.WriteString("\n")
				}
			}
		}
		sb.WriteString("Invoke a tool by replying with the format `tool:<name> <json arguments>` when necessary.\n")
	}

	if subagents := a.SubAgents(); len(subagents) > 0 {
		sb.WriteString("\nSpecialist sub-agents:\n")
		for _, sa := range subagents {
			sb.WriteString(fmt.Sprintf("- %s: %s\n", sa.Name(), sa.Description()))
		}
		sb.WriteString("Delegate by replying with `subagent:<name> <task>` when it improves the answer.\n")
	}

	sb.WriteString("\nConversation memory:\n")
	if len(records) == 0 {
		sb.WriteString("(no stored memory)\n")
	} else {
		for i, rec := range records {
			role := metadataRole(rec.Metadata)
			content := strings.TrimSpace(rec.Content)
			if content == "" {
				continue
			}
			sb.WriteString(fmt.Sprintf("%d. [%s] %s\n", i+1, role, content))
		}
	}

	sb.WriteString("\nCurrent user message:\n")
	sb.WriteString(strings.TrimSpace(userInput))
	sb.WriteString("\n")

	if trimmed := strings.TrimSpace(plannerNotes); trimmed != "" {
		sb.WriteString("\nInternal planner notes (for model use only, do NOT expose to the user):\n")
		sb.WriteString(trimmed)
		sb.WriteString("\n")
	}

	sb.WriteString("\nCompose the best possible assistant reply.\n")
	return sb.String(), nil
}

func (a *Agent) runPlanner(ctx context.Context, sessionID, userInput string, records []memory.MemoryRecord) (string, error) {
	if a.planner == nil {
		return "", nil
	}

	output, err := a.planner.Plan(ctx, PlannerInput{
		SessionID: sessionID,
		UserInput: userInput,
		Context:   records,
	})
	if err != nil {
		return "", err
	}

	formatted := output.Format()
	if strings.TrimSpace(formatted) == "" {
		return "", nil
	}

	a.storeMemory(sessionID, "planner", formatted, map[string]string{"source": "planner"})
	return formatted, nil
}

func (a *Agent) handleCommand(ctx context.Context, sessionID, userInput string) (bool, string, map[string]string, error) {
	trimmed := strings.TrimSpace(userInput)
	lower := strings.ToLower(trimmed)

	switch {
	case strings.HasPrefix(lower, "tool:"):
		payload := strings.TrimSpace(trimmed[len("tool:"):])
		if payload == "" {
			return true, "", nil, errors.New("tool name is missing")
		}
		name, args := splitCommand(payload)
		tool, spec, ok := a.lookupTool(name)
		if !ok {
			return true, "", nil, fmt.Errorf("unknown tool: %s", name)
		}
		arguments := parseToolArguments(args)
		response, err := tool.Invoke(ctx, ToolRequest{SessionID: sessionID, Arguments: arguments})
		if err != nil {
			return true, "", nil, err
		}
		metadata := map[string]string{"tool": spec.Name}
		for k, v := range response.Metadata {
			if strings.TrimSpace(k) == "" || strings.TrimSpace(v) == "" {
				continue
			}
			metadata[k] = v
		}
		a.storeMemory(sessionID, "tool", fmt.Sprintf("%s => %s", spec.Name, strings.TrimSpace(response.Content)), metadata)
		return true, response.Content, metadata, nil
	case strings.HasPrefix(lower, "subagent:"):
		payload := strings.TrimSpace(trimmed[len("subagent:"):])
		if payload == "" {
			return true, "", nil, errors.New("subagent name is missing")
		}
		name, args := splitCommand(payload)
		sa, ok := a.lookupSubAgent(name)
		if !ok {
			return true, "", nil, fmt.Errorf("unknown subagent: %s", name)
		}
		result, err := sa.Run(ctx, args)
		if err != nil {
			return true, "", nil, err
		}
		meta := map[string]string{"subagent": sa.Name()}
		a.storeMemory(sessionID, "subagent", fmt.Sprintf("%s => %s", sa.Name(), strings.TrimSpace(result)), meta)
		return true, result, meta, nil
	default:
		return false, "", nil, nil
	}
}

func parseToolArguments(raw string) map[string]any {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return map[string]any{}
	}
	var payload map[string]any
	if strings.HasPrefix(raw, "{") {
		if err := json.Unmarshal([]byte(raw), &payload); err == nil {
			return payload
		}
	}
	if strings.HasPrefix(raw, "[") {
		var arr []any
		if err := json.Unmarshal([]byte(raw), &arr); err == nil {
			return map[string]any{"items": arr}
		}
	}
	return map[string]any{"input": raw}
}

func (a *Agent) storeMemory(sessionID, role, content string, extra map[string]string) {
	if strings.TrimSpace(content) == "" {
		return
	}

	meta := map[string]string{"role": role}
	for k, v := range extra {
		if strings.TrimSpace(k) == "" || strings.TrimSpace(v) == "" {
			continue
		}
		meta[k] = v
	}
	metaBytes, _ := json.Marshal(meta)
	embedding, err := a.memory.Embedder.Embed(context.Background(), content)
	if err != nil {
		return
	}

	a.mu.Lock()
	defer a.mu.Unlock()
	a.memory.AddShortTerm(sessionID, content, string(metaBytes), embedding)
}

func (a *Agent) lookupTool(name string) (Tool, ToolSpec, bool) {
	if a.toolCatalog == nil {
		return nil, ToolSpec{}, false
	}
	return a.toolCatalog.Lookup(name)
}

func (a *Agent) lookupSubAgent(name string) (SubAgent, bool) {
	if a.subAgentDirectory == nil {
		return nil, false
	}
	return a.subAgentDirectory.Lookup(name)
}

// ToolSpecs returns the registered tool specifications in deterministic order.
func (a *Agent) ToolSpecs() []ToolSpec {
	if a.toolCatalog == nil {
		return nil
	}
	return a.toolCatalog.Specs()
}

// Tools returns the registered tools in deterministic order.
func (a *Agent) Tools() []Tool {
	if a.toolCatalog == nil {
		return nil
	}
	return a.toolCatalog.Tools()
}

// SubAgents returns all registered sub-agents in deterministic order.
func (a *Agent) SubAgents() []SubAgent {
	if a.subAgentDirectory == nil {
		return nil
	}
	return a.subAgentDirectory.All()
}

func metadataRole(metadata string) string {
	if metadata == "" {
		return "unknown"
	}
	var payload map[string]any
	if err := json.Unmarshal([]byte(metadata), &payload); err != nil {
		return "unknown"
	}
	if role, ok := payload["role"].(string); ok && role != "" {
		return role
	}
	return "unknown"
}

func splitCommand(payload string) (name string, args string) {
	parts := strings.Fields(payload)
	if len(parts) == 0 {
		return "", ""
	}
	name = parts[0]
	if len(payload) > len(name) {
		args = strings.TrimSpace(payload[len(name):])
	}
	return name, args
}
