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
)

const defaultSystemPrompt = "You are the primary coordinator for an AI agent team. Provide concise, accurate answers and explain when you call tools or delegate work to specialist sub-agents."

// Agent orchestrates model calls, memory, tools, and sub-agents.
type Agent struct {
	model        models.Agent
	memory       *memory.SessionMemory
	systemPrompt string
	contextLimit int

	tools     map[string]Tool
	toolOrder []Tool

	subagents     map[string]SubAgent
	subagentOrder []SubAgent

	mu sync.Mutex
}

// Options configure a new Agent.
type Options struct {
	Model        models.Agent
	Memory       *memory.SessionMemory
	SystemPrompt string
	ContextLimit int
	Tools        []Tool
	SubAgents    []SubAgent
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

	a := &Agent{
		model:        opts.Model,
		memory:       opts.Memory,
		systemPrompt: systemPrompt,
		contextLimit: ctxLimit,
		tools:        make(map[string]Tool),
		subagents:    make(map[string]SubAgent),
	}

	for _, tool := range opts.Tools {
		if tool == nil {
			continue
		}
		key := strings.ToLower(tool.Name())
		if key == "" {
			continue
		}
		a.tools[key] = tool
		a.toolOrder = append(a.toolOrder, tool)
	}

	for _, sa := range opts.SubAgents {
		if sa == nil {
			continue
		}
		key := strings.ToLower(sa.Name())
		if key == "" {
			continue
		}
		a.subagents[key] = sa
		a.subagentOrder = append(a.subagentOrder, sa)
	}

	return a, nil
}

// Respond processes a user message, optionally invoking tools or sub-agents.
func (a *Agent) Respond(ctx context.Context, sessionID, userInput string) (string, error) {
	if strings.TrimSpace(userInput) == "" {
		return "", errors.New("user input is empty")
	}

	a.storeMemory(sessionID, "user", userInput, nil)

	if handled, output, err := a.handleCommand(ctx, sessionID, userInput); handled {
		if err != nil {
			a.storeMemory(sessionID, "assistant", fmt.Sprintf("tool error: %v", err), map[string]string{"source": "tool"})
			return "", err
		}
		a.storeMemory(sessionID, "assistant", output, map[string]string{"source": "tool"})
		return output, nil
	}

	prompt, err := a.buildPrompt(ctx, sessionID, userInput)
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

func (a *Agent) buildPrompt(ctx context.Context, sessionID, userInput string) (string, error) {
	records, err := a.memory.RetrieveContext(ctx, sessionID, userInput, a.contextLimit)
	if err != nil {
		return "", err
	}

	var sb strings.Builder
	sb.WriteString(a.systemPrompt)

	if len(a.toolOrder) > 0 {
		sb.WriteString("\n\nAvailable tools:\n")
		for _, tool := range a.toolOrder {
			sb.WriteString(fmt.Sprintf("- %s: %s\n", tool.Name(), tool.Description()))
		}
		sb.WriteString("Invoke a tool by replying with the exact format `tool:<name> <input>` if necessary.\n")
	}

	if len(a.subagentOrder) > 0 {
		sb.WriteString("\nSpecialist sub-agents:\n")
		for _, sa := range a.subagentOrder {
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
	sb.WriteString("\n\nCompose the best possible assistant reply.\n")
	return sb.String(), nil
}

func (a *Agent) handleCommand(ctx context.Context, sessionID, userInput string) (bool, string, error) {
	trimmed := strings.TrimSpace(userInput)
	lower := strings.ToLower(trimmed)

	switch {
	case strings.HasPrefix(lower, "tool:"):
		payload := strings.TrimSpace(trimmed[len("tool:"):])
		if payload == "" {
			return true, "", errors.New("tool name is missing")
		}
		name, args := splitCommand(payload)
		tool, ok := a.tools[strings.ToLower(name)]
		if !ok {
			return true, "", fmt.Errorf("unknown tool: %s", name)
		}
		result, err := tool.Run(ctx, args)
		if err != nil {
			return true, "", err
		}
		a.storeMemory(sessionID, "tool", fmt.Sprintf("%s => %s", tool.Name(), strings.TrimSpace(result)), map[string]string{"tool": tool.Name()})
		return true, result, nil
	case strings.HasPrefix(lower, "subagent:"):
		payload := strings.TrimSpace(trimmed[len("subagent:"):])
		if payload == "" {
			return true, "", errors.New("subagent name is missing")
		}
		name, args := splitCommand(payload)
		sa, ok := a.subagents[strings.ToLower(name)]
		if !ok {
			return true, "", fmt.Errorf("unknown subagent: %s", name)
		}
		result, err := sa.Run(ctx, args)
		if err != nil {
			return true, "", err
		}
		a.storeMemory(sessionID, "subagent", fmt.Sprintf("%s => %s", sa.Name(), strings.TrimSpace(result)), map[string]string{"subagent": sa.Name()})
		return true, result, nil
	default:
		return false, "", nil
	}
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
	embedding := memory.VertexAIEmbedding(content)

	a.mu.Lock()
	defer a.mu.Unlock()
	a.memory.AddShortTerm(sessionID, content, string(metaBytes), embedding)
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
