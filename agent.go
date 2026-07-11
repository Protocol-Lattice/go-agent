package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/Protocol-Lattice/go-agent/src/memory"
	"github.com/Protocol-Lattice/go-agent/src/models"
	"github.com/universal-tool-calling-protocol/go-utcp"
	"github.com/universal-tool-calling-protocol/go-utcp/src/plugins/codemode"
	"github.com/universal-tool-calling-protocol/go-utcp/src/tools"
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

	mu sync.Mutex

	toolMu           sync.RWMutex
	toolSpecsCache   []tools.Tool
	toolSpecsExpiry  time.Time
	toolPromptCache  string
	toolPromptKey    string
	toolPromptExpiry time.Time

	Shared   *memory.SharedSession
	CodeMode *codemode.CodeModeUTCP

	AllowUnsafeTools bool
	Guardrails       *OutputGuardrails
	InputGuardrails  *InputGuardrails
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
	CodeMode          *codemode.CodeModeUTCP
	Shared            *memory.SharedSession
	AllowUnsafeTools  bool
	Guardrails        *OutputGuardrails
	InputGuardrails   *InputGuardrails
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
		Shared:            opts.Shared,
		CodeMode:          opts.CodeMode,
		AllowUnsafeTools:  opts.AllowUnsafeTools,
		Guardrails:        opts.Guardrails,
		InputGuardrails:   opts.InputGuardrails,
	}

	return a, nil
}

func (a *Agent) Generate(ctx context.Context, sessionID, userInput string) (any, error) {
	if a.InputGuardrails != nil {
		transformed, err := a.InputGuardrails.ValidateAndTransform(ctx, userInput)
		if err != nil {
			return "", err
		}
		userInput = transformed
	}

	trimmed := strings.TrimSpace(userInput)
	if trimmed == "" {
		return "", errors.New("user input is empty")
	}

	// -------------------------------------------------------------
	// PREFETCH: Start context retrieval and tool discovery in parallel
	// -------------------------------------------------------------
	var (
		prefetchWG sync.WaitGroup
		records    []memory.MemoryRecord
	)

	prefetchWG.Add(1)
	go func() {
		defer prefetchWG.Done()
		records, _ = a.retrieveContext(ctx, sessionID, userInput, a.contextLimit)
	}()

	// ToolSpecs discovery is internally cached and thread-safe.
	_ = a.ToolSpecs()

	// ---------------------------------------------
	// 0. DIRECT TOOL INVOCATION (bypass everything)
	// ---------------------------------------------
	if toolName, args, ok := a.detectDirectToolCall(trimmed); ok {
		// It's a direct tool call, execute it.
		result, err := a.executeTool(ctx, sessionID, toolName, args)
		if err != nil {
			return "", err
		}
		return fmt.Sprint(result), nil
	}

	// If the input is JSON but not a direct tool call, we should treat it as a normal prompt.
	// We can detect this by checking if it's a JSON object but `detectDirectToolCall` failed.
	var jsonData map[string]any
	if strings.HasPrefix(trimmed, "{") && json.Unmarshal([]byte(trimmed), &jsonData) == nil {
		// It's a JSON object but not a tool call, so we proceed to treat it as a regular prompt.
		// The logic below will handle storing it and sending it to the LLM.
	}

	// ---------------------------------------------
	// 1. SUBAGENT COMMANDS (subagent:researcher ...)
	// ---------------------------------------------
	if handled, out, meta, err := a.handleCommand(ctx, sessionID, userInput); handled {
		if err != nil {
			return "", err
		}
		a.storeMemory(sessionID, "subagent", out, meta)
		return out, nil
	}

	// ---------------------------------------------
	// 2. CODEMODE (Go-like DSL)
	// ---------------------------------------------
	if a.CodeMode != nil {
		if handled, output, err := a.CodeMode.CallTool(ctx, userInput); handled {
			if err != nil {
				return "", err
			}
			return output, nil
		}
	}

	// ---------------------------------------------
	// 3. TOOL ORCHESTRATOR (normal UTCP tools)
	// ---------------------------------------------
	prefetchWG.Wait() // Ensure memory is ready for orchestrator
	if handled, output, err := a.toolOrchestrator(ctx, sessionID, userInput, records); handled {
		if err != nil {
			return "", err
		}
		// Tool executed → do NOT store user memory
		return output, nil
	}

	// ---------------------------------------------
	// 5. STORE USER MEMORY (ONLY after toolOrchestrator failed)
	// ---------------------------------------------
	a.storeMemory(sessionID, "user", userInput, nil)

	// If the user input looks like a tool call, but wasn't handled above,
	// we can reasonably assume it was a malformed/unrecognized tool call.
	// We return an empty response rather than falling through to LLM completion.
	if a.userLooksLikeToolCall(trimmed) {
		return "", nil
	}

	// ---------------------------------------------
	// 6. LLM COMPLETION
	// ---------------------------------------------
	// Build LLM prompt without tools/subagents:
	var sb strings.Builder
	sb.Grow(4096)

	sb.WriteString(a.systemPrompt)
	sb.WriteString("\n\nConversation memory (TOON):\n")
	sb.WriteString(a.renderMemory(records))

	sb.WriteString("\n\nUser: ")
	sb.WriteString(sanitizeInput(userInput))
	sb.WriteString("\n\n")

	prompt := sb.String()

	files, _ := a.RetrieveAttachmentFiles(ctx, sessionID, a.contextLimit)

	var completion any
	var err error
	if len(files) > 0 {
		completion, err = a.model.GenerateWithFiles(ctx, prompt, files)
	} else {
		completion, err = a.model.Generate(ctx, prompt)
	}
	if err != nil {
		return "", err
	}

	finalText := fmt.Sprint(completion)
	if a.Guardrails != nil {
		validatedText, gErr := a.Guardrails.ValidateAndRepair(ctx, finalText)
		if gErr != nil {
			return "", gErr
		}
		finalText = validatedText
		completion = finalText
	}

	a.storeMemory(sessionID, "assistant", finalText, nil)
	return completion, nil
}

// GenerateWithFiles sends the user message plus in-memory files to the model
// without ingesting them into long-term memory. Use this when you already have
// file bytes (e.g., uploaded via API) and want the model to consider them
// ephemerally for this turn only.
// GenerateWithFiles runs the full orchestration pipeline (direct tool →
// subagent → CodeMode → UTCP tool loop) before falling back to a file-aware
// model call. Files are forwarded to the planner so tools can be selected with
// attachment context, but tool execution still uses the normal UTCP arguments.
func (a *Agent) GenerateWithFiles(
	ctx context.Context,
	sessionID string,
	userInput string,
	files []models.File,
) (string, error) {
	if a.InputGuardrails != nil {
		transformed, err := a.InputGuardrails.ValidateAndTransform(ctx, userInput)
		if err != nil {
			return "", err
		}
		userInput = transformed
	}

	trimmed := strings.TrimSpace(userInput)
	if trimmed == "" && len(files) == 0 {
		return "", errors.New("both user input and files are empty")
	}

	existingFiles, _ := a.RetrieveAttachmentFiles(ctx, sessionID, a.contextLimit)

	allFiles := make([]models.File, 0, len(existingFiles)+len(files))
	allFiles = append(allFiles, existingFiles...)
	allFiles = append(allFiles, files...)

	fileBacked := len(allFiles) > 0

	var (
		prefetchWG sync.WaitGroup
		records    []memory.MemoryRecord
	)

	prefetchWG.Add(1)
	go func() {
		defer prefetchWG.Done()
		records, _ = a.retrieveContext(ctx, sessionID, userInput, a.contextLimit)
	}()

	_ = a.ToolSpecs()

	// Direct tool calls are only safe for text-only requests.
	// File-backed requests must go through the file-aware orchestration path.
	if trimmed != "" && !fileBacked {
		if toolName, args, ok := a.detectDirectToolCall(trimmed); ok {
			result, err := a.executeTool(ctx, sessionID, toolName, args)
			if err != nil {
				return "", err
			}
			return fmt.Sprint(result), nil
		}
	}

	if handled, out, meta, err := a.handleCommand(ctx, sessionID, userInput); handled {
		if err != nil {
			return "", err
		}
		a.storeMemory(sessionID, "subagent", out, meta)
		return out, nil
	}

	// Direct CodeMode does not receive files.
	// If files are present, CodeMode must be disabled unless it receives full attachment context.
	if trimmed != "" && !fileBacked && a.CodeMode != nil {
		if handled, output, err := a.CodeMode.CallTool(ctx, userInput); handled {
			if err != nil {
				return "", err
			}
			return fmt.Sprint(output), nil
		}
	}

	prefetchWG.Wait()

	if len(files) > 0 {
		a.storeAttachmentMemories(sessionID, files)
	}

	orchestratorInput := userInput
	if fileBacked {
		var ob strings.Builder
		ob.Grow(len(userInput) + 4096)

		ob.WriteString("FILE-BACKED REQUEST\n")
		ob.WriteString("Use the attached workspace files as the source of truth.\n")
		ob.WriteString("Do not use CodeMode unless the full attachment context is included.\n")
		ob.WriteString("Do not invent files, packages, APIs, commands, or project structure.\n\n")
		ob.WriteString(fileBackedWorkspaceRules(allFiles))
		ob.WriteString("\n")

		if len(existingFiles) > 0 {
			ob.WriteString(a.buildAttachmentPrompt("Session attachments rehydrated", existingFiles))
			ob.WriteString("\n")
		}

		if len(files) > 0 {
			ob.WriteString(a.buildAttachmentPrompt("Files provided for this turn", files))
			ob.WriteString("\n")
		}

		if trimmed != "" {
			ob.WriteString("User instruction:\n")
			ob.WriteString(sanitizeInput(userInput))
			ob.WriteString("\n")
		} else {
			ob.WriteString("User instruction:\nAnalyze the provided files.\n")
		}

		orchestratorInput = ob.String()
	}

	if handled, output, err := a.toolOrchestrator(ctx, sessionID, orchestratorInput, records, allFiles...); handled {
		if err != nil {
			return "", err
		}
		return output, nil
	}

	if trimmed != "" {
		a.storeMemory(sessionID, "user", userInput, nil)
	}

	if trimmed != "" && !fileBacked && a.userLooksLikeToolCall(trimmed) {
		return "", nil
	}

	var sb strings.Builder
	sb.Grow(4096)

	if strings.TrimSpace(a.systemPrompt) != "" {
		sb.WriteString(strings.TrimSpace(a.systemPrompt))
		sb.WriteString("\n\n")
	}

	sb.WriteString("Conversation memory (TOON):\n")
	sb.WriteString(a.renderMemory(records))
	sb.WriteString("\n\n")

	if fileBacked {
		sb.WriteString(fileBackedWorkspaceRules(allFiles))
		sb.WriteString("\n")
	}

	if len(existingFiles) > 0 {
		sb.WriteString(a.buildAttachmentPrompt("Session attachments rehydrated", existingFiles))
		sb.WriteString("\n")
	}

	if len(files) > 0 {
		sb.WriteString(a.buildAttachmentPrompt("Files provided for this turn", files))
		sb.WriteString("\n")
	}

	if trimmed != "" {
		sb.WriteString("User: ")
		sb.WriteString(sanitizeInput(userInput))
		sb.WriteString("\n")
	} else {
		sb.WriteString("User: Analyze the provided files.\n")
	}

	prompt := sb.String()

	var (
		completion any
		err        error
	)

	if fileBacked {
		completion, err = a.model.GenerateWithFiles(ctx, prompt, allFiles)
	} else {
		completion, err = a.model.Generate(ctx, prompt)
	}
	if err != nil {
		return "", err
	}

	response := fmt.Sprint(completion)
	if a.Guardrails != nil {
		validated, gErr := a.Guardrails.ValidateAndRepair(ctx, response)
		if gErr != nil {
			return "", gErr
		}
		response = validated
	}

	a.storeMemory(sessionID, "assistant", response, nil)
	return response, nil
}
