package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/Raezil/go-agent-development-kit/pkg/memory"
	"github.com/Raezil/go-agent-development-kit/pkg/models"
	"github.com/Raezil/go-agent-development-kit/pkg/upload"
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
	mu                sync.Mutex
	Shared            *memory.SharedSession
	Ingest            *upload.Ingestor
}

// Attachment describes a file to ingest alongside a user message.
// Prefer Path if you already have a file on disk; otherwise use Name+Reader.
type Attachment struct {
	// One of Path OR (Name+Reader) must be set.
	Path   string
	Name   string
	MIME   string            // optional, let Ingestor sniff if empty
	Reader io.Reader         // optional, used when Path == ""
	Tags   []string          // optional
	TTL    *time.Duration    // optional override
	Meta   map[string]string // optional
	Scope  upload.Scope      // default: upload.ScopeSpace (shared space / swarm)
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
	Ingest            *upload.Ingestor

	Shared *memory.SharedSession
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
		Ingest:            opts.Ingest,
	}

	return a, nil
}

// Generate processes a user message, optionally invoking tools or sub-agents.
func (a *Agent) Generate(ctx context.Context, sessionID, userInput string) (string, error) {
	if strings.TrimSpace(userInput) == "" {
		return "", errors.New("user input is empty")
	}

	a.storeMemory(sessionID, "user", userInput, nil)

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
	queryType := classifyQuery(userInput)

	switch queryType {
	case QueryMath:
		return fmt.Sprintf("%s\n\nCurrent user message:\n%s\n\nCompose the best possible assistant reply.\n",
			a.systemPrompt, strings.TrimSpace(userInput)), nil

	case QueryShortFactoid:
		records, err := a.retrieveContext(ctx, sessionID, userInput, min(a.contextLimit/2, 3))
		if err != nil {
			return "", fmt.Errorf("retrieve context: %w", err)
		}
		return a.buildFullPrompt(userInput, records), nil

	case QueryComplex:
		records, err := a.retrieveContext(ctx, sessionID, userInput, a.contextLimit)
		if err != nil {
			return "", fmt.Errorf("retrieve context: %w", err)
		}
		return a.buildFullPrompt(userInput, records), nil
	}

	return "", nil
}

func (a *Agent) buildFullPrompt(userInput string, records []memory.MemoryRecord) string {
	var sb strings.Builder
	sb.Grow(4096)

	sb.WriteString(a.systemPrompt)

	if tools := a.renderTools(); tools != "" {
		sb.WriteString("\n\n")
		sb.WriteString(tools)
	}
	if sub := a.renderSubAgents(); sub != "" {
		sb.WriteString("\n\n")
		sb.WriteString(sub)
	}

	sb.WriteString("\n\nConversation memory:\n")
	sb.WriteString(a.renderMemory(records))

	sb.WriteString("\n\nCurrent user message:\n")
	sb.WriteString(strings.TrimSpace(userInput))
	sb.WriteString("\n\nCompose the best possible assistant reply.\n")

	return sb.String()
}

// renderTools formats the available tool specs into a prompt-friendly block.
func (a *Agent) renderTools() string {
	specs := a.ToolSpecs()
	if len(specs) == 0 {
		return ""
	}

	var sb strings.Builder
	sb.WriteString("Available tools:\n")
	for _, spec := range specs {
		sb.WriteString(fmt.Sprintf("- %s: %s\n", spec.Name, spec.Description))
		if len(spec.InputSchema) > 0 {
			if schemaJSON, err := json.MarshalIndent(spec.InputSchema, "  ", "  "); err == nil {
				sb.WriteString("  Input schema: ")
				sb.Write(schemaJSON)
				sb.WriteString("\n")
			}
		}
		if len(spec.Examples) > 0 {
			sb.WriteString("  Examples:\n")
			for _, ex := range spec.Examples {
				if exJSON, err := json.MarshalIndent(ex, "    ", "  "); err == nil {
					sb.Write(exJSON)
					sb.WriteString("\n")
				}
			}
		}
	}
	sb.WriteString("Invoke a tool with: `tool:<name> <json arguments>`\n")
	return sb.String()
}

// renderSubAgents formats specialist sub-agents into a prompt-friendly block.
func (a *Agent) renderSubAgents() string {
	subagents := a.SubAgents()
	if len(subagents) == 0 {
		return ""
	}

	var sb strings.Builder
	sb.WriteString("Specialist sub-agents:\n")
	for _, sa := range subagents {
		sb.WriteString(fmt.Sprintf("- %s: %s\n", sa.Name(), sa.Description()))
	}
	sb.WriteString("Delegate with: `subagent:<name> <task>`\n")
	return sb.String()
}

// renderMemory formats retrieved memory records into a clean, token-efficient list.
func (a *Agent) renderMemory(records []memory.MemoryRecord) string {
	if len(records) == 0 {
		return "(no stored memory)\n"
	}

	var sb strings.Builder
	for i, rec := range records {
		content := strings.TrimSpace(rec.Content)
		if content == "" {
			continue
		}
		role := metadataRole(rec.Metadata)
		sb.WriteString(fmt.Sprintf("%d. [%s] %s\n", i+1, role, escapePromptContent(content)))
	}
	return sb.String()
}

// escapePromptContent safely escapes content that might break formatting.
func escapePromptContent(s string) string {
	s = strings.ReplaceAll(s, "`", "'")
	return s
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
	a.mu.Lock()
	if a.Shared != nil {
		a.Shared.AddShortLocal(content, meta)
		for _, space := range a.Shared.Spaces() {
			if err := a.Shared.AddShortTo(space, content, meta); err != nil {
				continue
			}
		}
		a.mu.Unlock()
		return
	}
	a.mu.Unlock()

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

func (a *Agent) retrieveContext(ctx context.Context, sessionID, query string, limit int) ([]memory.MemoryRecord, error) {
	if a.Shared != nil {
		return a.Shared.Retrieve(ctx, query, limit)
	}
	return a.memory.RetrieveContext(ctx, sessionID, query, limit)
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

func (a *Agent) SetSharedSpaces(shared *memory.SharedSession) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.Shared = shared
}

// EnsureSpaceGrants gives the provided sessionID writer access to each space.
// This mirrors how tests set up spaces: mem.Spaces.Grant(space, session, role, ttl).
func (a *Agent) EnsureSpaceGrants(sessionID string, spaces []string) {
	if a == nil || a.memory == nil {
		return
	}
	for _, s := range spaces {
		s = strings.TrimSpace(s)
		if s == "" {
			continue
		}
		a.memory.Spaces.Grant(s, sessionID, memory.SpaceRoleWriter, 0)
	}
}

// SessionMemory exposes the underlying session memory (useful for advanced setup/tests).
func (a *Agent) SessionMemory() *memory.SessionMemory {
	return a.memory
}

// GenerateWithAttachments ingests files, updates memory context, and then generates.
func (a *Agent) GenerateWithAttachments(ctx context.Context, sessionID, userInput string, atts []Attachment) (string, error) {
	if strings.TrimSpace(userInput) == "" && len(atts) == 0 {
		return "", errors.New("both user input and attachments are empty")
	}
	// Guard: require ingestor only if there are attachments
	if len(atts) > 0 && a.Ingest == nil {
		return "", errors.New("attachments provided but Agent.Ingest is nil")
	}

	// 1) Persist the user message first (so it’s part of history regardless of ingest)
	if strings.TrimSpace(userInput) != "" {
		a.storeMemory(sessionID, "user", userInput, nil)
	}

	// 2) Ingest files (if any) into the session’s space (SharedSession or sessionID).
	if len(atts) > 0 {
		ingestedIDs, ingestMeta, err := a.ingestAttachments(ctx, sessionID, atts)
		if err != nil {
			// Write a structured error into memory to keep provenance
			a.storeMemory(sessionID, "assistant",
				fmt.Sprintf("attachment ingest failed: %v", err),
				map[string]string{"source": "ingest", "severity": "error"},
			)
			return "", err
		}
		// Log a short assistant message summarizing what was ingested
		a.storeMemory(sessionID, "assistant",
			fmt.Sprintf("Ingested %d attachment chunk(s) for context.", len(ingestedIDs)),
			ingestMeta,
		)
	}

	// 3) Build prompt with fresh memory (now includes ingested content)
	prompt, err := a.buildPrompt(ctx, sessionID, userInput)
	if err != nil {
		return "", err
	}

	// 4) Generate
	completion, err := a.model.Generate(ctx, prompt)
	if err != nil {
		return "", err
	}

	response := fmt.Sprint(completion)
	a.storeMemory(sessionID, "assistant", response, nil)
	return response, nil
}

// ingestAttachments is a small utility that streams each attachment into the Ingestor.
// It chooses a reasonable default scope (shared space) and tags everything with session/sessionID.
func (a *Agent) ingestAttachments(ctx context.Context, sessionID string, atts []Attachment) ([]string, map[string]string, error) {
	// Serialize concurrent ingests across one Agent to avoid surprising races in custom stores/embedders.
	a.mu.Lock()
	defer a.mu.Unlock()

	space := sessionID
	scopeDefault := upload.ScopeSpace
	allIDs := make([]string, 0, len(atts)*2)

	start := time.Now()
	for _, att := range atts {
		var name string
		var r io.Reader
		var closeFn func() error

		if att.Path != "" {
			f, err := os.Open(att.Path)
			if err != nil {
				return nil, nil, fmt.Errorf("open %q: %w", att.Path, err)
			}
			r = f
			name = baseName(att.Path)
			closeFn = f.Close
		} else if att.Reader != nil && att.Name != "" {
			r = att.Reader
			name = att.Name
		} else {
			return nil, nil, errors.New("each attachment needs Path or (Name+Reader)")
		}

		opts := upload.IngestOptions{
			Space: space,
			Scope: firstNonZeroScope(att.Scope, scopeDefault),
			Tags:  append([]string{"session:" + sessionID}, att.Tags...),
			TTL:   att.TTL,
			Meta:  att.Meta,
		}

		ids, err := a.Ingest.IngestReader(name, att.MIME, r, opts)
		if closeFn != nil {
			_ = closeFn()
		}
		if err != nil {
			return nil, nil, fmt.Errorf("ingest %q failed: %w", name, err)
		}
		allIDs = append(allIDs, ids...)
	}

	meta := map[string]string{
		"source":       "ingest",
		"ingest_space": space,
		"ingest_scope": string(scopeDefault),
		"elapsed_ms":   fmt.Sprintf("%d", time.Since(start).Milliseconds()),
	}
	return allIDs, meta, nil
}

// helpers

func baseName(p string) string {
	// no path import here to keep this tiny
	i := strings.LastIndexAny(p, `/\`)
	if i < 0 {
		return p
	}
	return p[i+1:]
}

func firstNonZeroScope(s, def upload.Scope) upload.Scope {
	if s == "" {
		return def
	}
	return s
}
