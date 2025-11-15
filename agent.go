package runtime

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/Protocol-Lattice/go-agent/src/memory"
	"github.com/Protocol-Lattice/go-agent/src/models"
	"github.com/alpkeskin/gotoon"
	"github.com/universal-tool-calling-protocol/go-utcp"
	"github.com/universal-tool-calling-protocol/go-utcp/src/plugins/chain"
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
	mu                sync.Mutex
	Shared            *memory.SharedSession
	CodeMode          *codemode.CodeModeUTCP
	CodeChain         *chain.UtcpChainClient
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
	CodeChain         *chain.UtcpChainClient
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
		CodeChain:         opts.CodeChain,
	}

	return a, nil
}

// codeModeOrchestrator decides whether CodeMode should execute userInput,
// runs it, stores TOON-encoded memory, and returns the final output.
func (a *Agent) codeModeOrchestrator(
	ctx context.Context,
	sessionID string,
	userInput string,
) (bool, string, error) {

	// If CodeMode is not active, skip.
	if a.CodeMode == nil {
		return false, "", nil
	}

	// Decide if this input looks like “code to execute”.
	// You can refine this heuristic over time.
	trim := strings.TrimSpace(userInput)
	if !strings.HasPrefix(trim, "```go") &&
		!strings.Contains(trim, "main(") &&
		!strings.Contains(trim, "{") {
		// Not CodeMode input — let LLM handle normally.
		return false, "", nil
	}

	// Strip code fences if present.
	code := trim
	if strings.HasPrefix(trim, "```") {
		code = strings.TrimPrefix(trim, "```go")
		code = strings.TrimPrefix(code, "```")
		code = strings.TrimSuffix(code, "```")
		code = strings.TrimSpace(code)
	}

	// Execute via CodeModeUTCP.
	result, err := a.CodeMode.Execute(ctx, codemode.CodeModeArgs{
		Code:    code,
		Timeout: 20000,
	})
	if err != nil {
		// Store error so LLM has context.
		a.storeMemory(sessionID, "assistant",
			fmt.Sprintf("CodeMode error: %v", err),
			map[string]string{"source": "codemode"},
		)
		return true, "", err
	}

	output := fmt.Sprint(result.Value)

	// Encode TOON output
	toonBytes, _ := gotoon.Encode(result.Value)
	full := fmt.Sprintf("%s\n\n.toon:\n%s", output, string(toonBytes))

	// Store memory
	a.storeMemory(sessionID, "assistant", full,
		map[string]string{"source": "codemode"},
	)

	return true, full, nil
}

// Flush persists session memory into the long-term store.
func (a *Agent) Flush(ctx context.Context, sessionID string) error {
	return a.memory.FlushToLongTerm(ctx, sessionID)
}

func (a *Agent) executeTool(ctx context.Context, sessionID, toolName string, args map[string]any) (any, error) {
	// 1. Try to find and invoke the tool locally first.
	if tool, _, ok := a.lookupTool(toolName); ok {
		resp, err := tool.Invoke(ctx, ToolRequest{
			SessionID: sessionID,
			Arguments: args,
		})
		if err != nil {
			return nil, err
		}
		return resp.Content, nil
	}

	// 2. If not found locally, and a UTCP client exists, try calling it remotely.
	if a.UTCPClient != nil {
		return a.UTCPClient.CallTool(ctx, toolName, args)
	}

	return nil, fmt.Errorf("unknown tool: %s", toolName)
}

// buildPrompt assembles the full assistant prompt for normal LLM generation.
// It does NOT include Toon markup. It NEVER formats for tool calls.
// It simply injects system prompt, retrieved memory, and file context.
func (a *Agent) buildPrompt(
	ctx context.Context,
	sessionID string,
	userInput string,
) (string, error) {

	// Detect query type to choose retrieval depth.
	queryType := classifyQuery(userInput)
	var records []memory.MemoryRecord
	var err error

	switch queryType {

	case QueryMath:
		// Skip heavy retrieval; math needs no context.

	case QueryShortFactoid:
		records, err = a.retrieveContext(ctx, sessionID, userInput, min(a.contextLimit/2, 3))
		if err != nil {
			return "", fmt.Errorf("retrieve context: %w", err)
		}

	case QueryComplex:
		records, err = a.retrieveContext(ctx, sessionID, userInput, a.contextLimit)
		if err != nil {
			return "", fmt.Errorf("retrieve context: %w", err)
		}

	default:
		// Unknown → no retrieval
	}

	// Build LLM prompt without tools/subagents:
	// Tools are only exposed inside the toolOrchestrator,
	// not during normal generation.
	var sb strings.Builder
	sb.Grow(4096)

	sb.WriteString(a.systemPrompt)
	sb.WriteString("\n\nConversation memory (TOON):\n")
	sb.WriteString(a.renderMemory(records))

	sb.WriteString("\n\nUser: ")
	sb.WriteString(strings.TrimSpace(userInput))
	sb.WriteString("\n\n") // no forced persona label

	// Rehydrate attachments
	files, _ := a.RetrieveAttachmentFiles(ctx, sessionID, a.contextLimit)
	if len(files) > 0 {
		sb.WriteString(a.buildAttachmentPrompt("Session attachments (rehydrated)", files))
	}

	return sb.String(), nil
}

func (a *Agent) renderTools() string {
	toolList := a.ToolSpecs()
	if len(toolList) == 0 {
		return ""
	}

	var sb strings.Builder
	sb.WriteString("Available UTCP tools:\n")

	for _, t := range toolList {
		sb.WriteString(fmt.Sprintf("- %s: %s\n", t.Name, t.Description))

		// -----------------------------
		// Input arguments (InputSchema)
		// -----------------------------
		props := t.Inputs.Properties
		required := map[string]bool{}
		for _, r := range t.Inputs.Required {
			required[r] = true
		}

		if len(props) > 0 {
			sb.WriteString("  args:\n")

			for name, spec := range props {
				typ := "any"

				if m, ok := spec.(map[string]any); ok {
					if tval, ok := m["type"].(string); ok {
						typ = tval
					}
				}

				if required[name] {
					sb.WriteString(fmt.Sprintf("    - %s (%s, required)\n", name, typ))
				} else {
					sb.WriteString(fmt.Sprintf("    - %s (%s)\n", name, typ))
				}
			}
		}

		// -----------------------------
		// Output schema (optional)
		// -----------------------------
		out := t.Outputs
		if len(out.Properties) > 0 || out.Type != "" {
			if toon := encodeTOONBlock(out); toon != "" {
				sb.WriteString("  returns (TOON):\n")
				sb.WriteString(indentBlock(toon, "    "))
				sb.WriteString("\n")
			}
		}
	}

	return sb.String()
}

func (a *Agent) buildFullPrompt(userInput string, records []memory.MemoryRecord) string {
	var sb strings.Builder
	sb.Grow(4096)

	sb.WriteString(a.systemPrompt)
	if tools := a.renderTools(); tools != "" {
		sb.WriteString("\n\nAvailable Tools:\n")
		sb.WriteString(tools)
	}
	if sub := a.renderSubAgents(); sub != "" {
		sb.WriteString("\n\n")
		sb.WriteString(sub)
	}

	sb.WriteString("\n\nConversation memory (TOON):\n")
	sb.WriteString(a.renderMemory(records))

	sb.WriteString("\n\nCurrent user message:\n")
	sb.WriteString(strings.TrimSpace(userInput))
	sb.WriteString("\n\nCompose the best possible assistant reply.\n")

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

	entries := make([]map[string]any, 0, len(records))
	var fallback strings.Builder
	counter := 0
	for _, rec := range records {
		content := strings.TrimSpace(rec.Content)
		if content == "" {
			continue
		}
		counter++
		role := metadataRole(rec.Metadata)
		space := rec.Space
		if space == "" {
			space = rec.SessionID
		}
		entry := map[string]any{
			"id":          counter,
			"role":        role,
			"space":       space,
			"score":       rec.Score,
			"importance":  rec.Importance,
			"source":      rec.Source,
			"summary":     rec.Summary,
			"content":     content,
			"last_update": rec.LastEmbedded.UTC().Format(time.RFC3339Nano),
		}
		if rec.LastEmbedded.IsZero() {
			delete(entry, "last_update")
		}
		entries = append(entries, entry)
		fallback.WriteString(fmt.Sprintf("%d. [%s] %s\n", counter, role, escapePromptContent(content)))
	}
	if len(entries) == 0 {
		return "(no stored memory)\n"
	}
	if toon := encodeTOONBlock(map[string]any{"memories": entries}); toon != "" {
		return toon + "\n"
	}
	return fallback.String()
}

// escapePromptContent safely escapes content that might break formatting.
func escapePromptContent(s string) string {
	s = strings.ReplaceAll(s, "`", "'")
	return s
}

func (a *Agent) detectDirectToolCall(s string) (string, map[string]any, bool) {
	s = strings.TrimSpace(s)
	lower := strings.ToLower(s)

	// ---------------------------------------------
	// Case 1: Raw JSON: {"tool": "...", "arguments": {...}}
	// ---------------------------------------------
	if strings.HasPrefix(s, "{") && strings.Contains(s, "\"tool\"") {
		var payload struct {
			Tool      string         `json:"tool"`
			Arguments map[string]any `json:"arguments"`
		}
		if err := json.Unmarshal([]byte(s), &payload); err == nil && payload.Tool != "" {
			return payload.Tool, payload.Arguments, true
		}
	}

	// ---------------------------------------------
	// Case 2: DSL: tool: echo { ... }
	// ---------------------------------------------
	if strings.HasPrefix(lower, "tool:") {
		rest := strings.TrimSpace(s[len("tool:"):])
		parts := strings.Fields(rest)
		if len(parts) >= 2 {
			tool := parts[0]
			argsStr := strings.TrimSpace(rest[len(tool):])

			var args map[string]any
			_ = json.Unmarshal([]byte(argsStr), &args) // best-effort

			return tool, args, true
		}
	}

	// ---------------------------------------------
	// Case 3: Shorthand: echo { ... }
	// ---------------------------------------------
	parts := strings.Fields(s)
	if len(parts) >= 2 {
		tool := strings.TrimSpace(parts[0])
		argsStr := strings.TrimSpace(s[len(tool):])

		var args map[string]any
		if err := json.Unmarshal([]byte(argsStr), &args); err == nil {
			return tool, args, true
		}
	}

	return "", nil, false
}

func (a *Agent) handleCommand(ctx context.Context, sessionID, userInput string) (bool, string, map[string]string, error) {
	trimmed := strings.TrimSpace(userInput)
	lower := strings.ToLower(trimmed)

	switch {
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
	if a == nil || strings.TrimSpace(content) == "" {
		return
	}

	// Build metadata safely.
	meta := map[string]string{}
	if rs := strings.TrimSpace(role); rs != "" {
		meta["role"] = rs
	}
	if extra != nil {
		for k, v := range extra {
			ks, vs := strings.TrimSpace(k), strings.TrimSpace(v)
			if ks != "" && vs != "" {
				meta[ks] = vs
			}
		}
	}

	// Snapshot pointers without holding the lock during external calls.
	a.mu.Lock()
	shared := a.Shared
	mem := a.memory
	a.mu.Unlock()

	// 1) Best-effort write to shared spaces (doesn't require embedder).
	if shared != nil {
		shared.AddShortLocal(content, meta)
		for _, space := range shared.Spaces() {
			_ = shared.AddShortTo(space, content, meta) // ignore per-space errors
		}
	}

	// 2) Write to session memory if available.
	if mem == nil || mem.Embedder == nil {
		return // nothing else to do; avoid panic
	}

	// Compute embedding with a small timeout to avoid hanging the call.
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	embedding, err := mem.Embedder.Embed(ctx, content)
	if err != nil {
		return // silent drop on embed failure; consider logging if desired
	}

	metaBytes, _ := json.Marshal(meta)

	// Append to short-term memory under lock.
	a.mu.Lock()
	defer a.mu.Unlock()
	mem.AddShortTerm(sessionID, content, string(metaBytes), embedding)
}

func (a *Agent) storeAttachmentMemories(sessionID string, files []models.File) {
	for i, file := range files {
		name := strings.TrimSpace(file.Name)
		if name == "" {
			name = fmt.Sprintf("file_%d", i+1)
		}
		mime := strings.TrimSpace(file.MIME)
		content := buildAttachmentMemoryContent(name, mime, file.Data)
		extra := map[string]string{
			"source":   "file_upload",
			"filename": name,
		}
		if mime != "" {
			extra["mime"] = mime
		}
		if size := len(file.Data); size > 0 {
			extra["size_bytes"] = strconv.Itoa(size)
		}
		if len(file.Data) > 0 {
			extra["data_base64"] = base64.StdEncoding.EncodeToString(file.Data)
		}
		if isTextAttachment(mime, file.Data) {
			extra["text"] = "true"
		} else {
			extra["text"] = "false"
		}
		a.storeMemory(sessionID, "attachment", content, extra)
	}
}

// RetrieveAttachmentFiles returns attachment files stored for the session.
// It reconstructs the original bytes from base64-encoded metadata, making it
// suitable for binary assets such as images and videos.
func (a *Agent) RetrieveAttachmentFiles(ctx context.Context, sessionID string, limit int) ([]models.File, error) {
	if a == nil || a.memory == nil {
		return nil, nil
	}
	if limit <= 0 {
		limit = a.contextLimit
		if limit <= 0 {
			limit = 8
		}
	}

	records, err := a.memory.RetrieveContext(ctx, sessionID, "", limit)
	if err != nil {
		return nil, err
	}

	var attachments []models.File
	for _, record := range records {
		if metadataRole(record.Metadata) != "attachment" {
			continue
		}
		file, ok := attachmentFromRecord(record)
		if !ok {
			continue
		}
		attachments = append(attachments, file)
	}

	return attachments, nil
}

func attachmentFromRecord(record memory.MemoryRecord) (models.File, bool) {
	if strings.TrimSpace(record.Metadata) == "" {
		return models.File{}, false
	}

	var payload map[string]any
	if err := json.Unmarshal([]byte(record.Metadata), &payload); err != nil {
		return models.File{}, false
	}

	name, _ := payload["filename"].(string)
	mime, _ := payload["mime"].(string)
	dataB64, _ := payload["data_base64"].(string)
	if name == "" {
		name = "attachment"
	}

	var data []byte
	if dataB64 != "" {
		raw, err := base64.StdEncoding.DecodeString(dataB64)
		if err != nil {
			return models.File{}, false
		}
		data = raw
	} else {
		data = extractTextAttachment(record.Content)
	}

	return models.File{Name: name, MIME: mime, Data: data}, true
}

func extractTextAttachment(content string) []byte {
	idx := strings.Index(content, ":\n")
	if idx == -1 {
		return nil
	}
	return []byte(content[idx+2:])
}

func isTextAttachment(mime string, data []byte) bool {
	mt := strings.ToLower(strings.TrimSpace(mime))
	switch {
	case strings.HasPrefix(mt, "text/"):
		return true
	case mt == "application/json",
		mt == "application/xml",
		mt == "application/x-yaml",
		mt == "application/yaml",
		mt == "text/markdown",
		mt == "text/x-markdown":
		return true
	}
	if len(data) == 0 {
		return true
	}
	return utf8.Valid(data)
}

func renderUtcpToolsForPrompt(list []tools.Tool) string {
	if len(list) == 0 {
		return "No UTCP tools available.\n"
	}

	// Deterministic ordering
	sort.Slice(list, func(i, j int) bool {
		return strings.ToLower(list[i].Name) < strings.ToLower(list[j].Name)
	})

	var sb strings.Builder
	sb.WriteString("Available UTCP tools:\n")

	for _, t := range list {
		// ---- Basic metadata ----
		sb.WriteString(fmt.Sprintf("- %s: %s\n", t.Name, t.Description))

		// ---- Input schema ----
		props := t.Inputs.Properties
		required := map[string]bool{}
		for _, r := range t.Inputs.Required {
			required[r] = true
		}

		if len(props) > 0 {
			sb.WriteString("  args:\n")

			// deterministic order of args
			keys := make([]string, 0, len(props))
			for k := range props {
				keys = append(keys, k)
			}
			sort.Strings(keys)

			for _, name := range keys {
				spec := props[name]

				typ := "any"
				if mm, ok := spec.(map[string]any); ok {
					if tt, ok := mm["type"].(string); ok {
						typ = tt
					}
				}

				if required[name] {
					sb.WriteString(fmt.Sprintf("    - %s (%s, required)\n", name, typ))
				} else {
					sb.WriteString(fmt.Sprintf("    - %s (%s)\n", name, typ))
				}
			}
		}

		// ---- Output schema (TOON-encoded) ----
		out := t.Outputs
		if len(out.Properties) > 0 || out.Type != "" {
			if toon := encodeTOONBlock(out); toon != "" {
				sb.WriteString("  returns (TOON):\n")
				sb.WriteString(indentBlock(toon, "    "))
				sb.WriteString("\n")
			}
		}
	}

	return sb.String()
}

func buildAttachmentMemoryContent(name, mime string, data []byte) string {
	display := strings.TrimSpace(name)
	if display == "" {
		display = "attachment"
	}
	descriptor := display
	if m := strings.TrimSpace(mime); m != "" {
		descriptor = fmt.Sprintf("%s (%s)", display, m)
	}
	if len(data) == 0 {
		return fmt.Sprintf("Attachment %s [empty file]", descriptor)
	}
	if isTextAttachment(mime, data) {
		var sb strings.Builder
		sb.Grow(len(data) + len(descriptor) + 32)
		sb.WriteString("Attachment ")
		sb.WriteString(descriptor)
		sb.WriteString(":\n")
		sb.Write(data)
		return sb.String()
	}
	return fmt.Sprintf("Attachment %s [%d bytes of non-text content]", descriptor, len(data))
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
func (a *Agent) ToolSpecs() []tools.Tool {
	var allSpecs []tools.Tool
	seen := make(map[string]bool)

	// 2. Get UTCP tool specs and merge
	if a.UTCPClient != nil {
		utcpTools, _ := a.UTCPClient.SearchTools("", 50)
		for _, tool := range utcpTools {
			key := strings.ToLower(tool.Name)
			if !seen[key] {
				allSpecs = append(allSpecs, tool)
				seen[key] = true
			}
		}
	}
	return allSpecs
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

func (a *Agent) Generate(ctx context.Context, sessionID, userInput string) (string, error) {
	trimmed := strings.TrimSpace(userInput)
	if trimmed == "" {
		return "", errors.New("user input is empty")
	}

	// ---------------------------------------------
	// 0. DIRECT TOOL INVOCATION (bypass LLM entirely)
	// ---------------------------------------------
	if toolName, args, ok := a.detectDirectToolCall(trimmed); ok {
		// Execute tool immediately
		result, err := a.executeTool(ctx, sessionID, toolName, args)
		if err != nil {
			return "", err
		}

		// RAW output — NO toon, NO coordinator, NO assistant prefix
		return fmt.Sprint(result), nil
	}

	// ---------------------------------------------
	// 1. Normal: store user memory
	// ---------------------------------------------
	a.storeMemory(sessionID, "user", userInput, nil)

	// ---------------------------------------------
	// 2. Tool orchestrator (LLM decides a tool)
	// ---------------------------------------------
	if handled, output, err := a.toolOrchestrator(ctx, sessionID, userInput); handled {
		if err != nil {
			return "", err
		}
		return output, nil // toolOrchestrator already returns RAW tool output
	}
	// ---------------------------------------------
	// 0B. SUBAGENT COMMANDS (e.g., subagent:researcher ...)
	// ---------------------------------------------
	if handled, out, meta, err := a.handleCommand(ctx, sessionID, userInput); handled {
		if err != nil {
			return "", err
		}
		a.storeMemory(sessionID, "subagent", out, meta)
		return out, nil
	}

	// ---------------------------------------------
	// 3. CodeChain (pre-CodeMode)
	// ---------------------------------------------
	if a.CodeChain != nil {
		if handled, output, err := a.codeChainOrchestrator(ctx, sessionID, userInput); handled {
			if err != nil {
				return "", err
			}
			return output, nil
		}
	}

	// ---------------------------------------------
	// 4. CodeMode
	// ---------------------------------------------
	if a.CodeMode != nil {
		if handled, output, err := a.codeModeOrchestrator(ctx, sessionID, userInput); handled {
			if err != nil {
				return "", err
			}
			return output, nil
		}
	}

	// ---------------------------------------------
	// 5. Standard LLM response
	// ---------------------------------------------
	prompt, err := a.buildPrompt(ctx, sessionID, userInput)
	if err != nil {
		return "", err
	}

	files, _ := a.RetrieveAttachmentFiles(ctx, sessionID, a.contextLimit)

	var completion any
	if len(files) > 0 {
		completion, err = a.model.GenerateWithFiles(ctx, prompt, files)
	} else {
		completion, err = a.model.Generate(ctx, prompt)
	}
	if err != nil {
		return "", err
	}

	response := fmt.Sprint(completion)

	// ---------------------------------------------
	// 6. TOON for assistant messages only
	// ---------------------------------------------
	toonBytes, _ := gotoon.Encode(completion)
	full := fmt.Sprintf("%s\n\n.toon:\n%s", response, string(toonBytes))

	a.storeMemory(sessionID, "assistant", full, nil)
	return full, nil
}

// SessionMemory exposes the underlying session memory (useful for advanced setup/tests).
func (a *Agent) SessionMemory() *memory.SessionMemory {
	return a.memory
}

// GenerateWithFiles sends the user message plus in-memory files to the model
// without ingesting them into long-term memory. Use this when you already have
// file bytes (e.g., uploaded via API) and want the model to consider them
// ephemerally for this turn only.
func (a *Agent) GenerateWithFiles(
	ctx context.Context,
	sessionID string,
	userInput string,
	files []models.File,
) (string, error) {
	if strings.TrimSpace(userInput) == "" && len(files) == 0 {
		return "", errors.New("both user input and files are empty")
	}

	if strings.TrimSpace(userInput) != "" {
		a.storeMemory(sessionID, "user", userInput, nil)
	}

	// Build base prompt
	prompt, err := a.buildPrompt(ctx, sessionID, userInput)
	if err != nil {
		return "", err
	}

	// Persist new this-turn files
	if len(files) > 0 {
		a.storeAttachmentMemories(sessionID, files)
		prompt += a.buildAttachmentPrompt("Files provided for this turn", files)
	}

	// Rehydrate old files
	existing, _ := a.RetrieveAttachmentFiles(ctx, sessionID, a.contextLimit)
	if len(existing) > 0 {
		prompt += a.buildAttachmentPrompt("Session attachments (rehydrated)", existing)
	}

	// Model call
	completion, err := a.model.GenerateWithFiles(
		ctx,
		prompt,
		append(existing, files...),
	)
	if err != nil {
		return "", err
	}

	response := fmt.Sprint(completion)

	// ---------------------------------------
	// 🔵 NEW: Add TOON-encoded output
	// ---------------------------------------
	toonBytes, _ := gotoon.Encode(completion)
	full := fmt.Sprintf("%s\n\n.toon:\n%s", response, string(toonBytes))

	// store TOON-enhanced version
	a.storeMemory(sessionID, "assistant", full, nil)

	return full, nil
}

// buildAttachmentPrompt renders a compact, token-conscious list of files.
// It never inlines non-text bytes. For text files, it shows a short preview.
func (a *Agent) buildAttachmentPrompt(title string, files []models.File) string {
	if len(files) == 0 {
		return ""
	}
	var fallback strings.Builder
	entries := make([]map[string]any, 0, len(files))
	for i, f := range files {
		name := strings.TrimSpace(f.Name)
		if name == "" {
			name = fmt.Sprintf("attachment_%d", i+1)
		}
		mime := strings.TrimSpace(f.MIME)
		if mime == "" {
			mime = "application/octet-stream"
		}
		sizeBytes := len(f.Data)
		isText := isTextAttachment(mime, f.Data)
		entry := map[string]any{
			"id":         i + 1,
			"name":       name,
			"mime":       mime,
			"size_bytes": sizeBytes,
			"text":       isText,
		}
		if isText && len(f.Data) > 0 {
			entry["preview"] = previewText(mime, f.Data)
		}
		entries = append(entries, entry)

		fallback.WriteString(fmt.Sprintf("- %s (%s, %s)", name, mime, humanSize(sizeBytes)))
		if isText && len(f.Data) > 0 {
			fallback.WriteString("\n  preview:\n  ")
			fallback.WriteString(escapePromptContent(previewText(mime, f.Data)))
		}
		fallback.WriteString("\n")
	}

	var sb strings.Builder
	sb.WriteString("\n\n")
	sb.WriteString(title)
	sb.WriteString(":\n")
	if toon := encodeTOONBlock(map[string]any{"files": entries}); toon != "" {
		sb.WriteString(indentBlock(toon, "  "))
		sb.WriteString("\n")
	} else {
		sb.WriteString(fallback.String())
	}
	return sb.String()
}

func humanSize(n int) string {
	const (
		KB = 1024
		MB = 1024 * KB
		GB = 1024 * MB
	)
	switch {
	case n >= GB:
		return fmt.Sprintf("%.2f GB", float64(n)/float64(GB))
	case n >= MB:
		return fmt.Sprintf("%.2f MB", float64(n)/float64(MB))
	case n >= KB:
		return fmt.Sprintf("%.2f KB", float64(n)/float64(KB))
	default:
		return fmt.Sprintf("%d B", n)
	}
}

// previewText returns a short snippet from text attachments (max ~1KB) to save tokens.
func previewText(_ string, data []byte) string {
	const maxPreview = 1024
	txt := string(data)
	return truncate(txt, maxPreview)
}

func truncate(s string, max int) string {
	if len(s) <= max {
		return s
	}
	// Try to cut on a boundary to avoid mid-rune issues for safety.
	if max > 3 {
		return s[:max-3] + "..."
	}
	return s[:max]
}

func encodeTOONBlock(value any) string {
	if value == nil {
		return ""
	}
	if encoded, err := gotoon.Encode(value); err == nil {
		return strings.TrimSpace(encoded)
	}
	if fallback, err := json.MarshalIndent(value, "", "  "); err == nil {
		return strings.TrimSpace(string(fallback))
	}
	return ""
}

func indentBlock(text, prefix string) string {
	text = strings.TrimRight(text, "\n")
	if text == "" {
		return ""
	}
	lines := strings.Split(text, "\n")
	for i := range lines {
		lines[i] = prefix + lines[i]
	}
	return strings.Join(lines, "\n")
}

type ToolChoice struct {
	UseTool   bool           `json:"use_tool"`
	ToolName  string         `json:"tool_name"`
	Arguments map[string]any `json:"arguments"`
	Reason    string         `json:"reason"`
}

// codeChainOrchestrator decides whether a UTCP chain should run,
// executes it via UtcpChainClient, encodes output in TOON, stores memory,
// and returns the final result.
func (a *Agent) codeChainOrchestrator(
	ctx context.Context,
	sessionID string,
	userInput string,
) (bool, string, error) {

	// Chain client not enabled → skip
	if a.CodeChain == nil {
		return false, "", nil
	}

	trim := strings.TrimSpace(userInput)
	lower := strings.ToLower(trim)

	// ---- 1. Detection heuristics ----
	// You can refine these later.
	isChain :=
		strings.HasPrefix(lower, "chain:") ||
			strings.Contains(lower, "\"tool_name\"") ||
			strings.Contains(lower, "\"steps\"") ||
			strings.Contains(lower, "use_previous")

	if !isChain {
		return false, "", nil
	}

	// ---- 2. Extract JSON payload ----
	payload := strings.TrimSpace(strings.TrimPrefix(lower, "chain:"))
	if payload == "" {
		return true, "", errors.New("chain: missing steps payload")
	}

	var steps []chain.ChainStep
	var wrapper struct {
		Steps   []chain.ChainStep `json:"steps"`
		Timeout int               `json:"timeout"`
	}

	// Accept direct []ChainStep or wrapped object { steps: [...] }
	if err := json.Unmarshal([]byte(payload), &steps); err != nil {
		if err2 := json.Unmarshal([]byte(payload), &wrapper); err2 != nil {
			return true, "", fmt.Errorf("chain: invalid steps JSON: %v", err)
		}
		steps = wrapper.Steps
	}

	timeout := 20 * time.Second
	if wrapper.Timeout > 0 {
		timeout = time.Duration(wrapper.Timeout) * time.Millisecond
	}

	// ---- 3. Execute chain ----
	result, err := a.CodeChain.CallToolChain(ctx, steps, timeout)
	if err != nil {
		a.storeMemory(sessionID, "assistant",
			fmt.Sprintf("CodeChain error: %v", err),
			map[string]string{"source": "codechain"},
		)
		return true, "", err
	}

	// ---- 4. Convert result → string ----
	var out string
	if b, err := json.MarshalIndent(result, "", "  "); err == nil {
		out = string(b)
	} else {
		out = fmt.Sprint(result)
	}

	// ---- 5. TOON encode + store ----
	toonBytes, _ := gotoon.Encode(result)
	full := fmt.Sprintf("%s\n\n.toon:\n%s", out, string(toonBytes))

	a.storeMemory(sessionID, "assistant", full,
		map[string]string{"source": "codechain"},
	)

	return true, full, nil
}

func (a *Agent) toolOrchestrator(
	ctx context.Context,
	sessionID string,
	userInput string,
) (bool, string, error) {

	// Collect merged local + UTCP tools
	toolList := a.ToolSpecs()
	if len(toolList) == 0 {
		return false, "", nil
	}

	// -----------------------------
	// 1. Build LLM tool-selection prompt
	// -----------------------------
	var sb strings.Builder
	sb.WriteString("UTCP Tools:\n")
	for _, t := range toolList {
		sb.WriteString(fmt.Sprintf("- %s: %s\n", t.Name, t.Description))
	}

	toolDesc := renderUtcpToolsForPrompt(toolList)

	choicePrompt := fmt.Sprintf(`
You are a UTCP tool selection engine.

A user asked:
%q

You have access to these UTCP tools:
%s

You can also discover tools dynamically using:
search_tools("<query>", <limit>)

Example:
search_tools("file", 10)

Think step-by-step whether ANY tool should be used.

Return ONLY a JSON object EXACTLY like this:

{
  "use_tool": true|false,
  "tool_name": "name or empty",
  "arguments": { },
  "reason": "short explanation"
}

Return ONLY JSON. No explanations.
`, userInput, toolDesc)

	// -----------------------------
	// 2. Query LLM
	// -----------------------------
	raw, err := a.model.Generate(ctx, choicePrompt)
	if err != nil {
		return false, "", err
	}

	response := strings.TrimSpace(fmt.Sprint(raw))

	// -----------------------------
	// 3. EXTRACT AND VALIDATE JSON
	// -----------------------------
	jsonStr := extractJSON(response)
	if jsonStr == "" {
		return false, "", nil
	}

	var tc ToolChoice
	if err := json.Unmarshal([]byte(jsonStr), &tc); err != nil {
		return false, "", nil
	}

	if !tc.UseTool {
		return false, "", nil
	}
	if strings.TrimSpace(tc.ToolName) == "" {
		return false, "", nil
	}

	// -----------------------------
	// 4. Validate tool exists
	// -----------------------------
	exists := false
	for _, t := range toolList {
		if t.Name == tc.ToolName {
			exists = true
			break
		}
	}
	if !exists {
		return true, "", fmt.Errorf("UTCP tool unknown: %s", tc.ToolName)
	}

	// -----------------------------
	// 5. Execute UTCP or local tool
	// IMPORTANT: always return RAW result.
	// -----------------------------
	result, err := a.executeTool(ctx, sessionID, tc.ToolName, tc.Arguments)
	if err != nil {
		// still store memory in TOON for context
		a.storeMemory(sessionID, "assistant",
			fmt.Sprintf("tool %s error: %v", tc.ToolName, err),
			map[string]string{"source": "tool_orchestrator"},
		)
		return true, "", err
	}

	// -----------------------------
	// 6. RAW OUTPUT (NO TOON FORMATTING!)
	// -----------------------------
	rawOut := fmt.Sprint(result)

	// Store TOON version separately in memory (allowed)
	toonBytes, _ := gotoon.Encode(rawOut)
	store := fmt.Sprintf("%s\n\n.toon:\n%s", rawOut, string(toonBytes))

	a.storeMemory(sessionID, "assistant", store, map[string]string{
		"tool":   tc.ToolName,
		"source": "tool_orchestrator",
	})

	// -----------------------------
	// 7. Return RAW tool output (what UTCP tests expect)
	// -----------------------------
	return true, rawOut, nil
}

// extractJSON attempts to extract valid JSON from a response that may contain
// markdown code fences, extra text, or concatenated content.
func extractJSON(response string) string {
	response = strings.TrimSpace(response)

	// Case 1: Pure JSON (starts and ends with braces)
	if strings.HasPrefix(response, "{") && strings.HasSuffix(response, "}") {
		return response
	}

	// Case 2: JSON wrapped in markdown code fence
	// ```json\n{...}\n```
	if strings.Contains(response, "```") {
		// Remove opening fence
		response = strings.TrimSpace(response)
		response = strings.TrimPrefix(response, "```json")
		response = strings.TrimPrefix(response, "```")
		response = strings.TrimSpace(response)

		// Remove closing fence
		if idx := strings.Index(response, "```"); idx != -1 {
			response = response[:idx]
		}
		response = strings.TrimSpace(response)

		if strings.HasPrefix(response, "{") && strings.HasSuffix(response, "}") {
			return response
		}
	}

	// Case 3: JSON followed by extra content (e.g., " | prompt text")
	// Find the first { and try to extract a complete JSON object
	startIdx := strings.Index(response, "{")
	if startIdx == -1 {
		return ""
	}

	// Find the matching closing brace
	depth := 0
	inString := false
	escaped := false

	for i := startIdx; i < len(response); i++ {
		ch := response[i]

		if escaped {
			escaped = false
			continue
		}

		if ch == '\\' {
			escaped = true
			continue
		}

		if ch == '"' {
			inString = !inString
			continue
		}

		if inString {
			continue
		}

		if ch == '{' {
			depth++
		} else if ch == '}' {
			depth--
			if depth == 0 {
				// Found the matching closing brace
				candidate := response[startIdx : i+1]
				// Validate it's actually valid JSON
				var test interface{}
				if json.Unmarshal([]byte(candidate), &test) == nil {
					return candidate
				}
			}
		}
	}

	return ""
}
