package agent

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
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

const (
	defaultSystemPrompt = "You are the primary coordinator for an AI agent team. Provide concise, accurate answers and explain when you call tools or delegate work to specialist sub-agents."
	defaultToolCacheTTL = 30 * time.Second
)

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

	Shared    *memory.SharedSession
	CodeMode  *codemode.CodeModeUTCP
	CodeChain *chain.UtcpChainClient
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

// userLooksLikeToolCall returns true if the user *likely* meant to call a tool.
func (a *Agent) userLooksLikeToolCall(s string) bool {
	s = strings.TrimSpace(strings.ToLower(s))

	// Looks like: echo {...}
	if strings.Contains(s, "{") && strings.Contains(s, "}") {
		parts := strings.Fields(s)
		if len(parts) > 0 {
			tool := parts[0]
			for _, t := range a.ToolSpecs() {
				if strings.ToLower(t.Name) == tool {
					return true
				}
			}
		}
	}

	// Looks like: tool: echo {...}
	if strings.HasPrefix(s, "tool:") {
		return true
	}

	// Looks like: {"tool": "echo", ...}
	if strings.HasPrefix(s, "{") && strings.Contains(s, "\"tool\"") {
		return true
	}

	return false
}

func (a *Agent) decideIfToolsNeeded(
	ctx context.Context,
	query string,
	tools string,
) (bool, error) {

	prompt := fmt.Sprintf(`
Decide if the following user query requires using ANY UTCP tools.

USER QUERY:
%q

AVAILABLE UTCP TOOLS:
%s

Respond ONLY in JSON:
{ "needs": true } or { "needs": false }
`, query, tools)

	raw, err := a.model.Generate(ctx, prompt)
	if err != nil {
		return false, err
	}

	jsonStr := extractJSON(fmt.Sprint(raw))
	if jsonStr == "" {
		return false, nil
	}

	var resp struct {
		Needs bool `json:"needs"`
	}

	if err := json.Unmarshal([]byte(jsonStr), &resp); err != nil {
		return false, nil
	}

	return resp.Needs, nil
}

func (a *Agent) selectTools(
	ctx context.Context,
	query string,
	tools string,
) ([]string, error) {

	prompt := fmt.Sprintf(`
Select ALL UTCP tools that match the user's intent.

USER QUERY:
%q

AVAILABLE UTCP TOOLS:
%s

Respond ONLY in JSON:
{
  "tools": ["provider.tool", ...]
}

Rules:
- Use ONLY names listed above.
- NO modifications, NO guessing.
- If multiple tools apply, include all.
`, query, tools)

	raw, err := a.model.Generate(ctx, prompt)
	if err != nil {
		return nil, err
	}

	jsonStr := extractJSON(fmt.Sprint(raw))
	if jsonStr == "" {
		return nil, nil
	}

	var resp struct {
		Tools []string `json:"tools"`
	}

	_ = json.Unmarshal([]byte(jsonStr), &resp)
	return resp.Tools, nil
}

func (a *Agent) generateSnippet(
	ctx context.Context,
	query string,
	tools []string,
	toolSpecs string,
) (string, bool, error) {

	toolsJSON, _ := json.Marshal(tools)

	prompt := fmt.Sprintf(`
Generate a Go snippet that uses ONLY the following UTCP tools:

%v

USER QUERY:
%q

TOOL SPECS:
%s

------------------------------------------------------------
SNIPPET RULES
------------------------------------------------------------
- Use ONLY the tool names listed above.
- Use EXACT input keys from the tool schemas. Do NOT invent new fields.
- Use these exact helper functions:
  - codemode.CallTool(name, args)
  - codemode.CallToolStream(name, args)
  - codemode.SearchTools(query, limit)
  - codemode.Sprintf(format, ...), codemode.Errorf(format, ...)
- No imports, no package — ONLY Go statements.
- Don't Declare 'var __out'
- The final result MUST be assigned to '__out', containing all intermediate and final results.
- If ANY streaming tool is used, set "stream": true.

------------------------------------------------------------
CHAINING (NON-STREAMING) — STRICT RULES
------------------------------------------------------------
To pass output of one tool into another:

1. Call the tool:
    r1, err := codemode.CallTool("<tool>", map[string]any{
        "a": 5,
        "b": 7,
    })
    if err != nil { return err }

2. Extract value using EXACT output-schema keys:
    var sum any
    if m, ok := r1.(map[string]any); ok {
        sum = m["result"]   // key MUST match schema
    }

3. Use this value as input to the next tool:
    r2, err := codemode.CallTool("<next_tool>", map[string]any{
        "a": sum,
        "b": 3,
    })

4. The final line must set:
    __out = map[string]any{ // USE = NOT :=
        "sum": sum,
        "product": r2,
    }

------------------------------------------------------------
STREAMING TOOLS — STRICT RULES
------------------------------------------------------------
When calling a streaming tool:

1. Start the stream:
    stream, err := codemode.CallToolStream("<stream_tool>", map[string]any{
        "input": "hello",
    })
    if err != nil { return err }

2. Read chunks in a loop:
    var items []any
    for {
        chunk, err := stream.Next()
        if err != nil { break }
        items = append(items, chunk)
    }

3. You may chain streaming results into non-streaming tools:
    r2, err := codemode.CallTool("provider.summarize", map[string]any{
        "values": items,
    })

4. Or output directly:
    __out = items

------------------------------------------------------------
Respond ONLY in JSON:
{
  "code": "<go snippet>",
  "stream": false
}
`, string(toolsJSON), query, toolSpecs)

	raw, err := a.model.Generate(ctx, prompt)
	if err != nil {
		return "", false, err
	}

	jsonStr := extractJSON(fmt.Sprint(raw))
	if jsonStr == "" {
		return "", false, fmt.Errorf("snippet empty")
	}

	var resp struct {
		Code   string `json:"code"`
		Stream bool   `json:"stream"`
	}

	if err := json.Unmarshal([]byte(jsonStr), &resp); err != nil {
		return "", false, err
	}
	if !isValidSnippet(resp.Code) {
		log.Println("Skipping invalid snippet:", resp.Code)
		return "", false, fmt.Errorf("snippet validation failed")
	}

	return resp.Code, resp.Stream, nil
}

// Flush persists session memory into the long-term store.
func (a *Agent) Flush(ctx context.Context, sessionID string) error {
	return a.memory.FlushToLongTerm(ctx, sessionID)
}

func (a *Agent) executeTool(
	ctx context.Context,
	sessionID, toolName string,
	args map[string]any,
) (any, error) {

	if args == nil {
		args = map[string]any{}
	}

	// ---------------------------------------------
	// 1. REMOTE UTCP TOOL
	// If "stream": true → CallToolStream
	// else → CallTool
	// ---------------------------------------------
	if a.UTCPClient != nil {

		// streaming request?
		if streamFlag, ok := args["stream"].(bool); ok && streamFlag {

			stream, err := a.UTCPClient.CallToolStream(ctx, toolName, args)
			if err != nil {
				return nil, err
			}
			if stream == nil {
				return nil, fmt.Errorf("CallToolStream returned nil stream for %s", toolName)
			}

			// Accumulate streamed chunks into a single string
			var sb strings.Builder
			for {
				chunk, err := stream.Next()
				if err != nil {
					break
				}

				if chunk != nil {
					sb.WriteString(fmt.Sprint(chunk))
				}
			}

			return sb.String(), nil
		}

		// Non-streaming remote call
		return a.UTCPClient.CallTool(ctx, toolName, args)
	}

	// ---------------------------------------------
	// 3. Unknown tool
	// ---------------------------------------------
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

	// Build a lowercase lookup of all registered tool names
	valid := make(map[string]string)      // lowerName → exactName
	prefixes := make(map[string]struct{}) // provider prefixes
	bases := make(map[string][]string)    // short name → list of full names

	for _, spec := range a.ToolSpecs() {
		exact := spec.Name
		lowerName := strings.ToLower(exact)
		valid[lowerName] = exact

		// Collect prefix (provider)
		if parts := strings.Split(lowerName, "."); len(parts) >= 2 {
			prefixes[parts[0]] = struct{}{}
			short := parts[len(parts)-1]
			bases[short] = append(bases[short], exact)
		}
	}

	// helper: try matching tool name dynamically using full registry
	normalize := func(name string) (string, bool) {
		nameLower := strings.ToLower(strings.TrimSpace(name))

		// 1) Exact match
		if exact, ok := valid[nameLower]; ok {
			return exact, true
		}

		// 2) Match by fully-qualified suffix (e.g. "math.add")
		for fullLower, exact := range valid {
			if strings.HasSuffix(fullLower, "."+nameLower) {
				return exact, true
			}
		}

		// 3) Match by base (last segment only)
		if list, ok := bases[nameLower]; ok && len(list) > 0 {
			// if multiple tools share the same short name, choose the first or return false
			return list[0], true
		}

		return "", false
	}

	// ---------------------------------------------------------
	// Case 1: Raw JSON {"tool":"...", "arguments":{...}}
	// ---------------------------------------------------------
	if strings.HasPrefix(s, "{") && strings.Contains(s, "\"tool\"") {
		var payload struct {
			Tool      string         `json:"tool"`
			Arguments map[string]any `json:"arguments"`
		}
		if err := json.Unmarshal([]byte(s), &payload); err == nil && payload.Tool != "" {
			if real, ok := normalize(payload.Tool); ok {
				return real, payload.Arguments, ok
			}
			// Fallback: if UTCP client exists, accept raw tool name
			if a.UTCPClient != nil {
				return payload.Tool, payload.Arguments, true
			}

			return "", nil, false
		}
	}

	// ---------------------------------------------------------
	// Case 2: DSL: tool: echo { ... }
	// ---------------------------------------------------------
	if strings.HasPrefix(lower, "tool:") {
		rest := strings.TrimSpace(s[len("tool:"):])
		parts := strings.Fields(rest)
		if len(parts) >= 2 {
			tool := parts[0]
			argsStr := strings.TrimSpace(rest[len(tool):])

			var args map[string]any
			_ = json.Unmarshal([]byte(argsStr), &args)

			if real, ok := normalize(tool); ok {
				return real, args, ok
			}
			// Fallback for UTCP client
			if a.UTCPClient != nil {
				return tool, args, true
			}
			return "", nil, false
		}
	}

	// ---------------------------------------------------------
	// Case 3: Shorthand: echo { ... }
	// ---------------------------------------------------------
	parts := strings.Fields(s)
	if len(parts) >= 2 {
		tool := strings.TrimSpace(parts[0])
		argsStr := strings.TrimSpace(s[len(tool):])

		var args map[string]any
		if err := json.Unmarshal([]byte(argsStr), &args); err == nil {
			if real, ok := normalize(tool); ok {
				return real, args, ok
			}
			// Fallback for UTCP client
			if a.UTCPClient != nil {
				return tool, args, true
			}
			return "", nil, false
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

func toolCacheTTL() time.Duration {
	raw := strings.TrimSpace(os.Getenv("utcp_tool_cache_ttl_ms"))
	if raw == "" {
		return defaultToolCacheTTL
	}
	ms, err := strconv.Atoi(raw)
	if err != nil || ms <= 0 {
		return defaultToolCacheTTL
	}
	return time.Duration(ms) * time.Millisecond
}

func toolListSignature(specs []tools.Tool) string {
	if len(specs) == 0 {
		return ""
	}

	var sb strings.Builder
	sb.Grow(len(specs) * 32)

	for _, t := range specs {
		sb.WriteString(strings.ToLower(strings.TrimSpace(t.Name)))
		sb.WriteByte('|')
		sb.WriteString(strings.TrimSpace(t.Description))
		sb.WriteByte('|')
		sb.WriteString(strconv.Itoa(len(t.Inputs.Properties)))
		sb.WriteByte('|')
		sb.WriteString(strconv.Itoa(len(t.Inputs.Required)))
		sb.WriteByte(';')
	}
	return sb.String()
}

func (a *Agent) cachedToolPrompt(specs []tools.Tool) string {
	if len(specs) == 0 {
		return ""
	}

	key := toolListSignature(specs)
	now := time.Now()

	a.toolMu.RLock()
	prompt := a.toolPromptCache
	cacheKey := a.toolPromptKey
	promptExpiry := a.toolPromptExpiry
	specExpiry := a.toolSpecsExpiry
	a.toolMu.RUnlock()

	if prompt != "" && key == cacheKey && (promptExpiry.IsZero() || now.Before(promptExpiry)) {
		return prompt
	}

	rendered := renderUtcpToolsForPrompt(specs)
	expiry := specExpiry
	if expiry.IsZero() || now.After(expiry) {
		expiry = now.Add(toolCacheTTL())
	}

	a.toolMu.Lock()
	a.toolPromptCache = rendered
	a.toolPromptKey = key
	a.toolPromptExpiry = expiry
	a.toolMu.Unlock()

	return rendered
}

func renderUtcpToolsForPrompt(specs []tools.Tool) string {
	var sb strings.Builder

	sb.WriteString("------------------------------------------------------------\n")
	sb.WriteString("UTCP TOOL REFERENCE (INPUT + OUTPUT SCHEMAS)\n")
	sb.WriteString("Use EXACT field names listed below. Do NOT invent new keys.\n")
	sb.WriteString("------------------------------------------------------------\n\n")

	for _, t := range specs {

		sb.WriteString(fmt.Sprintf("TOOL: %s\n", t.Name))
		sb.WriteString(fmt.Sprintf("DESCRIPTION: %s\n\n", t.Description))

		// -------------------------------
		// INPUT FIELD LIST
		// -------------------------------
		sb.WriteString("INPUT FIELDS (USE EXACTLY THESE KEYS):\n")

		if len(t.Inputs.Properties) == 0 {
			sb.WriteString("- (no fields)\n")
		} else {
			for key, raw := range t.Inputs.Properties {

				// Try to extract "type" from nested schema if present
				propType := "any"
				if m, ok := raw.(map[string]any); ok {
					if v, ok := m["type"]; ok {
						if s, ok := v.(string); ok {
							propType = s
						}
					}
				}

				sb.WriteString(fmt.Sprintf("- %s: %s\n", key, propType))
			}
		}

		// Required field list
		if len(t.Inputs.Required) > 0 {
			sb.WriteString("\nREQUIRED FIELDS:\n")
			for _, r := range t.Inputs.Required {
				sb.WriteString(fmt.Sprintf("- %s\n", r))
			}
		}

		sb.WriteString("\n")

		// Full JSON schema for LLM clarity
		inBytes, _ := json.MarshalIndent(t.Inputs, "", "  ")
		sb.WriteString("FULL INPUT SCHEMA (JSON):\n")
		sb.WriteString(string(inBytes))
		sb.WriteString("\n\n")

		// -------------------------------
		// OUTPUT SCHEMA
		// -------------------------------
		sb.WriteString("OUTPUT SCHEMA (EXACT SHAPE RETURNED BY TOOL):\n")

		if t.Outputs.Type != "" || len(t.Outputs.Properties) > 0 {
			outBytes, _ := json.MarshalIndent(t.Outputs, "", "  ")
			sb.WriteString(string(outBytes))
		} else {
			// Generic fallback
			sb.WriteString("{ \"result\": <any> }\n")
		}

		sb.WriteString("\n")
		sb.WriteString("------------------------------------------------------------\n\n")
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
	now := time.Now()

	a.toolMu.RLock()
	if a.toolSpecsCache != nil && (a.toolSpecsExpiry.IsZero() || now.Before(a.toolSpecsExpiry)) {
		specs := append([]tools.Tool(nil), a.toolSpecsCache...)
		a.toolMu.RUnlock()
		return specs
	}
	a.toolMu.RUnlock()

	var allSpecs []tools.Tool
	seen := make(map[string]bool)

	// 1. Local tools registered via ToolCatalog
	if a.toolCatalog != nil {
		for _, spec := range a.toolCatalog.Specs() {
			name := strings.TrimSpace(spec.Name)
			if name == "" {
				continue
			}
			key := strings.ToLower(name)
			if seen[key] {
				continue
			}

			allSpecs = append(allSpecs, tools.Tool{
				Name:        name,
				Description: spec.Description,
				Inputs: tools.ToolInputOutputSchema{
					Type:       "object",
					Properties: spec.InputSchema,
				},
			})
			seen[key] = true
		}
	}

	// 2. Built-in CodeMode tool (if available)
	if a.CodeMode != nil {
		if cmTools, err := a.CodeMode.Tools(); err == nil {
			for _, t := range cmTools {
				key := strings.ToLower(strings.TrimSpace(t.Name))
				if key == "" || seen[key] {
					continue
				}
				allSpecs = append(allSpecs, t)
				seen[key] = true
			}
		}
	}

	limit, err := strconv.Atoi(os.Getenv("utcp_search_tools_limit"))
	if err != nil {
		limit = 50
	}
	if limit == 0 {
		limit = 50
	}

	// 3. Get UTCP tool specs and merge
	if a.UTCPClient != nil {
		utcpTools, _ := a.UTCPClient.SearchTools("", limit)
		for _, tool := range utcpTools {
			key := strings.ToLower(tool.Name)
			if !seen[key] {
				allSpecs = append(allSpecs, tool)
				seen[key] = true
			}
		}
	}

	a.toolMu.Lock()
	a.toolSpecsCache = append([]tools.Tool(nil), allSpecs...)
	a.toolSpecsExpiry = now.Add(toolCacheTTL())
	a.toolPromptCache = ""
	a.toolPromptKey = ""
	a.toolPromptExpiry = time.Time{}
	a.toolMu.Unlock()

	return append([]tools.Tool(nil), allSpecs...)
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

func (a *Agent) Generate(ctx context.Context, sessionID, userInput string) (any, error) {
	trimmed := strings.TrimSpace(userInput)
	if trimmed == "" {
		return "", errors.New("user input is empty")
	}

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

	// -------------------------------------------------------------
	// 3. Chain Orchestrator (LLM decides a multi-step chain execution)
	// -------------------------------------------------------------
	if handled, output, err := a.codeChainOrchestrator(ctx, sessionID, userInput); handled {
		return output, err
	}
	// ---------------------------------------------
	// 4. TOOL ORCHESTRATOR (normal UTCP tools)
	// ---------------------------------------------
	if handled, output, err := a.toolOrchestrator(ctx, sessionID, userInput); handled {
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

	a.storeMemory(sessionID, "assistant", fmt.Sprintf("%s", completion), nil)
	return completion, nil
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

// In the toolOrchestrator function, modify the JSON parsing section:

func (a *Agent) toolOrchestrator(
	ctx context.Context,
	sessionID string,
	userInput string,
) (bool, string, error) {
	if strings.Contains(userInput, `"tool_name": "codemode.run_code"`) {
		return false, "", nil
	}

	// FAST PATH: Skip LLM call for obvious non-tool queries
	// This saves 1-3 seconds per request!
	lowerInput := strings.ToLower(strings.TrimSpace(userInput))

	// Skip if input looks like a natural question/statement
	if !a.likelyNeedsToolCall(lowerInput) {
		return false, "", nil
	}

	// Collect merged local + UTCP tools
	toolList := a.ToolSpecs()
	if len(toolList) == 0 {
		return false, "", nil
	}

	// Add codemode.run_code as a discoverable tool if CodeMode is enabled
	if a.CodeMode != nil {
		toolList = append(toolList, tools.Tool{
			Name:        "codemode.run_code",
			Description: "Execute Go code with access to UTCP tools via CallTool() and CallToolStream()",
			Inputs: tools.ToolInputOutputSchema{
				Type: "object",
				Properties: map[string]any{
					"code": map[string]any{
						"type":        "string",
						"description": "Go code to execute",
					},
					"timeout": map[string]any{
						"type":        "integer",
						"description": "Timeout in milliseconds",
					},
				},
				Required: []string{"code"},
			},
		})
	}

	// Build tool selection prompt
	toolDesc := a.cachedToolPrompt(toolList)

	choicePrompt := fmt.Sprintf(`
You are a UTCP tool selection engine.

A user asked:
%q

You have access to these UTCP tools:
%s

Think step-by-step whether ANY tool should be used.

Return ONLY a JSON object EXACTLY like this:

{
  "use_tool": true|false,
  "tool_name": "name or empty",
  "arguments": { },
  "stream": true|false
}

Return ONLY JSON. No explanations.
`, userInput, toolDesc)

	// Query LLM
	raw, err := a.model.Generate(ctx, choicePrompt)
	if err != nil {
		return false, "", err
	}

	response := strings.TrimSpace(fmt.Sprint(raw))

	// Extract and validate JSON
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

	// Handle codemode.run_code specially
	if tc.ToolName == "codemode.run_code" && a.CodeMode != nil {
		code, _ := tc.Arguments["code"].(string)
		timeout, ok := tc.Arguments["timeout"].(float64)
		if !ok {
			timeout = 200000
		}

		result, err := a.CodeMode.Execute(ctx, codemode.CodeModeArgs{
			Code:    code,
			Timeout: int(timeout),
		})
		if err != nil {
			a.storeMemory(sessionID, "assistant",
				fmt.Sprintf("CodeMode error: %v", err),
				map[string]string{"source": "codemode"},
			)
			return true, "", err
		}

		rawOut := fmt.Sprint(result)
		toonBytes, _ := gotoon.Encode(rawOut)
		full := fmt.Sprintf("%s\n\n.toon:\n%s", rawOut, string(toonBytes))

		a.storeMemory(sessionID, "assistant", full, map[string]string{
			"source": "codemode",
		})

		return true, rawOut, nil
	}

	// Validate tool exists
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

	// Execute UTCP or local tool
	result, err := a.executeTool(ctx, sessionID, tc.ToolName, tc.Arguments)
	if err != nil {
		a.storeMemory(sessionID, "assistant",
			fmt.Sprintf("tool %s error: %v", tc.ToolName, err),
			map[string]string{"source": "tool_orchestrator"},
		)
		return true, "", err
	}

	// Return RAW output
	rawOut := fmt.Sprint(result)

	// Store TOON version in memory
	toonBytes, _ := gotoon.Encode(rawOut)
	store := fmt.Sprintf("%s\n\n.toon:\n%s", rawOut, string(toonBytes))

	a.storeMemory(sessionID, "assistant", store, map[string]string{
		"tool":   tc.ToolName,
		"source": "tool_orchestrator",
	})

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

// likelyNeedsToolCall uses fast heuristics to determine if input likely needs a tool.
// This AVOIDS expensive LLM calls for obvious non-tool queries.
// EXTREMELY CONSERVATIVE: only filters pure informational questions.
func (a *Agent) likelyNeedsToolCall(lowerInput string) bool {
	// ONLY filter out EXPLICIT pure informational questions
	// Examples: "what is X?", "explain Y", "why does Z"

	// Check for pure question patterns WITHOUT any action words
	pureQuestionStarters := []string{
		"what is ", "what are ", "what does ", "what's ",
		"why is ", "why are ", "why does ", "why do ",
		"who is ", "who are ", "who was ",
		"when is ", "when was ", "when did ",
		"where is ", "where are ", "where was ",
		"explain ", "describe ", "define ",
		"tell me about ", "tell me what ",
	}

	for _, starter := range pureQuestionStarters {
		if strings.HasPrefix(lowerInput, starter) {
			// Even pure questions might need tools if they mention specific actions
			hasActionWord := strings.Contains(lowerInput, " search") ||
				strings.Contains(lowerInput, " find") ||
				strings.Contains(lowerInput, " get") ||
				strings.Contains(lowerInput, " list") ||
				strings.Contains(lowerInput, " show") ||
				strings.Contains(lowerInput, " files")

			if !hasActionWord {
				// Pure informational question - skip tool orchestration
				return false
			}
		}
	}

	// For EVERYTHING else, allow tool orchestration
	// This includes: commands, greetings, tool requests, ambiguous queries, etc.
	// Better to make an unnecessary LLM call than miss a tool request
	return true
}

func isValidSnippet(code string) bool {
	// invalid if LLM emits standalone maps like: map[value:hello world]
	if strings.Contains(code, "map[value:") {
		return false
	}

	// invalid if no __out assignment exists
	if !strings.Contains(code, "__out") {
		return false
	}

	return true
}
