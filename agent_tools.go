package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/universal-tool-calling-protocol/go-utcp/src/plugins/codemode"
	"github.com/universal-tool-calling-protocol/go-utcp/src/tools"
)

const defaultToolCacheTTL = 30 * time.Second

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

func (a *Agent) executeTool(
	ctx context.Context,
	sessionID, toolName string,
	args map[string]any,
) (any, error) {
	if args == nil {
		args = map[string]any{}
	}

	// 0. Built-in CodeMode tool.
	// ToolSpecs exposes codemode.run_code from a.CodeMode, so execution must
	// also route it here instead of forwarding it to the UTCP client.
	if toolName == codemode.CodeModeToolName || toolName == "codemode.run_code" {
		if a.CodeMode == nil {
			return nil, fmt.Errorf("codemode is not configured")
		}
		if !a.AllowUnsafeTools {
			return nil, fmt.Errorf("unauthorized tool execution: %s is restricted", toolName)
		}

		code, ok := args["code"].(string)
		if !ok || strings.TrimSpace(code) == "" {
			return nil, fmt.Errorf("codemode.run_code requires non-empty string field: code")
		}

		timeout := 30000
		switch v := args["timeout"].(type) {
		case int:
			timeout = v
		case int64:
			timeout = int(v)
		case float64:
			timeout = int(v)
		case json.Number:
			if n, err := v.Int64(); err == nil {
				timeout = int(n)
			}
		}

		result, err := a.CodeMode.Execute(ctx, codemode.CodeModeArgs{
			Code:    code,
			Timeout: timeout,
		})
		if err != nil {
			return nil, err
		}
		if strings.TrimSpace(result.Stderr) != "" {
			return nil, fmt.Errorf("codemode script produced stderr: %s", result.Stderr)
		}

		if result.Stdout != "" {
			return map[string]any{
				"value":  result.Value,
				"stdout": result.Stdout,
			}, nil
		}

		return result.Value, nil
	}

	// 1. Locally registered tool.
	if tool, _, ok := a.lookupTool(toolName); ok {
		response, err := tool.Invoke(ctx, ToolRequest{
			SessionID: sessionID,
			Arguments: args,
		})
		if err != nil {
			return nil, err
		}
		return response.Content, nil
	}

	// 2. Remote UTCP tool.
	if a.UTCPClient != nil {
		if streamFlag, ok := args["stream"].(bool); ok && streamFlag {
			stream, err := a.UTCPClient.CallToolStream(ctx, toolName, args)
			if err != nil {
				return nil, err
			}
			if stream == nil {
				return nil, fmt.Errorf("CallToolStream returned nil stream for %s", toolName)
			}

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

		return a.UTCPClient.CallTool(ctx, toolName, args)
	}

	return nil, fmt.Errorf("unknown tool: %s", toolName)
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

			inputs := tools.ToolInputOutputSchema{Type: "object"}
			if encoded, err := json.Marshal(spec.InputSchema); err == nil {
				_ = json.Unmarshal(encoded, &inputs)
			}
			if strings.TrimSpace(inputs.Type) == "" {
				inputs.Type = "object"
			}

			allSpecs = append(allSpecs, tools.Tool{
				Name:        name,
				Description: spec.Description,
				Inputs:      inputs,
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

// likelyNeedsToolCall uses fast heuristics to determine if input likely needs a tool.
// This AVOIDS expensive LLM calls for obvious non-tool queries.
// EXTREMELY CONSERVATIVE: only filters pure informational questions.
func (a *Agent) likelyNeedsToolCall(lowerInput string) bool {
	// 0. Skip for very short inputs or greetings
	if len(lowerInput) < 2 {
		return false
	}
	greetings := []string{"hello", "hi", "hey", "good morning", "good afternoon", "thanks", "thank you"}
	for _, g := range greetings {
		if lowerInput == g || strings.HasPrefix(lowerInput, g+" ") || strings.HasPrefix(lowerInput, g+",") {
			return false
		}
	}

	// 1. Check for pure informational question patterns WITHOUT any action words
	pureQuestionStarters := []string{
		"what is ", "what are ", "what does ", "what's ",
		"why is ", "why are ", "why does ", "why do ",
		"who is ", "who are ", "who was ",
		"when is ", "when was ", "when did ",
		"where is ", "where are ", "where was ",
		"how is ", "how are ", "how does ",
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
				strings.Contains(lowerInput, " files") ||
				strings.Contains(lowerInput, " run") ||
				strings.Contains(lowerInput, " exec")

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
