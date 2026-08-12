package main

import (
	"context"
	"embed"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io/fs"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	agent "github.com/Protocol-Lattice/go-agent"
	"github.com/Protocol-Lattice/go-agent/src/memory"
	"github.com/Protocol-Lattice/go-agent/src/models"
	utcp "github.com/universal-tool-calling-protocol/go-utcp"
	"github.com/universal-tool-calling-protocol/go-utcp/src/plugins/codemode"
	"github.com/universal-tool-calling-protocol/go-utcp/src/tools"
)

//go:embed web/*
var webFS embed.FS

var (
	flagAddr     = flag.String("addr", ":8080", "Listen address")
	flagProvider = flag.String("provider", "dummy", "LLM provider: dummy|gemini|openai|anthropic|ollama|openrouter|vertex")
	flagModel    = flag.String("model", "local:", "Model ID")
	flagSystem   = flag.String("system", "You are a helpful assistant. You can orchestrate UTCP tools and delegate work to specialist sub-agents.", "System prompt")
	flagTimeout  = flag.Duration("timeout", 60*time.Second, "Per-request timeout")
	flagContext  = flag.Int("context", 8, "Max memory records retrieved per turn")
)

type agentRuntime struct {
	mu        sync.RWMutex
	ag        *agent.Agent
	utcp      utcp.UtcpClientInterface
	provider  string
	model     string
	subagents map[string]*agent.Agent
}

func (rt *agentRuntime) current() (*agent.Agent, utcp.UtcpClientInterface, string, string) {
	rt.mu.RLock()
	defer rt.mu.RUnlock()
	return rt.ag, rt.utcp, rt.provider, rt.model
}

func newModel(ctx context.Context, provider, model string) (models.Agent, error) {
	if provider == "dummy" {
		return models.NewDummyLLM(model), nil
	}
	return models.NewLLMProvider(ctx, provider, model, "")
}

func main() {
	flag.Parse()
	ag, client, err := buildAgent(context.Background())
	if err != nil {
		log.Fatalf("build agent: %v", err)
	}

	runtime := &agentRuntime{ag: ag, utcp: client, provider: strings.ToLower(*flagProvider), model: *flagModel, subagents: make(map[string]*agent.Agent)}
	persisted, err := loadPersistedSubagents(context.Background(), client, runtime.provider, runtime.model)
	if err != nil {
		log.Printf("warning: failed to load persisted subagents: %v", err)
	} else {
		runtime.subagents = persisted
		log.Printf("loaded %d persisted subagent(s)", len(persisted))
	}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /", handleWeb)
	mux.Handle("POST /chat", withTimeout(*flagTimeout, handleChat(runtime)))
	mux.Handle("POST /stream", withTimeout(*flagTimeout, handleStream(runtime)))
	mux.HandleFunc("GET /health", handleHealth)
	mux.HandleFunc("GET /api/skills", handleSkills(runtime))
	mux.HandleFunc("GET /api/tools", handleTools(runtime))
	mux.HandleFunc("GET /api/subagents", handleSubagents(runtime))
	mux.Handle("POST /api/subagents", withTimeout(60*time.Second, handleCreateSubagent(runtime)))
	log.Printf("gateway listening on %s (provider=%s model=%s, utcp=enabled, codemode=enabled)", *flagAddr, *flagProvider, *flagModel)
	if err := http.ListenAndServe(*flagAddr, mux); err != nil {
		log.Fatal(err)
	}
}

func handleWeb(w http.ResponseWriter, r *http.Request) {
	path := strings.TrimPrefix(r.URL.Path, "/")
	if path == "" { path = "web/index.html" }
	if !strings.HasPrefix(path, "web/") { http.NotFound(w, r); return }
	data, err := fs.ReadFile(webFS, path)
	if err != nil { http.NotFound(w, r); return }
	contentType := "text/plain; charset=utf-8"
	switch { case strings.HasSuffix(path, ".html"): contentType = "text/html; charset=utf-8"; case strings.HasSuffix(path, ".css"): contentType = "text/css; charset=utf-8"; case strings.HasSuffix(path, ".js"): contentType = "text/javascript; charset=utf-8"; case strings.HasSuffix(path, ".json"): contentType = "application/json; charset=utf-8" }
	w.Header().Set("Content-Type", contentType)
	_, _ = w.Write(data)
}

func handleSkills(runtime *agentRuntime) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) { ag, _, _, _ := runtime.current(); writeJSON(w, 200, map[string]any{"skills": ag.WebUISkills()}) }
}

func handleTools(runtime *agentRuntime) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		ag, client, _, _ := runtime.current()
		toolSpecs := ag.WebUITools()
		if client != nil { if utcpTools, err := client.SearchTools("", 100); err == nil { for _, tool := range utcpTools { toolSpecs = append(toolSpecs, utcpToolSpec(tool)) } } }
		writeJSON(w, 200, map[string]any{"tools": toolSpecs})
	}
}

func handleSubagents(runtime *agentRuntime) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) { runtime.mu.RLock(); items := make([]map[string]any, 0, len(runtime.subagents)); for name := range runtime.subagents { items = append(items, map[string]any{"name": name}) }; runtime.mu.RUnlock(); writeJSON(w, 200, map[string]any{"subagents": items}) }
}

func utcpToolSpec(tool tools.Tool) agent.ToolSpec {
	schema := map[string]any{"type": tool.Inputs.Type, "properties": tool.Inputs.Properties}
	if len(tool.Inputs.Required) > 0 { schema["required"] = tool.Inputs.Required }
	if tool.Inputs.Description != "" { schema["description"] = tool.Inputs.Description }
	if tool.Inputs.Title != "" { schema["title"] = tool.Inputs.Title }
	if tool.Inputs.Items != nil { schema["items"] = tool.Inputs.Items }
	if len(tool.Inputs.Enum) > 0 { schema["enum"] = tool.Inputs.Enum }
	if tool.Inputs.Format != "" { schema["format"] = tool.Inputs.Format }
	if tool.Inputs.Minimum != nil { schema["minimum"] = *tool.Inputs.Minimum }
	if tool.Inputs.Maximum != nil { schema["maximum"] = *tool.Inputs.Maximum }
	return agent.ToolSpec{Name: tool.Name, Description: tool.Description, InputSchema: schema}
}

type createSubagentRequest struct { Name string `json:"name"`; Description string `json:"description"`; SystemPrompt string `json:"system_prompt"`; Provider string `json:"provider"`; Model string `json:"model"` }

func handleCreateSubagent(runtime *agentRuntime) http.HandlerFunc {
	return func(w http.ResponseWriter, req *http.Request) {
		var body createSubagentRequest
		if err := json.NewDecoder(req.Body).Decode(&body); err != nil { writeError(w, 400, "invalid JSON: "+err.Error()); return }
		body.Name, body.Description, body.SystemPrompt = strings.TrimSpace(body.Name), strings.TrimSpace(body.Description), strings.TrimSpace(body.SystemPrompt)
		if body.Name == "" { writeError(w, 400, "name is required"); return }
		if strings.ContainsAny(body.Name, " .:/\\") { writeError(w, 400, "name may only contain letters, numbers, '-' and '_'"); return }
		if body.Description == "" { body.Description = "Sub-agent created from the go-agent WebUI" }
		if body.SystemPrompt == "" { body.SystemPrompt = body.Description }
		runtime.mu.Lock(); if _, exists := runtime.subagents[body.Name]; exists { runtime.mu.Unlock(); writeError(w, 409, "sub-agent already exists"); return }; provider, model, client := runtime.provider, runtime.model, runtime.utcp; runtime.mu.Unlock()
		if body.Provider != "" { provider = strings.ToLower(strings.TrimSpace(body.Provider)) }; if body.Model != "" { model = strings.TrimSpace(body.Model) }
		llm, err := newModel(req.Context(), provider, model); if err != nil { writeError(w, 400, fmt.Sprintf("create model: %v", err)); return }
		mem := memory.NewSessionMemory(memory.NewMemoryBankWithStore(memory.NewInMemoryStore()), *flagContext)
		sa, err := agent.New(agent.Options{Model: llm, Memory: mem, SystemPrompt: body.SystemPrompt, ContextLimit: *flagContext, UTCPClient: client, CodeMode: codemode.NewCodeModeUTCP(client, llm)}); if err != nil { writeError(w, 500, err.Error()); return }
		if err := sa.RegisterAsUTCPProvider(req.Context(), client, body.Name, body.Description); err != nil { writeError(w, 500, fmt.Sprintf("register UTCP provider: %v", err)); return }
		runtime.mu.Lock(); runtime.subagents[body.Name] = sa; runtime.mu.Unlock()
		writeJSON(w, 201, map[string]any{"ok": true, "name": body.Name, "description": body.Description, "provider": provider, "model": model, "tool": body.Name})
	}
}

func buildAgent(ctx context.Context) (*agent.Agent, utcp.UtcpClientInterface, error) {
	model, err := newModel(ctx, strings.ToLower(*flagProvider), *flagModel); if err != nil { return nil, nil, fmt.Errorf("create model (%s): %w", *flagProvider, err) }
	providersPath := filepath.Join("cmd", "gateway", "web", "providers.json"); if _, statErr := os.Stat(providersPath); statErr != nil { providersPath = "" }
	client, err := utcp.NewUTCPClient(ctx, &utcp.UtcpClientConfig{ProvidersFilePath: providersPath}, nil, nil); if err != nil { return nil, nil, fmt.Errorf("create UTCP client: %w", err) }
	codeMode := codemode.NewCodeModeUTCP(client, model)
	mem := memory.NewSessionMemory(memory.NewMemoryBankWithStore(memory.NewInMemoryStore()), *flagContext)
	ag, err := agent.New(agent.Options{Model: model, Memory: mem, SystemPrompt: *flagSystem, ContextLimit: *flagContext, UTCPClient: client, CodeMode: codeMode}); if err != nil { return nil, nil, err }
	return ag, client, nil
}

type chatRequest struct { Session string `json:"session"`; Message string `json:"message"` }
type chatResponse struct { Response string `json:"response"`; Session string `json:"session"` }

func handleChat(runtime *agentRuntime) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) { var req chatRequest; if err := json.NewDecoder(r.Body).Decode(&req); err != nil { writeError(w, 400, "invalid JSON: "+err.Error()); return }; if err := validateRequest(req); err != nil { writeError(w, 400, err.Error()); return }; ag, _, _, _ := runtime.current(); out, err := ag.Generate(r.Context(), req.Session, req.Message); if err != nil { writeError(w, 500, err.Error()); return }; writeJSON(w, 200, chatResponse{Response: fmt.Sprint(out), Session: req.Session}) }
}

func isCodeModeRequest(message string) bool {
	lower := strings.ToLower(strings.TrimSpace(message))
	for _, marker := range []string{"codemode", "code mode", "run code", "execute go code", "execute code"} { if strings.Contains(lower, marker) { return true } }
	return false
}

func handleStream(runtime *agentRuntime) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req chatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil { writeError(w, 400, "invalid JSON: "+err.Error()); return }
		if err := validateRequest(req); err != nil { writeError(w, 400, err.Error()); return }
		ag, _, _, _ := runtime.current()
		flusher, ok := w.(http.Flusher); if !ok { writeError(w, 500, "streaming not supported by transport"); return }
		w.Header().Set("Content-Type", "text/event-stream; charset=utf-8"); w.Header().Set("Cache-Control", "no-cache, no-store, must-revalidate"); w.Header().Set("X-Accel-Buffering", "no"); w.WriteHeader(http.StatusOK)

		codeMode := isCodeModeRequest(req.Message)
		if codeMode {
			writeSSE(w, map[string]any{"type":"workflow_start","run_id":req.Session+"-codemode","workflow":"codemode"});
			writeSSE(w, map[string]any{"type":"tool_start","run_id":req.Session+"-codemode","step_id":"codemode.run_code","tool":"codemode.run_code"}); flusher.Flush()
		}

		ch, err := ag.GenerateStream(r.Context(), req.Session, req.Message)
		if err != nil { if codeMode { writeSSE(w, map[string]any{"type":"tool_error","tool":"codemode.run_code","step_id":"codemode.run_code","error":err.Error()}); writeSSE(w, map[string]any{"type":"workflow_error","run_id":req.Session+"-codemode","error":err.Error()}) }; writeSSE(w, map[string]any{"error": err.Error()}); flusher.Flush(); return }

		for chunk := range ch {
			if chunk.Err != nil { if codeMode { writeSSE(w, map[string]any{"type":"tool_error","tool":"codemode.run_code","step_id":"codemode.run_code","error":chunk.Err.Error()}) }; writeSSE(w, map[string]any{"error": chunk.Err.Error()}); flusher.Flush(); return }
			if chunk.Delta != "" { writeSSE(w, map[string]any{"delta": chunk.Delta}); flusher.Flush() }
			if chunk.Done { if codeMode { writeSSE(w, map[string]any{"type":"tool_result","tool":"codemode.run_code","step_id":"codemode.run_code","result":chunk.FullText}); writeSSE(w, map[string]any{"type":"workflow_complete","run_id":req.Session+"-codemode"}) }; writeSSE(w, map[string]any{"done": true}); flusher.Flush(); return }
		}
		if codeMode { writeSSE(w, map[string]any{"type":"tool_result","tool":"codemode.run_code","step_id":"codemode.run_code","result":""}); writeSSE(w, map[string]any{"type":"workflow_complete","run_id":req.Session+"-codemode"}) }
		writeSSE(w, map[string]any{"done": true}); flusher.Flush()
	}
}

func writeSSE(w http.ResponseWriter, v any) { data, err := json.Marshal(v); if err != nil { return }; fmt.Fprintf(w, "data: %s\n\n", data) }
func handleHealth(w http.ResponseWriter, _ *http.Request) { writeJSON(w, 200, map[string]any{"ok": true}) }
func validateRequest(req chatRequest) error { if strings.TrimSpace(req.Session) == "" { return errors.New("session is required") }; if strings.TrimSpace(req.Message) == "" { return errors.New("message is required") }; return nil }
func withTimeout(d time.Duration, h http.Handler) http.Handler { return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { ctx, cancel := context.WithTimeout(r.Context(), d); defer cancel(); h.ServeHTTP(w, r.WithContext(ctx)) }) }
func writeJSON(w http.ResponseWriter, status int, v any) { w.Header().Set("Content-Type", "application/json"); w.WriteHeader(status); _ = json.NewEncoder(w).Encode(v) }
func writeError(w http.ResponseWriter, status int, msg string) { writeJSON(w, status, map[string]string{"error": msg}) }
