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
	"strings"
	"time"

	agent "github.com/Protocol-Lattice/go-agent"
	"github.com/Protocol-Lattice/go-agent/src/memory"
	"github.com/Protocol-Lattice/go-agent/src/models"
)

//go:embed web/*
var webFS embed.FS

var (
	flagAddr = flag.String("addr", ":8080", "Listen address")
	flagProvider = flag.String("provider", "dummy", "LLM provider: dummy|gemini|openai|anthropic|ollama")
	flagModel = flag.String("model", "local:", "Model ID for the selected provider")
	flagSystem = flag.String("system", "You are a helpful assistant.", "System prompt")
	flagTimeout = flag.Duration("timeout", 60*time.Second, "Per-request timeout")
	flagContext = flag.Int("context", 8, "Max memory records retrieved per turn")
)

func main() {
	flag.Parse()
	ag, err := buildAgent(context.Background())
	if err != nil { log.Fatalf("build agent: %v", err) }

	mux := http.NewServeMux()
	// A single GET / handler serves the UI and its embedded assets. Registering
	// a methodless /web/ pattern together with GET / conflicts in Go 1.26.
	mux.HandleFunc("GET /", handleWeb)
	mux.Handle("POST /chat", withTimeout(*flagTimeout, handleChat(ag)))
	mux.Handle("POST /stream", withTimeout(*flagTimeout, handleStream(ag)))
	mux.HandleFunc("GET /health", handleHealth)

	log.Printf("gateway listening on %s (provider=%s model=%s)", *flagAddr, *flagProvider, *flagModel)
	if err := http.ListenAndServe(*flagAddr, mux); err != nil { log.Fatal(err) }
}

func handleWeb(w http.ResponseWriter, r *http.Request) {
	path := strings.TrimPrefix(r.URL.Path, "/")
	if path == "" { path = "web/index.html" }
	if !strings.HasPrefix(path, "web/") { http.NotFound(w, r); return }
	data, err := fs.ReadFile(webFS, path)
	if err != nil { http.NotFound(w, r); return }

	contentType := "text/plain; charset=utf-8"
	switch {
	case strings.HasSuffix(path, ".html"): contentType = "text/html; charset=utf-8"
	case strings.HasSuffix(path, ".css"): contentType = "text/css; charset=utf-8"
	case strings.HasSuffix(path, ".js"): contentType = "text/javascript; charset=utf-8"
	}
	w.Header().Set("Content-Type", contentType)
	_, _ = w.Write(data)
}

func buildAgent(ctx context.Context) (*agent.Agent, error) {
	var model models.Agent
	var err error
	provider := strings.ToLower(*flagProvider)
	if provider == "dummy" { model = models.NewDummyLLM(*flagModel) } else {
		model, err = models.NewLLMProvider(ctx, provider, *flagModel, "")
		if err != nil { return nil, fmt.Errorf("create model (%s): %w", provider, err) }
	}
	mem := memory.NewSessionMemory(memory.NewMemoryBankWithStore(memory.NewInMemoryStore()), *flagContext)
	return agent.New(agent.Options{Model: model, Memory: mem, SystemPrompt: *flagSystem, ContextLimit: *flagContext})
}

type chatRequest struct { Session string `json:"session"`; Message string `json:"message"` }
type chatResponse struct { Response string `json:"response"`; Session string `json:"session"` }

func handleChat(ag *agent.Agent) http.HandlerFunc { return func(w http.ResponseWriter, r *http.Request) {
	var req chatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil { writeError(w, 400, "invalid JSON: "+err.Error()); return }
	if err := validateRequest(req); err != nil { writeError(w, 400, err.Error()); return }
	out, err := ag.Generate(r.Context(), req.Session, req.Message)
	if err != nil { writeError(w, 500, err.Error()); return }
	writeJSON(w, 200, chatResponse{Response: fmt.Sprint(out), Session: req.Session})
} }

func handleStream(ag *agent.Agent) http.HandlerFunc { return func(w http.ResponseWriter, r *http.Request) {
	var req chatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil { writeError(w, 400, "invalid JSON: "+err.Error()); return }
	if err := validateRequest(req); err != nil { writeError(w, 400, err.Error()); return }
	flusher, ok := w.(http.Flusher); if !ok { writeError(w, 500, "streaming not supported by transport"); return }
	w.Header().Set("Content-Type", "text/event-stream"); w.Header().Set("Cache-Control", "no-cache"); w.Header().Set("Connection", "keep-alive"); w.WriteHeader(http.StatusOK)
	ch, err := ag.GenerateStream(r.Context(), req.Session, req.Message)
	if err != nil { fmt.Fprintf(w, "data: error: %s\n\n", err.Error()); flusher.Flush(); return }
	for chunk := range ch {
		if chunk.Err != nil { fmt.Fprintf(w, "data: error: %s\n\n", chunk.Err.Error()); flusher.Flush(); return }
		if chunk.Done { fmt.Fprint(w, "data: [DONE]\n\n"); flusher.Flush(); return }
		if chunk.Delta != "" { fmt.Fprintf(w, "data: %s\n\n", chunk.Delta); flusher.Flush() }
	}
} }

func handleHealth(w http.ResponseWriter, _ *http.Request) { writeJSON(w, 200, map[string]any{"ok": true}) }
func validateRequest(req chatRequest) error { if strings.TrimSpace(req.Session) == "" { return errors.New("session is required") }; if strings.TrimSpace(req.Message) == "" { return errors.New("message is required") }; return nil }
func withTimeout(d time.Duration, h http.Handler) http.Handler { return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { ctx, cancel := context.WithTimeout(r.Context(), d); defer cancel(); h.ServeHTTP(w, r.WithContext(ctx)) }) }
func writeJSON(w http.ResponseWriter, status int, v any) { w.Header().Set("Content-Type", "application/json"); w.WriteHeader(status); _ = json.NewEncoder(w).Encode(v) }
func writeError(w http.ResponseWriter, status int, msg string) { writeJSON(w, status, map[string]string{"error": msg}) }
