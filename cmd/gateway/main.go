// cmd/gateway — HTTP gateway that exposes a go-agent as a REST API.
//
// Endpoints:
//
//	POST /chat        synchronous chat: {session, message} → {response}
//	POST /stream      SSE streaming:    {session, message} → text/event-stream
//	GET  /health      liveness check:   → {ok: true}
//
// Examples (no API key required — uses dummy model by default):
//
//	go run .
//	curl -s -X POST http://localhost:8080/chat \
//	     -H "Content-Type: application/json" \
//	     -d '{"session":"alice","message":"Hello!"}'
//
//	# Switch to a real provider:
//	export GOOGLE_API_KEY=...
//	go run . -provider gemini -model gemini-2.5-flash
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	agent "github.com/Protocol-Lattice/go-agent"
	"github.com/Protocol-Lattice/go-agent/src/memory"
	"github.com/Protocol-Lattice/go-agent/src/models"
)

var (
	flagAddr     = flag.String("addr", ":8080", "Listen address")
	flagProvider = flag.String("provider", "dummy", "LLM provider: dummy|gemini|openai|anthropic|ollama")
	flagModel    = flag.String("model", "local:", "Model ID for the selected provider")
	flagSystem   = flag.String("system", "You are a helpful assistant.", "System prompt")
	flagTimeout  = flag.Duration("timeout", 60*time.Second, "Per-request timeout")
	flagContext  = flag.Int("context", 8, "Max memory records retrieved per turn")
)

func main() {
	flag.Parse()

	ctx := context.Background()

	ag, err := buildAgent(ctx)
	if err != nil {
		log.Fatalf("build agent: %v", err)
	}

	mux := http.NewServeMux()
	mux.Handle("POST /chat", withTimeout(*flagTimeout, handleChat(ag)))
	mux.Handle("POST /stream", withTimeout(*flagTimeout, handleStream(ag)))
	mux.HandleFunc("GET /health", handleHealth)

	log.Printf("gateway listening on %s (provider=%s model=%s)", *flagAddr, *flagProvider, *flagModel)
	if err := http.ListenAndServe(*flagAddr, mux); err != nil {
		log.Fatal(err)
	}
}

// buildAgent constructs the agent with in-memory storage.
// Swap modules.InMemoryMemoryModule for InPostgresMemory / InQdrantMemory
// to add persistence without changing any other code.
func buildAgent(ctx context.Context) (*agent.Agent, error) {
	var model models.Agent
	var err error

	provider := strings.ToLower(*flagProvider)
	if provider == "dummy" {
		model = models.NewDummyLLM(*flagModel)
	} else {
		model, err = models.NewLLMProvider(ctx, provider, *flagModel, "")
		if err != nil {
			return nil, fmt.Errorf("create model (%s): %w", provider, err)
		}
	}

	mem := memory.NewSessionMemory(
		memory.NewMemoryBankWithStore(memory.NewInMemoryStore()),
		*flagContext,
	)

	return agent.New(agent.Options{
		Model:        model,
		Memory:       mem,
		SystemPrompt: *flagSystem,
		ContextLimit: *flagContext,
	})
}

// chatRequest is the JSON body for POST /chat and POST /stream.
type chatRequest struct {
	Session string `json:"session"`
	Message string `json:"message"`
}

// chatResponse is the JSON body returned by POST /chat.
type chatResponse struct {
	Response string `json:"response"`
	Session  string `json:"session"`
}

func handleChat(ag *agent.Agent) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req chatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeError(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
			return
		}
		if err := validateRequest(req); err != nil {
			writeError(w, http.StatusBadRequest, err.Error())
			return
		}

		out, err := ag.Generate(r.Context(), req.Session, req.Message)
		if err != nil {
			writeError(w, http.StatusInternalServerError, err.Error())
			return
		}

		writeJSON(w, http.StatusOK, chatResponse{
			Response: fmt.Sprint(out),
			Session:  req.Session,
		})
	}
}

// handleStream serves Server-Sent Events so the client receives tokens as they arrive.
//
// Event format:
//
//	data: <token>\n\n       — incremental text chunk
//	data: [DONE]\n\n        — stream finished
func handleStream(ag *agent.Agent) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req chatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeError(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
			return
		}
		if err := validateRequest(req); err != nil {
			writeError(w, http.StatusBadRequest, err.Error())
			return
		}

		flusher, ok := w.(http.Flusher)
		if !ok {
			writeError(w, http.StatusInternalServerError, "streaming not supported by transport")
			return
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.WriteHeader(http.StatusOK)

		ch, err := ag.GenerateStream(r.Context(), req.Session, req.Message)
		if err != nil {
			fmt.Fprintf(w, "data: error: %s\n\n", err.Error())
			flusher.Flush()
			return
		}

		for chunk := range ch {
			if chunk.Err != nil {
				fmt.Fprintf(w, "data: error: %s\n\n", chunk.Err.Error())
				flusher.Flush()
				return
			}
			if chunk.Done {
				fmt.Fprintf(w, "data: [DONE]\n\n")
				flusher.Flush()
				return
			}
			if chunk.Delta != "" {
				fmt.Fprintf(w, "data: %s\n\n", chunk.Delta)
				flusher.Flush()
			}
		}
	}
}

func handleHealth(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"ok": true})
}

func validateRequest(req chatRequest) error {
	if strings.TrimSpace(req.Session) == "" {
		return errors.New("session is required")
	}
	if strings.TrimSpace(req.Message) == "" {
		return errors.New("message is required")
	}
	return nil
}

func withTimeout(d time.Duration, h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ctx, cancel := context.WithTimeout(r.Context(), d)
		defer cancel()
		h.ServeHTTP(w, r.WithContext(ctx))
	})
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}
