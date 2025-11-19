package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"

	adkmodules "github.com/Protocol-Lattice/go-agent/src/adk/modules"

	"net/http"
	"os"
	"time"

	"github.com/Protocol-Lattice/go-agent/src/adk"
	"github.com/Protocol-Lattice/go-agent/src/memory"
	"github.com/Protocol-Lattice/go-agent/src/memory/engine"
	"github.com/Protocol-Lattice/go-agent/src/models"
	"github.com/Protocol-Lattice/go-agent/src/subagents"
	"github.com/universal-tool-calling-protocol/go-utcp"
)

var discovered bool

func startServer(addr string) {
	http.HandleFunc("/tools", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		raw, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, fmt.Sprintf("failed to read body: %v", err), http.StatusBadRequest)
			return
		}
		defer r.Body.Close()

		// Discovery: first empty-body => discovery
		if len(raw) == 0 && !discovered {
			discovered = true
			// Read discovery response from tools.json
			data, err := os.ReadFile("tools.json")
			if err != nil {
				log.Printf("Failed to read tools.json: %v", err)
				return
			}
			var discoveryResponse map[string]interface{}
			if err := json.Unmarshal(data, &discoveryResponse); err != nil {
				log.Printf("Failed to unmarshal tools.json: %v", err)
				return
			}

			w.Header().Set("Content-Type", "application/json")
			if err := json.NewEncoder(w).Encode(discoveryResponse); err != nil {
				log.Printf("Failed to encode discovery response: %v", err)
			}
			return
		}

		// Try to parse the JSON
		var probe map[string]interface{}
		if err := json.Unmarshal(raw, &probe); err != nil {
			http.Error(w, fmt.Sprintf("invalid JSON: %v", err), http.StatusBadRequest)
			return
		}

		// Standard tool call (has "tool" field)
		if toolName, hasToolField := probe["tool"].(string); hasToolField && toolName != "" {
			// Empty-body after discovery => timestamp call
			if len(raw) == 0 {
				log.Printf("Empty body – timestamp call")
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(map[string]any{"result": time.Now().Format(time.RFC3339)})
				return
			}

			var req struct {
				Tool string                 `json:"tool"`
				Args map[string]interface{} `json:"args"`
			}
			if err := json.Unmarshal(raw, &req); err != nil {
				http.Error(w, fmt.Sprintf("invalid JSON for tool call: %v", err), http.StatusBadRequest)
				return
			}

			log.Printf("Standard tool call: %s with args: %v", req.Tool, req.Args)
			w.Header().Set("Content-Type", "application/json")

			switch req.Tool {

			case "http.echo":
				msg, ok := req.Args["message"].(string)
				if !ok {
					http.Error(w, "missing 'message' argument for http.echo", http.StatusBadRequest)
					return
				}
				json.NewEncoder(w).Encode(map[string]any{"result": msg})

			case "http.timestamp":
				json.NewEncoder(w).Encode(map[string]any{"result": time.Now().Format(time.RFC3339)})

			case "http.math.add":
				a, aOk := req.Args["a"].(float64)
				b, bOk := req.Args["b"].(float64)
				if !aOk || !bOk {
					http.Error(w, "missing 'a' or 'b' arguments for http.math.add", http.StatusBadRequest)
					return
				}
				json.NewEncoder(w).Encode(map[string]any{"result": a + b})

			case "http.math.multiply":
				a, aOk := req.Args["a"].(float64)
				b, bOk := req.Args["b"].(float64)
				if !aOk || !bOk {
					http.Error(w, "missing 'a' or 'b' arguments for http.math.multiply", http.StatusBadRequest)
					return
				}
				json.NewEncoder(w).Encode(map[string]any{"result": a * b})

			case "http.string.concat":
				prefix, pOk := req.Args["prefix"].(string)
				value, vOk := req.Args["value"].(string)
				if !pOk || !vOk {
					http.Error(w, "missing 'prefix' or 'value' arguments for http.string.concat", http.StatusBadRequest)
					return
				}
				json.NewEncoder(w).Encode(map[string]any{"result": prefix + value})

			default:
				http.Error(w, fmt.Sprintf("unknown tool: %s", req.Tool), http.StatusBadRequest)
				return
			}

			return
		}

		// Direct echo call (has "message" field)
		if _, hasMessage := probe["message"]; hasMessage {
			log.Printf("Direct echo call with args: %v", probe)
			w.Header().Set("Content-Type", "application/json")
			msg, _ := probe["message"].(string)
			json.NewEncoder(w).Encode(map[string]any{"result": msg})
			return
		}

		// Handle math.add call from UTCP http transport (body is just args)
		if _, hasA := probe["a"]; hasA {
			if _, hasB := probe["b"]; hasB {
				log.Printf("Direct math.add call with args: %v", probe)
				a, _ := probe["a"].(float64)
				b, _ := probe["b"].(float64)
				json.NewEncoder(w).Encode(map[string]any{"sum": a + b})
				return
			}
		}

		// Unknown request format
		log.Printf("Unknown request format: %v", probe)
		http.Error(w, "unknown request format", http.StatusBadRequest)
	})

	log.Printf("HTTP mock server on %s", addr)
	log.Fatal(http.ListenAndServe(addr, nil))

}

func main() {
	go startServer(":8080")
	time.Sleep(200 * time.Millisecond)
	ctx := context.Background()
	cfg := &utcp.UtcpClientConfig{ProvidersFilePath: "provider.json"}
	client, err := utcp.NewUTCPClient(ctx, cfg, nil, nil)
	if err != nil {
		log.Fatalf("client error: %v", err)
	}

	modelName := flag.String("model", "gemini-2.5-flash", "Gemini model ID")
	flag.Parse()

	// --- Runtime (shared) ---
	researcherModel, err := models.NewGeminiLLM(ctx, *modelName, "")
	if err != nil {
		log.Fatalf("failed to create researcher model: %v", err)
	}
	memOpts := engine.DefaultOptions()
	model, err := models.NewGeminiLLM(ctx, *modelName, "")
	if err != nil {
		log.Fatalf("failed to create model: %v", err)
	}
	kit, err := adk.New(ctx,
		adk.WithDefaultSystemPrompt(""),
		adk.WithSubAgents(subagents.NewResearcher(researcherModel)),
		adk.WithModules(
			adkmodules.NewModelModule("gemini-model", func(_ context.Context) (models.Agent, error) {
				return model, nil
			}),
			adkmodules.InMemoryMemoryModule(100000, memory.AutoEmbedder(), &memOpts),
		),
		adk.WithCodeModeUtcp(client, model),
		adk.WithUTCP(client),
	)

	if err != nil {
		log.Fatalf("failed to initialise kit: %v", err)
	}
	ag, err := kit.BuildAgent(ctx)
	if err != nil {
		log.Fatalf("failed to initialise kit: %v", err)
	}
	resp, err := ag.Generate(ctx, "test", prompt)
	if err != nil {
		log.Fatalf("failed to generate: %v", err)
	}
	fmt.Println(resp)
}

var prompt = `
Echo a greeting: Call the http.echo tool with the message "hello world"\n
Get the current time: Call the http.timestamp tool to retrieve the current server timestamp\n
Add two numbers: Use the http.math.add tool to add 5 and 7 together\n
Multiply the result: Take the sum from step 3 and multiply it by 3 using the http.math.multiply tool\n
Format as a string: Use the http.string.concat tool to prepend the text "Number: " to the multiplication result from step 4\n
`
