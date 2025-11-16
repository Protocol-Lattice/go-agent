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

		// Empty-body after discovery => timestamp call
		if len(raw) == 0 {
			log.Printf("Empty body – timestamp call")
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]any{"result": time.Now().Format(time.RFC3339)})
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

			case "echo":
				msg, _ := req.Args["message"].(string)
				json.NewEncoder(w).Encode(map[string]any{"result": msg})

			case "timestamp":
				json.NewEncoder(w).Encode(map[string]any{"result": time.Now().Format(time.RFC3339)})

			case "math.add":
				a, _ := req.Args["a"].(float64)
				b, _ := req.Args["b"].(float64)
				json.NewEncoder(w).Encode(map[string]any{"sum": a + b})

			case "math.multiply":
				a, _ := req.Args["a"].(float64)
				b, _ := req.Args["b"].(float64)
				json.NewEncoder(w).Encode(map[string]any{"product": a * b})

			case "string.concat":
				prefix, _ := req.Args["prefix"].(string)
				value, _ := req.Args["value"].(string)
				json.NewEncoder(w).Encode(map[string]any{"result": prefix + value})

			case "stream.echo":
				// streaming: return chunks in sequence
				// You can simplify and return single chunk
				json.NewEncoder(w).Encode(map[string]any{
					"stream": []any{"A", "B", "C"},
				})

			default:
				http.Error(w, "unknown tool", http.StatusNotFound)
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
	researcherModel, err := models.NewGeminiLLM(ctx, *modelName, "Research summary:")
	if err != nil {
		log.Fatalf("failed to create researcher model: %v", err)
	}
	memOpts := engine.DefaultOptions()

	kit, err := adk.New(ctx,
		adk.WithDefaultSystemPrompt(""),
		adk.WithSubAgents(subagents.NewResearcher(researcherModel)),
		adk.WithModules(
			adkmodules.NewModelModule("gemini-model", func(_ context.Context) (models.Agent, error) {
				return models.NewGeminiLLM(ctx, *modelName, "")
			}),
			adkmodules.InMemoryMemoryModule(100000, memory.AutoEmbedder(), &memOpts),
		),
		adk.WithCodeModeUtcp(client),
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
Echo a greeting: Call the http.echo tool with the message "hello world"
Get the current time: Call the http.timestamp tool to retrieve the current server timestamp
Add two numbers: Use the http.math.add tool to add 5 and 7 together
Multiply the result: Take the sum from step 3 and multiply it by 3 using the http.math.multiply tool
Format as a string: Use the http.string.concat tool to prepend the text "Number: " to the multiplication result from step 4
`
