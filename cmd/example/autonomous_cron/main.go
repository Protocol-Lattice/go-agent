package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/Protocol-Lattice/go-agent"
	"github.com/Protocol-Lattice/go-agent/cmd/example/internal/filestore"
	"github.com/Protocol-Lattice/go-agent/src/adk"
	"github.com/Protocol-Lattice/go-agent/src/adk/modules"
	"github.com/Protocol-Lattice/go-agent/src/memory"
	"github.com/Protocol-Lattice/go-agent/src/memory/engine"
	"github.com/Protocol-Lattice/go-agent/src/models"
	"github.com/universal-tool-calling-protocol/go-utcp"
)

func main() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	fmt.Println("=== Autonomous Agent Cron Loop with Terminal User Input ===")
	fmt.Println("Demonstrates a continuously running agent with an interactive CLI.")
	fmt.Println()

	// 1. Create UTCP Client
	client, err := utcp.NewUTCPClient(ctx, &utcp.UtcpClientConfig{}, nil, nil)
	if err != nil {
		log.Fatalf("Failed to create UTCP client: %v", err)
	}

	// 2. Create and register a specialist tool agent
	monitorModel, err := models.NewGeminiLLM(ctx, "gemini-3-flash-preview", "You are SystemMonitor")
	if err != nil {
		log.Fatalf("Failed to create monitor model: %v", err)
	}
	analyst, err := agent.New(agent.Options{
		Model:        monitorModel,
		Memory:       memory.NewSessionMemory(memory.NewMemoryBankWithStore(memory.NewInMemoryStore()), 8),
		SystemPrompt: "You monitor system health and perform background tasks.",
	})
	if err != nil {
		log.Fatalf("Failed to create tool agent: %v", err)
	}

	err = analyst.RegisterAsUTCPProvider(ctx, client, "local.monitor", "Monitors system resources")
	if err != nil {
		log.Fatalf("Failed to register tool: %v", err)
	}

	// 3. Create Orchestrator with CodeMode
	orchestratorModel, err := models.NewGeminiLLM(ctx, "gemini-3-flash-preview", "You are Orchestrator")
	if err != nil {
		log.Fatalf("Failed to create orchestrator model: %v", err)
	}
	memOpts := engine.DefaultOptions()
	kit, err := adk.New(ctx,
		adk.WithDefaultSystemPrompt("You orchestrate workflow tools. You receive prompts from users and automated cron triggers."),
		adk.WithModules(
			modules.NewModelModule("model", func(_ context.Context) (models.Agent, error) {
				return orchestratorModel, nil
			}),
			filestore.FileBackedMemoryModule("agent_memory.json", 10000, memory.AutoEmbedder(), &memOpts),
		),
		adk.WithCodeModeUtcp(client, orchestratorModel),
	)
	if err != nil {
		log.Fatalf("Failed to initialize ADK: %v", err)
	}

	orchestrator, err := kit.BuildAgent(ctx)
	if err != nil {
		log.Fatalf("Failed to build orchestrator: %v", err)
	}

	// 4. Setup channels for the loop
	userInputCh := make(chan string)

	// 5. Start a goroutine to read from stdin
	go func() {
		fmt.Println("\nType your message below (type 'exit' to quit):")
		scanner := bufio.NewScanner(os.Stdin)
		for {
			fmt.Print("\n> ")
			if !scanner.Scan() {
				break
			}
			text := strings.TrimSpace(scanner.Text())
			if text == "exit" || text == "quit" {
				cancel()
				return
			}
			if text != "" {
				userInputCh <- text
			}
		}
	}()

	// 6. Setup the cron ticker (e.g., tick every 10 seconds)
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	fmt.Println("Agent is now running in an autonomous loop...")

	sessionID := "autonomous-session-1"

	// 7. Event Loop
	for {
		select {
		case <-ctx.Done():
			fmt.Println("\nExiting autonomous loop...")
			return

		case input := <-userInputCh:
			// Ensure user prompt doesn't get messed up with background output
			fmt.Printf("\n[USER INPUT] Sending to orchestrator: %s\n", input)
			resp, generateErr := orchestrator.Generate(ctx, sessionID, input)
			if generateErr != nil {
				log.Printf("Orchestrator error on user input: %v", generateErr)
			} else {
				fmt.Printf("[AGENT] Response: %v\n> ", resp)
			}

		case t := <-ticker.C:
			// Autonomous background task execution
			fmt.Printf("\n\n--- [CRON EVENT TICK: %s] ---\n", t.Format("15:04:05"))
			backgroundPrompt := "[CRON] Perform background assessment and invoke local.monitor if necessary."
			fmt.Printf("Executing background prompt: %s\n", backgroundPrompt)

			resp, generateErr := orchestrator.Generate(ctx, sessionID, backgroundPrompt)
			if generateErr != nil {
				log.Printf("Orchestrator error on cron tick: %v", generateErr)
			} else {
				fmt.Printf("[AGENT BACKGROUND] Output: %v\n> ", resp)
			}
		}
	}
}
