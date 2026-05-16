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
	"github.com/Protocol-Lattice/go-agent/src/adk"
	"github.com/Protocol-Lattice/go-agent/src/adk/modules"
	"github.com/Protocol-Lattice/go-agent/src/memory"
	"github.com/Protocol-Lattice/go-agent/src/memory/engine"
	"github.com/Protocol-Lattice/go-agent/src/models"
	"github.com/universal-tool-calling-protocol/go-utcp"
)

const clawSystemPrompt = `You are Claw, a persistent, autonomous AI personal assistant.
Your goal is to help the user with any task by orchestrating tools.

Rules:
1. You have access to tools registered via UTCP.
2. You have specialist agents available as UTCP tools:
   - agent.researcher: For deep research and fact-finding.
   - agent.builder: For architecture and implementation planning.
3. You run in a continuous loop, receiving both user messages and background cron events.
4. When a cron event happens, assess the current state and decide if any background action is needed.
5. Be proactive, professional, and efficient.`

func main() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	fmt.Println("=== Claw Autonomous Agent (Extended from Cron Example) ===")
	fmt.Println("Features: UTCP integration, Persistent loop.")
	fmt.Println()

	// 1. Create UTCP Client
	client, err := utcp.NewUTCPClient(ctx, &utcp.UtcpClientConfig{}, nil, nil)
	if err != nil {
		log.Fatalf("Failed to create UTCP client: %v", err)
	}

	// 2. Create and register specialist agents as UTCP tools
	if err := registerSpecialists(ctx, client); err != nil {
		log.Printf("Warning: failed to register specialist agents: %v", err)
	}

	// 3. Create Orchestrator with CodeMode
	orchestratorModel, err := models.NewGeminiLLM(ctx, "gemini-3-flash-preview", "Claw:")
	if err != nil {
		log.Fatalf("Failed to create orchestrator model: %v", err)
	}
	memOpts := engine.DefaultOptions()
	kit, err := adk.New(ctx,
		adk.WithDefaultSystemPrompt(clawSystemPrompt),
		adk.WithModules(
			modules.NewModelModule("model", func(_ context.Context) (models.Agent, error) {
				return orchestratorModel, nil
			}),
			FileBackedMemoryModule("cmd/example/claw_cron/claw_memory.json", 10000, memory.AutoEmbedder(), &memOpts),
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
		fmt.Println("\nClaw Gateway Ready. Type 'exit' to quit.")
		scanner := bufio.NewScanner(os.Stdin)
		for {
			fmt.Print("\nClaw > ")
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

	// 6. Setup the cron ticker
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	sessionID := "claw-session-main"

	// 7. Event Loop
	for {
		select {
		case <-ctx.Done():
			fmt.Println("\nShutting down OpenClaw Gateway...")
			return

		case input := <-userInputCh:
			fmt.Printf("\n[USER] %s\n", input)
			resp, generateErr := orchestrator.Generate(ctx, sessionID, input)
			if generateErr != nil {
				log.Printf("Error: %v", generateErr)
			} else {
				fmt.Printf("\n[CLAW] %v\n", resp)
			}

		case t := <-ticker.C:
			fmt.Printf("\n[SYSTEM TICK: %s] Performing background assessment...\n", t.Format("15:04:05"))
			backgroundPrompt := "[BACKGROUND TASK] Analyze current context. If any task is pending or needs proactive action, use your tools. Otherwise, return 'IDLE'."

			resp, generateErr := orchestrator.Generate(ctx, sessionID, backgroundPrompt)
			if generateErr != nil {
				log.Printf("Background Error: %v", generateErr)
			} else {
				output := fmt.Sprint(resp)
				if !strings.Contains(strings.ToUpper(output), "IDLE") {
					fmt.Printf("\n[CLAW PROACTIVE] %v\n", output)
				}
			}
		}
	}
}
func registerSpecialists(ctx context.Context, client utcp.UtcpClientInterface) error {
	// Researcher Agent
	researcherModel, err := models.NewGeminiLLM(ctx, "gemini-3-flash-preview", "Researcher:")
	if err != nil {
		return fmt.Errorf("create researcher model: %w", err)
	}
	researcher, err := agent.New(agent.Options{
		Model:        researcherModel,
		Memory:       memory.NewSessionMemory(memory.NewMemoryBankWithStore(memory.NewInMemoryStore()), 10),
		SystemPrompt: "You are a research specialist. Clarify unknowns and provide concrete facts.",
	})
	if err != nil {
		return fmt.Errorf("create researcher agent: %w", err)
	}
	if err := researcher.RegisterAsUTCPProvider(ctx, client, "agent.researcher", "Specialized research agent"); err != nil {
		return fmt.Errorf("register researcher: %w", err)
	}
	fmt.Println("Registered specialist: agent.researcher")

	// Builder Agent
	builderModel, err := models.NewGeminiLLM(ctx, "gemini-3-flash-preview", "Builder:")
	if err != nil {
		return fmt.Errorf("create builder model: %w", err)
	}
	builder, err := agent.New(agent.Options{
		Model:        builderModel,
		Memory:       memory.NewSessionMemory(memory.NewMemoryBankWithStore(memory.NewInMemoryStore()), 10),
		SystemPrompt: "You are an implementation specialist. Turn requirements into architecture and code plans.",
	})
	if err != nil {
		return fmt.Errorf("create builder agent: %w", err)
	}
	if err := builder.RegisterAsUTCPProvider(ctx, client, "agent.builder", "Specialized implementation agent"); err != nil {
		return fmt.Errorf("register builder: %w", err)
	}
	fmt.Println("Registered specialist: agent.builder")

	return nil
}
