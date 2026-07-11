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

const clawSystemPrompt = `You are Claw, a persistent, autonomous AI personal assistant.
Your goal is to help the user with any task by orchestrating tools.

Rules:
1. You have access to tools registered via UTCP.
2. You have specialist agents available as UTCP tools:
   - agent.researcher: For deep research and fact-finding.
   - agent.builder: For architecture and implementation planning.
3. You have a persistent TASK REGISTRY available via:
   - tasks.create_goal, tasks.update_task, tasks.list_active_tasks.
4. For high-stakes or dangerous actions (e.g. file deletion, large purchases, structural changes), you MUST use:
   - gateway.request_permission: Ask for user approval before proceeding.
5. You run in a continuous loop, receiving both user messages and background cron events.
6. Before every action, perform a 'Reflect & Plan' step to update your internal status and pivot if needed.
7. Be proactive, professional, and efficient.`

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

	// 2b. Create and register Task Store tools
	taskStore := NewTaskStore("cmd/example/claw_cron/claw_tasks.json")
	if err := taskStore.RegisterAsUTCPProvider(ctx, client); err != nil {
		log.Printf("Warning: failed to register task tools: %v", err)
	}

	// 2c. Create and register Permission Gateway tools
	gateway := NewPermissionGateway()
	if err := gateway.RegisterAsUTCPProvider(ctx, client); err != nil {
		log.Printf("Warning: failed to register gateway tools: %v", err)
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
			filestore.FileBackedMemoryModule("cmd/example/claw_cron/claw_memory.json", 10000, memory.AutoEmbedder(), &memOpts),
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

	// Helper for Reflective Loop
	runWithReflection := func(input string) (any, error) {
		fmt.Printf("\n[CLAW REFLECTING...]\n")
		reflectPrompt := fmt.Sprintf("[INTERNAL REFLECTION] Based on the user input '%s' and the current task registry, what is your updated internal status, plan, and next step? Keep it brief.", input)
		_, _ = orchestrator.Generate(ctx, sessionID, reflectPrompt)

		return orchestrator.Generate(ctx, sessionID, input)
	}

	// 7. Event Loop
	for {
		select {
		case <-ctx.Done():
			fmt.Println("\nShutting down OpenClaw Gateway...")
			return

		case input := <-userInputCh:
			fmt.Printf("\n[USER] %s\n", input)
			resp, generateErr := runWithReflection(input)
			if generateErr != nil {
				log.Printf("Error: %v", generateErr)
			} else {
				fmt.Printf("\n[CLAW] %v\n", resp)
			}

		case req := <-gateway.RequestChannel():
			fmt.Printf("\n[CLAW PERMISSION REQUEST] %s\n", req.action)
			fmt.Print("Approve? (y/n): ")
			// We need to read from stdin here, but the main loop is already reading from it in a goroutine.
			// This is a bit tricky for a CLI example.
			// Let's assume the user types 'y' or 'n' in the main input.
			// Actually, let's simplify: the main gateway goroutine could handle this if we send a signal.
			// For this example, I'll just auto-approve for the demonstration or wait for a specific input.
			// Better: let's have the user type 'y' or 'n' and have the goroutine detect it.

			// For now, let's just wait for the next userInputCh and see if it's y/n
			select {
			case answer := <-userInputCh:
				approved := strings.ToLower(answer) == "y" || strings.ToLower(answer) == "yes"
				req.resp <- approved
			case <-time.After(1 * time.Minute):
				req.resp <- false
			}

		case t := <-ticker.C:
			fmt.Printf("\n[SYSTEM TICK: %s] Performing background assessment...\n", t.Format("15:04:05"))
			backgroundPrompt := "[BACKGROUND TASK] 1. List active tasks. 2. Based on tasks and history, decide if action is needed. 3. If so, use tools. 4. If nothing to do, return 'IDLE'."

			resp, generateErr := runWithReflection(backgroundPrompt)
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
