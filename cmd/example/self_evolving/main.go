package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/Protocol-Lattice/go-agent"
	"github.com/Protocol-Lattice/go-agent/src/memory"
	"github.com/Protocol-Lattice/go-agent/src/memory/engine"
	"github.com/Protocol-Lattice/go-agent/src/models"
	"github.com/Protocol-Lattice/go-agent/src/selfevolve"
	"github.com/universal-tool-calling-protocol/go-utcp"
	"github.com/universal-tool-calling-protocol/go-utcp/src/plugins/codemode"
)

func main() {
	verbose := flag.Bool("verbose", false, "Enable verbose logging")
	flag.Parse()

	ctx := context.Background()

	// Setup API key
	apiKey := os.Getenv("GOOGLE_API_KEY")
	if apiKey == "" {
		apiKey = os.Getenv("GEMINI_API_KEY")
	}
	if apiKey == "" {
		log.Fatal("GOOGLE_API_KEY or GEMINI_API_KEY environment variable required")
	}

	fmt.Println("╔══════════════════════════════════════════════════════════════╗")
	fmt.Println("║     SELF-EVOLVING AGENTS WITH CODEMODE                       ║")
	fmt.Println("║     Autonomous Agent Retraining Demo                         ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════╝")
	fmt.Println()

	// 1. Create the base model for the agent
	baseModel, err := models.NewGeminiLLM(ctx, "gemini-2.0-flash-exp", "Task:")
	if err != nil {
		log.Fatalf("Failed to create base model: %v", err)
	}

	// 2. Create the optimizer model (used for evaluation and prompt optimization)
	optimizerModel, err := models.NewGeminiLLM(ctx, "gemini-2.0-flash-exp", "Evaluation:")
	if err != nil {
		log.Fatalf("Failed to create optimizer model: %v", err)
	}

	// 3. Setup memory
	memStore := memory.NewInMemoryStore()
	memOpts := engine.DefaultOptions()
	memEngine := memory.NewEngine(memStore, memOpts).WithEmbedder(memory.AutoEmbedder())
	sessionMemory := memory.NewSessionMemory(
		memory.NewMemoryBankWithStore(memStore),
		8,
	).WithEngine(memEngine)

	// 4. Setup UTCP client with codemode
	utcpClient, err := utcp.NewUTCPClient(ctx, nil, nil, nil)
	if err != nil {
		log.Fatalf("Failed to create UTCP client: %v", err)
	}

	// Initialize codemode
	codeMode := codemode.NewCodeModeUTCP(utcpClient, baseModel)

	// 5. Create the base agent
	initialPrompt := `You are a helpful assistant that provides concise, accurate summaries.
Your summaries should be clear and informative.`

	baseAgent, err := agent.New(agent.Options{
		Model:        baseModel,
		Memory:       sessionMemory,
		SystemPrompt: initialPrompt,
		ContextLimit: 8,
		UTCPClient:   utcpClient,
		CodeMode:     codeMode,
	})
	if err != nil {
		log.Fatalf("Failed to create base agent: %v", err)
	}

	// 6. Create custom evaluators
	lengthEvaluator := selfevolve.NewLLMAsJudgeEvaluator(
		optimizerModel,
		"length_check",
		"The summary should be concise (under 100 words) but comprehensive",
		0.7,
	)

	qualityEvaluator := selfevolve.NewLLMAsJudgeEvaluator(
		optimizerModel,
		"quality_check",
		"The summary should capture key points accurately and be well-structured",
		0.8,
	)

	// 7. Configure evolution
	config := &selfevolve.EvolutionConfig{
		MaxRetries:    3,
		TargetScore:   0.8,
		Evaluators:    []selfevolve.Evaluator{lengthEvaluator, qualityEvaluator},
		EnableLogging: *verbose,
		StopOnSuccess: true,
	}

	// 8. Create the self-evolving agent
	evolvingAgent := selfevolve.NewEvolvingAgent(
		baseAgent,
		optimizerModel,
		initialPrompt,
		"gemini-2.0-flash-exp",
		config,
	)

	fmt.Println("🚀 Starting Self-Evolution Loop")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println()

	// 9. Test dataset - simulate incoming tasks
	testTasks := []struct {
		name  string
		input string
	}{
		{
			name: "AI Agents Overview",
			input: `Summarize the following:
AI agents are autonomous software systems that can perceive their environment, make decisions, 
and take actions to achieve specific goals. They use large language models (LLMs) to understand 
natural language, reason about tasks, and generate responses. Modern AI agents can use tools, 
access external data sources, and even coordinate with other agents to solve complex problems. 
The field is rapidly evolving with new capabilities like self-reflection, planning, and 
autonomous retraining being actively developed.`,
		},
		{
			name: "Self-Evolving Systems",
			input: `Summarize the following:
Self-evolving AI systems represent a paradigm shift in machine learning. Instead of requiring 
manual intervention for improvements, these systems can automatically evaluate their performance, 
identify weaknesses, and optimize their behavior. This is achieved through techniques like 
LLM-as-a-judge evaluation, meta-prompting for prompt optimization, and continuous feedback loops. 
The OpenAI cookbook demonstrates how agents can autonomously retrain by running evaluations, 
collecting feedback, and generating improved prompts. This approach enables agents to adapt to 
new scenarios and improve over time without human supervision.`,
		},
		{
			name: "UTCP Protocol",
			input: `Summarize the following:
The Universal Tool Calling Protocol (UTCP) is a standardized framework for enabling AI agents 
to discover and invoke tools across different platforms and languages. It provides a unified 
interface for tool registration, discovery, and execution. UTCP supports both synchronous and 
streaming tool calls, making it suitable for a wide range of applications. The protocol enables 
agents to be exposed as tools themselves, creating powerful hierarchical and mesh architectures. 
CodeMode is a UTCP plugin that allows agents to orchestrate tool calls through generated Go code, 
providing a natural and flexible way to chain multiple tools together.`,
		},
	}

	// 10. Run the evolution loop
	sessionID := "self-evolve-demo"

	for i, task := range testTasks {
		fmt.Printf("📋 Task %d: %s\n", i+1, task.name)
		fmt.Println(strings.Repeat("─", 60))

		output, err := evolvingAgent.Generate(ctx, sessionID, task.input)
		if err != nil {
			log.Printf("❌ Task %d failed: %v\n", i+1, err)
			continue
		}

		fmt.Printf("✅ Summary: %s\n", output)
		fmt.Println()
	}

	// 11. Print evolution summary
	evolvingAgent.PrintSummary()

	// 12. Show the best prompt
	bestPrompt := evolvingAgent.GetBestPrompt()
	fmt.Println("\n🏆 BEST PERFORMING PROMPT")
	fmt.Println(strings.Repeat("=", 80))
	fmt.Printf("Version: v%d\n", bestPrompt.Version)
	fmt.Printf("Score: %.3f\n", bestPrompt.Score)
	fmt.Printf("Timestamp: %s\n", bestPrompt.Timestamp.Format("2006-01-02 15:04:05"))
	fmt.Println("\nPrompt Text:")
	fmt.Println(bestPrompt.Prompt)
	fmt.Println(strings.Repeat("=", 80))

	// 13. Demonstrate codemode integration
	fmt.Println("\n🔧 CODEMODE INTEGRATION DEMO")
	fmt.Println(strings.Repeat("=", 80))
	fmt.Println("The evolved agent can now use codemode to orchestrate UTCP tools...")
	fmt.Println()

	// Apply the best prompt to the agent
	evolvingAgent.RollbackToBest()

	// Use the evolved agent with codemode
	codemodeQuery := "Use codemode to search for tools related to 'summarization' and call the best one"
	fmt.Printf("Query: %s\n", codemodeQuery)

	result, err := baseAgent.Generate(ctx, sessionID, codemodeQuery)
	if err != nil {
		log.Printf("Codemode demo failed: %v\n", err)
	} else {
		fmt.Printf("Result: %s\n", result)
	}

	fmt.Println("\n✨ Self-Evolution Complete!")
}
