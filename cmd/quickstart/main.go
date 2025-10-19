package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/Raezil/go-agent-development-kit/pkg/adk"
	adkmodules "github.com/Raezil/go-agent-development-kit/pkg/adk/modules"
	"github.com/Raezil/go-agent-development-kit/pkg/agent"
	"github.com/Raezil/go-agent-development-kit/pkg/memory"
	"github.com/Raezil/go-agent-development-kit/pkg/models"
	"github.com/Raezil/go-agent-development-kit/pkg/tools"
)

func main() {
	qdrantURL := flag.String("qdrant-url", "http://localhost:6333", "Qdrant base URL")
	qdrantCollection := flag.String("qdrant-collection", "adk_memories", "Qdrant collection name")
	flag.Parse()
	ctx := context.Background()

	adk, err := adk.New(ctx,
		adk.WithDefaultSystemPrompt("You orchestrate a helpful assistant team."),
		adk.WithModules(
			adkmodules.NewModelModule("gemini-model", func(_ context.Context) (models.Agent, error) {
				researcherModel, err := models.NewGeminiLLM(ctx, "gemini-2.5-pro", "Research summary:")
				if err != nil {
					return nil, err
				}
				return researcherModel, nil
			}),
			adkmodules.InQdrantMemory(100000, *qdrantURL, *qdrantCollection, memory.AutoEmbedder()),
			adkmodules.NewToolModule("essentials", adkmodules.StaticToolProvider([]agent.Tool{&tools.EchoTool{}}, nil)),
		),
	)
	if err != nil {
		log.Fatalf("failed to initialise kit: %v", err)
	}

	agent, err := adk.BuildAgent(ctx)
	if err != nil {
		log.Fatalf("failed to build agent: %v", err)
	}

	fmt.Println("Agent Development Kit quickstart. Type a message and press enter (empty line exits).")
	reader := bufio.NewReader(os.Stdin)

	for {
		fmt.Print("> ")
		line, err := reader.ReadString('\n')
		if err != nil {
			log.Fatalf("read input: %v", err)
		}
		line = strings.TrimSpace(line)
		if line == "" {
			fmt.Println("Goodbye!")
			return
		}

		response, err := agent.Respond(ctx, "quickstart-session", line)
		if err != nil {
			fmt.Printf("error: %v\n", err)
			continue
		}
		fmt.Printf("%s\n", response)
	}
}
