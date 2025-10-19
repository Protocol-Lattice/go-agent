package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/Raezil/go-agent-development-kit/pkg/agent"
	"github.com/Raezil/go-agent-development-kit/pkg/kit"
	kitmodules "github.com/Raezil/go-agent-development-kit/pkg/kit/modules"
	"github.com/Raezil/go-agent-development-kit/pkg/models"
	"github.com/Raezil/go-agent-development-kit/pkg/tools"
)

func main() {
	ctx := context.Background()

	kitInstance, err := kit.New(ctx,
		kit.WithDefaultSystemPrompt("You orchestrate a helpful assistant team."),
		kit.WithModules(
			kitmodules.NewModelModule("dummy-model", func(_ context.Context) (models.Agent, error) {
				return models.NewDummyLLM("Quickstart agent:"), nil
			}),
			kitmodules.InMemoryMemoryModule(8),
			kitmodules.NewToolModule("essentials", kitmodules.StaticToolProvider([]agent.Tool{&tools.EchoTool{}}, nil)),
		),
	)
	if err != nil {
		log.Fatalf("failed to initialise kit: %v", err)
	}

	agentInstance, err := kitInstance.BuildAgent(ctx)
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

		response, err := agentInstance.Respond(ctx, "quickstart-session", line)
		if err != nil {
			fmt.Printf("error: %v\n", err)
			continue
		}
		fmt.Printf("%s\n", response)
	}
}
