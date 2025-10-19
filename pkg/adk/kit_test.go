package adk_test

import (
	"context"
	"strings"
	"testing"

	"github.com/Raezil/go-agent-development-kit/pkg/adk"
	kitmodules "github.com/Raezil/go-agent-development-kit/pkg/adk/modules"
	"github.com/Raezil/go-agent-development-kit/pkg/agent"
	"github.com/Raezil/go-agent-development-kit/pkg/models"
	"github.com/Raezil/go-agent-development-kit/pkg/subagents"
	"github.com/Raezil/go-agent-development-kit/pkg/tools"
)

func TestKitBuildAgent(t *testing.T) {
	t.Parallel()

	ctx := context.Background()

	researcherModel := models.NewDummyLLM("Researcher reply:")

	kitInstance, err := adk.New(ctx,
		adk.WithDefaultContextLimit(6),
		adk.WithModules(
			kitmodules.NewModelModule("coordinator", kitmodules.StaticModelProvider(models.NewDummyLLM("Coordinator:"))),
			kitmodules.InMemoryMemoryModule(4),
			kitmodules.NewToolModule("echo", kitmodules.StaticToolProvider([]agent.Tool{&tools.EchoTool{}}, nil)),
			kitmodules.NewSubAgentModule("researcher", kitmodules.StaticSubAgentProvider([]agent.SubAgent{subagents.NewResearcher(researcherModel)}, nil)),
		),
	)
	if err != nil {
		t.Fatalf("kit.New: %v", err)
	}

	built, err := kitInstance.BuildAgent(ctx)
	if err != nil {
		t.Fatalf("BuildAgent: %v", err)
	}

	response, err := built.Respond(ctx, "session", "hello world")
	if err != nil {
		t.Fatalf("Respond: %v", err)
	}
	if !strings.Contains(response, "Coordinator:") {
		t.Fatalf("expected coordinator prefix, got %q", response)
	}

	// Ensure tool registry works by issuing a command.
	toolResponse, err := built.Respond(ctx, "session", "tool:echo {\"input\": \"ping\"}")
	if err != nil {
		t.Fatalf("tool invocation failed: %v", err)
	}
	if strings.TrimSpace(toolResponse) != "ping" {
		t.Fatalf("unexpected tool response %q", toolResponse)
	}

	// Ensure sub-agent invocation path is configured.
	saResponse, err := built.Respond(ctx, "session", "subagent:researcher Summarise the impact of refactoring")
	if err != nil {
		t.Fatalf("subagent invocation failed: %v", err)
	}
	if !strings.Contains(saResponse, "Researcher reply:") {
		t.Fatalf("expected researcher prefix in %q", saResponse)
	}
}
