package adk_test

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/Raezil/go-agent-development-kit/pkg/adk"
	kitmodules "github.com/Raezil/go-agent-development-kit/pkg/adk/modules"
	"github.com/Raezil/go-agent-development-kit/pkg/agent"
	"github.com/Raezil/go-agent-development-kit/pkg/helpers"
	"github.com/Raezil/go-agent-development-kit/pkg/memory"
	"github.com/Raezil/go-agent-development-kit/pkg/models"
	"github.com/Raezil/go-agent-development-kit/pkg/subagents"
	"github.com/Raezil/go-agent-development-kit/pkg/tools"
)

const (
	DefaultMemorySimWeight        = 0.55
	DefaultMemoryImportanceWeight = 0.25
	DefaultMemoryRecencyWeight    = 0.15
	DefaultMemorySourceWeight     = 0.05
	DefaultMemoryMMRLambda        = 0.7
	DefaultMemoryClusterSim       = 0.83
	DefaultMemoryDriftThreshold   = 0.90
	DefaultMemoryDuplicateSim     = 0.97
	DefaultMemoryMaxSize          = 200000
)

// Default durations are kept as variables because time.Duration is not allowed as a const type.
var (
	DefaultMemoryHalfLife = 72 * time.Hour
	DefaultMemoryTTL      = 720 * time.Hour
)

// Optional source boost and toggles.
const (
	DefaultMemorySourceBoost      = ""
	DefaultMemoryDisableSummaries = false
)

func DefaultMemoryOptions() memory.Options {
	return memory.Options{
		Weights: memory.ScoreWeights{
			Similarity: DefaultMemorySimWeight,
			Importance: DefaultMemoryImportanceWeight,
			Recency:    DefaultMemoryRecencyWeight,
			Source:     DefaultMemorySourceWeight,
		},
		LambdaMMR:           DefaultMemoryMMRLambda,
		HalfLife:            DefaultMemoryHalfLife,
		ClusterSimilarity:   DefaultMemoryClusterSim,
		DriftThreshold:      DefaultMemoryDriftThreshold,
		DuplicateSimilarity: DefaultMemoryDuplicateSim,
		TTL:                 DefaultMemoryTTL,
		MaxSize:             DefaultMemoryMaxSize,
		SourceBoost:         helpers.ParseSourceBoostFlag(DefaultMemorySourceBoost),
		EnableSummaries:     !DefaultMemoryDisableSummaries,
	}
}

func TestKitBuildAgent(t *testing.T) {
	t.Parallel()

	ctx := context.Background()

	researcherModel := models.NewDummyLLM("Researcher reply:")

	memoryOpts := DefaultMemoryOptions()
	kitInstance, err := adk.New(ctx,
		adk.WithDefaultContextLimit(6),
		adk.WithModules(
			kitmodules.NewModelModule("coordinator", kitmodules.StaticModelProvider(models.NewDummyLLM("Coordinator:"))),
			kitmodules.InMemoryMemoryModule(4, memory.DummyEmbedder{}, &memoryOpts),
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
