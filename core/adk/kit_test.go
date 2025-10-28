package adk_test

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/Protocol-Lattice/agent/core/adk"
	kitmodules "github.com/Protocol-Lattice/agent/core/adk/modules"
	agent "github.com/Protocol-Lattice/agent/core/agentic"
	"github.com/Protocol-Lattice/agent/core/helpers"
	"github.com/Protocol-Lattice/agent/core/memory"
	"github.com/Protocol-Lattice/agent/core/models"
	"github.com/Protocol-Lattice/agent/core/subagents"
	"github.com/Protocol-Lattice/agent/core/tools"
)

const (
	DefaultMemorySimWeight        = 0.45
	DefaultMemoryKeywordWeight    = 0.20
	DefaultMemoryImportanceWeight = 0.20
	DefaultMemoryRecencyWeight    = 0.10
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
			Keywords:   DefaultMemoryKeywordWeight,
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

	response, err := built.Generate(ctx, "session", "hello world")
	if err != nil {
		t.Fatalf("Respond: %v", err)
	}
	if !strings.Contains(response, "Coordinator:") {
		t.Fatalf("expected coordinator prefix, got %q", response)
	}

	// Ensure tool registry works by issuing a command.
	toolResponse, err := built.Generate(ctx, "session", "tool:echo {\"input\": \"ping\"}")
	if err != nil {
		t.Fatalf("tool invocation failed: %v", err)
	}
	if strings.TrimSpace(toolResponse) != "ping" {
		t.Fatalf("unexpected tool response %q", toolResponse)
	}

	// Ensure sub-agent invocation path is configured.
	saResponse, err := built.Generate(ctx, "session", "subagent:researcher Summarise the impact of refactoring")
	if err != nil {
		t.Fatalf("subagent invocation failed: %v", err)
	}
	if !strings.Contains(saResponse, "Researcher reply:") {
		t.Fatalf("expected researcher prefix in %q", saResponse)
	}
}

func TestKitWithSubAgentsOption(t *testing.T) {
	t.Parallel()

	ctx := context.Background()

	researcherModel := models.NewDummyLLM("Researcher reply:")
	memoryOpts := DefaultMemoryOptions()

	kitInstance, err := adk.New(ctx,
		adk.WithDefaultContextLimit(6),
		adk.WithModules(
			kitmodules.NewModelModule("coordinator", kitmodules.StaticModelProvider(models.NewDummyLLM("Coordinator:"))),
			kitmodules.InMemoryMemoryModule(4, memory.DummyEmbedder{}, &memoryOpts),
		),
		adk.WithSubAgents(subagents.NewResearcher(researcherModel)),
	)
	if err != nil {
		t.Fatalf("kit.New: %v", err)
	}

	built, err := kitInstance.BuildAgent(ctx)
	if err != nil {
		t.Fatalf("BuildAgent: %v", err)
	}

	subAgents := built.SubAgents()
	if len(subAgents) != 1 {
		t.Fatalf("expected 1 subagent, got %d", len(subAgents))
	}
	if name := subAgents[0].Name(); name != "researcher" {
		t.Fatalf("unexpected subagent name %q", name)
	}
}

func TestKitSharedSession(t *testing.T) {
	t.Parallel()

	ctx := context.Background()

	memoryOpts := DefaultMemoryOptions()
	kitInstance, err := adk.New(ctx,
		adk.WithModules(
			kitmodules.NewModelModule("coordinator", kitmodules.StaticModelProvider(models.NewDummyLLM("Coordinator:"))),
			kitmodules.InMemoryMemoryModule(4, memory.DummyEmbedder{}, &memoryOpts),
		),
	)
	if err != nil {
		t.Fatalf("kit.New: %v", err)
	}

	provider := kitInstance.MemoryProvider()
	if provider == nil {
		t.Fatalf("expected memory provider")
	}
	bundle, err := provider(ctx)
	if err != nil {
		t.Fatalf("memory provider: %v", err)
	}
	if bundle.Session == nil {
		t.Fatalf("memory bundle missing session")
	}

	if err := bundle.Session.Spaces.Grant("team:shared", "agent:alpha", memory.SpaceRoleAdmin, 0); err != nil {
		t.Fatalf("grant alpha: %v", err)
	}
	if err := bundle.Session.Spaces.Grant("team:shared", "agent:beta", memory.SpaceRoleAdmin, 0); err != nil {
		t.Fatalf("grant beta: %v", err)
	}

	alpha, err := kitInstance.NewSharedSession(ctx, "agent:alpha", "team:shared")
	if err != nil {
		t.Fatalf("SharedSession alpha: %v", err)
	}
	beta, err := kitInstance.NewSharedSession(ctx, "agent:beta", "team:shared")
	if err != nil {
		t.Fatalf("SharedSession beta: %v", err)
	}

	if err := alpha.AddShortTo("team:shared", "Shared context about refactoring", map[string]string{"source": "test"}); err != nil {
		t.Fatalf("alpha AddShortTo: %v", err)
	}

	records, err := beta.Retrieve(ctx, "refactoring", 5)
	if err != nil {
		t.Fatalf("beta Retrieve: %v", err)
	}
	if len(records) == 0 {
		t.Fatalf("expected shared records, got none")
	}

	found := false
	for _, rec := range records {
		if strings.Contains(rec.Content, "Shared context") {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("shared memory not retrieved: %+v", records)
	}
}
