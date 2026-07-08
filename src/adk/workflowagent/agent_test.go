package workflowagent_test

import (
	"context"
	"strings"
	"testing"

	goagent "github.com/Protocol-Lattice/go-agent"
	"github.com/Protocol-Lattice/go-agent/src/adk/workflow"
	"github.com/Protocol-Lattice/go-agent/src/adk/workflowagent"
)

func TestWorkflowAgentGenerate(t *testing.T) {
	t.Parallel()

	step := workflow.NewFunctionNode[string, string]("step",
		func(_ workflow.Context, input string) (string, error) {
			return "done: " + input, nil
		},
		workflow.NodeConfig{},
	)

	root, err := workflowagent.New(workflowagent.Config{
		Name:        "root_agent",
		Description: "test graph",
		Edges:       workflow.Chain(workflow.Start, step),
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	out, err := root.Generate(context.Background(), "session", "task")
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if out != "done: task" {
		t.Fatalf("unexpected output: %q", out)
	}
}

func TestWorkflowAgentImplementsSubAgent(t *testing.T) {
	t.Parallel()

	var _ goagent.SubAgent = (*workflowagent.Agent)(nil)

	step := workflow.NewFunctionNode[string, string]("step",
		func(_ workflow.Context, input string) (string, error) {
			return strings.ToUpper(input), nil
		},
		workflow.NodeConfig{},
	)
	root, err := workflowagent.New(workflowagent.Config{
		Name:  "upper_agent",
		Edges: workflow.Chain(workflow.Start, step),
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	out, err := root.Run(context.Background(), "hello")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out != "HELLO" {
		t.Fatalf("unexpected output: %q", out)
	}
	if root.Description() != "upper_agent" {
		t.Fatalf("expected name fallback description, got %q", root.Description())
	}
}
