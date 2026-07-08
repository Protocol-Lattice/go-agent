package workflow_test

import (
	"context"
	"strings"
	"testing"

	goagent "github.com/Protocol-Lattice/go-agent"
	"github.com/Protocol-Lattice/go-agent/src/adk/workflow"
)

func TestGraphRunsSequentialFunctionNodes(t *testing.T) {
	t.Parallel()

	upper := workflow.NewFunctionNode[string, string]("upper",
		func(_ workflow.Context, input string) (string, error) {
			return strings.ToUpper(input), nil
		},
		workflow.NodeConfig{},
	)
	suffix := workflow.NewFunctionNode[string, string]("suffix",
		func(_ workflow.Context, input string) (string, error) {
			return input + " IS AWESOME!", nil
		},
		workflow.NodeConfig{},
	)

	graph, err := workflow.NewGraph(workflow.Chain(workflow.Start, upper, suffix), workflow.GraphConfig{})
	if err != nil {
		t.Fatalf("NewGraph: %v", err)
	}

	out, err := graph.Run(context.Background(), "session", "go-agent")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out != "GO-AGENT IS AWESOME!" {
		t.Fatalf("unexpected output: %q", out)
	}
}

func TestGraphRoutesFromEmittedEvent(t *testing.T) {
	t.Parallel()

	classify := workflow.NewEmittingFunctionNode[string, any]("classify",
		func(_ workflow.Context, input string, emit workflow.EmitFunc) (any, error) {
			route := "LOGISTICS"
			if strings.Contains(strings.ToLower(input), "bug") {
				route = "BUG"
			}
			return nil, emit(&workflow.Event{
				Output: input,
				Routes: []any{route},
			})
		},
		workflow.NodeConfig{},
	)
	bug := workflow.NewFunctionNode[string, string]("bug",
		func(_ workflow.Context, input string) (string, error) {
			return "Handling bug: " + input, nil
		},
		workflow.NodeConfig{},
	)
	logistics := workflow.NewFunctionNode[string, string]("logistics",
		func(_ workflow.Context, input string) (string, error) {
			return "Handling logistics: " + input, nil
		},
		workflow.NodeConfig{},
	)

	edges := workflow.Concat(
		workflow.Chain(workflow.Start, classify),
		[]workflow.Edge{
			{From: classify, To: bug, Route: workflow.StringRoute("BUG")},
			{From: classify, To: logistics, Route: workflow.Default},
		},
	)

	graph, err := workflow.NewGraph(edges, workflow.GraphConfig{})
	if err != nil {
		t.Fatalf("NewGraph: %v", err)
	}

	out, err := graph.Run(context.Background(), "session", "bug in checkout")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out != "Handling bug: bug in checkout" {
		t.Fatalf("unexpected route output: %q", out)
	}
}

func TestAgentNodeWrapsSessionGenerator(t *testing.T) {
	t.Parallel()

	runner := &fakeSessionGenerator{}
	node := workflow.NewAgentNode("agent", runner, workflow.NodeConfig{})
	graph, err := workflow.NewGraph(workflow.Chain(workflow.Start, node), workflow.GraphConfig{})
	if err != nil {
		t.Fatalf("NewGraph: %v", err)
	}

	out, err := graph.Run(context.Background(), "s1", "hello")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out != "s1:hello" {
		t.Fatalf("unexpected output: %q", out)
	}
}

func TestToolNodePassesMapArguments(t *testing.T) {
	t.Parallel()

	tool := &echoTool{}
	node := workflow.NewToolNode("echo", tool, workflow.NodeConfig{})
	graph, err := workflow.NewGraph(workflow.Chain(workflow.Start, node), workflow.GraphConfig{})
	if err != nil {
		t.Fatalf("NewGraph: %v", err)
	}

	out, err := graph.Run(context.Background(), "s1", map[string]any{"input": "hello"})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out != "hello" {
		t.Fatalf("unexpected output: %q", out)
	}
	if tool.last.SessionID != "s1" {
		t.Fatalf("expected session id to be passed, got %q", tool.last.SessionID)
	}
}

type fakeSessionGenerator struct{}

func (f *fakeSessionGenerator) Generate(_ context.Context, sessionID, input string) (any, error) {
	return sessionID + ":" + input, nil
}

type echoTool struct {
	last goagent.ToolRequest
}

func (t *echoTool) Spec() goagent.ToolSpec {
	return goagent.ToolSpec{
		Name:        "echo",
		Description: "echoes input",
		InputSchema: map[string]any{
			"type": "object",
		},
	}
}

func (t *echoTool) Invoke(_ context.Context, req goagent.ToolRequest) (goagent.ToolResponse, error) {
	t.last = req
	return goagent.ToolResponse{Content: req.Arguments["input"].(string)}, nil
}
