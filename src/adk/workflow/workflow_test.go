package workflow_test

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

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

func TestGraphJoinsFanOutBranchOutputs(t *testing.T) {
	t.Parallel()

	left := workflow.NewFunctionNode[string, string]("left",
		func(_ workflow.Context, input string) (string, error) {
			return "left:" + input, nil
		},
		workflow.NodeConfig{},
	)
	right := workflow.NewFunctionNode[string, string]("right",
		func(_ workflow.Context, input string) (string, error) {
			return "right:" + input, nil
		},
		workflow.NodeConfig{},
	)
	joinCalls := 0
	join := workflow.NewJoinNode("join",
		func(_ workflow.Context, inputs map[string]any) (any, error) {
			joinCalls++
			return inputs["left"].(string) + " + " + inputs["right"].(string), nil
		},
		workflow.NodeConfig{},
	)

	graph, err := workflow.NewGraph([]workflow.Edge{
		{From: workflow.Start, To: left},
		{From: workflow.Start, To: right},
		{From: left, To: join},
		{From: right, To: join},
	}, workflow.GraphConfig{})
	if err != nil {
		t.Fatalf("NewGraph: %v", err)
	}

	out, err := graph.Run(context.Background(), "session", "work")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out != "left:work + right:work" {
		t.Fatalf("unexpected output: %q", out)
	}
	if joinCalls != 1 {
		t.Fatalf("expected join to run once, got %d calls", joinCalls)
	}
}

func TestGraphReportsIncompleteJoin(t *testing.T) {
	t.Parallel()

	router := workflow.NewEmittingFunctionNode[string, any]("router",
		func(_ workflow.Context, input string, emit workflow.EmitFunc) (any, error) {
			return nil, emit(&workflow.Event{Output: input, Routes: []any{"left"}})
		},
		workflow.NodeConfig{},
	)
	left := workflow.NewFunctionNode[string, string]("left",
		func(_ workflow.Context, input string) (string, error) { return input, nil },
		workflow.NodeConfig{},
	)
	right := workflow.NewFunctionNode[string, string]("right",
		func(_ workflow.Context, input string) (string, error) { return input, nil },
		workflow.NodeConfig{},
	)
	joinCalls := 0
	join := workflow.NewJoinNode("join",
		func(_ workflow.Context, inputs map[string]any) (any, error) {
			joinCalls++
			return inputs, nil
		},
		workflow.NodeConfig{},
	)

	graph, err := workflow.NewGraph([]workflow.Edge{
		{From: workflow.Start, To: router},
		{From: router, To: left, Route: workflow.StringRoute("left")},
		{From: router, To: right, Route: workflow.StringRoute("right")},
		{From: left, To: join},
		{From: right, To: join},
	}, workflow.GraphConfig{})
	if err != nil {
		t.Fatalf("NewGraph: %v", err)
	}

	_, err = graph.Run(context.Background(), "session", "work")
	if err == nil || !strings.Contains(err.Error(), "did not receive outputs from right") {
		t.Fatalf("expected incomplete join error, got %v", err)
	}
	if joinCalls != 0 {
		t.Fatalf("join should not run with incomplete inputs, got %d calls", joinCalls)
	}
}

func TestGraphJoinTimeout(t *testing.T) {
	t.Parallel()

	left := workflow.NewFunctionNode[string, string]("left",
		func(_ workflow.Context, input string) (string, error) { return input, nil },
		workflow.NodeConfig{},
	)
	delay := workflow.NewFunctionNode[string, string]("delay",
		func(_ workflow.Context, input string) (string, error) {
			time.Sleep(20 * time.Millisecond)
			return input, nil
		},
		workflow.NodeConfig{},
	)
	right := workflow.NewFunctionNode[string, string]("right",
		func(_ workflow.Context, input string) (string, error) { return input, nil },
		workflow.NodeConfig{},
	)
	join := workflow.NewJoinNode("join",
		func(_ workflow.Context, inputs map[string]any) (any, error) { return inputs, nil },
		workflow.NodeConfig{},
	)

	graph, err := workflow.NewGraph([]workflow.Edge{
		{From: workflow.Start, To: left},
		{From: left, To: join},
		{From: left, To: delay},
		{From: delay, To: right},
		{From: right, To: join},
	}, workflow.GraphConfig{JoinTimeout: time.Millisecond})
	if err != nil {
		t.Fatalf("NewGraph: %v", err)
	}

	_, err = graph.Run(context.Background(), "session", "work")
	if err == nil || !strings.Contains(err.Error(), "timed out") {
		t.Fatalf("expected join timeout, got %v", err)
	}
}

func TestGraphRespectsCanceledContext(t *testing.T) {
	t.Parallel()

	node := workflow.NewFunctionNode[string, string]("node",
		func(_ workflow.Context, input string) (string, error) { return input, nil },
		workflow.NodeConfig{},
	)
	graph, err := workflow.NewGraph(workflow.Chain(workflow.Start, node), workflow.GraphConfig{})
	if err != nil {
		t.Fatalf("NewGraph: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err = graph.Run(ctx, "session", "work")
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context cancellation, got %v", err)
	}
}

func TestNewGraphRejectsJoinWithOnePredecessor(t *testing.T) {
	t.Parallel()

	step := workflow.NewFunctionNode[string, string]("step",
		func(_ workflow.Context, input string) (string, error) { return input, nil },
		workflow.NodeConfig{},
	)
	join := workflow.NewJoinNode("join",
		func(_ workflow.Context, inputs map[string]any) (any, error) { return inputs, nil },
		workflow.NodeConfig{},
	)

	_, err := workflow.NewGraph(workflow.Chain(workflow.Start, step, join), workflow.GraphConfig{})
	if err == nil || !strings.Contains(err.Error(), "at least two direct predecessors") {
		t.Fatalf("expected join validation error, got %v", err)
	}
}

func TestGraphDurableRunResumesFromFailedNode(t *testing.T) {
	t.Parallel()

	prepareCalls := 0
	flakyCalls := 0
	finishCalls := 0
	prepare := workflow.NewFunctionNode[string, string]("prepare",
		func(ctx workflow.Context, input string) (string, error) {
			prepareCalls++
			ctx.State["prepared"] = "yes"
			return input + ":prepared", nil
		},
		workflow.NodeConfig{},
	)
	flaky := workflow.NewFunctionNode[string, string]("flaky",
		func(_ workflow.Context, input string) (string, error) {
			flakyCalls++
			if flakyCalls == 1 {
				return "", errors.New("temporary downstream outage")
			}
			return input + ":retried", nil
		},
		workflow.NodeConfig{},
	)
	finish := workflow.NewFunctionNode[string, string]("finish",
		func(ctx workflow.Context, input string) (string, error) {
			finishCalls++
			return input + ":" + ctx.State["prepared"].(string), nil
		},
		workflow.NodeConfig{},
	)

	graph, err := workflow.NewGraph(workflow.Chain(workflow.Start, prepare, flaky, finish), workflow.GraphConfig{})
	if err != nil {
		t.Fatalf("NewGraph: %v", err)
	}
	store := workflow.NewInMemoryRunStore()

	_, err = graph.StartRun(context.Background(), store, "retry-run", "session", "task")
	if err == nil || !strings.Contains(err.Error(), "temporary downstream outage") {
		t.Fatalf("expected persisted transient error, got %v", err)
	}
	state, err := store.Load(context.Background(), "retry-run")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if state.Status != workflow.RunStatusRunning {
		t.Fatalf("status = %q, want running", state.Status)
	}
	if len(state.Queue) != 1 || state.Queue[0].Node != "flaky" {
		t.Fatalf("pending queue = %#v, want flaky node", state.Queue)
	}
	if state.ContextState["prepared"] != "yes" {
		t.Fatalf("context state was not checkpointed: %#v", state.ContextState)
	}

	out, err := graph.ResumeRun(context.Background(), store, "retry-run")
	if err != nil {
		t.Fatalf("ResumeRun: %v", err)
	}
	if out != "task:prepared:retried:yes" {
		t.Fatalf("unexpected resumed output: %q", out)
	}
	if prepareCalls != 1 || flakyCalls != 2 || finishCalls != 1 {
		t.Fatalf("unexpected calls: prepare=%d flaky=%d finish=%d", prepareCalls, flakyCalls, finishCalls)
	}

	state, err = store.Load(context.Background(), "retry-run")
	if err != nil {
		t.Fatalf("Load completed run: %v", err)
	}
	if state.Status != workflow.RunStatusCompleted || state.Result != "task:prepared:retried:yes" {
		t.Fatalf("completed state = %#v", state)
	}

	out, err = graph.ResumeRun(context.Background(), store, "retry-run")
	if err != nil {
		t.Fatalf("ResumeRun completed run: %v", err)
	}
	if out != "task:prepared:retried:yes" || prepareCalls != 1 || flakyCalls != 2 || finishCalls != 1 {
		t.Fatalf("completed run should not execute nodes again; output=%q calls=%d/%d/%d", out, prepareCalls, flakyCalls, finishCalls)
	}
}

func TestGraphDurableRunSurvivesFileStoreRestart(t *testing.T) {
	t.Parallel()

	type payload struct {
		Name string `json:"name"`
	}

	attempts := 0
	prepare := workflow.NewFunctionNode[payload, payload]("prepare",
		func(ctx workflow.Context, input payload) (payload, error) {
			ctx.State["source"] = "durable"
			input.Name = strings.ToUpper(input.Name)
			return input, nil
		},
		workflow.NodeConfig{},
	)
	flaky := workflow.NewFunctionNode[payload, string]("flaky",
		func(ctx workflow.Context, input payload) (string, error) {
			attempts++
			if attempts == 1 {
				return "", errors.New("restart required")
			}
			return input.Name + ":" + ctx.State["source"].(string), nil
		},
		workflow.NodeConfig{},
	)
	graph, err := workflow.NewGraph(workflow.Chain(workflow.Start, prepare, flaky), workflow.GraphConfig{})
	if err != nil {
		t.Fatalf("NewGraph: %v", err)
	}

	dir := t.TempDir()
	store, err := workflow.NewFileRunStore(dir)
	if err != nil {
		t.Fatalf("NewFileRunStore: %v", err)
	}
	_, err = graph.StartRun(context.Background(), store, "file-restart", "session", payload{Name: "agent"})
	if err == nil || !strings.Contains(err.Error(), "restart required") {
		t.Fatalf("expected transient failure, got %v", err)
	}

	reopened, err := workflow.NewFileRunStore(dir)
	if err != nil {
		t.Fatalf("reopen file store: %v", err)
	}
	out, err := graph.ResumeRun(context.Background(), reopened, "file-restart")
	if err != nil {
		t.Fatalf("ResumeRun after restart: %v", err)
	}
	if out != "AGENT:durable" {
		t.Fatalf("unexpected resumed output: %q", out)
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
