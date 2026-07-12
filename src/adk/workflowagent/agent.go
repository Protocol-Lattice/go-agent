package workflowagent

import (
	"context"
	"fmt"
	"strings"

	"github.com/Protocol-Lattice/go-agent/src/adk/workflow"
)

// Config describes a graph-backed agent.
type Config struct {
	Name        string
	Description string
	Edges       []workflow.Edge
	MaxSteps    int
}

// Agent executes a workflow graph behind an agent-like Generate and Run API.
type Agent struct {
	name        string
	description string
	graph       *workflow.Graph
}

// New builds a graph-backed agent from workflow edges.
func New(config Config) (*Agent, error) {
	name := strings.TrimSpace(config.Name)
	if name == "" {
		return nil, fmt.Errorf("workflow agent requires a name")
	}
	graph, err := workflow.NewGraph(config.Edges, workflow.GraphConfig{MaxSteps: config.MaxSteps})
	if err != nil {
		return nil, err
	}
	return &Agent{
		name:        name,
		description: strings.TrimSpace(config.Description),
		graph:       graph,
	}, nil
}

func (a *Agent) Name() string {
	if a == nil {
		return ""
	}
	return a.name
}

func (a *Agent) Description() string {
	if a == nil {
		return ""
	}
	if a.description != "" {
		return a.description
	}
	return a.name
}

// Generate runs the workflow graph with session-aware context.
func (a *Agent) Generate(ctx context.Context, sessionID, input string) (any, error) {
	if a == nil || a.graph == nil {
		return nil, fmt.Errorf("nil workflow agent")
	}
	return a.graph.Run(ctx, sessionID, input)
}

// StartRun creates and executes a durable graph run. Use ResumeRun after a
// transient node failure or process restart.
func (a *Agent) StartRun(ctx context.Context, store workflow.RunStore, runID, sessionID string, input any) (any, error) {
	if a == nil || a.graph == nil {
		return nil, fmt.Errorf("nil workflow agent")
	}
	return a.graph.StartRun(ctx, store, runID, sessionID, input)
}

// ResumeRun continues a durable graph run from its most recently saved node
// transition. Completed runs return their saved result without re-executing.
func (a *Agent) ResumeRun(ctx context.Context, store workflow.RunStore, runID string) (any, error) {
	if a == nil || a.graph == nil {
		return nil, fmt.Errorf("nil workflow agent")
	}
	return a.graph.ResumeRun(ctx, store, runID)
}

// Run lets a workflow agent satisfy the root agent.SubAgent interface.
func (a *Agent) Run(ctx context.Context, input string) (string, error) {
	out, err := a.Generate(ctx, "", input)
	if err != nil {
		return "", err
	}
	return fmt.Sprint(out), nil
}

func (a *Agent) Graph() *workflow.Graph {
	if a == nil {
		return nil
	}
	return a.graph
}
