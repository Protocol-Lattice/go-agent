package workflow

import (
	"context"
	"fmt"
	"strings"
)

const defaultMaxSteps = 128

// GraphConfig configures workflow graph execution.
type GraphConfig struct {
	MaxSteps int
}

// Graph executes workflow nodes according to explicit edges and routes.
type Graph struct {
	edges   []Edge
	adj     map[string][]Edge
	maxStep int
}

// NewGraph validates and constructs a workflow graph.
func NewGraph(edges []Edge, config GraphConfig) (*Graph, error) {
	if len(edges) == 0 {
		return nil, fmt.Errorf("workflow graph requires at least one edge")
	}
	maxSteps := config.MaxSteps
	if maxSteps <= 0 {
		maxSteps = defaultMaxSteps
	}

	adj := make(map[string][]Edge)
	hasStart := false
	for i, edge := range edges {
		if edge.From == nil {
			return nil, fmt.Errorf("workflow edge %d has nil From", i)
		}
		if edge.To == nil {
			return nil, fmt.Errorf("workflow edge %d has nil To", i)
		}
		from := nodeKey(edge.From)
		to := nodeKey(edge.To)
		if from == "" {
			return nil, fmt.Errorf("workflow edge %d has empty From node name", i)
		}
		if to == "" {
			return nil, fmt.Errorf("workflow edge %d has empty To node name", i)
		}
		if to == nodeKey(Start) {
			return nil, fmt.Errorf("workflow edge %d targets START", i)
		}
		if from == nodeKey(Start) {
			hasStart = true
		}
		adj[from] = append(adj[from], edge)
	}
	if !hasStart {
		return nil, fmt.Errorf("workflow graph requires an edge from START")
	}

	copied := make([]Edge, len(edges))
	copy(copied, edges)
	return &Graph{edges: copied, adj: adj, maxStep: maxSteps}, nil
}

// Edges returns a copy of graph edges.
func (g *Graph) Edges() []Edge {
	if g == nil || len(g.edges) == 0 {
		return nil
	}
	out := make([]Edge, len(g.edges))
	copy(out, g.edges)
	return out
}

// Run executes the graph from START with input as the first node input.
func (g *Graph) Run(ctx context.Context, sessionID string, input any) (any, error) {
	if g == nil {
		return nil, fmt.Errorf("nil workflow graph")
	}

	wctx := newContext(ctx, sessionID)
	queue := make([]runItem, 0)
	for _, edge := range g.matchingEdges(Start, Event{Output: input}) {
		queue = append(queue, runItem{node: edge.To, input: input})
	}
	if len(queue) == 0 {
		return nil, fmt.Errorf("workflow graph has no START target")
	}

	finals := make([]any, 0, 1)
	steps := 0
	for len(queue) > 0 {
		if steps >= g.maxStep {
			return nil, fmt.Errorf("workflow graph exceeded max steps %d", g.maxStep)
		}
		item := queue[0]
		queue = queue[1:]
		steps++

		events, err := item.node.run(wctx, item.input)
		if err != nil {
			return nil, fmt.Errorf("workflow node %s: %w", item.node.Name(), err)
		}
		for _, ev := range events {
			matches := g.matchingEdges(item.node, ev)
			if len(matches) == 0 {
				finals = append(finals, eventResult(ev))
				continue
			}
			for _, edge := range matches {
				queue = append(queue, runItem{node: edge.To, input: ev.Output})
			}
		}
	}

	switch len(finals) {
	case 0:
		return nil, nil
	case 1:
		return finals[0], nil
	default:
		return finals, nil
	}
}

type runItem struct {
	node  Node
	input any
}

func (g *Graph) matchingEdges(from Node, ev Event) []Edge {
	if g == nil || from == nil {
		return nil
	}
	edges := g.adj[nodeKey(from)]
	if len(edges) == 0 {
		return nil
	}

	var matches []Edge
	var defaults []Edge
	for _, edge := range edges {
		if edge.Route == nil {
			matches = append(matches, edge)
			continue
		}
		if edge.Route.isDefault() {
			defaults = append(defaults, edge)
			continue
		}
		if edge.Route.Match(ev) {
			matches = append(matches, edge)
		}
	}
	if len(matches) > 0 {
		return matches
	}
	for _, edge := range defaults {
		if edge.Route.Match(ev) {
			matches = append(matches, edge)
		}
	}
	return matches
}

func nodeKey(node Node) string {
	if node == nil {
		return ""
	}
	return strings.TrimSpace(node.Name())
}
