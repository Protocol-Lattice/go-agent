package workflow

import (
	"context"
	"fmt"
	"strings"
	"time"
)

const defaultMaxSteps = 128

// GraphConfig configures workflow graph execution.
type GraphConfig struct {
	MaxSteps    int
	JoinTimeout time.Duration
}

// Graph executes workflow nodes according to explicit edges and routes.
type Graph struct {
	edges       []Edge
	adj         map[string][]Edge
	joins       map[string]joinSpec
	maxStep     int
	joinTimeout time.Duration
}

type joinSpec struct {
	sources map[string]struct{}
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
	joins := make(map[string]joinSpec)
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
		if isJoinNode(edge.To) {
			if from == nodeKey(Start) {
				return nil, fmt.Errorf("workflow join node %s cannot receive input from START", to)
			}
			spec := joins[to]
			if spec.sources == nil {
				spec.sources = make(map[string]struct{})
			}
			if _, exists := spec.sources[from]; exists {
				return nil, fmt.Errorf("workflow join node %s has duplicate predecessor %s", to, from)
			}
			spec.sources[from] = struct{}{}
			joins[to] = spec
		}
		adj[from] = append(adj[from], edge)
	}
	if !hasStart {
		return nil, fmt.Errorf("workflow graph requires an edge from START")
	}
	for name, spec := range joins {
		if len(spec.sources) < 2 {
			return nil, fmt.Errorf("workflow join node %s requires at least two direct predecessors", name)
		}
	}

	copied := make([]Edge, len(edges))
	copy(copied, edges)
	return &Graph{
		edges:       copied,
		adj:         adj,
		joins:       joins,
		maxStep:     maxSteps,
		joinTimeout: config.JoinTimeout,
	}, nil
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
		queue = append(queue, runItem{from: Start, node: edge.To, input: input})
	}
	if len(queue) == 0 {
		return nil, fmt.Errorf("workflow graph has no START target")
	}

	finals := make([]any, 0, 1)
	joinStates := make(map[string]*joinState, len(g.joins))
	steps := 0
	for len(queue) > 0 {
		if err := wctx.Err(); err != nil {
			return nil, err
		}
		if err := g.joinTimeoutError(joinStates); err != nil {
			return nil, err
		}
		if steps >= g.maxStep {
			return nil, fmt.Errorf("workflow graph exceeded max steps %d", g.maxStep)
		}
		item := queue[0]
		queue = queue[1:]
		steps++

		if isJoinNode(item.node) {
			joinedInput, ready, err := g.recordJoinInput(item, joinStates)
			if err != nil {
				return nil, err
			}
			if !ready {
				continue
			}
			item.input = joinedInput
		}

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
				queue = append(queue, runItem{from: item.node, node: edge.To, input: ev.Output})
			}
		}
	}
	if err := g.incompleteJoinError(joinStates); err != nil {
		return nil, err
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
	from  Node
	node  Node
	input any
}

type joinState struct {
	values       map[string]any
	firstArrival time.Time
}

func isJoinNode(node Node) bool {
	_, ok := node.(interface{ isJoinNode() })
	return ok
}

func (g *Graph) recordJoinInput(item runItem, states map[string]*joinState) (map[string]any, bool, error) {
	name := nodeKey(item.node)
	spec, ok := g.joins[name]
	if !ok {
		return nil, false, fmt.Errorf("workflow join node %s is not configured", name)
	}
	source := nodeKey(item.from)
	if _, expected := spec.sources[source]; !expected {
		return nil, false, fmt.Errorf("workflow join node %s received output from unexpected predecessor %s", name, source)
	}

	state := states[name]
	if state == nil {
		state = &joinState{values: make(map[string]any, len(spec.sources)), firstArrival: time.Now()}
		states[name] = state
	}
	if _, duplicate := state.values[source]; duplicate {
		return nil, false, fmt.Errorf("workflow join node %s received multiple outputs from predecessor %s", name, source)
	}
	state.values[source] = item.input
	if len(state.values) != len(spec.sources) {
		return nil, false, nil
	}

	inputs := make(map[string]any, len(state.values))
	for source, output := range state.values {
		inputs[source] = output
	}
	return inputs, true, nil
}

func (g *Graph) joinTimeoutError(states map[string]*joinState) error {
	if g.joinTimeout <= 0 {
		return nil
	}
	now := time.Now()
	for name, state := range states {
		if state == nil || len(state.values) == 0 || now.Sub(state.firstArrival) <= g.joinTimeout {
			continue
		}
		return fmt.Errorf("workflow join node %s timed out after %s", name, g.joinTimeout)
	}
	return nil
}

func (g *Graph) incompleteJoinError(states map[string]*joinState) error {
	for name, state := range states {
		if state == nil || len(state.values) == 0 {
			continue
		}
		spec := g.joins[name]
		missing := make([]string, 0, len(spec.sources)-len(state.values))
		for source := range spec.sources {
			if _, received := state.values[source]; !received {
				missing = append(missing, source)
			}
		}
		if len(missing) > 0 {
			return fmt.Errorf("workflow join node %s did not receive outputs from %s", name, strings.Join(missing, ", "))
		}
	}
	return nil
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
