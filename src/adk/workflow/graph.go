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
	nodes       map[string]Node
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
	nodes := make(map[string]Node)
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
		} else if _, exists := nodes[from]; !exists {
			nodes[from] = edge.From
		}
		if _, exists := nodes[to]; !exists {
			nodes[to] = edge.To
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
		nodes:       nodes,
		joins:       joins,
		maxStep:     maxSteps,
		joinTimeout: config.JoinTimeout,
	}, nil
}

// StartRun creates and executes a durable workflow run. Every completed node
// transition is saved to store before the next node begins. If execution is
// interrupted or a node returns an error, call ResumeRun with the same ID.
func (g *Graph) StartRun(ctx context.Context, store RunStore, runID, sessionID string, input any) (any, error) {
	if g == nil {
		return nil, fmt.Errorf("nil workflow graph")
	}
	if store == nil {
		return nil, fmt.Errorf("workflow run store is nil")
	}
	if err := validateRunID(runID); err != nil {
		return nil, err
	}
	if ctx == nil {
		ctx = context.Background()
	}

	invocation := newContext(ctx, sessionID)
	state := &RunState{
		ID:           runID,
		SessionID:    sessionID,
		InvocationID: invocation.InvocationID,
		Status:       RunStatusRunning,
		Queue:        g.initialQueue(input),
		JoinStates:   make(map[string]JoinState, len(g.joins)),
		ContextState: invocation.State,
		CreatedAt:    time.Now().UTC(),
		UpdatedAt:    time.Now().UTC(),
	}
	if len(state.Queue) == 0 {
		return nil, fmt.Errorf("workflow graph has no START target")
	}
	if err := store.Create(ctx, state); err != nil {
		return nil, fmt.Errorf("create workflow run %s: %w", runID, err)
	}
	return g.executeRun(ctx, store, state)
}

// ResumeRun loads a previously started durable run and continues it. Resuming
// a completed run returns its persisted result without invoking nodes again.
func (g *Graph) ResumeRun(ctx context.Context, store RunStore, runID string) (any, error) {
	if g == nil {
		return nil, fmt.Errorf("nil workflow graph")
	}
	if store == nil {
		return nil, fmt.Errorf("workflow run store is nil")
	}
	if err := validateRunID(runID); err != nil {
		return nil, err
	}
	if ctx == nil {
		ctx = context.Background()
	}
	state, err := store.Load(ctx, runID)
	if err != nil {
		return nil, fmt.Errorf("load workflow run %s: %w", runID, err)
	}
	if state.Status == RunStatusCompleted {
		return state.Result, nil
	}
	if state.Status == RunStatusFailed {
		return nil, fmt.Errorf("workflow run %s failed: %s", runID, state.LastError)
	}
	if state.Status != RunStatusRunning {
		return nil, fmt.Errorf("workflow run %s has unknown status %q", runID, state.Status)
	}
	return g.executeRun(ctx, store, state)
}

func (g *Graph) initialQueue(input any) []QueuedNode {
	queue := make([]QueuedNode, 0)
	for _, edge := range g.matchingEdges(Start, Event{Output: input}) {
		queue = append(queue, QueuedNode{From: nodeKey(Start), Node: nodeKey(edge.To), Input: input})
	}
	return queue
}

func (g *Graph) executeRun(ctx context.Context, store RunStore, state *RunState) (any, error) {
	if state == nil {
		return nil, fmt.Errorf("nil workflow run state")
	}
	if state.InvocationID == "" {
		state.InvocationID = newContext(ctx, state.SessionID).InvocationID
	}
	if state.ContextState == nil {
		state.ContextState = make(map[string]any)
	}
	if state.JoinStates == nil {
		state.JoinStates = make(map[string]JoinState, len(g.joins))
	}
	wctx := Context{
		Context:      ctx,
		SessionID:    state.SessionID,
		InvocationID: state.InvocationID,
		State:        state.ContextState,
	}

	for len(state.Queue) > 0 {
		if err := wctx.Err(); err != nil {
			return nil, err
		}
		if err := g.durableJoinTimeoutError(state.JoinStates); err != nil {
			return g.failRun(ctx, store, state, err)
		}
		if state.Steps >= g.maxStep {
			return g.failRun(ctx, store, state, fmt.Errorf("workflow graph exceeded max steps %d", g.maxStep))
		}

		item := state.Queue[0]
		node, ok := g.nodes[item.Node]
		if !ok || node == nil {
			return g.failRun(ctx, store, state, fmt.Errorf("workflow run references unknown node %q", item.Node))
		}

		input := item.Input
		if isJoinNode(node) {
			joinedInput, ready, err := g.recordDurableJoinInput(item, state.JoinStates)
			if err != nil {
				return g.failRun(ctx, store, state, err)
			}
			if !ready {
				state.Queue = state.Queue[1:]
				state.Steps++
				state.LastError = ""
				if err := saveRun(ctx, store, state); err != nil {
					return nil, err
				}
				continue
			}
			input = joinedInput
		}

		events, err := node.run(wctx, input)
		if err != nil {
			state.LastError = fmt.Sprintf("workflow node %s: %v", node.Name(), err)
			if saveErr := saveRun(ctx, store, state); saveErr != nil {
				return nil, fmt.Errorf("%s (also failed to checkpoint: %w)", state.LastError, saveErr)
			}
			return nil, fmt.Errorf("%s", state.LastError)
		}

		state.Queue = state.Queue[1:]
		for _, ev := range events {
			matches := g.matchingEdges(node, ev)
			if len(matches) == 0 {
				state.Finals = append(state.Finals, eventResult(ev))
				continue
			}
			for _, edge := range matches {
				state.Queue = append(state.Queue, QueuedNode{
					From:  nodeKey(node),
					Node:  nodeKey(edge.To),
					Input: ev.Output,
				})
			}
		}
		state.Steps++
		state.LastError = ""
		if err := saveRun(ctx, store, state); err != nil {
			return nil, err
		}
	}

	if err := g.durableIncompleteJoinError(state.JoinStates); err != nil {
		return g.failRun(ctx, store, state, err)
	}
	state.Result = durableResult(state.Finals)
	state.Status = RunStatusCompleted
	state.LastError = ""
	if err := saveRun(ctx, store, state); err != nil {
		return nil, err
	}
	return state.Result, nil
}

func (g *Graph) recordDurableJoinInput(item QueuedNode, states map[string]JoinState) (map[string]any, bool, error) {
	spec, ok := g.joins[item.Node]
	if !ok {
		return nil, false, fmt.Errorf("workflow join node %s is not configured", item.Node)
	}
	if _, expected := spec.sources[item.From]; !expected {
		return nil, false, fmt.Errorf("workflow join node %s received output from unexpected predecessor %s", item.Node, item.From)
	}

	state, ok := states[item.Node]
	if !ok {
		state = JoinState{Values: make(map[string]any, len(spec.sources)), FirstArrival: time.Now().UTC()}
	}
	if _, duplicate := state.Values[item.From]; duplicate {
		return nil, false, fmt.Errorf("workflow join node %s received multiple outputs from predecessor %s", item.Node, item.From)
	}
	state.Values[item.From] = item.Input
	states[item.Node] = state
	if len(state.Values) != len(spec.sources) {
		return nil, false, nil
	}

	inputs := make(map[string]any, len(state.Values))
	for source, output := range state.Values {
		inputs[source] = output
	}
	return inputs, true, nil
}

func (g *Graph) durableJoinTimeoutError(states map[string]JoinState) error {
	if g.joinTimeout <= 0 {
		return nil
	}
	now := time.Now()
	for name, state := range states {
		if len(state.Values) == 0 || now.Sub(state.FirstArrival) <= g.joinTimeout {
			continue
		}
		return fmt.Errorf("workflow join node %s timed out after %s", name, g.joinTimeout)
	}
	return nil
}

func (g *Graph) durableIncompleteJoinError(states map[string]JoinState) error {
	for name, state := range states {
		if len(state.Values) == 0 {
			continue
		}
		spec := g.joins[name]
		missing := make([]string, 0, len(spec.sources)-len(state.Values))
		for source := range spec.sources {
			if _, received := state.Values[source]; !received {
				missing = append(missing, source)
			}
		}
		if len(missing) > 0 {
			return fmt.Errorf("workflow join node %s did not receive outputs from %s", name, strings.Join(missing, ", "))
		}
	}
	return nil
}

func (g *Graph) failRun(ctx context.Context, store RunStore, state *RunState, runErr error) (any, error) {
	state.Status = RunStatusFailed
	state.LastError = runErr.Error()
	if err := saveRun(ctx, store, state); err != nil {
		return nil, fmt.Errorf("%w (also failed to checkpoint: %v)", runErr, err)
	}
	return nil, runErr
}

func saveRun(ctx context.Context, store RunStore, state *RunState) error {
	state.UpdatedAt = time.Now().UTC()
	if err := store.Save(ctx, state); err != nil {
		return fmt.Errorf("checkpoint workflow run %s: %w", state.ID, err)
	}
	return nil
}

func durableResult(finals []any) any {
	switch len(finals) {
	case 0:
		return nil
	case 1:
		return finals[0]
	default:
		return finals
	}
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
