package workflow

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"

	goagent "github.com/Protocol-Lattice/go-agent"
)

// NodeConfig carries metadata for workflow nodes.
type NodeConfig struct {
	Description string
}

// Node is an executable step in a workflow graph.
type Node interface {
	Name() string
	Description() string
	run(ctx Context, input any) ([]Event, error)
}

type startNode struct{}

// Start is the synthetic graph entry point used in edges and chains.
var Start Node = startNode{}

func (startNode) Name() string        { return "START" }
func (startNode) Description() string { return "workflow start" }
func (startNode) run(Context, any) ([]Event, error) {
	return nil, fmt.Errorf("workflow start node is not executable")
}

type functionNode[I, O any] struct {
	name   string
	fn     func(Context, I) (O, error)
	config NodeConfig
}

// NewFunctionNode wraps a plain Go function as a workflow node. The returned
// value is forwarded to successor nodes through Event.Output.
func NewFunctionNode[I, O any](name string, fn func(Context, I) (O, error), config NodeConfig) Node {
	return &functionNode[I, O]{
		name:   strings.TrimSpace(name),
		fn:     fn,
		config: config,
	}
}

func (n *functionNode[I, O]) Name() string        { return n.name }
func (n *functionNode[I, O]) Description() string { return n.config.Description }

func (n *functionNode[I, O]) run(ctx Context, input any) ([]Event, error) {
	if strings.TrimSpace(n.name) == "" {
		return nil, fmt.Errorf("workflow function node has empty name")
	}
	if n.fn == nil {
		return nil, fmt.Errorf("workflow function node %s has nil function", n.name)
	}
	typedInput, err := coerceInput[I](input)
	if err != nil {
		return nil, fmt.Errorf("workflow node %s input: %w", n.name, err)
	}
	out, err := n.fn(ctx, typedInput)
	if err != nil {
		return nil, err
	}
	return []Event{{Output: out}}, nil
}

type emittingFunctionNode[I, O any] struct {
	name   string
	fn     func(Context, I, EmitFunc) (O, error)
	config NodeConfig
}

// NewEmittingFunctionNode wraps a function that can emit explicit Events.
// Returning nil and emitting no events suppresses automatic output.
func NewEmittingFunctionNode[I, O any](name string, fn func(Context, I, EmitFunc) (O, error), config NodeConfig) Node {
	return &emittingFunctionNode[I, O]{
		name:   strings.TrimSpace(name),
		fn:     fn,
		config: config,
	}
}

func (n *emittingFunctionNode[I, O]) Name() string        { return n.name }
func (n *emittingFunctionNode[I, O]) Description() string { return n.config.Description }

func (n *emittingFunctionNode[I, O]) run(ctx Context, input any) ([]Event, error) {
	if strings.TrimSpace(n.name) == "" {
		return nil, fmt.Errorf("workflow emitting function node has empty name")
	}
	if n.fn == nil {
		return nil, fmt.Errorf("workflow emitting function node %s has nil function", n.name)
	}
	typedInput, err := coerceInput[I](input)
	if err != nil {
		return nil, fmt.Errorf("workflow node %s input: %w", n.name, err)
	}

	var events []Event
	emit := func(ev *Event) error {
		if ev == nil {
			return nil
		}
		events = append(events, *ev)
		return nil
	}

	out, err := n.fn(ctx, typedInput, emit)
	if err != nil {
		return nil, err
	}
	if !isNil(out) {
		events = append(events, Event{Output: out})
	}
	return events, nil
}

// SessionGenerator is the minimal surface needed to wrap an agent node.
type SessionGenerator interface {
	Generate(ctx context.Context, sessionID, input string) (any, error)
}

type agentNode struct {
	name   string
	agent  SessionGenerator
	config NodeConfig
}

// NewAgentNode wraps a session-aware agent as a graph node.
func NewAgentNode(name string, runner SessionGenerator, config NodeConfig) Node {
	return &agentNode{
		name:   strings.TrimSpace(name),
		agent:  runner,
		config: config,
	}
}

func (n *agentNode) Name() string        { return n.name }
func (n *agentNode) Description() string { return n.config.Description }

func (n *agentNode) run(ctx Context, input any) ([]Event, error) {
	if strings.TrimSpace(n.name) == "" {
		return nil, fmt.Errorf("workflow agent node has empty name")
	}
	if n.agent == nil {
		return nil, fmt.Errorf("workflow agent node %s has nil agent", n.name)
	}
	out, err := n.agent.Generate(ctx.Context, ctx.SessionID, textInput(input))
	if err != nil {
		return nil, err
	}
	return []Event{{Output: out}}, nil
}

type toolNode struct {
	name   string
	tool   goagent.Tool
	config NodeConfig
}

// NewToolNode wraps an agent.Tool as a graph node. Map inputs are passed as
// arguments; other inputs are passed as the "input" argument.
func NewToolNode(name string, tool goagent.Tool, config NodeConfig) Node {
	return &toolNode{
		name:   strings.TrimSpace(name),
		tool:   tool,
		config: config,
	}
}

func (n *toolNode) Name() string        { return n.name }
func (n *toolNode) Description() string { return n.config.Description }

func (n *toolNode) run(ctx Context, input any) ([]Event, error) {
	if strings.TrimSpace(n.name) == "" {
		return nil, fmt.Errorf("workflow tool node has empty name")
	}
	if n.tool == nil {
		return nil, fmt.Errorf("workflow tool node %s has nil tool", n.name)
	}

	args := map[string]any{}
	if m, ok := input.(map[string]any); ok {
		for k, v := range m {
			args[k] = v
		}
	} else {
		args["input"] = input
	}

	resp, err := n.tool.Invoke(ctx.Context, goagent.ToolRequest{
		SessionID: ctx.SessionID,
		Arguments: args,
	})
	if err != nil {
		return nil, err
	}
	return []Event{{
		Output:   resp.Content,
		Metadata: resp.Metadata,
	}}, nil
}

func coerceInput[I any](input any) (I, error) {
	var out I
	if input == nil {
		return out, nil
	}

	target := reflect.TypeOf((*I)(nil)).Elem()
	value := reflect.ValueOf(input)
	if value.Type().AssignableTo(target) {
		reflect.ValueOf(&out).Elem().Set(value)
		return out, nil
	}
	if value.Type().ConvertibleTo(target) {
		reflect.ValueOf(&out).Elem().Set(value.Convert(target))
		return out, nil
	}
	return out, fmt.Errorf("expected %s, got %T", target.String(), input)
}

func isNil[T any](v T) bool {
	rv := reflect.ValueOf(v)
	if !rv.IsValid() {
		return true
	}
	switch rv.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return rv.IsNil()
	default:
		return false
	}
}

func textInput(input any) string {
	switch v := input.(type) {
	case nil:
		return ""
	case string:
		return v
	case []byte:
		return string(v)
	default:
		b, err := json.Marshal(v)
		if err == nil {
			return string(b)
		}
		return fmt.Sprint(v)
	}
}
