package runtime

import (
	"fmt"
	"strings"
	"sync"
)

// StaticToolCatalog is the default in-memory implementation of ToolCatalog used by the runtime.
type StaticToolCatalog struct {
	mu    sync.RWMutex
	tools map[string]Tool
	specs map[string]ToolSpec
	order []string
}

// NewStaticToolCatalog constructs a catalog seeded with the provided tools.
func NewStaticToolCatalog(tools []Tool) *StaticToolCatalog {
	catalog := &StaticToolCatalog{
		tools: make(map[string]Tool),
		specs: make(map[string]ToolSpec),
	}
	for _, tool := range tools {
		_ = catalog.Register(tool) // skip invalid entries silently to match legacy behaviour
	}
	return catalog
}

// Register adds a tool to the catalog using a lower-cased key. Duplicate names return an error.
func (c *StaticToolCatalog) Register(tool Tool) error {
	if tool == nil {
		return fmt.Errorf("tool is nil")
	}
	spec := tool.Spec()
	key := strings.ToLower(strings.TrimSpace(spec.Name))
	if key == "" {
		return fmt.Errorf("tool name is empty")
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	if _, exists := c.tools[key]; exists {
		return fmt.Errorf("tool %s already registered", spec.Name)
	}
	c.tools[key] = tool
	c.specs[key] = spec
	c.order = append(c.order, key)
	return nil
}

// Lookup returns the tool and its specification if present.
func (c *StaticToolCatalog) Lookup(name string) (Tool, ToolSpec, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	key := strings.ToLower(strings.TrimSpace(name))
	tool, ok := c.tools[key]
	if !ok {
		return nil, ToolSpec{}, false
	}
	return tool, c.specs[key], true
}

// Specs returns a snapshot of the tool specifications in registration order.
func (c *StaticToolCatalog) Specs() []ToolSpec {
	c.mu.RLock()
	defer c.mu.RUnlock()

	specs := make([]ToolSpec, 0, len(c.order))
	for _, key := range c.order {
		specs = append(specs, c.specs[key])
	}
	return specs
}

// Tools returns the registered tools in order.
func (c *StaticToolCatalog) Tools() []Tool {
	c.mu.RLock()
	defer c.mu.RUnlock()

	tools := make([]Tool, 0, len(c.order))
	for _, key := range c.order {
		tools = append(tools, c.tools[key])
	}
	return tools
}

// StaticSubAgentDirectory is the default SubAgentDirectory implementation.
type StaticSubAgentDirectory struct {
	mu        sync.RWMutex
	subagents map[string]SubAgent
	order     []string
}

// NewStaticSubAgentDirectory constructs a directory from the provided sub-agents.
func NewStaticSubAgentDirectory(subagents []SubAgent) *StaticSubAgentDirectory {
	dir := &StaticSubAgentDirectory{
		subagents: make(map[string]SubAgent),
	}
	for _, sa := range subagents {
		_ = dir.Register(sa)
	}
	return dir
}

// Register adds a sub-agent to the directory. Duplicate names return an error.
func (d *StaticSubAgentDirectory) Register(subAgent SubAgent) error {
	if subAgent == nil {
		return fmt.Errorf("sub-agent is nil")
	}
	key := strings.ToLower(strings.TrimSpace(subAgent.Name()))
	if key == "" {
		return fmt.Errorf("sub-agent name is empty")
	}

	d.mu.Lock()
	defer d.mu.Unlock()

	if _, exists := d.subagents[key]; exists {
		return fmt.Errorf("sub-agent %s already registered", subAgent.Name())
	}
	d.subagents[key] = subAgent
	d.order = append(d.order, key)
	return nil
}

// Lookup retrieves a sub-agent by name.
func (d *StaticSubAgentDirectory) Lookup(name string) (SubAgent, bool) {
	d.mu.RLock()
	defer d.mu.RUnlock()

	key := strings.ToLower(strings.TrimSpace(name))
	sa, ok := d.subagents[key]
	return sa, ok
}

// All returns the registered sub-agents in registration order.
func (d *StaticSubAgentDirectory) All() []SubAgent {
	d.mu.RLock()
	defer d.mu.RUnlock()

	subagents := make([]SubAgent, 0, len(d.order))
	for _, key := range d.order {
		subagents = append(subagents, d.subagents[key])
	}
	return subagents
}
