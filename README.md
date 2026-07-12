# go-agent

[![Go Version](https://img.shields.io/badge/Go-1.25-00ADD8?logo=go&logoColor=white)](https://go.dev/dl/)
[![CI Status](https://github.com/Protocol-Lattice/go-agent/actions/workflows/go.yml/badge.svg)](https://github.com/Protocol-Lattice/go-agent/actions/workflows/go.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/Protocol-Lattice/go-agent.svg)](https://pkg.go.dev/github.com/Protocol-Lattice/go-agent)
[![Go Report Card](https://goreportcard.com/badge/github.com/Protocol-Lattice/go-agent)](https://goreportcard.com/report/github.com/Protocol-Lattice/go-agent)

go-agent is a Go framework for building AI agents with pluggable LLM providers, memory, file context, guardrails, UTCP tool orchestration, and multi-agent coordination.

Use it when you want agent runtime pieces that stay idiomatic in Go:

- A small `agent.Agent` core with `Generate`, `GenerateWithFiles`, and `GenerateStream`
- Provider adapters for Gemini, OpenAI, Anthropic, Ollama, and a local dummy model
- Short-term memory plus vector-store backed long-term memory
- ADK modules for wiring models, memory, tools, sub-agents, CodeMode, and UTCP
- Agent-as-tool patterns for specialist agents and hierarchical workflows
- Input/output guardrails and checkpoint/restore support

## Install

```bash
go get github.com/Protocol-Lattice/go-agent
```

For this repository:

```bash
git clone https://github.com/Protocol-Lattice/go-agent.git
cd go-agent
go test ./...
```

The module currently targets Go `1.25.10`.

## Quick Start

This example runs without API keys. It uses the dummy model and in-memory storage, so it is safe for tests and local wiring checks.

```go
package main

import (
	"context"
	"fmt"
	"log"

	agent "github.com/Protocol-Lattice/go-agent"
	"github.com/Protocol-Lattice/go-agent/src/memory"
	"github.com/Protocol-Lattice/go-agent/src/models"
)

func main() {
	ctx := context.Background()

	mem := memory.NewSessionMemory(
		memory.NewMemoryBankWithStore(memory.NewInMemoryStore()),
		8,
	)

	a, err := agent.New(agent.Options{
		Model:        models.NewDummyLLM("local:"),
		Memory:       mem,
		SystemPrompt: "You are concise and helpful.",
	})
	if err != nil {
		log.Fatal(err)
	}

	out, err := a.Generate(ctx, "demo-session", "Say hello in one sentence.")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(out)
}
```

## Real Model Providers

Use `models.NewLLMProvider` when you want provider selection from configuration or flags.

```go
model, err := models.NewLLMProvider(ctx, "openai", "gpt-4o-mini", "")
if err != nil {
	log.Fatal(err)
}
```

Supported provider names:

| Provider | Aliases | Required environment |
| --- | --- | --- |
| Gemini | `gemini`, `google` | `GOOGLE_API_KEY` or `GEMINI_API_KEY` |
| OpenAI | `openai` | `OPENAI_API_KEY` or `OPENAI_KEY` |
| Anthropic | `anthropic`, `claude` | `ANTHROPIC_API_KEY` |
| Ollama | `ollama` | optional `OLLAMA_HOST`, defaults to `http://localhost:11434` |
| OpenRouter | `openrouter` | `OPENROUTER_API_KEY` or `OPENROUTER_KEY` |

Embeddings are selected with `memory.AutoEmbedder()`.

| Variable | Purpose |
| --- | --- |
| `ADK_EMBED_PROVIDER` | `openai`, `google`, `gemini`, `ollama`, `claude`, `anthropic`, or `fastembed` |
| `ADK_EMBED_MODEL` | Provider-specific embedding model |

If no embedding provider can be created, Lattice falls back to `DummyEmbedder`.

## ADK Setup

For applications, prefer the ADK when you want dependency injection around model, memory, tools, and runtime features.

```go
package main

import (
	"context"
	"log"

	"github.com/Protocol-Lattice/go-agent/src/adk"
	"github.com/Protocol-Lattice/go-agent/src/adk/modules"
	"github.com/Protocol-Lattice/go-agent/src/memory"
	"github.com/Protocol-Lattice/go-agent/src/models"
)

func main() {
	ctx := context.Background()
	memOpts := memory.DefaultOptions()

	kit, err := adk.New(ctx,
		adk.WithDefaultSystemPrompt("You coordinate a helpful assistant."),
		adk.WithModules(
			modules.NewModelModule("llm", func(ctx context.Context) (models.Agent, error) {
				return models.NewLLMProvider(ctx, "openai", "gpt-4o-mini", "")
			}),
			modules.InMemoryMemoryModule(8, memory.AutoEmbedder(), &memOpts),
		),
	)
	if err != nil {
		log.Fatal(err)
	}

	a, err := kit.BuildAgent(ctx)
	if err != nil {
		log.Fatal(err)
	}

	_, _ = a.Generate(ctx, "user-123", "Draft a short project update.")
}
```

Use direct `agent.New` for small programs and tests. Use `adk.New` once you need reusable modules, shared sessions, provider selection, or UTCP runtime wiring.

## Graph Workflows

Graph workflows give you ADK Go v2-style deterministic control flow: define nodes, wire them with edges, and pass each node's output to the next node. Function nodes, emitting router nodes, session-aware agent nodes, and `agent.Tool` nodes can be mixed in the same graph.

For fan-out work, use `NewJoinNode` as a barrier: it waits for one output from
each direct predecessor, then gives the reducer a `map[string]any` keyed by
node name. Set `GraphConfig.JoinTimeout` to bound how long a partially-filled
join may wait; graph cancellation is also respected.

```go
package main

import (
	"context"
	"fmt"
	"strings"

	"github.com/Protocol-Lattice/go-agent/src/adk/workflow"
	"github.com/Protocol-Lattice/go-agent/src/adk/workflowagent"
)

func main() {
	classify := workflow.NewEmittingFunctionNode[string, any]("classify",
		func(_ workflow.Context, input string, emit workflow.EmitFunc) (any, error) {
			route := "LOGISTICS"
			if strings.Contains(strings.ToLower(input), "bug") {
				route = "BUG"
			}
			return nil, emit(&workflow.Event{Output: input, Routes: []any{route}})
		},
		workflow.NodeConfig{},
	)

	bug := workflow.NewFunctionNode[string, string]("bug",
		func(_ workflow.Context, input string) (string, error) {
			return "Handling bug: " + input, nil
		},
		workflow.NodeConfig{},
	)

	fallback := workflow.NewFunctionNode[string, string]("fallback",
		func(_ workflow.Context, input string) (string, error) {
			return "Handling request: " + input, nil
		},
		workflow.NodeConfig{},
	)

	root, err := workflowagent.New(workflowagent.Config{
		Name: "routing_workflow",
		Edges: workflow.Concat(
			workflow.Chain(workflow.Start, classify),
			[]workflow.Edge{
				{From: classify, To: bug, Route: workflow.StringRoute("BUG")},
				{From: classify, To: fallback, Route: workflow.Default},
			},
		),
	})
	if err != nil {
		panic(err)
	}

	out, err := root.Generate(context.Background(), "demo-session", "bug in checkout")
	if err != nil {
		panic(err)
	}
	fmt.Println(out)
}
```

See `cmd/example/graph_workflow` for a runnable no-key example.

### Durable Workflow Runs

For multi-step work that must survive a process restart or a transient node
failure, execute the graph through a `workflow.RunStore`. Each completed node
transition is checkpointed. Resume the same run ID to continue from its saved
queue; a completed run returns its saved result without invoking nodes again.

```go
store, err := workflow.NewFileRunStore("./workflow-runs")
if err != nil {
	log.Fatal(err)
}

out, err := graph.StartRun(ctx, store, "invoice-1042", "customer-7", input)
if err != nil {
	// Resolve transient dependencies, restart the process, then continue.
	out, err = graph.ResumeRun(ctx, store, "invoice-1042")
}
if err != nil {
	log.Fatal(err)
}
fmt.Println(out)
```

`workflow.NewInMemoryRunStore()` is available for tests. `FileRunStore` uses
one atomically replaced JSON file per run; production applications can provide
a database-backed `workflow.RunStore`. Persisted inputs, outputs, join values,
and `workflow.Context.State` must be JSON-serializable. Execution is
at-least-once: a crash after a node side effect but before its checkpoint may
invoke that node again, so side-effecting nodes should be idempotent.

## Memory

Every agent needs a `*memory.SessionMemory`. The session layer keeps recent conversation turns and can retrieve long-term records from a vector store.

Common backends:

| Backend | Constructor or module |
| --- | --- |
| In-memory | `memory.NewInMemoryStore()` or `modules.InMemoryMemoryModule(...)` |
| PostgreSQL + pgvector | `memory.NewPostgresStore(...)` or `modules.InPostgresMemory(...)` |
| Qdrant | `memory.NewQdrantStore(...)` or `modules.InQdrantMemory(...)` |
| MongoDB | `memory.NewMongoStore(...)` or `modules.InMongoMemory(...)` |
| Neo4j | `memory.NewNeo4jStore(...)` or `modules.InNeo4jMemory(...)` |

Minimal in-memory setup:

```go
mem := memory.NewSessionMemory(
	memory.NewMemoryBankWithStore(memory.NewInMemoryStore()),
	8,
)
```

Persistent stores that support schema setup implement `memory.SchemaInitializer`.

```go
store, err := memory.NewPostgresStore(ctx, connStr)
if err != nil {
	log.Fatal(err)
}
defer store.Close()

if err := store.CreateSchema(ctx, ""); err != nil {
	log.Fatal(err)
}
```

## File Context

Use `GenerateWithFiles` when you already have file bytes in memory. Text files are included in the prompt context; supported image/video MIME types are passed through provider-specific paths where available.

```go
files := []models.File{
	{
		Name: "notes.md",
		MIME: "text/markdown",
		Data: []byte("# Notes\nShip the README update."),
	},
}

out, err := a.GenerateWithFiles(ctx, "demo-session", "Summarize this file.", files)
```

## Tools

Tools are small Go interfaces with a JSON-schema-like spec and an invocation function.

```go
type EchoTool struct{}

func (EchoTool) Spec() agent.ToolSpec {
	return agent.ToolSpec{
		Name:        "echo",
		Description: "Returns the input text.",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"input": map[string]any{
					"type": "string",
				},
			},
			"required": []string{"input"},
		},
	}
}

func (EchoTool) Invoke(ctx context.Context, req agent.ToolRequest) (agent.ToolResponse, error) {
	return agent.ToolResponse{Content: fmt.Sprint(req.Arguments["input"])}, nil
}
```

Register tools directly when constructing an agent to keep them in the agent catalog and expose them through `a.Tools()` or ADK tool bundles:

```go
a, err := agent.New(agent.Options{
	Model:  model,
	Memory: mem,
	Tools:  []agent.Tool{EchoTool{}},
})
```

For model-selected tool execution across providers and processes, wire execution through UTCP. Agents can also be exposed as UTCP tools.

Models that implement `models.ToolCallingAgent` use provider-native tool calls automatically. The OpenAI adapter supports this path; other models continue through the prompt-based planner. Native tool calls are not cached because they may execute side effects.

## Agents As Tools

Any `*agent.Agent` can be wrapped as a local `agent.Tool`.

```go
researcher, _ := agent.New(agent.Options{
	Model:        researcherModel,
	Memory:       researcherMemory,
	SystemPrompt: "You are a research specialist.",
})

manager, _ := agent.New(agent.Options{
	Model:        managerModel,
	Memory:       managerMemory,
	SystemPrompt: "You delegate research work.",
	Tools: []agent.Tool{
		researcher.AsTool("researcher", "Delegates research to a specialist agent."),
	},
})
```

You can also register an agent as a UTCP provider:

```go
client, err := utcp.NewUTCPClient(ctx, &utcp.UtcpClientConfig{}, nil, nil)
if err != nil {
	log.Fatal(err)
}

if err := researcher.RegisterAsUTCPProvider(
	ctx,
	client,
	"agent.researcher",
	"Specialist research agent",
); err != nil {
	log.Fatal(err)
}

result, err := client.CallTool(ctx, "agent.researcher", map[string]any{
	"instruction": "Find three facts about pgvector.",
})
```

## Guardrails

Input guardrails validate or transform user input before the model call. Output guardrails validate or repair model responses before they are returned.

```go
inputGuardrails := &agent.InputGuardrails{
	SafetyPolicies: []agent.InputSafetyPolicy{
		agent.NewPromptInjectionDetectorPolicy(nil),
	},
	Transformers: []agent.InputTransformer{
		agent.NewPIIMaskerTransformer(true, true, false, false),
	},
}

outputPolicy, err := agent.NewRegexBlocklistPolicy([]string{
	`(?i)\bpassword\s*=`,
})
if err != nil {
	log.Fatal(err)
}

a, err := agent.New(agent.Options{
	Model:           model,
	Memory:          mem,
	InputGuardrails: inputGuardrails,
	Guardrails: &agent.OutputGuardrails{
		SafetyPolicies: []agent.SafetyPolicy{outputPolicy},
	},
})
```

See `cmd/example/guardrails` for a complete runnable example.

## Checkpoint And Restore

Checkpointing serializes the agent system prompt, short-term memory, shared-space memberships, and timestamp.

```go
data, err := a.Checkpoint()
if err != nil {
	log.Fatal(err)
}

restored, err := agent.New(agent.Options{
	Model:  model,
	Memory: freshMemory,
})
if err != nil {
	log.Fatal(err)
}

if err := restored.Restore(data); err != nil {
	log.Fatal(err)
}
```

See `cmd/example/checkpoint` for a disk-backed example.

## CodeMode

Lattice can integrate with UTCP CodeMode and chain execution:

- `adk.WithUTCP(client)` makes remote/discovered UTCP tools available to the agent.
- `adk.WithCodeModeUtcp(client, model)` enables Go-code tool orchestration through CodeMode.
- `Agent.AllowUnsafeTools` must be enabled before `codemode.run_code` can execute.

Use these features only in trusted environments. CodeMode executes generated Go snippets through the configured UTCP runtime.

## Examples

No-key examples:

```bash
go run ./cmd/example/composability
go run ./cmd/example/guardrails
go run ./cmd/example/checkpoint
```

Provider-backed examples:

```bash
# Requires GOOGLE_API_KEY or GEMINI_API_KEY by default.
go run ./cmd/example/codemode

# Requires provider credentials and a Qdrant instance unless flags are changed.
go run ./cmd/app -provider openai -model gpt-4o-mini -message "Summarize this project"

# Requires provider credentials and PostgreSQL + pgvector unless flags are changed.
go run ./cmd/example -provider openai -model gpt-4o-mini -message "Summarize this project"
```

Specialized workflows:

| Path | Demonstrates |
| --- | --- |
| `cmd/example/agent_as_tool` | Registering an agent as a UTCP tool |
| `cmd/example/agent_as_utcp_codemode` | Orchestrating agent tools through CodeMode |
| `cmd/example/codemode_utcp_workflow` | Analyst/writer/reviewer workflow |
| `cmd/example/autonomous_agent` | Configurable multi-agent coordinator |
| `cmd/example/autonomous_cron` | Autonomous periodic task pattern |
| `cmd/example/claw_cron` | Task store, permission gateway, and specialist agents |
| `cmd/codemode` | CodeMode CLI wiring |

## Repository Layout

```text
.
|-- agent.go                 # Core Agent runtime
|-- agent_stream.go          # Streaming responses
|-- agent_tool.go            # Agent-as-tool and UTCP provider adapters
|-- input_guardrails.go      # Input validation and transforms
|-- safety_policies.go       # Output safety policies
|-- catalog.go               # Tool and sub-agent registries
|-- src/
|   |-- adk/                 # Agent Development Kit and modules
|   |-- cache/               # LRU cache utilities
|   |-- concurrent/          # Worker pool helpers
|   |-- helpers/             # Small CLI/config helpers
|   |-- memory/              # Session memory, engine, stores, embedders
|   |-- models/              # LLM provider adapters
|   |-- subagents/           # Built-in specialist agents
|   `-- swarm/               # Multi-agent coordination primitives
`-- cmd/
    |-- app/                 # Qdrant-backed CLI
    |-- codemode/            # CodeMode CLI
    `-- example/             # Runnable examples
```

## Development

```bash
# Run all tests.
go test ./...

# Run one package.
go test ./src/memory/engine

# Run one test.
go test ./... -run TestCheckpoint

# Format changed Go files.
gofmt -w path/to/file.go
```

FastEmbed support is behind the `fastembed` build tag:

```bash
go test -tags fastembed ./src/memory/embed
```

## Adding Components

Add a model provider by implementing `src/models.Agent`:

```go
type Agent interface {
	Generate(context.Context, string) (any, error)
	GenerateWithFiles(context.Context, string, []File) (any, error)
	GenerateStream(context.Context, string) (<-chan StreamChunk, error)
}
```

Add a memory backend by implementing `memory.VectorStore`. Add `memory.SchemaInitializer` if the backend needs schema/bootstrap support.

Add a tool by implementing `agent.Tool`, then register it through `agent.Options`, an ADK tool provider, or a UTCP provider depending on how it should be discovered and executed.

## Troubleshooting

### Missing API Key

Provider constructors fail when required keys are missing. Set the matching environment variable or use `models.NewDummyLLM` for local tests.

### No Long-Term Memory Results

Check that the session uses a store-backed `MemoryBank`, an embedder is configured, and records have been flushed or stored through the memory engine.

### PostgreSQL Vector Errors

For pgvector-backed memory, enable the extension:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Then run the store schema initializer:

```go
_ = store.CreateSchema(ctx, "")
```

### Tool Not Found

Confirm the tool name exactly matches the registered UTCP tool name. Fully qualified names such as `agent.researcher` are preferred when multiple providers expose similar tools.

## License

See [LICENSE](./LICENSE).
