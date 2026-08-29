# go-agent

[![Go Version](https://img.shields.io/badge/Go-1.25.10-00ADD8?logo=go&logoColor=white)](https://go.dev/dl/)
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
- Composable retry, timeout, rate-limit, and token-budget model middleware

## Local Skills

Agents automatically load local instructions from `.skills` in the process working directory. Use either the conventional `SKILL.md` layout or Markdown files directly in `.skills`:

```text
.skills/
├── code-review/
│   └── SKILL.md
└── release.md
```

Each skill is added to the system instructions for normal, streaming, file-backed, and tool-planning requests. `SKILL.md` may include optional YAML-style front matter:

```markdown
---
name: release
description: Prepare safe releases
---
Run the full test suite before proposing a release.
```

Set `Options.SkillsDir` to use another directory, call `agent.ReloadSkills()` after editing a long-running agent's files, or set `DisableSkills: true` to opt out.

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
| Vertex AI | `vertex`, `vertexai`, `vertex-ai` | `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION` (or `GOOGLE_CLOUD_REGION`), and Application Default Credentials |
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

Vertex AI uses the Google GenAI SDK and Application Default Credentials. For
local development, authenticate with `gcloud auth application-default login`,
then set the project and location before selecting the `vertex` provider:

```bash
export GOOGLE_CLOUD_PROJECT="my-project"
export GOOGLE_CLOUD_LOCATION="global"
```

## Model Middleware

Wrap any `models.Agent` with production policies before passing it to
`agent.New` or returning it from an ADK model provider.

```go
package main

import (
	"context"
	"log"
	"time"

	"github.com/Protocol-Lattice/go-agent/src/models"
	modelmw "github.com/Protocol-Lattice/go-agent/src/models/middleware"
)

func buildModel(ctx context.Context) models.Agent {
	base, err := models.NewLLMProvider(ctx, "openai", "gpt-4o-mini", "")
	if err != nil {
		log.Fatal(err)
	}

	budget, err := modelmw.NewTokenBudget(50_000, nil)
	if err != nil {
		log.Fatal(err)
	}

	model, err := modelmw.Wrap(
		base,
		modelmw.TimeoutPolicy{Duration: 30 * time.Second},
		modelmw.RetryPolicy{
			MaxAttempts:    3,
			InitialBackoff: 200 * time.Millisecond,
			MaxBackoff:     2 * time.Second,
		},
		modelmw.RateLimitPolicy{
			Requests: 60,
			Per:      time.Minute,
			Burst:    5,
			Mode:     modelmw.RateLimitWait,
		},
		modelmw.TokenBudgetPolicy{Budget: budget},
	)
	if err != nil {
		log.Fatal(err)
	}
	return model
}
```

Middleware is listed outermost first. In the order above, the timeout covers
the complete operation, including retry backoff. Every retry consumes a rate
limit permit and an estimated input-token charge.

`RateLimitWait` waits for capacity and respects context cancellation;
`RateLimitReject` returns `middleware.ErrRateLimitExceeded` immediately.
Retry middleware retries stream setup failures only, because restarting a
stream after chunks have been delivered could duplicate output.

Token budgets are concurrency-safe. Associate a budget with one request or
workflow through its context to override the policy's fallback budget:

```go
requestBudget, _ := modelmw.NewTokenBudget(8_000, nil)
runCtx := modelmw.ContextWithTokenBudget(ctx, requestBudget)
```

The default estimator uses approximately one token per four UTF-8 bytes. Pass
a provider-specific `modelmw.TokenEstimator` when exact tokenizer behavior is
required. Until provider usage metadata is normalized, budgets are estimates:
input is rejected before a call, streaming stops before forwarding the chunk
that crosses the budget, and an oversized non-streaming response is accounted
for but returned as `middleware.ErrTokenBudgetExceeded`.

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

Enable tool-orchestration diagnostics with `AGENT_ORCHESTRATOR_LOG=1` (`true`,
`yes`, and `on` are also accepted). The logs distinguish planner latency,
dispatch overhead, local/CodeMode/UTCP execution, stream setup, chunk count,
completion, and errors. Prompt text, argument values, and tool results are not
written to the log. Observed UTCP `tool_result` events also include
`duration_ms`, so gateways can expose the same timing to streaming clients.

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

Add a model policy by implementing `middleware.Middleware`; use
`middleware.MiddlewareFunc` for small wrappers.

## Workspace intelligence

`go-agent` includes a repository-aware Workspace Intelligence layer for coding agents. It builds a structural index of the codebase and can combine symbol search, imports, dependencies, embeddings, and source context into a bounded context for an agent.

```text
Repository
   │
   ├── AST parser
   ├── Symbol index
   ├── Import/dependency graph
   ├── Embeddings (optional)
   └── File metadata
          │
          ▼
    Hybrid Context Builder
          │
          ▼
        Agent
```

## Basic usage

Build an index for a Go repository:

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/Protocol-Lattice/go-agent/workspace"
)

func main() {
    ctx := context.Background()

    index := workspace.NewIndex(workspace.DefaultConfig("."))
    if err := index.Build(ctx); err != nil {
        log.Fatal(err)
    }

    results := index.SearchSymbols(ctx, "Authenticate", 10)
    for _, result := range results {
        fmt.Printf("%s %s:%d (score=%d)\n",
            result.Symbol.Kind,
            result.Symbol.File,
            result.Symbol.StartLine,
            result.Score,
        )
    }
}
```

The default configuration indexes Go files and ignores `.git`, `vendor`, `node_modules`, `dist`, `build`, and `tmp` directories.

## Building agent context

`BuildContext` turns a natural-language task into a bounded set of relevant source files. Structural symbol matches are used first, and semantic retrieval can be enabled when an embedder is configured.

```go
ctx, err := index.BuildContext(context.Background(), workspace.ContextRequest{
    Query:      "Fix the authentication timeout bug",
    MaxBytes:   32 << 10,
    MaxFiles:   8,
    MaxResults: 20,
    Semantic:   true,
})
if err != nil {
    log.Fatal(err)
}

for _, file := range ctx.Files {
    fmt.Printf("--- %s ---\n%s\n", file.Path, file.Content)
}
```

The context builder:

1. searches indexed symbols;
2. optionally performs semantic/vector search;
3. selects relevant files;
4. expands the selection by one dependency hop;
5. enforces a deterministic file and byte budget.

This lets a coding agent retrieve focused repository context instead of loading the whole workspace.

## Semantic search

Embeddings are optional. The workspace package uses a small `Embedder` interface, so an application can connect any embedding provider without coupling the index to a particular vendor.

```go
type Embedder interface {
    Embed(context.Context, string) ([]float32, error)
}
```

Configure the embedder before building the index:

```go
index := workspace.NewIndex(workspace.DefaultConfig("."))
index.SetEmbedder(myEmbedder)

if err := index.Build(ctx); err != nil {
    log.Fatal(err)
}

results, err := index.SearchSemantic(ctx, "authentication timeout", 10)
if err != nil {
    log.Fatal(err)
}
```

Semantic search is optional; symbol and structural search continue to work without an embedding provider.

## Incremental updates

For long-running coding agents, re-index only changed files:

```go
err := index.Update(ctx, []workspace.FileChange{
    {
        Path: "internal/auth/service.go",
        Kind: workspace.ChangeModified,
    },
})
if err != nil {
    log.Fatal(err)
}
```

Deleted files are removed from the file, symbol, import, and embedding indexes:

```go
err := index.Update(ctx, []workspace.FileChange{
    {
        Path: "internal/auth/legacy.go",
        Kind: workspace.ChangeDeleted,
    },
})
```

Only modified files are parsed and, when semantic indexing is enabled, re-embedded.

## Live workspace watcher

Use `workspace.Watcher` to keep the index synchronized with a working tree:

```go
watcher := &workspace.Watcher{
    Index:    index,
    Interval: 500 * time.Millisecond,
}

go func() {
    if err := watcher.Run(ctx); err != nil && !errors.Is(err, context.Canceled) {
        log.Printf("workspace watcher: %v", err)
    }
}()
```

The watcher uses dependency-free polling. It detects created, modified, and deleted files and forwards changes to `Index.Update`.

## Dependencies

The index also exposes imports and module-aware dependency relationships:

```go
imports := index.Imports("internal/auth/service.go")

for _, path := range imports {
    fmt.Println(path)
}

for _, dependency := range index.Dependencies("internal/auth/service.go") {
    fmt.Println("depends on:", dependency)
}
```

This structural graph can be used by higher-level context ranking to expand from a matched symbol into the implementation and its directly related packages.

## Recommended agent pipeline

For coding agents, the intended architecture is:

```text
User task
   │
   ▼
Workspace Index
   ├── symbols
   ├── imports
   ├── dependencies
   └── embeddings
   │
   ▼
Context Builder
   ├── lexical relevance
   ├── semantic relevance
   ├── dependency expansion
   └── token/byte budget
   │
   ▼
Agent
   │
   ├── plan
   ├── edit
   ├── test
   └── validate
```

For production coding agents, keep the index alive for the lifetime of the agent and run the watcher alongside the agent execution loop. This avoids rebuilding the repository index after every request.

## Autonomous UTCP agent CLI

This project provides an OpenClaw-like CLI on top of `github.com/Protocol-Lattice/go-agent`:

- UTCP codemode via `WithCodeModeUtcp(...)`
- Specialist agents registered as UTCP tools via `RegisterAsUTCPProvider(...)`
- Command-driven UX: `agent`, `loop`, `chat`, `tools`, `doctor`

## Commands

- `agent`: single-turn execution (default target is `coordinator`)
- `loop`: autonomous multi-step execution until `AUTONOMOUS_DONE`, with bounded self-healing recovery after execution failures
- `chat`: interactive REPL with runtime agent switching
- `tools`: list configured tools or live registered UTCP tools
- `doctor`: validate provider/model/env setup

## Specialist UTCP Tools

Default tools are prefixed with `local_` and registered at runtime:

- `local_researcher.run`
- `local_builder.run`
- `local_reviewer.run`

## Common Flags

- `--provider` (default from `LLM_PROVIDER`, fallback `gemini`)
- `--model` (default from `LLM_MODEL`, fallback `gemini-3-flash-preview`)
- `--session-id` (default from `AGENT_SESSION`, fallback `autonomous-session`)
- `--context-window` (default `20`)
- `--max-recoveries` (default `2`; total self-healing recovery budget for the loop, `0` disables recovery)
- `--tool-prefix` (default from `UTCP_TOOL_PREFIX`, fallback `local.`)

## Usage

```bash
# Single turn through coordinator
go run ./cmd/example/autonomous_agent agent \
  --message "Draft rollout plan for UTCP migration"

# Single turn through specialist
go run ./cmd/example/autonomous_agent agent \
  --agent reviewer \
  --message "What could fail in this deploy plan?"

# Autonomous loop
go run ./cmd/example/autonomous_agent loop \
  --goal "Design and verify a UTCP-based repository triage workflow" \
  --max-steps 8

# Interactive mode
go run ./cmd/example/autonomous_agent chat --goal "Prepare release plan"

# Tools
go run ./cmd/example/autonomous_agent tools
go run ./cmd/example/autonomous_agent tools --live

# Environment checks
go run ./cmd/example/autonomous_agent doctor
```

## Chat Commands

Inside `chat`:

- `/help`
- `/tools`
- `/agent <name>`
- `/exit`

## Notes

- You must provide provider credentials through environment variables expected by your selected provider.
- Runtime now validates provider credentials before bootstrapping agents and reports missing keys explicitly.
- If `ADK_EMBED_PROVIDER` is unset, it is inferred for known providers (`gemini`, `openai`, `ollama`); otherwise set it manually.
- `loop` completes when the model emits `AUTONOMOUS_DONE`; otherwise it exits at `--max-steps`. Failed iterations consume the loop's `--max-recoveries` budget without consuming another step. Each recovery includes the failure in the scratchpad so the coordinator can choose a different approach. Context cancellation and deadline errors are never retried.
- `agent --thinking`, `agent --local`, and `agent --deliver` are included for OpenClaw-like UX compatibility.

## Next steps

- Run the no-key graph workflow to learn the node and edge model.
- Use the autonomous CLI when you want a ready-made coordinator and specialist setup.
- Read `docs/workspace-intelligence.md` before building a repository-aware coding agent.
- Browse `cmd/example` for runnable patterns rather than copying partial snippets, then run `go test ./...` before upgrading dependencies or changing runtime behavior.

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

See [LICENSE](https://github.com/Protocol-Lattice/go-agent/blob/main/LICENSE).
