# go-agent

[![Go version](https://img.shields.io/badge/Go-1.25-00ADD8?logo=go&logoColor=white)](https://go.dev/dl/)
[![CI](https://github.com/Protocol-Lattice/go-agent/actions/workflows/go.yml/badge.svg)](https://github.com/Protocol-Lattice/go-agent/actions/workflows/go.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/Protocol-Lattice/go-agent.svg)](https://pkg.go.dev/github.com/Protocol-Lattice/go-agent)

Build AI agents in Go without giving up control of the runtime. `go-agent`
provides a small agent core, model providers, memory, tools, guardrails, and
deterministic workflows that compose as normal Go code.

## Start here

Requires Go `1.25.10` or newer.

```bash
go get github.com/Protocol-Lattice/go-agent
```

This complete example needs no API key. It uses the in-memory store and dummy
model, so it is also a useful wiring test.

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
	mem := memory.NewSessionMemory(
		memory.NewMemoryBankWithStore(memory.NewInMemoryStore()),
		8,
	)

	a, err := agent.New(agent.Options{
		Model:        models.NewDummyLLM("local: "),
		Memory:       mem,
		SystemPrompt: "You are concise and helpful.",
	})
	if err != nil {
		log.Fatal(err)
	}

	response, err := a.Generate(context.Background(), "demo", "Say hello in one sentence.")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(response)
}
```

For a real model, replace the dummy model:

```go
model, err := models.NewLLMProvider(ctx, "openai", "gpt-4o-mini", "")
```

## What to use when

| Need | Start with |
| --- | --- |
| One agent, tests, or a small service | `agent.New(agent.Options{...})` |
| Reusable application wiring | `adk.New(...)` with modules |
| A model call with text, images, or files | `Agent.GenerateWithFiles(...)` |
| Streaming output | `Agent.GenerateStream(...)` |
| Long-lived conversation context | `memory.SessionMemory` |
| Tool use or specialist agents | `Tool`, `SubAgent`, or UTCP / CodeMode |
| Deterministic multi-step flow | `src/adk/workflow` |
| Retries, timeouts, quotas | `src/models/middleware` |

## Providers

Create a provider through `models.NewLLMProvider`. Set the matching credential
before starting your application.

| Provider | Names | Environment |
| --- | --- | --- |
| Gemini | `gemini`, `google` | `GOOGLE_API_KEY` or `GEMINI_API_KEY` |
| Vertex AI | `vertex`, `vertexai`, `vertex-ai` | Application Default Credentials, `GOOGLE_CLOUD_PROJECT`, and `GOOGLE_CLOUD_LOCATION` |
| OpenAI | `openai` | `OPENAI_API_KEY` or `OPENAI_KEY` |
| Anthropic | `anthropic`, `claude` | `ANTHROPIC_API_KEY` |
| Ollama | `ollama` | Optional `OLLAMA_HOST` (defaults to `http://localhost:11434`) |
| OpenRouter | `openrouter` | `OPENROUTER_API_KEY` or `OPENROUTER_KEY` |

For semantic memory, select an embedder with `memory.AutoEmbedder()`. Set
`ADK_EMBED_PROVIDER` and, optionally, `ADK_EMBED_MODEL` when automatic
selection is not appropriate.

## Build an application

Use the ADK once an application needs configurable models, memory, tools, or
shared runtime setup:

```go
kit, err := adk.New(ctx,
	adk.WithDefaultSystemPrompt("You are a helpful assistant."),
	adk.WithModules(
		modules.NewModelModule("llm", func(ctx context.Context) (models.Agent, error) {
			return models.NewLLMProvider(ctx, "openai", "gpt-4o-mini", "")
		}),
		modules.InMemoryMemoryModule(8, memory.AutoEmbedder(), nil),
	),
)
if err != nil {
	return err
}

a, err := kit.BuildAgent(ctx)
if err != nil {
	return err
}
```

Wrap the model with `src/models/middleware` before passing it to the agent
when running in production. The middleware package provides timeout, retry,
rate-limit, and token-budget policies.

## Core capabilities

### Memory and files

Every agent receives a `*memory.SessionMemory`. Use an in-memory store for
tests, or connect PostgreSQL/pgvector, Qdrant, MongoDB, or Neo4j for persistent
semantic memory.

Pass files already held in memory to `GenerateWithFiles`:

```go
files := []models.File{{
	Name: "notes.md",
	MIME: "text/markdown",
	Data: []byte("# Release notes\nShip the README."),
}}

response, err := a.GenerateWithFiles(ctx, "demo", "Summarize this file.", files)
```

### Tools, agents, and workflows

An agent can invoke regular Go tools, delegate to subagents, or use UTCP tools.
Use CodeMode when a model should plan and execute tool-calling Go code. See the
runnable examples below for each pattern.

For explicit, inspectable control flow, build a graph with
`src/adk/workflow`. Graphs support branching, fan-out/join, cancellation, and
durable runs through a `workflow.RunStore`.

### Local skills and guardrails

The agent loads instructions from `.skills` in the working directory. Add a
`SKILL.md` or a Markdown file, set `Options.SkillsDir` for a different
location, or set `DisableSkills: true` to turn the feature off.

Use input and output guardrails to block unsafe requests, prompt injections,
or disallowed model output. Policies include regex blocklists, PII masking,
and optional LLM evaluation.

## Examples

Run any example from the repository root:

| Example | Demonstrates | Command |
| --- | --- | --- |
| [Graph workflow](cmd/example/graph_workflow) | Routing and deterministic workflows | `go run ./cmd/example/graph_workflow` |
| [CodeMode](cmd/example/codemode) | Agent-directed UTCP tool calls | `go run ./cmd/example/codemode` |
| [Multi-agent workflow](cmd/example/codemode_utcp_workflow) | Specialist-agent orchestration | `go run ./cmd/example/codemode_utcp_workflow` |
| [Checkpoint](cmd/example/checkpoint) | Save and restore agent state | `go run ./cmd/example/checkpoint` |
| [Guardrails](cmd/example/guardrails) | Input and output policies | `go run ./cmd/example/guardrails` |
| [Autonomous CLI](cmd/example/autonomous_agent) | Coordinator, specialists, and a self-healing loop | `go run ./cmd/example/autonomous_agent --help` |

More detail is available in [the examples guide](cmd/example/README.md).

## Development

```bash
# Clone and verify the repository.
git clone https://github.com/Protocol-Lattice/go-agent.git
cd go-agent
go test ./...

# Format changed Go files.
gofmt -w path/to/file.go
```

The public API reference is available on [pkg.go.dev](https://pkg.go.dev/github.com/Protocol-Lattice/go-agent).

## License

See [LICENSE](LICENSE).

