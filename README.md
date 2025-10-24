# Go Lattice - Agent Development Kit

[![Go Version](https://img.shields.io/badge/Go-1.25-00ADD8?logo=go&logoColor=white)](https://go.dev/dl/)
[![CI Status](https://github.com/Raezil/lattice-agent/actions/workflows/go.yml/badge.svg)](https://github.com/Raezil/lattice-agent/actions/workflows/go.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/Raezil/lattice-agent.svg)](https://pkg.go.dev/github.com/Raezil/lattice-agent)
[![Go Report Card](https://goreportcard.com/badge/github.com/Raezil/lattice-agent)](https://goreportcard.com/report/github.com/Raezil/lattice-agent)

Build production-ready AI agents in Go with a batteries-included toolkit. Lattice-Agent wraps language
models, tool execution, retrieval-augmented memory, and multi-agent coordination behind pragmatic
interfaces so you can focus on domain logic instead of orchestration plumbing.

---

## Table of Contents
1. [Why lattice-agent?](#why-lattice-agent)
2. [Feature Highlights](#feature-highlights)
3. [Package Tour](#package-tour)
4. [Quick Start](#quick-start)
5. [Configuring Providers](#configuring-providers)
6. [Memory Engine](#memory-engine)
7. [Shared Spaces (Swarm)](#shared-spaces-swarm)
8. [Development Workflow](#development-workflow)
9. [Troubleshooting](#troubleshooting)
10. [Contributing](#contributing)
11. [License](#license)

---

## Why lattice-agent?
Modern agents demand more than a single model call—they need deterministic session orchestration,
reusable tool adapters, rich memory, and safe extensibility. GO-ADK delivers:

- **Composable runtime** – stitch together models, tools, sub-agents, and memory with minimal glue.
- **Opinionated defaults** – sensible conventions that stay out of the way when you customise.
- **Production-ready patterns** – test-friendly abstractions, UTCP support, and pluggable storage.

Whether you are experimenting locally or embedding agents into an existing service, GO-ADK provides a
consistent foundation.

## Feature Highlights
- **Modular agent kit**: `pkg/adk` exposes a module system for assembling deployments with declarative
  options.
- **Coordinator & specialists**: `pkg/agent` hosts the core runtime. `pkg/subagents` shows how to wire
  persona-specific specialists through a `ToolCatalog` and `SubAgentDirectory`.
- **Tooling ecosystem**: implement the `agent.Tool` interface and register tooling under `pkg/tools` to
  make it available to every agent automatically.
- **Retrieval-augmented memory**: `pkg/memory` layers importance scoring, weighted retrieval, MMR, and
  pruning over pluggable vector stores (PostgreSQL + pgvector, Qdrant, in-memory).
- **Model abstraction**: `pkg/models` provides adapters for Gemini 2.5 Pro and dummy/offline models.
  Anthropic, Ollama, and future providers can plug in via the same interface.
- **Universal Tool Calling Protocol (UTCP)**: first-class UTCP support for portable tool invocation.
- **Command-line demos**: `cmd/demo` illustrates tool usage, delegation, and memory persistence from
  the terminal.

## Package Tour
```
cmd/
├── demo              # Interactive CLI showcasing tools, delegation, and memory
├── quickstart        # Minimal sample wired through the high-level kit
└── team              # Multi-agent coordination examples
pkg/
├── adk               # Modular Agent Development Kit and module interfaces
├── agent             # Coordinator runtime, routing, and delegation logic
├── memory            # Memory engine, vector-store adapters, pruning strategies
├── models            # Gemini, Ollama, Anthropic, and dummy model adapters
├── subagents         # Specialist personas built on top of the runtime
└── tools             # Built-in tools (echo, calculator, time, etc.)
```

## Quick Start
### Prerequisites
- **Go** 1.22 or newer (1.25 recommended)
- **PostgreSQL** 15+ with the [`pgvector`](https://github.com/pgvector/pgvector) extension (optional
  for local experimentation, required for long-term memory persistence)
- **Gemini API key** exported as `GOOGLE_API_KEY` or `GEMINI_API_KEY`

### Install
```bash
git clone https://github.com/Raezil/lattice-agent.git
cd go-agent-development-kit
go mod download
```

### Configure Environment
Set up API keys and database connectivity as needed:

| Variable | Description |
| --- | --- |
| `GOOGLE_API_KEY` / `GEMINI_API_KEY` | Credentials for Gemini models. |
| `DATABASE_URL` | PostgreSQL DSN (e.g. `postgres://admin:admin@localhost:5432/ragdb?sslmode=disable`). |
| `ADK_EMBED_PROVIDER` | Optional embedding provider override (defaults to Gemini for the demo). |

```bash
export GOOGLE_API_KEY="<your-gemini-api-key>"
export DATABASE_URL="postgres://admin:admin@localhost:5432/ragdb?sslmode=disable"
export ADK_EMBED_PROVIDER="gemini"
```

## Configuring Providers
The modular kit lets you declare models, memory, and tools with a few options:

```go
adkAgent, err := adk.New(ctx,
    kit.WithModules(
        modules.NewModelModule("coordinator", modules.StaticModelProvider(models.NewDummyLLM("Coordinator:"))),
        modules.InMemoryMemoryModule(8),
        modules.NewToolModule("default-tools", modules.StaticToolProvider([]agent.Tool{&tools.EchoTool{}}, nil)),
    ),
)
if err != nil {
    log.Fatal(err)
}
agent, err := adkAgent.BuildAgent(ctx)
```

Providers can register models, memory engines, tool catalogs, sub-agents, and even the runtime. Combine
them with `kit.WithAgentOptions` to share defaults (system prompts, UTCP clients, etc.) across builds.

## Memory Engine
`pkg/memory` offers an `Engine` that composes retrieval heuristics, clustering, and pruning atop any
`VectorStore` implementation.

```go
store := memory.NewInMemoryStore() // or postgres/qdrant implementation
engine := memory.NewEngine(store, memory.Options{}).
    WithEmbedder(memory.DummyEmbedder{})

sessionMemory := memory.NewSessionMemory(memory.NewMemoryBankWithStore(store), 8).
    WithEngine(engine)
```

Schema migrations rely on online-safe `ALTER TABLE ... IF NOT EXISTS` statements to avoid downtime.

## Shared Spaces (Swarm)
Coordinate multiple agents that **share memory across named spaces** (e.g. `team:core`, `team:project-x`).
Swarm builds on `memory.SharedSession` to read and write memories visible to everyone in the same space.

Use it when:
- Cooperating agents (e.g. *alpha*, *beta*, *researcher*) need to see each other’s latest notes.
- You want shared RAG context from a team space while keeping local short-term memory.
- You need explicit access control—grant read/write permissions per space and per session ID.

## Development Workflow
1. Update Go code or modules.
2. Format using `gofmt` (or your editor tooling).
3. Run the test suite:
   ```bash
   go test ./...
   ```
4. Update documentation and examples when adding new tools, models, or memory backends.

## Troubleshooting
- **Missing pgvector extension** – Ensure `CREATE EXTENSION vector;` runs before connecting; otherwise
  the Postgres-backed memory store cannot create vector columns.
- **API key issues** – Confirm `GOOGLE_API_KEY` or `GEMINI_API_KEY` is set in the same shell where you
  run the demo.
- **Tool discovery** – Check that tool names are unique when registering them in catalogs/directories.
- **Deterministic tests** – Use the dummy model adapter for repeatable orchestration logic in unit tests.

## Contributing
Issues and pull requests are welcome! Please update the README and examples when contributing new
models, tools, or memory backends so the community benefits from your additions.

## License
This project is licensed under the [Apache 2.0 License](./LICENSE).

## 🙏 Acknowledgments
- Google’s [Agent Development Kit (Python)](https://github.com/google/adk-python) for foundational ideas.
