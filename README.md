# Go Agent Development Kit

[![Go Version](https://img.shields.io/badge/Go-1.25-00ADD8?logo=go&logoColor=white)](https://go.dev/dl/) [![CI Status](https://github.com/Raezil/go-agent-development-kit/actions/workflows/go.yml/badge.svg)](https://github.com/Raezil/go-agent-development-kit/actions/workflows/go.yml) [![Go Reference](https://pkg.go.dev/badge/github.com/Raezil/go-agent-development-kit.svg)](https://pkg.go.dev/github.com/Raezil/go-agent-development-kit) [![Go Report Card](https://goreportcard.com/badge/github.com/Raezil/go-agent-development-kit)](https://goreportcard.com/report/github.com/Raezil/go-agent-development-kit)



Build production-ready AI agents in Go with a batteries-included toolkit. The Go Agent Development Kit (GO-ADK) wraps language models, tool execution, retrieval-augmented memory, and multi-agent coordination behind a pragmatic API so you can focus on domain-specific logic instead of orchestration plumbing.

---

## Table of Contents
1. [Why GO-ADK?](#why-gadk)
2. [Key Features](#key-features)
3. [Architecture Overview](#architecture-overview)
4. [Quick Start](#quick-start)
5. [Advanced Memory Engine](#advanced-memory-engine)
6. [Swarm (Shared Spaces)](#swarm-shared-spaces)
7. [Development Workflow](#development-workflow)
8. [Troubleshooting Tips](#troubleshooting-tips)
9. [Contributing](#contributing)

---

## Why GO-ADK?
Modern agents demand more than a single model call—they require deterministic session orchestration, reusable tool adapters, rich memory, and safe extensibility. GADK provides:

- A composable runtime that stitches together models, tools, memory, and sub-agents.
- Opinionated defaults that stay out of your way when you need to customise.
- Production-friendly patterns
Whether you are experimenting locally or embedding agents inside an existing service, GADK provides a consistent foundation.

## Key Features
- **Runtime orchestration** – `pkg/runtime` exposes a single entry point for constructing an agent runtime with configurable models, tools, memory engines, and sub-agent registries. A thread-safe session manager keeps execution deterministic.
- **Modular Agent Development Kit** – `pkg/adk` layers a pluggable module system on top of the existing adk, memory, model, and tool abstractions so you can compose deployments with a few declarative options.
- **Coordinator + specialists** – `pkg/agent` contains the core coordinator logic, while `pkg/subagents` demonstrates how to plug in specialist personas (for example, a researcher) through a `ToolCatalog` and `SubAgentDirectory` abstraction.
- **Tooling ecosystem** – Implement the `agent.Tool` interface and register implementations (echo, calculator, clock, etc.) under `pkg/tools`. Tools become available to the coordinator prompt automatically.
- **Retrieval-augmented memory** – `pkg/memory` layers importance scoring, weighted retrieval (similarity, recency, source, importance), maximal marginal relevance, summarisation, and pruning strategies over pluggable vector stores (PostgreSQL + pgvector, Qdrant, in-memory).
- **Model abstraction** – `pkg/models` defines a slim `Generate(ctx, prompt)` interface with adapters for Gemini 2.5 Pro and dummy models for offline testing. Other providers (Anthropic, Ollama) can slot in via the same contract.
- **Universal Tool Calling Protocol (UTCP)** – Native UTCP support enables the runtime to describe and invoke tools through a modern, provider-agnostic contract.
- **Command-line demo** – `cmd/demo` showcases tool usage, delegation, and memory persistence through a configurable CLI conversation.

## Architecture Overview
```
cmd/demo              # CLI entry point that configures and drives the runtime
cmd/quickstart        # Zero-config sample wired through the high-level kit
pkg/
├── adk               # Modular Agent Development Kit and module interfaces
├── agent            # Coordinator agent, tool routing, sub-agent delegation
├── memory           # Memory engine, vector-store adapters, pruning strategies
├── models           # Gemini, Anthropic, Ollama, Dummy model adapters
├── subagents        # Example researcher persona powered by an LLM
└── tools            # Built-in tools (echo, calculator, time)
```

## Quick Start
### Prerequisites
- **Go** 1.22+
- **PostgreSQL** 15+ with the [`pgvector`](https://github.com/pgvector/pgvector) extension (optional for local experimentation, required for long-term memory persistence)
- **Gemini API key** exported as `GOOGLE_API_KEY` or `GEMINI_API_KEY`

### Clone and Install
```bash
git clone https://github.com/Raezil/go-agent-development-kit.git
cd go-agent-development-kit
go mod download
```

### Configure Environment Variables
| Variable | Description |
| --- | --- |
| `GOOGLE_API_KEY` / `GEMINI_API_KEY` | Credentials for Gemini models. |
| `DATABASE_URL` | PostgreSQL DSN (e.g. `postgres://admin:admin@localhost:5432/ragdb?sslmode=disable`). |
| `ADK_EMBED_PROVIDER` | Embedding provider override (defaults to Gemini for the demo). |

```bash
export GOOGLE_API_KEY="<your-gemini-api-key>"
export DATABASE_URL="postgres://admin:admin@localhost:5432/ragdb?sslmode=disable"
export ADK_EMBED_PROVIDER="gemini"
```

### Provision PostgreSQL (Optional but Recommended)
Install the `pgvector` extension and let the CLI bootstrap schemas automatically:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### Building with Modules
The `pkg/adk` package introduces a lightweight module system so you can provision capabilities declaratively:

```go
adk_agent, _ := adk.New(ctx,
    kit.WithModules(
        modules.NewModelModule("coordinator", modules.StaticModelProvider(models.NewDummyLLM("Coordinator:"))),
        modules.InMemoryMemoryModule(8),
        modules.NewToolModule("default-tools", modules.StaticToolProvider([]agent.Tool{&tools.EchoTool{}}, nil)),
    ),
)
agent, _ := adk_agent.BuildAgent(ctx)
```

Modules can register custom providers for models, memory, tools, sub-agents, or even the runtime itself. Combine them with `kit.WithAgentOptions` to push shared defaults (system prompts, UTCP clients, etc.) across every agent you build.

## Advanced Memory Engine
The `pkg/memory` package exposes an `Engine` that composes retrieval heuristics, clustering, and pruning on top of any `VectorStore`.

```go
store := memory.NewInMemoryStore() // or postgres/qdrant implementation
opts := memory.Options{}
engine := memory.NewEngine(store, opts).
    WithEmbedder(memory.DummyEmbedder{})

sessionMemory := memory.NewSessionMemory(memory.NewMemoryBankWithStore(store), 8).
    WithEngine(engine)
```


Tune retrieval with runtime flags (no YAML required):
```bash
go run ./cmd/demo \
  --memory-sim-weight=0.6 \
  --memory-recency-weight=0.2 \
  --memory-half-life=48h \
  --memory-source-boost="pagerduty=1.0,slack=0.6"
```
Schema migrations rely on online-safe `ALTER TABLE ... IF NOT EXISTS` statements to avoid downtime.

## Swarm (Shared Spaces)

Coordinate multiple agents that **share memory across named spaces** (e.g., `team:core`, `team:project-x`).  
Swarm uses `memory.SharedSession` on top of the session memory engine to read/write memories visible to everyone in the same space.

### When to use
- You run several cooperating agents (e.g., *alpha*, *beta*, *researcher*) and want them to see each other’s latest notes.
- You want *shared RAG context* retrieved from a team space while still keeping local, per-session short‑term memory.
- You need explicit access control: grant write/read access per space and per session ID.

## Development Workflow
1. Write or update Go code.
2. Format as needed (`gofmt` or your editor tooling).
3. Run the full test suite:
   ```bash
   go test ./...
   ```
4. Update documentation and examples when adding new tools, models, or memory backends.

## Troubleshooting Tips
- **Missing pgvector extension** – Ensure `CREATE EXTENSION vector;` runs before connecting. Without it the Postgres-backed memory store cannot create required columns.
- **API key issues** – Confirm `GOOGLE_API_KEY` or `GEMINI_API_KEY` is exported in the same shell where you run the demo.
- **Tool discovery** – When adding tools or sub-agents, verify their names are unique to avoid collisions in the catalog/directory registries.
- **Deterministic tests** – Use the dummy model adapter for repeatable results when writing unit tests around orchestration logic.

## Contributing
Issues and pull requests are welcome! Please update the README and examples when contributing new models, tools, or memory backends so the community benefits from your additions.

## 🙏 Acknowledgments

**Upstream inspiration**
- Google’s [Agent Development Kit (Python)](https://github.com/googleapis/agent-development-kit) — the conceptual baseline for runtime composition, modules, and memory patterns.
