# go-agent

[![Go Version](https://img.shields.io/badge/Go-1.25-00ADD8?logo=go&logoColor=white)](https://go.dev/dl/)
[![CI Status](https://github.com/Protocol-Lattice/go-agent/actions/workflows/go.yml/badge.svg)](https://github.com/Protocol-Lattice/go-agent/actions/workflows/go.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/Protocol-Lattice/go-agent.svg)](https://pkg.go.dev/github.com/Protocol-Lattice/go-agent)
[![Go Report Card](https://goreportcard.com/badge/github.com/Protocol-Lattice/go-agent)](https://goreportcard.com/report/github.com/Protocol-Lattice/go-agent)

go-agent is a Go framework for building AI agents with pluggable LLM providers, memory, file context, guardrails, UTCP tool orchestration, skills, CodeMode, and multi-agent coordination.

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
│   ├── SKILL.md
│   └── ...
├── codemode/
│   └── SKILL.md
└── refactor-readme/
    └── SKILL.md
```

Each skill can include optional YAML-style front matter:

```markdown
---
name: release
description: Prepare safe releases
---
Run the full test suite before proposing a release.
```

Skills are routed deterministically before request execution. A matched request gets one primary skill; declared dependencies are resolved separately. The active skill then scopes the available tools and supplies its instructions to the execution layer.

For repository mutation work, the recommended workflow is:

```text
user request
    ↓
primary skill
    ↓
CodeMode
    ↓
filesystem.read / filesystem.list / filesystem.search
    ↓
filesystem.patch or filesystem.write
    ↓
verification read/test
```

The runtime emits the skill and tool execution events in the same request-scoped workflow stream, so clients can display the actual execution order instead of reconstructing it from model text.

The built-in `refactor-readme` skill is an example of a mutation-oriented workflow: it requires reading `README.md`, inspecting the repository as needed, making a real filesystem mutation, and verifying the resulting README. Mutation-oriented skills automatically receive the standard filesystem mutation tools when they expose `filesystem.read`.

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