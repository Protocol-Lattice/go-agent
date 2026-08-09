# Agent Arena

Agent Arena is the evaluation layer for `go-agent`. It separates agent execution from evaluation so the same tasks can benchmark different agents, models, swarms, or remote runtimes.

## Quick start

```go
package main

import (
    "context"
    "fmt"

    "github.com/Protocol-Lattice/go-agent/arena"
)

func main() {
    runner := myRunner{}
    tasks := []arena.Task{
        {
            Name:      "capital",
            Input:     "What is the capital of France?",
            Evaluator: arena.ContainsEvaluator{Required: []string{"Paris"}},
        },
    }

    results := (&arena.Arena{Runner: runner}).RunAll(context.Background(), tasks, 4)
    fmt.Printf("%+v\n", arena.Summarize(results))
}
```

## Compare agents

```go
suite := arena.RunSuite(ctx, tasks, []arena.Competitor{
    {Name: "agent-a", Runner: runnerA},
    {Name: "agent-b", Runner: runnerB},
}, 4)

for _, entry := range arena.RankSuite(suite) {
    fmt.Printf("%s: %.2f\n", entry.Name, entry.Summary.AverageScore)
}
```

## Native go-agent

Use `arena.AgentRunner` to benchmark a normal `*agent.Agent`:

```go
runner := arena.AgentRunner{Agent: myAgent}
```

Each task gets an isolated `arena:<task-name>` session unless `SessionID` is explicitly supplied.

## Built-in evaluators

- `ExactEvaluator` — normalized exact output match.
- `ContainsEvaluator` — all required fragments must be present; partial scores are supported.
- `FuncEvaluator` — custom boolean/score evaluation.
- `ScoreEvaluator` — custom score function with automatic clamping to `[0, 1]`.

The result model also tracks duration, tokens, tool calls, retries, cost, feedback, and metadata when the runner provides those metrics.
