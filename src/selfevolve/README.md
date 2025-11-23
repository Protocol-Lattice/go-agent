# Self-Evolving Agents

This package implements **self-evolving agents** - AI agents that can autonomously improve their performance through evaluation, feedback, and prompt optimization. Inspired by the [OpenAI Cookbook on Autonomous Agent Retraining](https://cookbook.openai.com/examples/partners/self_evolving_agents/autonomous_agent_retraining).

## Overview

Self-evolving agents address a critical limitation in traditional AI systems: they reach a plateau after initial deployment because they depend on humans to diagnose edge cases and correct failures. This package enables agents to:

1. **Evaluate their own outputs** using LLM-as-a-judge or custom evaluators
2. **Collect feedback** on what went wrong and why
3. **Optimize their prompts** through meta-prompting
4. **Track performance** across iterations
5. **Automatically improve** without human intervention

## Architecture

The self-evolution loop consists of four key components:

### 1. Baseline Agent
The starting point - any `agent.Agent` instance with an initial system prompt.

### 2. Evaluator
Assesses agent outputs against defined criteria. Can be:
- **LLM-as-a-Judge**: Uses an LLM to score outputs
- **Composite Evaluator**: Combines multiple evaluators
- **Custom Evaluator**: Implement the `Evaluator` interface

### 3. Prompt Optimizer
Uses meta-prompting to generate improved system prompts based on evaluation feedback.

### 4. Evolution Loop
Orchestrates the process:
```
For each task:
  For attempt in 1..MaxRetries:
    1. Generate output with current prompt
    2. Evaluate the output
    3. If score >= target: SUCCESS, continue to next task
    4. If score < target: Optimize prompt, retry
  If all attempts fail: Keep best attempt, move to next task
```

## Quick Start

```go
package main

import (
    "context"
    "log"
    
    "github.com/Protocol-Lattice/go-agent"
    "github.com/Protocol-Lattice/go-agent/src/selfevolve"
    "github.com/Protocol-Lattice/go-agent/src/models"
)

func main() {
    ctx := context.Background()
    
    // 1. Create your base agent
    baseAgent, _ := agent.New(agent.Options{
        Model: yourModel,
        Memory: yourMemory,
        SystemPrompt: "You are a helpful assistant.",
    })
    
    // 2. Create an optimizer model (for evaluation & optimization)
    optimizerModel, _ := models.NewGeminiLLM(ctx, "gemini-2.0-flash-exp", "")
    
    // 3. Configure evolution
    config := &selfevolve.EvolutionConfig{
        MaxRetries:    3,
        TargetScore:   0.8,
        EnableLogging: true,
    }
    
    // 4. Create the evolving agent
    evolvingAgent := selfevolve.NewEvolvingAgent(
        baseAgent,
        optimizerModel,
        "You are a helpful assistant.",
        "gemini-2.0-flash-exp",
        config,
    )
    
    // 5. Use it like a normal agent - it evolves automatically!
    output, err := evolvingAgent.Generate(ctx, "session1", "Summarize AI agents")
    if err != nil {
        log.Fatal(err)
    }
    
    // 6. View evolution metrics
    evolvingAgent.PrintSummary()
}
```

## Core Components

### Evaluator Interface

```go
type Evaluator interface {
    // Evaluate scores an agent's output against defined criteria
    Evaluate(ctx context.Context, input, output string) (*EvaluationResult, error)
    
    // Name returns the evaluator's identifier
    Name() string
    
    // Criteria returns what this evaluator checks
    Criteria() string
}
```

### LLM-as-a-Judge Evaluator

Uses an LLM to evaluate outputs:

```go
evaluator := selfevolve.NewLLMAsJudgeEvaluator(
    optimizerModel,
    "quality_check",
    "The output should be accurate, helpful, and well-formatted",
    0.8, // threshold
)
```

### Composite Evaluator

Combines multiple evaluators:

```go
lengthEval := selfevolve.NewLLMAsJudgeEvaluator(
    model, "length", "Output should be concise (under 100 words)", 0.7,
)

qualityEval := selfevolve.NewLLMAsJudgeEvaluator(
    model, "quality", "Output should be accurate and complete", 0.8,
)

composite := selfevolve.NewCompositeEvaluator(
    []selfevolve.Evaluator{lengthEval, qualityEval},
    selfevolve.StrategyAverage, // or StrategyMinimum
)
```

### Evolution Configuration

```go
type EvolutionConfig struct {
    // MaxRetries: maximum optimization attempts per task
    MaxRetries int
    
    // TargetScore: minimum score to consider successful
    TargetScore float64
    
    // Evaluators: list of evaluators to use
    Evaluators []Evaluator
    
    // EnableLogging: enable detailed evolution logs
    EnableLogging bool
    
    // StopOnSuccess: stop optimizing once target is reached
    StopOnSuccess bool
}
```

## Advanced Usage

### Custom Evaluators

Implement your own evaluation logic:

```go
type CustomEvaluator struct {
    name     string
    criteria string
}

func (e *CustomEvaluator) Name() string {
    return e.name
}

func (e *CustomEvaluator) Criteria() string {
    return e.criteria
}

func (e *CustomEvaluator) Evaluate(ctx context.Context, input, output string) (*EvaluationResult, error) {
    // Your custom evaluation logic
    score := calculateScore(output)
    
    return &EvaluationResult{
        Score:     score,
        Passed:    score >= 0.8,
        Reasoning: "Custom evaluation reasoning",
        Criteria:  e.criteria,
    }, nil
}
```

### Accessing Evolution Metrics

```go
// Get metrics
metrics := evolvingAgent.GetMetrics()
fmt.Printf("Success Rate: %.1f%%\n", metrics["success_rate"].(float64) * 100)
fmt.Printf("Avg Retries: %.2f\n", metrics["avg_retries_per_task"])

// Get evolution log
log := evolvingAgent.GetEvolutionLog()
for _, entry := range log {
    fmt.Printf("Task %d, Attempt %d: Score=%.3f, Passed=%v\n",
        entry.TaskNumber, entry.Attempt, entry.Evaluation.Score, entry.Success)
}

// Get prompt history
history := evolvingAgent.GetPromptHistory()
for _, version := range history {
    fmt.Printf("v%d (Score: %.3f): %s\n", 
        version.Version, version.Score, version.Prompt)
}
```

### Manual Prompt Management

```go
// Get current prompt
current := evolvingAgent.GetCurrentPrompt()
fmt.Printf("Current: v%d (Score: %.3f)\n", current.Version, current.Score)

// Get best prompt
best := evolvingAgent.GetBestPrompt()
fmt.Printf("Best: v%d (Score: %.3f)\n", best.Version, best.Score)

// Rollback to best
evolvingAgent.RollbackToBest()

// Apply a specific prompt version
customPrompt := &selfevolve.PromptVersion{
    Version: 99,
    Prompt:  "Custom optimized prompt",
    Model:   "gemini-2.0-flash-exp",
}
evolvingAgent.ApplyPrompt(customPrompt)
```

## Integration with CodeMode

Self-evolving agents work seamlessly with CodeMode for UTCP tool orchestration:

```go
// Create agent with CodeMode
baseAgent, _ := agent.New(agent.Options{
    Model:      model,
    Memory:     memory,
    UTCPClient: utcpClient,
    CodeMode:   codemode.NewCodeModeUTCP(utcpClient),
})

// Wrap with self-evolution
evolvingAgent := selfevolve.NewEvolvingAgent(
    baseAgent,
    optimizerModel,
    initialPrompt,
    modelName,
    config,
)

// The agent will evolve its prompt while maintaining CodeMode capabilities
output, _ := evolvingAgent.Generate(ctx, sessionID, 
    "Use codemode to search and call the best summarization tool")
```

## Best Practices

### 1. Choose Appropriate Evaluators

- **Single criterion**: Use one `LLMAsJudgeEvaluator`
- **Multiple criteria**: Use `CompositeEvaluator` with `StrategyAverage`
- **All must pass**: Use `CompositeEvaluator` with `StrategyMinimum`

### 2. Set Realistic Targets

- Start with `TargetScore: 0.7` and adjust based on results
- Lower scores (0.6-0.7) for creative tasks
- Higher scores (0.8-0.9) for factual/structured tasks

### 3. Limit Retries

- `MaxRetries: 3` is usually sufficient
- More retries = more API calls and latency
- Monitor success rate to tune this value

### 4. Enable Logging During Development

```go
config := &selfevolve.EvolutionConfig{
    EnableLogging: true, // Detailed logs
}
```

Disable in production for better performance.

### 5. Monitor Evolution Metrics

```go
metrics := evolvingAgent.GetMetrics()
if metrics["success_rate"].(float64) < 0.5 {
    // Success rate too low - adjust evaluators or target score
}
```

## Example: Summarization Agent

See `cmd/example/self_evolving/main.go` for a complete example that:

1. Creates a summarization agent
2. Defines length and quality evaluators
3. Runs the evolution loop on multiple tasks
4. Tracks prompt improvements over time
5. Demonstrates CodeMode integration

Run it:

```bash
export GOOGLE_API_KEY=your-key
go run cmd/example/self_evolving/main.go --verbose
```

## Performance Considerations

- **API Costs**: Each evaluation and optimization requires LLM calls
- **Latency**: Evolution adds overhead (typically 2-3x base latency)
- **Caching**: Prompt versions are cached; rollback is instant
- **Parallelization**: Currently sequential; consider batching for production

## Comparison with OpenAI Cookbook

This implementation adapts the OpenAI cookbook's Python approach to Go:

| Feature | OpenAI Cookbook | go-agent/selfevolve |
|---------|----------------|---------------------|
| Language | Python | Go |
| Evaluation | OpenAI Evals API | LLM-as-a-Judge (any model) |
| Optimization | Meta-prompting | Meta-prompting |
| Tool Integration | Function calling | UTCP + CodeMode |
| Metrics | Dashboard | Programmatic API |
| Deployment | Notebook | Production-ready |

## Future Enhancements

- [ ] Parallel evaluation for multiple outputs
- [ ] Persistent prompt version storage
- [ ] A/B testing between prompt versions
- [ ] Genetic algorithm for prompt optimization (GEPA)
- [ ] Integration with observability platforms
- [ ] Automatic dataset generation from production logs

## References

- [OpenAI Cookbook: Self-Evolving Agents](https://cookbook.openai.com/examples/partners/self_evolving_agents/autonomous_agent_retraining)
- [LLM-as-a-Judge Pattern](https://arxiv.org/abs/2306.05685)
- [Meta-Prompting for Prompt Optimization](https://arxiv.org/abs/2401.12954)

## License

Apache 2.0 - See LICENSE file for details.
