// Package arena provides a small, deterministic evaluation harness for go-agent.
//
// Arena separates task execution from evaluation so the same task suite can be
// used with different agents, models, or runners. It also records enough
// execution metadata to compare correctness, latency, failures, retries,
// tokens, and cost.
package arena

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"sync"
	"time"
)

// Runner executes an arena task. Implementations may wrap agent.Agent, an
// HTTP-hosted agent, a swarm, or any other runtime.
type Runner interface {
	Run(context.Context, Task) (RunOutput, error)
}

// Task is a single benchmark/evaluation case.
type Task struct {
	Name        string
	Description string
	Input       string
	Metadata    map[string]string
	Evaluator   Evaluator
}

// RunOutput is the observable result of executing a task.
type RunOutput struct {
	Output       string
	InputTokens  int
	OutputTokens int
	ToolCalls    int
	Retries      int
	Metadata     map[string]string
}

// Result contains execution and evaluation information for one task.
type Result struct {
	TaskName     string
	Output       string
	Success      bool
	Score        float64
	Duration     time.Duration
	InputTokens  int
	OutputTokens int
	ToolCalls    int
	Retries      int
	Cost         float64
	Error        error
	Feedback     []string
	Metadata     map[string]string
}

// Evaluator evaluates a completed task run. Score must be in [0, 1].
type Evaluator interface {
	Evaluate(context.Context, Task, RunOutput) Evaluation
}

// Evaluation is the result returned by an Evaluator.
type Evaluation struct {
	Score    float64
	Success  bool
	Feedback []string
}

// FuncEvaluator adapts a function into an Evaluator.
type FuncEvaluator func(context.Context, Task, RunOutput) Evaluation

func (f FuncEvaluator) Evaluate(ctx context.Context, task Task, output RunOutput) Evaluation {
	return f(ctx, task, output)
}

// Arena executes tasks and aggregates their results.
type Arena struct {
	Runner Runner
	// CostPerInputToken and CostPerOutputToken are optional USD rates.
	CostPerInputToken  float64
	CostPerOutputToken float64
}

// Run executes one task.
func (a *Arena) Run(ctx context.Context, task Task) Result {
	started := time.Now()
	result := Result{TaskName: task.Name}

	if a == nil || a.Runner == nil {
		result.Error = errors.New("arena requires a runner")
		result.Duration = time.Since(started)
		return result
	}

	output, err := a.Runner.Run(ctx, task)
	result.Duration = time.Since(started)
	result.Output = output.Output
	result.InputTokens = output.InputTokens
	result.OutputTokens = output.OutputTokens
	result.ToolCalls = output.ToolCalls
	result.Retries = output.Retries
	result.Metadata = cloneMap(output.Metadata)
	result.Cost = float64(output.InputTokens)*a.CostPerInputToken + float64(output.OutputTokens)*a.CostPerOutputToken

	if err != nil {
		result.Error = err
		return result
	}

	if task.Evaluator == nil {
		result.Success = true
		result.Score = 1
		return result
	}

	evaluation := task.Evaluator.Evaluate(ctx, task, output)
	result.Score = clampScore(evaluation.Score)
	result.Success = evaluation.Success
	result.Feedback = append([]string(nil), evaluation.Feedback...)
	return result
}

// RunAll executes all tasks. With concurrency <= 1 tasks run sequentially.
// Results preserve task input order regardless of execution order.
func (a *Arena) RunAll(ctx context.Context, tasks []Task, concurrency int) []Result {
	if len(tasks) == 0 {
		return nil
	}
	if concurrency <= 1 {
		results := make([]Result, 0, len(tasks))
		for _, task := range tasks {
			results = append(results, a.Run(ctx, task))
		}
		return results
	}
	if concurrency > len(tasks) {
		concurrency = len(tasks)
	}

	results := make([]Result, len(tasks))
	jobs := make(chan int)
	var wg sync.WaitGroup
	wg.Add(concurrency)
	for worker := 0; worker < concurrency; worker++ {
		go func() {
			defer wg.Done()
			for index := range jobs {
				results[index] = a.Run(ctx, tasks[index])
			}
		}()
	}
	for index := range tasks {
		select {
		case jobs <- index:
		case <-ctx.Done():
			close(jobs)
			wg.Wait()
			return results
		}
	}
	close(jobs)
	wg.Wait()
	return results
}

// Summary aggregates a set of results.
type Summary struct {
	Tasks          int
	Passed         int
	Failed         int
	AverageScore   float64
	SuccessRate    float64
	TotalDuration  time.Duration
	AverageDuration time.Duration
	InputTokens    int
	OutputTokens   int
	ToolCalls      int
	Retries        int
	TotalCost      float64
}

// Summarize aggregates results into a leaderboard-friendly summary.
func Summarize(results []Result) Summary {
	var summary Summary
	summary.Tasks = len(results)
	if len(results) == 0 {
		return summary
	}
	for _, result := range results {
		if result.Success {
			summary.Passed++
		} else {
			summary.Failed++
		}
		summary.AverageScore += result.Score
		summary.TotalDuration += result.Duration
		summary.InputTokens += result.InputTokens
		summary.OutputTokens += result.OutputTokens
		summary.ToolCalls += result.ToolCalls
		summary.Retries += result.Retries
		summary.TotalCost += result.Cost
	}
	summary.AverageScore /= float64(len(results))
	summary.SuccessRate = float64(summary.Passed) / float64(len(results))
	summary.AverageDuration = summary.TotalDuration / time.Duration(len(results))
	return summary
}

// LeaderboardEntry is a comparable aggregate for one named runner.
type LeaderboardEntry struct {
	Name    string
	Summary Summary
}

// Rank returns entries sorted by score descending, then success rate,
// duration, and name. Sorting is deterministic.
func Rank(entries []LeaderboardEntry) []LeaderboardEntry {
	out := append([]LeaderboardEntry(nil), entries...)
	sort.SliceStable(out, func(i, j int) bool {
		a, b := out[i], out[j]
		if a.Summary.AverageScore != b.Summary.AverageScore {
			return a.Summary.AverageScore > b.Summary.AverageScore
		}
		if a.Summary.SuccessRate != b.Summary.SuccessRate {
			return a.Summary.SuccessRate > b.Summary.SuccessRate
		}
		if a.Summary.AverageDuration != b.Summary.AverageDuration {
			return a.Summary.AverageDuration < b.Summary.AverageDuration
		}
		return a.Name < b.Name
	})
	return out
}

// Validate checks the minimum task contract before a suite is executed.
func Validate(tasks []Task) error {
	seen := make(map[string]struct{}, len(tasks))
	for i, task := range tasks {
		if task.Name == "" {
			return fmt.Errorf("task %d has empty name", i)
		}
		if _, ok := seen[task.Name]; ok {
			return fmt.Errorf("duplicate task name %q", task.Name)
		}
		seen[task.Name] = struct{}{}
		if task.Evaluator == nil {
			return fmt.Errorf("task %q has no evaluator", task.Name)
		}
	}
	return nil
}

func clampScore(score float64) float64 {
	if score < 0 {
		return 0
	}
	if score > 1 {
		return 1
	}
	return score
}

func cloneMap(in map[string]string) map[string]string {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]string, len(in))
	for key, value := range in {
		out[key] = value
	}
	return out
}
