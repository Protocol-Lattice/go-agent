package arena

import (
	"context"
	"errors"
	"testing"
	"time"
)

type fakeRunner struct {
	outputs map[string]string
	err     error
}

func (r fakeRunner) Run(_ context.Context, task Task) (RunOutput, error) {
	if r.err != nil {
		return RunOutput{}, r.err
	}
	return RunOutput{
		Output:       r.outputs[task.Name],
		InputTokens:  10,
		OutputTokens: 5,
		ToolCalls:    2,
		Retries:      1,
	}, nil
}

func TestArenaRun(t *testing.T) {
	a := &Arena{
		Runner:             fakeRunner{outputs: map[string]string{"hello": "world"}},
		CostPerInputToken:  0.001,
		CostPerOutputToken: 0.002,
	}
	result := a.Run(context.Background(), Task{
		Name:      "hello",
		Input:     "say hello",
		Evaluator: ExactEvaluator{Expected: "world"},
	})

	if !result.Success || result.Score != 1 {
		t.Fatalf("expected passing result, got %+v", result)
	}
	if result.InputTokens != 10 || result.OutputTokens != 5 || result.ToolCalls != 2 || result.Retries != 1 {
		t.Fatalf("execution metrics not propagated: %+v", result)
	}
	if result.Cost != 0.02 {
		t.Fatalf("unexpected cost: %v", result.Cost)
	}
}

func TestContainsEvaluatorPartialScore(t *testing.T) {
	e := ContainsEvaluator{Required: []string{"Go", "arena", "missing"}}
	result := e.Evaluate(context.Background(), Task{}, RunOutput{Output: "Go agent arena"})
	if result.Success {
		t.Fatal("partial match must not pass")
	}
	if result.Score != 2.0/3.0 {
		t.Fatalf("unexpected score: %v", result.Score)
	}
	if len(result.Feedback) != 1 {
		t.Fatalf("expected one feedback item, got %v", result.Feedback)
	}
}

func TestRunAllPreservesOrder(t *testing.T) {
	a := &Arena{Runner: fakeRunner{outputs: map[string]string{"a": "A", "b": "B", "c": "C"}}}
	tasks := []Task{
		{Name: "a"},
		{Name: "b"},
		{Name: "c"},
	}
	results := a.RunAll(context.Background(), tasks, 3)
	for i, expected := range []string{"A", "B", "C"} {
		if results[i].Output != expected {
			t.Fatalf("result %d: expected %q, got %q", i, expected, results[i].Output)
		}
	}
}

func TestRunFailure(t *testing.T) {
	a := &Arena{Runner: fakeRunner{err: errors.New("boom")}}
	result := a.Run(context.Background(), Task{Name: "broken"})
	if result.Success || result.Error == nil {
		t.Fatalf("expected failed result, got %+v", result)
	}
}

func TestSummarizeAndRank(t *testing.T) {
	results := []Result{
		{Success: true, Score: 1, Duration: 2 * time.Second, InputTokens: 10, OutputTokens: 5, Cost: 0.1},
		{Success: false, Score: 0.5, Duration: 4 * time.Second, InputTokens: 20, OutputTokens: 10, Cost: 0.2},
	}
	summary := Summarize(results)
	if summary.Passed != 1 || summary.Failed != 1 || summary.SuccessRate != 0.5 {
		t.Fatalf("unexpected summary: %+v", summary)
	}
	if summary.AverageScore != 0.75 || summary.TotalCost != 0.3 {
		t.Fatalf("unexpected aggregate metrics: %+v", summary)
	}

	ranked := Rank([]LeaderboardEntry{
		{Name: "slow", Summary: Summary{AverageScore: 0.9, SuccessRate: 0.9, AverageDuration: 2 * time.Second}},
		{Name: "fast", Summary: Summary{AverageScore: 0.9, SuccessRate: 0.9, AverageDuration: time.Second}},
	})
	if ranked[0].Name != "fast" {
		t.Fatalf("expected fast runner first, got %s", ranked[0].Name)
	}
}

func TestValidate(t *testing.T) {
	if err := Validate([]Task{{Name: "a", Evaluator: ExactEvaluator{Expected: "x"}}}); err != nil {
		t.Fatalf("valid task rejected: %v", err)
	}
	if err := Validate([]Task{{Name: "a"}}); err == nil {
		t.Fatal("missing evaluator must fail validation")
	}
	if err := Validate([]Task{{Name: "a", Evaluator: ExactEvaluator{}}, {Name: "a", Evaluator: ExactEvaluator{}}}); err == nil {
		t.Fatal("duplicate task names must fail validation")
	}
}
