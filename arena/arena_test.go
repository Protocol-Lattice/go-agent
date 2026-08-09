package arena

import (
	"context"
	"errors"
	"testing"
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
