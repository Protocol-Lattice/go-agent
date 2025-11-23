package selfevolve

import (
	"context"
	"testing"
)

// MockEvaluator for testing
type MockEvaluator struct {
	name     string
	criteria string
	score    float64
	passed   bool
}

func (m *MockEvaluator) Name() string {
	return m.name
}

func (m *MockEvaluator) Criteria() string {
	return m.criteria
}

func (m *MockEvaluator) Evaluate(ctx context.Context, input string, output any) (*EvaluationResult, error) {
	return &EvaluationResult{
		Score:     m.score,
		Passed:    m.passed,
		Reasoning: "Mock evaluation",
		Criteria:  m.criteria,
	}, nil
}

func TestCompositeEvaluator_Average(t *testing.T) {
	ctx := context.Background()

	eval1 := &MockEvaluator{
		name:     "eval1",
		criteria: "test1",
		score:    0.8,
		passed:   true,
	}

	eval2 := &MockEvaluator{
		name:     "eval2",
		criteria: "test2",
		score:    0.6,
		passed:   false,
	}

	composite := NewCompositeEvaluator(
		[]Evaluator{eval1, eval2},
		StrategyAverage,
	)

	result, err := composite.Evaluate(ctx, "input", "output")
	if err != nil {
		t.Fatalf("Evaluate failed: %v", err)
	}

	expectedScore := (0.8 + 0.6) / 2.0
	if result.Score != expectedScore {
		t.Errorf("Expected score %.2f, got %.2f", expectedScore, result.Score)
	}

	if result.Passed {
		t.Error("Expected Passed to be false when any evaluator fails")
	}
}

func TestCompositeEvaluator_Minimum(t *testing.T) {
	ctx := context.Background()

	eval1 := &MockEvaluator{
		name:     "eval1",
		criteria: "test1",
		score:    0.9,
		passed:   true,
	}

	eval2 := &MockEvaluator{
		name:     "eval2",
		criteria: "test2",
		score:    0.7,
		passed:   true,
	}

	composite := NewCompositeEvaluator(
		[]Evaluator{eval1, eval2},
		StrategyMinimum,
	)

	result, err := composite.Evaluate(ctx, "input", "output")
	if err != nil {
		t.Fatalf("Evaluate failed: %v", err)
	}

	expectedScore := 0.7 // minimum
	if result.Score != expectedScore {
		t.Errorf("Expected score %.2f, got %.2f", expectedScore, result.Score)
	}

	if !result.Passed {
		t.Error("Expected Passed to be true when all evaluators pass")
	}
}

func TestPromptVersion(t *testing.T) {
	version := &PromptVersion{
		Version: 1,
		Prompt:  "Test prompt",
		Model:   "test-model",
		Score:   0.8,
	}

	if version.Version != 1 {
		t.Errorf("Expected version 1, got %d", version.Version)
	}

	if version.Score != 0.8 {
		t.Errorf("Expected score 0.8, got %.2f", version.Score)
	}
}

func TestExtractJSON(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "simple json",
			input:    `{"key": "value"}`,
			expected: `{"key": "value"}`,
		},
		{
			name:     "json with text before",
			input:    `Some text before {"key": "value"}`,
			expected: `{"key": "value"}`,
		},
		{
			name:     "json with text after",
			input:    `{"key": "value"} some text after`,
			expected: `{"key": "value"}`,
		},
		{
			name:     "nested json",
			input:    `{"outer": {"inner": "value"}}`,
			expected: `{"outer": {"inner": "value"}}`,
		},
		{
			name:     "no json",
			input:    `no json here`,
			expected: ``,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := extractJSON(tt.input)
			if result != tt.expected {
				t.Errorf("Expected %q, got %q", tt.expected, result)
			}
		})
	}
}
