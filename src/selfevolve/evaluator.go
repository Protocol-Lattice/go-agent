package selfevolve

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/Protocol-Lattice/go-agent/src/models"
)

// EvaluationResult represents the outcome of evaluating an agent's output
type EvaluationResult struct {
	Score     float64        `json:"score"`     // 0.0 to 1.0
	Passed    bool           `json:"passed"`    // Whether it meets the threshold
	Reasoning string         `json:"reasoning"` // Why this score was given
	Criteria  string         `json:"criteria"`  // What was being evaluated
	Metadata  map[string]any `json:"metadata"`  // Additional context
}

// Evaluator defines how to evaluate agent outputs
type Evaluator interface {
	// Evaluate scores an agent's output against defined criteria
	Evaluate(ctx context.Context, input string, output any) (*EvaluationResult, error)

	// Name returns the evaluator's identifier
	Name() string

	// Criteria returns what this evaluator checks
	Criteria() string
}

// LLMAsJudgeEvaluator uses an LLM to evaluate agent outputs
type LLMAsJudgeEvaluator struct {
	model     models.Agent
	name      string
	criteria  string
	threshold float64
	prompt    string
}

// NewLLMAsJudgeEvaluator creates an evaluator that uses an LLM to judge outputs
func NewLLMAsJudgeEvaluator(model models.Agent, name, criteria string, threshold float64) *LLMAsJudgeEvaluator {
	prompt := `You are an expert evaluator. Your task is to assess the quality of an AI agent's output.

EVALUATION CRITERIA:
%s

INPUT PROVIDED TO AGENT:
%s

AGENT'S OUTPUT:
%s

Evaluate the output against the criteria. Respond ONLY with valid JSON:
{
  "score": 0.85,
  "passed": true,
  "reasoning": "The output meets the criteria because...",
  "suggestions": "To improve further, consider..."
}

Rules:
- score: float between 0.0 and 1.0
- passed: true if score >= %.2f
- reasoning: explain your evaluation
- suggestions: actionable improvements
`

	return &LLMAsJudgeEvaluator{
		model:     model,
		name:      name,
		criteria:  criteria,
		threshold: threshold,
		prompt:    prompt,
	}
}

func (e *LLMAsJudgeEvaluator) Name() string {
	return e.name
}

func (e *LLMAsJudgeEvaluator) Criteria() string {
	return e.criteria
}

func (e *LLMAsJudgeEvaluator) Evaluate(ctx context.Context, input string, output any) (*EvaluationResult, error) {
	evalPrompt := fmt.Sprintf(e.prompt, e.criteria, input, output, e.threshold)

	response, err := e.model.Generate(ctx, evalPrompt)
	if err != nil {
		return nil, fmt.Errorf("LLM evaluation failed: %w", err)
	}

	// Extract JSON from response
	jsonStr := extractJSON(fmt.Sprint(response))
	if jsonStr == "" {
		return nil, fmt.Errorf("no JSON found in evaluation response")
	}

	var result struct {
		Score       float64 `json:"score"`
		Passed      bool    `json:"passed"`
		Reasoning   string  `json:"reasoning"`
		Suggestions string  `json:"suggestions"`
	}

	if err := json.Unmarshal([]byte(jsonStr), &result); err != nil {
		return nil, fmt.Errorf("failed to parse evaluation JSON: %w", err)
	}

	return &EvaluationResult{
		Score:     result.Score,
		Passed:    result.Passed,
		Reasoning: result.Reasoning,
		Criteria:  e.criteria,
		Metadata: map[string]any{
			"suggestions": result.Suggestions,
			"threshold":   e.threshold,
		},
	}, nil
}

// CompositeEvaluator combines multiple evaluators
type CompositeEvaluator struct {
	evaluators []Evaluator
	strategy   AggregationStrategy
}

// AggregationStrategy defines how to combine multiple evaluation results
type AggregationStrategy string

const (
	// StrategyAverage takes the average score
	StrategyAverage AggregationStrategy = "average"
	// StrategyMinimum requires all evaluators to pass
	StrategyMinimum AggregationStrategy = "minimum"
	// StrategyWeightedAverage uses weighted scores (requires metadata)
	StrategyWeightedAverage AggregationStrategy = "weighted"
)

// NewCompositeEvaluator creates an evaluator that combines multiple evaluators
func NewCompositeEvaluator(evaluators []Evaluator, strategy AggregationStrategy) *CompositeEvaluator {
	return &CompositeEvaluator{
		evaluators: evaluators,
		strategy:   strategy,
	}
}

func (c *CompositeEvaluator) Name() string {
	return "composite_evaluator"
}

func (c *CompositeEvaluator) Criteria() string {
	criteria := "Composite evaluation combining: "
	for i, eval := range c.evaluators {
		if i > 0 {
			criteria += ", "
		}
		criteria += eval.Name()
	}
	return criteria
}

func (c *CompositeEvaluator) Evaluate(ctx context.Context, input string, output any) (*EvaluationResult, error) {
	results := make([]*EvaluationResult, 0, len(c.evaluators))

	for _, evaluator := range c.evaluators {
		result, err := evaluator.Evaluate(ctx, input, output)
		if err != nil {
			return nil, fmt.Errorf("evaluator %s failed: %w", evaluator.Name(), err)
		}
		results = append(results, result)
	}

	return c.aggregate(results), nil
}

func (c *CompositeEvaluator) aggregate(results []*EvaluationResult) *EvaluationResult {
	if len(results) == 0 {
		return &EvaluationResult{Score: 0, Passed: false}
	}

	switch c.strategy {
	case StrategyMinimum:
		return c.aggregateMinimum(results)
	case StrategyAverage:
		fallthrough
	default:
		return c.aggregateAverage(results)
	}
}

func (c *CompositeEvaluator) aggregateAverage(results []*EvaluationResult) *EvaluationResult {
	var totalScore float64
	allPassed := true
	reasoning := ""

	for i, result := range results {
		totalScore += result.Score
		if !result.Passed {
			allPassed = false
		}
		if i > 0 {
			reasoning += "; "
		}
		reasoning += fmt.Sprintf("%s: %s", c.evaluators[i].Name(), result.Reasoning)
	}

	avgScore := totalScore / float64(len(results))

	return &EvaluationResult{
		Score:     avgScore,
		Passed:    allPassed,
		Reasoning: reasoning,
		Criteria:  c.Criteria(),
		Metadata: map[string]any{
			"individual_results": results,
			"strategy":           c.strategy,
		},
	}
}

func (c *CompositeEvaluator) aggregateMinimum(results []*EvaluationResult) *EvaluationResult {
	minScore := 1.0
	allPassed := true
	reasoning := ""

	for i, result := range results {
		if result.Score < minScore {
			minScore = result.Score
		}
		if !result.Passed {
			allPassed = false
		}
		if i > 0 {
			reasoning += "; "
		}
		reasoning += fmt.Sprintf("%s: %s", c.evaluators[i].Name(), result.Reasoning)
	}

	return &EvaluationResult{
		Score:     minScore,
		Passed:    allPassed,
		Reasoning: reasoning,
		Criteria:  c.Criteria(),
		Metadata: map[string]any{
			"individual_results": results,
			"strategy":           c.strategy,
		},
	}
}

// extractJSON extracts JSON object from a string
func extractJSON(s string) string {
	start := -1
	end := -1
	depth := 0

	for i, ch := range s {
		if ch == '{' {
			if start == -1 {
				start = i
			}
			depth++
		} else if ch == '}' {
			depth--
			if depth == 0 && start != -1 {
				end = i + 1
				break
			}
		}
	}

	if start != -1 && end != -1 {
		return s[start:end]
	}
	return ""
}
