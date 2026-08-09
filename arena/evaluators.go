package arena

import (
	"context"
	"strings"
)

// ExactEvaluator passes when the normalized output exactly matches Expected.
type ExactEvaluator struct {
	Expected string
}

func (e ExactEvaluator) Evaluate(_ context.Context, _ Task, output RunOutput) Evaluation {
	passed := strings.TrimSpace(output.Output) == strings.TrimSpace(e.Expected)
	if passed {
		return Evaluation{Score: 1, Success: true}
	}
	return Evaluation{Score: 0, Success: false, Feedback: []string{"output does not match expected value"}}
}

// ContainsEvaluator passes when all required fragments occur in the output.
type ContainsEvaluator struct {
	Required []string
}

func (e ContainsEvaluator) Evaluate(_ context.Context, _ Task, output RunOutput) Evaluation {
	if len(e.Required) == 0 {
		return Evaluation{Score: 1, Success: true}
	}
	text := strings.ToLower(output.Output)
	matched := 0
	feedback := make([]string, 0)
	for _, fragment := range e.Required {
		if strings.Contains(text, strings.ToLower(fragment)) {
			matched++
		} else {
			feedback = append(feedback, "missing required fragment: "+fragment)
		}
	}
	score := float64(matched) / float64(len(e.Required))
	return Evaluation{Score: score, Success: matched == len(e.Required), Feedback: feedback}
}

// ScoreEvaluator converts a scoring function into an Evaluator. The function
// returns a score in [0, 1] and optional feedback.
type ScoreEvaluator func(context.Context, Task, RunOutput) (float64, []string)

func (f ScoreEvaluator) Evaluate(ctx context.Context, task Task, output RunOutput) Evaluation {
	score, feedback := f(ctx, task, output)
	score = clampScore(score)
	return Evaluation{Score: score, Success: score >= 1, Feedback: feedback}
}
