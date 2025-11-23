package selfevolve

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/Protocol-Lattice/go-agent/src/models"
)

// PromptVersion represents a versioned system prompt
type PromptVersion struct {
	Version   int            `json:"version"`
	Prompt    string         `json:"prompt"`
	Model     string         `json:"model"`
	Timestamp time.Time      `json:"timestamp"`
	Metadata  map[string]any `json:"metadata"`
	Score     float64        `json:"score"`
}

// PromptOptimizer improves prompts based on evaluation feedback
type PromptOptimizer struct {
	model          models.Agent
	currentVersion *PromptVersion
	history        []*PromptVersion
	bestVersion    *PromptVersion
	metaPrompt     string
}

// NewPromptOptimizer creates a new prompt optimizer
func NewPromptOptimizer(model models.Agent, initialPrompt, modelName string) *PromptOptimizer {
	initial := &PromptVersion{
		Version:   1,
		Prompt:    initialPrompt,
		Model:     modelName,
		Timestamp: time.Now(),
		Metadata:  make(map[string]any),
		Score:     0.0,
	}

	metaPrompt := `You are a prompt optimization expert. Your task is to improve an AI agent's system prompt based on evaluation feedback.

CURRENT PROMPT:
%s

TASK INPUT:
%s

AGENT OUTPUT:
%s

EVALUATION FEEDBACK:
%s

IMPROVEMENT GOAL:
Create an improved version of the system prompt that addresses the feedback while maintaining the agent's core purpose.

Rules:
1. Keep the prompt concise and clear
2. Address specific issues mentioned in the feedback
3. Maintain the agent's original role and capabilities
4. Add specific instructions to prevent the identified issues
5. Use clear, actionable language

Respond ONLY with the improved prompt text. Do NOT include explanations or JSON.
`

	return &PromptOptimizer{
		model:          model,
		currentVersion: initial,
		history:        []*PromptVersion{initial},
		bestVersion:    initial,
		metaPrompt:     metaPrompt,
	}
}

// Current returns the current prompt version
func (p *PromptOptimizer) Current() *PromptVersion {
	return p.currentVersion
}

// Best returns the best performing prompt version
func (p *PromptOptimizer) Best() *PromptVersion {
	return p.bestVersion
}

// History returns all prompt versions
func (p *PromptOptimizer) History() []*PromptVersion {
	return p.history
}

// Optimize generates an improved prompt based on feedback
func (p *PromptOptimizer) Optimize(ctx context.Context, input, output string, feedback *EvaluationResult) (*PromptVersion, error) {
	// Collect feedback text
	feedbackText := fmt.Sprintf(
		"Score: %.2f\nPassed: %v\nReasoning: %s\nCriteria: %s",
		feedback.Score,
		feedback.Passed,
		feedback.Reasoning,
		feedback.Criteria,
	)

	if suggestions, ok := feedback.Metadata["suggestions"].(string); ok && suggestions != "" {
		feedbackText += fmt.Sprintf("\nSuggestions: %s", suggestions)
	}

	// Generate improved prompt
	optimizationPrompt := fmt.Sprintf(
		p.metaPrompt,
		p.currentVersion.Prompt,
		input,
		output,
		feedbackText,
	)

	response, err := p.model.Generate(ctx, optimizationPrompt)
	if err != nil {
		return nil, fmt.Errorf("prompt optimization failed: %w", err)
	}

	improvedPrompt := strings.TrimSpace(fmt.Sprint(response))
	if improvedPrompt == "" {
		return nil, fmt.Errorf("optimizer returned empty prompt")
	}

	// Create new version
	newVersion := &PromptVersion{
		Version:   p.currentVersion.Version + 1,
		Prompt:    improvedPrompt,
		Model:     p.currentVersion.Model,
		Timestamp: time.Now(),
		Metadata: map[string]any{
			"previous_version":    p.currentVersion.Version,
			"feedback_score":      feedback.Score,
			"optimization_reason": feedback.Reasoning,
		},
		Score: 0.0, // Will be set after evaluation
	}

	return newVersion, nil
}

// Update sets a new current version and tracks it in history
func (p *PromptOptimizer) Update(newVersion *PromptVersion) {
	p.currentVersion = newVersion
	p.history = append(p.history, newVersion)

	// Update best version if this one is better
	if newVersion.Score > p.bestVersion.Score {
		p.bestVersion = newVersion
	}
}

// UpdateScore updates the score for the current version
func (p *PromptOptimizer) UpdateScore(score float64) {
	p.currentVersion.Score = score

	// Update best version if needed
	if score > p.bestVersion.Score {
		p.bestVersion = p.currentVersion
	}
}

// Rollback reverts to the best performing version
func (p *PromptOptimizer) Rollback() {
	p.currentVersion = p.bestVersion
}

// RollbackToVersion reverts to a specific version
func (p *PromptOptimizer) RollbackToVersion(version int) error {
	for _, v := range p.history {
		if v.Version == version {
			p.currentVersion = v
			return nil
		}
	}
	return fmt.Errorf("version %d not found in history", version)
}
