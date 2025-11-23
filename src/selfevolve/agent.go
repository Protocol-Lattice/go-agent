package selfevolve

import (
	"context"
	"fmt"
	"log"
	"strings"
	"time"

	"github.com/Protocol-Lattice/go-agent"
	"github.com/Protocol-Lattice/go-agent/src/models"
)

// EvolutionConfig configures the self-evolution loop
type EvolutionConfig struct {
	// MaxRetries is the maximum number of optimization attempts per task
	MaxRetries int

	// TargetScore is the minimum score to consider a prompt successful
	TargetScore float64

	// Evaluators to use for scoring outputs
	Evaluators []Evaluator

	// EnableLogging enables detailed evolution logging
	EnableLogging bool

	// StopOnSuccess stops optimization once target score is reached
	StopOnSuccess bool
}

// DefaultEvolutionConfig returns sensible defaults
func DefaultEvolutionConfig() *EvolutionConfig {
	return &EvolutionConfig{
		MaxRetries:    3,
		TargetScore:   0.8,
		Evaluators:    []Evaluator{},
		EnableLogging: true,
		StopOnSuccess: true,
	}
}

// EvolvingAgent wraps an agent with self-evolution capabilities
type EvolvingAgent struct {
	agent     *agent.Agent
	optimizer *PromptOptimizer
	evaluator Evaluator
	config    *EvolutionConfig

	// Metrics
	totalTasks      int
	successfulTasks int
	totalRetries    int
	evolutionLog    []EvolutionEntry
}

// EvolutionEntry records a single evolution attempt
type EvolutionEntry struct {
	Timestamp     time.Time         `json:"timestamp"`
	TaskNumber    int               `json:"task_number"`
	Attempt       int               `json:"attempt"`
	PromptVersion int               `json:"prompt_version"`
	Input         string            `json:"input"`
	Output        string            `json:"output"`
	Evaluation    *EvaluationResult `json:"evaluation"`
	Success       bool              `json:"success"`
}

// NewEvolvingAgent creates a self-evolving agent
func NewEvolvingAgent(
	baseAgent *agent.Agent,
	optimizerModel models.Agent,
	initialPrompt string,
	modelName string,
	config *EvolutionConfig,
) *EvolvingAgent {
	if config == nil {
		config = DefaultEvolutionConfig()
	}

	optimizer := NewPromptOptimizer(optimizerModel, initialPrompt, modelName)

	// Create composite evaluator if multiple evaluators provided
	var evaluator Evaluator
	if len(config.Evaluators) == 0 {
		// Default evaluator
		evaluator = NewLLMAsJudgeEvaluator(
			optimizerModel,
			"default_quality",
			"The output should be accurate, helpful, and well-formatted",
			config.TargetScore,
		)
	} else if len(config.Evaluators) == 1 {
		evaluator = config.Evaluators[0]
	} else {
		evaluator = NewCompositeEvaluator(config.Evaluators, StrategyAverage)
	}

	return &EvolvingAgent{
		agent:        baseAgent,
		optimizer:    optimizer,
		evaluator:    evaluator,
		config:       config,
		evolutionLog: make([]EvolutionEntry, 0),
	}
}

// Generate runs the agent with self-evolution
func (e *EvolvingAgent) Generate(ctx context.Context, sessionID, input string) (string, error) {
	e.totalTasks++
	taskNum := e.totalTasks

	if e.config.EnableLogging {
		log.Printf("[Evolution] Task %d: Starting with prompt v%d", taskNum, e.optimizer.Current().Version)
	}

	var bestOutput string
	var bestScore float64
	success := false

	for attempt := 1; attempt <= e.config.MaxRetries; attempt++ {
		e.totalRetries++

		if e.config.EnableLogging {
			log.Printf("[Evolution] Task %d, Attempt %d: Generating output...", taskNum, attempt)
		}

		// Generate output with current prompt
		rawOutput, err := e.agent.Generate(ctx, sessionID, input)
		if err != nil {
			if e.config.EnableLogging {
				log.Printf("[Evolution] Task %d, Attempt %d: Generation failed: %v", taskNum, attempt, err)
			}
			continue
		}

		// Convert output to string
		output := fmt.Sprint(rawOutput)

		// Evaluate the output
		evaluation, err := e.evaluator.Evaluate(ctx, input, output)
		if err != nil {
			if e.config.EnableLogging {
				log.Printf("[Evolution] Task %d, Attempt %d: Evaluation failed: %v", taskNum, attempt, err)
			}
			continue
		}

		// Log this attempt
		entry := EvolutionEntry{
			Timestamp:     time.Now(),
			TaskNumber:    taskNum,
			Attempt:       attempt,
			PromptVersion: e.optimizer.Current().Version,
			Input:         input,
			Output:        output,
			Evaluation:    evaluation,
			Success:       evaluation.Passed,
		}
		e.evolutionLog = append(e.evolutionLog, entry)

		if e.config.EnableLogging {
			log.Printf("[Evolution] Task %d, Attempt %d: Score=%.3f, Passed=%v",
				taskNum, attempt, evaluation.Score, evaluation.Passed)
			log.Printf("[Evolution] Reasoning: %s", evaluation.Reasoning)
		}

		// Update optimizer score
		e.optimizer.UpdateScore(evaluation.Score)

		// Track best output
		if evaluation.Score > bestScore {
			bestScore = evaluation.Score
			bestOutput = output
		}

		// Check if we met the target
		if evaluation.Passed && evaluation.Score >= e.config.TargetScore {
			success = true
			e.successfulTasks++

			if e.config.EnableLogging {
				log.Printf("[Evolution] Task %d: SUCCESS with prompt v%d (score: %.3f)",
					taskNum, e.optimizer.Current().Version, evaluation.Score)
			}

			if e.config.StopOnSuccess {
				return output, nil
			}
		}

		// If not successful and we have more attempts, optimize the prompt
		if !success && attempt < e.config.MaxRetries {
			if e.config.EnableLogging {
				log.Printf("[Evolution] Task %d: Optimizing prompt (current v%d)...",
					taskNum, e.optimizer.Current().Version)
			}

			newVersion, err := e.optimizer.Optimize(ctx, input, output, evaluation)
			if err != nil {
				if e.config.EnableLogging {
					log.Printf("[Evolution] Task %d: Optimization failed: %v", taskNum, err)
				}
				continue
			}

			e.optimizer.Update(newVersion)

			if e.config.EnableLogging {
				log.Printf("[Evolution] Task %d: Updated to prompt v%d",
					taskNum, e.optimizer.Current().Version)
			}
		}
	}

	if !success {
		if e.config.EnableLogging {
			log.Printf("[Evolution] Task %d: FAILED after %d attempts (best score: %.3f)",
				taskNum, e.config.MaxRetries, bestScore)
			log.Printf("[Evolution] Keeping latest prompt v%d for next task",
				e.optimizer.Current().Version)
		}
	}

	return bestOutput, nil
}

// GetMetrics returns evolution metrics
func (e *EvolvingAgent) GetMetrics() map[string]any {
	successRate := 0.0
	if e.totalTasks > 0 {
		successRate = float64(e.successfulTasks) / float64(e.totalTasks)
	}

	avgRetriesPerTask := 0.0
	if e.totalTasks > 0 {
		avgRetriesPerTask = float64(e.totalRetries) / float64(e.totalTasks)
	}

	return map[string]any{
		"total_tasks":            e.totalTasks,
		"successful_tasks":       e.successfulTasks,
		"success_rate":           successRate,
		"total_retries":          e.totalRetries,
		"avg_retries_per_task":   avgRetriesPerTask,
		"current_prompt_version": e.optimizer.Current().Version,
		"best_prompt_version":    e.optimizer.Best().Version,
		"best_prompt_score":      e.optimizer.Best().Score,
	}
}

// GetEvolutionLog returns the complete evolution history
func (e *EvolvingAgent) GetEvolutionLog() []EvolutionEntry {
	return e.evolutionLog
}

// GetCurrentPrompt returns the current prompt version
func (e *EvolvingAgent) GetCurrentPrompt() *PromptVersion {
	return e.optimizer.Current()
}

// GetBestPrompt returns the best performing prompt version
func (e *EvolvingAgent) GetBestPrompt() *PromptVersion {
	return e.optimizer.Best()
}

// GetPromptHistory returns all prompt versions
func (e *EvolvingAgent) GetPromptHistory() []*PromptVersion {
	return e.optimizer.History()
}

// RollbackToBest reverts to the best performing prompt
func (e *EvolvingAgent) RollbackToBest() {
	e.optimizer.Rollback()
	if e.config.EnableLogging {
		log.Printf("[Evolution] Rolled back to best prompt v%d (score: %.3f)",
			e.optimizer.Best().Version, e.optimizer.Best().Score)
	}
}

// ApplyPrompt manually sets a specific prompt version
func (e *EvolvingAgent) ApplyPrompt(version *PromptVersion) {
	e.optimizer.Update(version)
	if e.config.EnableLogging {
		log.Printf("[Evolution] Applied prompt v%d", version.Version)
	}
}

// PrintSummary prints a summary of the evolution process
func (e *EvolvingAgent) PrintSummary() {
	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println("SELF-EVOLUTION SUMMARY")
	fmt.Println(strings.Repeat("=", 80))

	metrics := e.GetMetrics()
	fmt.Printf("Total Tasks: %d\n", metrics["total_tasks"])
	fmt.Printf("Successful Tasks: %d\n", metrics["successful_tasks"])
	fmt.Printf("Success Rate: %.1f%%\n", metrics["success_rate"].(float64)*100)
	fmt.Printf("Average Retries per Task: %.2f\n", metrics["avg_retries_per_task"])
	fmt.Printf("\nCurrent Prompt Version: v%d\n", metrics["current_prompt_version"])
	fmt.Printf("Best Prompt Version: v%d (score: %.3f)\n",
		metrics["best_prompt_version"], metrics["best_prompt_score"])

	fmt.Println("\n" + strings.Repeat("-", 80))
	fmt.Println("PROMPT EVOLUTION HISTORY")
	fmt.Println(strings.Repeat("-", 80))

	for _, version := range e.GetPromptHistory() {
		fmt.Printf("\nVersion %d (Score: %.3f) - %s\n",
			version.Version, version.Score, version.Timestamp.Format("15:04:05"))
		fmt.Printf("Prompt: %s\n", truncate(version.Prompt, 100))
	}

	fmt.Println("\n" + strings.Repeat("=", 80))
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen-3] + "..."
}
