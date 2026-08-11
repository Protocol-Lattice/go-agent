package main

import (
	"context"
	"errors"
	"strings"
	"testing"
)

func TestRunSelfHealingLoopRecoversFromExecutionFailure(t *testing.T) {
	var output strings.Builder
	var prompts []string
	calls := 0

	err := runSelfHealingLoop(context.Background(), 2, 1, "finish the task", &output,
		func(_ context.Context, prompt string) (string, string, error) {
			calls++
			prompts = append(prompts, prompt)
			if calls == 1 {
				return "", "coordinator", errors.New("specialist unavailable")
			}
			return "completed\n" + doneMarker, "coordinator", nil
		},
	)
	if err != nil {
		t.Fatalf("runSelfHealingLoop() error = %v, want nil", err)
	}
	if calls != 2 {
		t.Fatalf("execution calls = %d, want 2", calls)
	}
	if !strings.Contains(prompts[1], "Recovery context:") || !strings.Contains(prompts[1], "specialist unavailable") {
		t.Fatalf("recovery prompt = %q, want failure context", prompts[1])
	}
	if !strings.Contains(output.String(), "=== recovery 1/1 after step 1 (coordinator) ===") {
		t.Fatalf("output = %q, want recovery status", output.String())
	}
}

func TestRunSelfHealingLoopStopsAfterRecoveryBudget(t *testing.T) {
	failed := errors.New("provider unavailable")
	calls := 0

	err := runSelfHealingLoop(context.Background(), 3, 1, "finish the task", &strings.Builder{},
		func(context.Context, string) (string, string, error) {
			calls++
			return "", "", failed
		},
	)
	if !errors.Is(err, failed) {
		t.Fatalf("runSelfHealingLoop() error = %v, want wrapped %v", err, failed)
	}
	if !strings.Contains(err.Error(), "failed after 1 recovery attempts") {
		t.Fatalf("error = %q, want exhausted recovery budget", err)
	}
	if calls != 2 {
		t.Fatalf("execution calls = %d, want initial attempt plus one recovery", calls)
	}
}

func TestRunSelfHealingLoopDoesNotRetryContextCancellation(t *testing.T) {
	calls := 0
	err := runSelfHealingLoop(context.Background(), 3, 3, "finish the task", &strings.Builder{},
		func(context.Context, string) (string, string, error) {
			calls++
			return "", "coordinator", context.Canceled
		},
	)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("runSelfHealingLoop() error = %v, want wrapped context cancellation", err)
	}
	if calls != 1 {
		t.Fatalf("execution calls = %d, want 1", calls)
	}
}
