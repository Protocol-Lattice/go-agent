package agent

import (
	"context"
	"errors"
	"fmt"
	"regexp"
	"strings"
	"sync"

	"github.com/Protocol-Lattice/go-agent/src/memory"
	"github.com/Protocol-Lattice/go-agent/src/models"
)

var explicitSingleToolRequestPattern = regexp.MustCompile(
	`\b(?:run|call|invoke|execute|use)\s+(?:the\s+)?[a-z0-9][a-z0-9_.:/-]*\s+tool\b`,
)

// shouldUseDirectCodeMode keeps CodeMode as an explicit execution strategy.
// Broad repository work must go through the normal tool orchestrator so the
// planner can inspect, edit, validate, and repeat tool calls across many steps.
func shouldUseDirectCodeMode(input string) bool {
	if explicitlyRequestsCodeMode(input) {
		return true
	}

	// Keep compatibility with natural-language requests for one named tool,
	// such as "Run the echo tool". Plural or broad requests such as "inspect
	// the repository with the available tools" intentionally do not match.
	return explicitSingleToolRequestPattern.MatchString(strings.ToLower(strings.TrimSpace(input)))
}

func explicitlyRequestsCodeMode(input string) bool {
	lower := strings.ToLower(strings.TrimSpace(input))
	for _, marker := range []string{
		"codemode",
		"code mode",
		"run code",
		"execute go code",
		"execute code",
	} {
		if strings.Contains(lower, marker) {
			return true
		}
	}
	return false
}

// GenerateStream provides a streaming interface for the agent's generation process.
// It follows the same logic as Generate but returns a channel of chunks.
func (a *Agent) GenerateStream(ctx context.Context, sessionID, userInput string) (<-chan models.StreamChunk, error) {
	if a.InputGuardrails != nil {
		transformed, err := a.InputGuardrails.ValidateAndTransform(ctx, userInput)
		if err != nil {
			return nil, err
		}
		userInput = transformed
	}

	trimmed := strings.TrimSpace(userInput)
	if trimmed == "" {
		return nil, errors.New("user input is empty")
	}

	// Helper to wrap immediate result in a stream.
	immediateStream := func(val any, err error) (<-chan models.StreamChunk, error) {
		ch := make(chan models.StreamChunk, 1)
		if err != nil {
			ch <- models.StreamChunk{Err: err, Done: true}
		} else {
			str := fmt.Sprint(val)
			ch <- models.StreamChunk{Delta: str, FullText: str, Done: true}
		}
		close(ch)
		return ch, nil
	}

	// 0. DIRECT TOOL INVOCATION
	if toolName, args, ok := a.detectDirectToolCall(trimmed); ok {
		result, err := a.executeTool(ctx, sessionID, toolName, args)
		return immediateStream(result, err)
	}

	// 1. SUBAGENT COMMANDS
	if handled, out, meta, err := a.handleCommand(ctx, sessionID, userInput); handled {
		if err != nil {
			return nil, err
		}
		a.storeMemory(sessionID, "subagent", out, meta)
		return immediateStream(out, nil)
	}

	// Prefetch context while the agent decides whether tool orchestration is needed.
	prefetchCtx, cancelPrefetch := context.WithCancel(ctx)
	defer cancelPrefetch()
	var (
		prefetchWG sync.WaitGroup
		records    []memory.MemoryRecord
	)
	prefetchWG.Add(1)
	go func() {
		defer prefetchWG.Done()
		records, _ = a.retrieveContext(prefetchCtx, sessionID, userInput, a.contextLimit)
	}()

	// 2. CODEMODE
	// Do not automatically short-circuit every request into one CodeMode call.
	// In particular, repository-wide refactors need the multi-step tool loop.
	if a.CodeMode != nil && explicitlyRequestsCodeMode(trimmed) && !a.AllowUnsafeTools {
		return nil, fmt.Errorf("unauthorized tool execution: codemode.run_code is restricted")
	}
	if a.CodeMode != nil && a.AllowUnsafeTools && shouldUseDirectCodeMode(trimmed) {
		handled, output, err := a.CodeMode.CallTool(ctx, userInput)
		if err != nil {
			return immediateStream(output, err)
		}
		if handled {
			return immediateStream(output, err)
		}
	}

	// 3. TOOL ORCHESTRATOR
	// This is intentionally reached for broad tasks such as repository refactors.
	// The orchestrator can perform up to configuredToolLoopMaxSteps sequential
	// tool steps instead of returning after the first tool invocation.
	prefetchWG.Wait()
	handled, output, orchestratorErr := a.toolOrchestrator(ctx, sessionID, userInput, records)
	if orchestratorErr != nil {
		return nil, orchestratorErr
	}
	if handled {
		return immediateStream(output, nil)
	}

	// 5. STORE USER MEMORY
	a.storeMemory(sessionID, "user", userInput, nil)

	if a.userLooksLikeToolCall(trimmed) {
		return immediateStream("", nil)
	}

	// 6. LLM COMPLETION (Streaming)
	var sb strings.Builder
	sb.Grow(4096)
	sb.WriteString(a.systemInstructions())
	sb.WriteString("\n\nConversation memory (TOON):\n")
	sb.WriteString(a.renderMemory(records))
	sb.WriteString("\n\nUser: ")
	sb.WriteString(sanitizeInput(userInput))
	sb.WriteString("\n\n")

	prompt := sb.String()
	stream, err := a.model.GenerateStream(ctx, prompt)
	if err != nil {
		return nil, err
	}

	outCh := make(chan models.StreamChunk)

	if a.Guardrails != nil {
		go func() {
			defer close(outCh)
			var full strings.Builder
			for chunk := range stream {
				if chunk.Err != nil {
					outCh <- chunk
					return
				}
				if chunk.Delta != "" {
					full.WriteString(chunk.Delta)
				}
			}

			finalText := full.String()
			validatedText, gErr := a.Guardrails.ValidateAndRepair(ctx, finalText)
			if gErr != nil {
				outCh <- models.StreamChunk{Err: gErr, Done: true}
				return
			}
			outCh <- models.StreamChunk{Delta: validatedText, FullText: validatedText, Done: true}
			a.storeMemory(sessionID, "assistant", validatedText, nil)
		}()
	} else {
		go func() {
			defer close(outCh)
			var full strings.Builder
			for chunk := range stream {
				if chunk.Err != nil {
					outCh <- chunk
					return
				}
				if chunk.Delta != "" {
					full.WriteString(chunk.Delta)
				}
				outCh <- chunk
			}
			a.storeMemory(sessionID, "assistant", full.String(), nil)
		}()
	}

	return outCh, nil
}
