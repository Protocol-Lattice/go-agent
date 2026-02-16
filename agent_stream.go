package agent

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/Protocol-Lattice/go-agent/src/models"
)

// GenerateStream provides a streaming interface for the agent's generation process.
// It follows the same logic as Generate but returns a channel of chunks.
func (a *Agent) GenerateStream(ctx context.Context, sessionID, userInput string) (<-chan models.StreamChunk, error) {
	trimmed := strings.TrimSpace(userInput)
	if trimmed == "" {
		return nil, errors.New("user input is empty")
	}

	// Helper to wrap immediate result in a stream
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

	// 2. CODEMODE
	if a.CodeMode != nil {
		if handled, output, err := a.CodeMode.CallTool(ctx, userInput); handled {
			return immediateStream(output, err)
		}
	}

	// 3. Chain Orchestrator
	if handled, output, err := a.codeChainOrchestrator(ctx, sessionID, userInput); handled {
		return immediateStream(output, err)
	}

	// 4. TOOL ORCHESTRATOR
	if handled, output, err := a.toolOrchestrator(ctx, sessionID, userInput); handled {
		return immediateStream(output, err)
	}

	// 5. STORE USER MEMORY
	a.storeMemory(sessionID, "user", userInput, nil)

	// If it looked like a tool call but wasn't handled, return empty
	if a.userLooksLikeToolCall(trimmed) {
		return immediateStream("", nil)
	}

	// 6. LLM COMPLETION (Streaming)
	prompt, err := a.buildPrompt(ctx, sessionID, userInput)
	if err != nil {
		return nil, err
	}

	// Note: Currently GenerateStream does not support file attachments for streaming.
	// We proceed with text-only streaming.

	stream, err := a.model.GenerateStream(ctx, prompt)
	if err != nil {
		return nil, err
	}

	// Wrap the stream to intercept and store memory
	outCh := make(chan models.StreamChunk)
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
		// Store memory after completion
		finalText := full.String()
		a.storeMemory(sessionID, "assistant", finalText, nil)
	}()

	return outCh, nil
}
