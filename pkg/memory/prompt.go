package memory

import (
	"context"
	"fmt"
	"strconv"
	"strings"
)

// SystemPromptV2 is a concise, production-ready system message for RAG agents.
// It enforces grounding, avoids chain-of-thought, and standardizes output.
// SystemPromptV3 is a strict, production-ready system message for ADK agents.
// - Grounded RAG with explicit memory citations [M#]
// - No chain-of-thought; final answers + terse bullets
// - Clear output contracts (Answer / Key points / Sources)
// - Safe, engineering-first defaults (Go-first for code)
// - Tool/subagent orchestration is handled by the runtime; never roleplay calls
const SystemPrompt = `
You are an accurate, execution-focused AI agent working inside the ADK runtime.

Non-negotiables:
1) Grounding & honesty
   - Use ONLY the "Context" block provided by the runtime. If context is missing or insufficient, say what’s missing and stop.
   - Never invent facts, tools, files, or people. If unsure, say "Unknown".
   - Cite memories used with [M1], [M2], … matching the indices in "Context".

2) No hidden reasoning
   - Do NOT reveal chain-of-thought or step-by-step internal reasoning.
   - Provide the final answer with brief, verifiable points.

3) Tools & subagents
   - The runtime may call tools or subagents (UTCP) for you. Do not fabricate or describe tool calls.
   - Write your answer normally. If additional data would help, briefly state what to fetch next (one sentence).

4) Style & code
   - Engineering-friendly, concise. Prefer bullet points over prose walls.
   - When code is requested, provide minimal, correct, runnable examples (Go preferred). Include imports. Avoid pseudo-code.
   - Avoid over-abstraction; prefer small, composable examples.

5) Safety
   - Refuse clearly if the request is unsafe (illegal, harmful, personal data leakage). Offer a safer alternative.

6) Dates & clarity
   - Use explicit dates (YYYY-MM-DD). If the user says “today/yesterday/tomorrow”, resolve them explicitly.
   - Keep assumptions minimal; state them when needed.

7) Output format (default):
   - Answer
   - Key points (bullets)
   - Sources (e.g., [M1], [M3]) — omit if none used

8) JSON mode (optional):
   - If the user explicitly asks for JSON, respond with:
     {"answer": "...", "key_points": ["..."], "sources": ["M1","M3"]}

Remember: precise, grounded, terse.`

type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
)

type Message struct {
	Role    Role
	Content string
}

// PromptOpts lets callers tune tone/format or override the system prompt.
type PromptOpts struct {
	SystemPrompt   string // if empty, SystemPromptV2
	IncludeIndices bool   // show [M1], [M2] next to each context row (default true)
	Header         string // optional one-liner like "You are a helpful AI agent."
}

func DefaultPromptOpts() PromptOpts {
	return PromptOpts{
		SystemPrompt:   SystemPrompt,
		IncludeIndices: true,
		Header:         "You orchestrate specialists and tools to help the user build AI agents.",
	}
}

// BuildMessages assembles a grounded chat prompt: [system] + [user (with Context + Query)].
// It preserves your RetrieveContext() call and formatting but makes messages provider-agnostic.
func (sm *SessionMemory) BuildMessages(ctx context.Context, sessionID, query string, limit int, opts PromptOpts) ([]Message, error) {
	results, err := sm.RetrieveContext(ctx, sessionID, query, limit)
	if err != nil {
		return nil, fmt.Errorf("context retrieval failed: %w", err)
	}

	if opts.SystemPrompt == "" {
		opts.SystemPrompt = SystemPrompt
	}

	var b strings.Builder
	b.Grow(1024)

	// Optional short header atop the user message (kept separate from system to reduce tokens).
	if opts.Header != "" {
		b.WriteString(opts.Header)
		b.WriteString("\n\n")
	}

	// Context block
	b.WriteString("Context:\n")
	if len(results) == 0 {
		b.WriteString("(No relevant memories found)\n\n")
	} else {
		for i := range results {
			// Keep writes simple & fast; avoid fmt.Sprintf in hot loops.
			// Show memory index & content; trimming prevents runaway whitespace.
			b.WriteString(strconv.Itoa(i + 1))
			if opts.IncludeIndices {
				b.WriteString(". [M")
				b.WriteString(strconv.Itoa(i + 1))
				b.WriteString("] ")
			} else {
				b.WriteString(". ")
			}
			b.WriteString(strings.TrimSpace(results[i].Content))
			b.WriteByte('\n')
		}
		b.WriteByte('\n')
	}

	// User query
	b.WriteString("User Query:\n")
	b.WriteString(query)
	b.WriteString("\n\n")
	b.WriteString("Please respond using the requested Output format.\n")

	return []Message{
		{Role: RoleSystem, Content: opts.SystemPrompt},
		{Role: RoleUser, Content: b.String()},
	}, nil
}

// BuildPrompt remains for compatibility; internally uses BuildMessages and flattens.
func (sm *SessionMemory) BuildPrompt(ctx context.Context, sessionID, query string, limit int) (string, error) {
	msgs, err := sm.BuildMessages(ctx, sessionID, query, limit, DefaultPromptOpts())
	if err != nil {
		return "", err
	}

	// Simple “chatML-like” flattening for providers that expect a single string prompt.
	// You can customize this per adapter if needed.
	var sb strings.Builder
	sb.Grow(1024)

	for _, m := range msgs {
		switch m.Role {
		case RoleSystem:
			sb.WriteString("## System\n")
		case RoleUser:
			sb.WriteString("## User\n")
		case RoleAssistant:
			sb.WriteString("## Assistant\n")
		default:
			sb.WriteString("## ")
			sb.WriteString(string(m.Role))
			sb.WriteByte('\n')
		}
		sb.WriteString(m.Content)
		if !strings.HasSuffix(m.Content, "\n") {
			sb.WriteByte('\n')
		}
		sb.WriteByte('\n')
	}

	// The assistant section header cues some models that a reply should follow.
	sb.WriteString("## Assistant\n")
	return sb.String(), nil
}
