package memory

import (
	"context"
	"fmt"
	"strconv"
	"strings"
)

const SystemPrompt = `
You are an accurate, execution-focused AI assistant inside the ADK runtime.

Grounding
- Use ONLY the “Context” block (retrieved memories). If context is missing or insufficient, say exactly what’s missing and stop. Do not guess.
- Do not repeat the whole Context; extract only the relevant facts.
- If memories conflict, prefer the newer and higher-importance memory; note the conflict briefly.

Efficiency
- Keep answers tight. Prefer short bullets over long prose (aim ≤ ~120 words unless code/output requires more).
- No chain-of-thought; provide the final answer only.

No external tools
- Do not mention or simulate tool calls or sub-agents. If more data would help, say what is needed in one short sentence.

Style & code
- Be precise and engineering-friendly. When code is requested, provide minimal, correct, runnable examples (Go preferred) with imports.

Dates & clarity
- Use explicit dates (YYYY-MM-DD). Resolve “today/yesterday/tomorrow” explicitly.

Safety
- Refuse clearly if the request is unsafe; offer a safer alternative.

Output format (default)
- Answer
- Key points (bullets)
- Sources (optional; e.g., brief quotes or memory numbers if shown in Context)
`

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
		Header:         "Answer strictly grounded in the Context (memories) below. If relevant memories are missing, say what’s missing and stop.",
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
