package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/alpkeskin/gotoon"
	"github.com/universal-tool-calling-protocol/go-utcp/src/plugins/chain"
)

// codeChainOrchestrator lets the LLM decide whether to execute a multi-step UTCP chain.
// It mirrors the design and behavior of toolOrchestrator, but produces a []ChainStep
// and executes it via CodeChain (UTCP chain execution engine).

func (a *Agent) codeChainOrchestrator(
	ctx context.Context,
	sessionID string,
	userInput string,
) (bool, string, error) {

	if a.CodeChain == nil {
		return false, "", nil
	}

	// ----------------------------------------------------------
	// 1. Build chain-selection prompt (LLM chain planning engine)
	// ----------------------------------------------------------
	toolList := a.ToolSpecs()
	toolDesc := a.cachedToolPrompt(toolList)

	choicePrompt := fmt.Sprintf(`You are a UTCP Chain Planning Engine that constructs multi-step tool execution plans.

USER REQUEST:
%q

AVAILABLE UTCP TOOLS:
%s

OBJECTIVE:
Determine if the user's request requires a sequence of UTCP tool calls. If so, construct an optimal execution chain.

CHAIN CONSTRUCTION RULES:
RULES:
1. Tool names and parameters MUST exactly match the UTCP tools listed above.

You MUST use the exact tool names as discovered:

- "http.echo"
- "http.timestamp"
- "http.math.add"
- "http.math.multiply"
- "http.string.concat"
- "http.stream.echo"

NEVER shorten or remove the provider prefix.
NEVER use "echo" or "math.add" — they are INVALID.
If a user mentions a shorthand name like “add”, you MUST map it to the correct
fully-qualified tool name such as "http.math.add".
2. "inputs" MUST be a JSON object containing all required parameters for that tool
3. "use_previous" is true when this step consumes output from the previous step
4. "stream" is true ONLY if:
   - The tool explicitly supports streaming, AND
   - Streaming is beneficial for this use case
5. Steps should be ordered to satisfy data dependencies
6. Each step's inputs can reference previous step outputs via "use_previous": true
7. The first step always has "use_previous": false
IMPORTANT:
The "tool_name" MUST exactly match the tool name from discovery.
NEVER abbreviate, shorten, rename, or paraphrase tool names.

For example:
- Use "math.add", NOT "add"
- Use "math.multiply", NOT "multiply"
- Use "string.concat", NOT "concat"
- Use "stream.echo", NOT "echo_stream" or "streamecho"

If the user describes an operation using a shortened name,
you MUST map it to the EXACT tool name from the discovery list.

DECISION LOGIC:
- Single tool call needed → Create a chain with one step
- Multiple dependent tool calls → Create a chain with multiple steps ordered by dependency
- No tools needed → Set "use_chain": false with empty "steps" array

CHAINING EXAMPLES:
Example 1 - Sequential processing:
  Step 1: fetch_data → outputs raw data
  Step 2: process_data (use_previous: true) → receives raw data, outputs processed result

Example 2 - Independent then merge:
  Step 1: get_userinfo (use_previous: false)
  Step 2: enrich_data (use_previous: true) → uses userinfo output

Example 3 - Streaming final output:
  Step 1: generate_text (use_previous: false, stream: false)
  Step 2: format_output (use_previous: true, stream: true) → streams formatted result

OUTPUT FORMAT:
Respond with ONLY valid JSON. NO markdown code blocks. NO explanations. NO reasoning text.

When tool chain is needed:
{
  "use_chain": true,
  "steps": [
    {
      "tool_name": "<exact_tool_name>",
      "inputs": { "param1": "value1", "param2": "value2" },
      "use_previous": false,
      "stream": false
    },
    {
      "tool_name": "<next_tool_name>",
      "inputs": { "param": "value" },
      "use_previous": true,
      "stream": false
    }
  ],
  "timeout": 20000
}

When NO tools needed:
{
  "use_chain": false,
  "steps": [],
  "timeout": 20000
}

Analyze the request and respond with ONLY the JSON object:`, userInput, toolDesc)
	raw, err := a.model.Generate(ctx, choicePrompt)
	if err != nil {
		return false, "", nil
	}

	jsonStr := extractJSON(fmt.Sprint(raw))
	if jsonStr == "" {
		return false, "", nil
	}

	// ----------------------------------------------------------
	// 2. Parse JSON with all chain fields (including stream/use_previous)
	// ----------------------------------------------------------
	type chainStepJSON struct {
		ID          string         `json:"id"`
		ToolName    string         `json:"tool_name"`
		Inputs      map[string]any `json:"inputs"`
		UsePrevious bool           `json:"use_previous"`
		Stream      bool           `json:"stream"`
	}

	var parsed struct {
		Steps   []chainStepJSON `json:"steps"`
		Timeout int             `json:"timeout"`
	}

	if err := json.Unmarshal([]byte(jsonStr), &parsed); err != nil {
		return false, "", nil
	}
	timeout := time.Duration(parsed.Timeout) * time.Millisecond
	if timeout <= 0 {
		timeout = 20 * time.Second
	}

	// ----------------------------------------------------------
	// 3. Convert JSON → UTCP ChainStep via builder (correct)
	// ----------------------------------------------------------
	steps := make([]chain.ChainStep, len(parsed.Steps))
	for i, s := range parsed.Steps {
		steps[i] = chain.ChainStep{
			ToolName:    s.ToolName,
			Inputs:      s.Inputs,
			Stream:      s.Stream,
			UsePrevious: s.UsePrevious,
		}
	}

	// ----------------------------------------------------------
	// 4. Execute chain
	// ----------------------------------------------------------
	result, err := a.CodeChain.CallToolChain(ctx, steps, timeout)
	if err != nil {
		a.storeMemory(sessionID, "assistant",
			fmt.Sprintf("Chain error: %v", err),
			map[string]string{"source": "chain"},
		)
		return true, "", err
	}

	// ----------------------------------------------------------
	// 5. Encode result
	// ----------------------------------------------------------
	outBytes, _ := json.Marshal(result)
	rawOut := string(outBytes)

	toonBytes, _ := gotoon.Encode(rawOut)
	full := fmt.Sprintf("%s\n\n.toon:\n%s", rawOut, string(toonBytes))

	// ----------------------------------------------------------
	// 6. Store memory
	// ----------------------------------------------------------
	a.storeMemory(sessionID, "assistant", full, map[string]string{
		"source": "chain",
	})

	return true, rawOut, nil
}
