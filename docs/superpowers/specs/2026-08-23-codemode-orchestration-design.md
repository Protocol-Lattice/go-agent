# CodeMode Orchestration Design

## Goal

Make CodeMode orchestration deterministic and state-safe. The planner proposes actions, the canonical UTCP registry defines what can execute, the orchestrator enforces the current state, CodeMode executes only approved canonical tools, and verification determines whether a mutation actually completed the request.

## Contract

```text
request
  -> classify
  -> planner JSON
  -> parse + semantic validation
  -> canonical registry validation
  -> state/mutation gate
  -> CodeMode execution
  -> observe
  -> verify
  -> complete
```

Invalid planner JSON is repaired at most twice. Invalid plans never reach execution.

## Planner envelope

The planner returns exactly one JSON object:

```json
{"use_tool":true,"tool_name":"codemode.run_code","arguments":{"code":"..."},"reason":"next concrete action","final_answer":""}
```

or:

```json
{"use_tool":false,"tool_name":"","arguments":{},"reason":"complete","final_answer":"..."}
```

`tool_name` must be the planner-visible CodeMode tool for repository work. Canonical repository tools are invoked from inside CodeMode.

## Canonical registry

The UTCP registry is the only authority for tool existence and mutation capability. The planner may not invent tool names or aliases. CodeMode `CallTool` and `CallToolStream` calls must resolve against this registry before execution.

## State machine

```text
START -> CLASSIFY -> PLAN -> VALIDATE -> INSPECTION -> MUTATION -> VERIFY -> COMPLETE
```

For read-only requests, the mutation/verification states are not required. For mutation requests, completion requires both a successful mutation and verification.

## Mutation gate

When:

```text
requiresMutation == true
inspected == true
mutated == false
```

the next executable action must be mutation-capable. Read/list/search actions and read-only CodeMode programs are rejected. This gate is enforced in Go and is not merely a prompt instruction.

## Verification gate

After mutation, CodeMode may perform read-only verification. `complete` is rejected until the requested postcondition is verified.

## Error handling

Stable planner error categories include:

- `invalid_json`
- `invalid_action`
- `unknown_tool`
- `invalid_arguments`
- `empty_code`
- `mutation_required_next`
- `verification_required`

Planner repair is bounded to two attempts. Tool retries and total orchestration steps are also bounded.

## Regression requirements

Tests must cover invalid JSON, bounded repair, unknown canonical tools, CodeMode execution through canonical tools, mutation-required transitions, prevention of read loops, premature completion, verification, and max-step termination.

## Non-goals

This design does not add a second tool registry, autonomous tool-name translation, unrestricted CodeMode execution, or unbounded planner retries.
