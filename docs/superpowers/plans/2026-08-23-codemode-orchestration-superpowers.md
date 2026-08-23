# CodeMode Orchestration Superpowers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make CodeMode orchestration deterministic by enforcing a strict planner envelope, canonical UTCP tool validation, bounded planner repair, mutation gating, verification-aware completion, and regression coverage for the observed orchestration failures.

**Architecture:** Keep the existing `toolOrchestrator` as the execution boundary, but make planner parsing/validation and state transitions explicit and deterministic. CodeMode remains the only planner-visible execution tool; its `CallTool`/`CallToolStream` operations are constrained by the canonical UTCP registry. Mutation and verification are state gates rather than prompt-only instructions.

**Tech Stack:** Go, `encoding/json`, existing `go-utcp` tools/codemode packages, Go tests, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-08-23-codemode-orchestration-design.md` (implemented contract is also documented by the existing orchestration plan).

## Global Constraints

- CodeMode is the only planner/execution tool exposed for repository work.
- Canonical UTCP registry is the only source of valid tool names and capabilities.
- Invalid planner JSON must never reach execution.
- Planner repair is bounded to 2 attempts.
- When mutation is required, inspection is complete, and mutation has not happened, the next action must be mutation-capable.
- Mutation does not imply completion; mutation requests require verification before `complete`.
- No unregistered tool may execute through CodeMode.
- Retry loops are bounded.

---

### Task 1: Establish the canonical planner contract and parser tests

**Files:**
- Modify: `agent_tool_orchestration.go`
- Create/modify: `agent_tool_orchestration_contract_test.go`

**Interfaces:**
- `ToolChoice` remains the normalized planner envelope.
- `parseToolChoice(raw string) (ToolChoice, error)` is the single JSON parsing entry point.
- Validation errors use stable `planner_error=<code>` identifiers.

- [ ] **Step 1: Write failing tests** for empty/non-object planner output, malformed JSON, valid object extraction, invalid action, and empty CodeMode code.
- [ ] **Step 2: Run the focused tests and confirm the missing/incorrect contract behavior fails.**
- [ ] **Step 3: Implement strict parsing and semantic validation without allowing prose or invalid envelopes into execution.**
- [ ] **Step 4: Run the focused tests and confirm they pass.**
- [ ] **Step 5: Commit** with `test: lock down CodeMode planner contract` / implementation follow-up as appropriate.

---

### Task 2: Implement bounded planner repair

**Files:**
- Modify: `agent_tool_orchestration.go`
- Test: `agent_tool_orchestration_contract_test.go`

**Interfaces:**
- Planner repair is controlled by `defaultPlannerRepairAttempts = 2`.
- Repair prompts must demand exactly one JSON object and must not execute a partially parsed response.

- [ ] **Step 1: Add a failing test covering invalid JSON followed by a valid repaired response.**
- [ ] **Step 2: Add a failing test proving a third invalid response terminates with `invalid_json` rather than looping forever.**
- [ ] **Step 3: Implement the repair loop at the planner boundary.**
- [ ] **Step 4: Run the focused tests and confirm bounded repair behavior.**
- [ ] **Step 5: Commit** the bounded repair behavior.

---

### Task 3: Enforce canonical registry and CodeMode-only execution

**Files:**
- Modify: `agent_tool_orchestration.go`
- Test: `agent_tool_orchestration_contract_test.go`

**Interfaces:**
- `toolSpecExists`/canonical tool lookup remains the registry boundary.
- `validatePlannedTool(...)` rejects unknown tools before execution.
- CodeMode source may only invoke tools that exist in the canonical registry.

- [ ] **Step 1: Add failing tests for an unknown direct tool and an unknown CodeMode tool.**
- [ ] **Step 2: Run focused tests and confirm both are rejected.**
- [ ] **Step 3: Implement/strengthen registry validation and CodeMode call validation without introducing tool-name aliases.**
- [ ] **Step 4: Run focused tests and confirm both paths reject unknown tools.**
- [ ] **Step 5: Commit** canonical registry enforcement.

---

### Task 4: Make mutation gating deterministic

**Files:**
- Modify: `agent_tool_orchestration.go`
- Test: `agent_tool_orchestration_contract_test.go`

**Interfaces:**
- `orchestrationState` tracks `requiresMutation`, `inspected`, `mutated`, and `verified`.
- `validatePlannedTool(...)` is the mutation gate.
- Mutation capability comes from canonical tool metadata, never from planner text.

- [ ] **Step 1: Add failing tests for read-after-inspection, write-after-inspection, and CodeMode read-after-inspection.**
- [ ] **Step 2: Run tests and confirm the read paths are rejected for mutation-required requests.**
- [ ] **Step 3: Implement the deterministic mutation-only transition once inspection is complete.**
- [ ] **Step 4: Run focused tests and confirm mutation-capable actions are accepted.**
- [ ] **Step 5: Commit** the mutation gate.

---

### Task 5: Enforce verification-aware completion

**Files:**
- Modify: `agent_tool_orchestration.go`
- Test: `agent_tool_orchestration_contract_test.go`

**Interfaces:**
- `complete` is accepted only when request-specific completion criteria are satisfied.
- Mutation-required tasks require `mutated == true` and `verified == true` before completion.

- [ ] **Step 1: Add failing tests for premature `complete` before mutation and before verification.**
- [ ] **Step 2: Run tests and confirm premature completion is rejected.**
- [ ] **Step 3: Implement completion validation and post-mutation verification transition.**
- [ ] **Step 4: Run focused tests and confirm only verified mutation can complete.**
- [ ] **Step 5: Commit** verification-aware completion.

---

### Task 6: Add regression coverage for orchestration loops and step limits

**Files:**
- Modify: `agent_tool_orchestration_contract_test.go`
- Modify: `agent_tool_orchestration.go` only if required by tests

**Interfaces:**
- `configuredToolLoopMaxSteps()` remains the global bounded execution limit.
- Duplicate read-only planning after inspection must not create an unbounded loop.

- [ ] **Step 1: Add failing regression tests for `mutation_required_next`, duplicate read calls, and max-step termination.**
- [ ] **Step 2: Run focused tests and verify they expose the regression.**
- [ ] **Step 3: Implement only the minimum state/loop change required.**
- [ ] **Step 4: Run the complete Go test suite.**
- [ ] **Step 5: Commit** regression coverage and loop hardening.

---

### Task 7: Final verification and review

**Files:**
- Modify only if a verification failure requires a targeted fix.

- [ ] **Step 1: Run `gofmt` on modified Go files.**
- [ ] **Step 2: Run `go test ./...`.**
- [ ] **Step 3: Run `go vet ./...`.**
- [ ] **Step 4: Review the complete diff against the CodeMode orchestration specification.**
- [ ] **Step 5: Confirm GitHub Actions for the branch are green before claiming completion.**
- [ ] **Step 6: Request final code review before merge.**

## Verification Matrix

| Requirement | Verification |
|---|---|
| Strict JSON | parser contract tests |
| Invalid JSON repair | bounded repair tests |
| Unknown tool rejection | registry tests |
| CodeMode-only execution | planner/tool contract tests |
| Mutation gate | state transition tests |
| No read loop | regression test |
| Mutation before completion | completion tests |
| Verification before completion | verification tests |
| Bounded execution | max-step test |
| Repository health | `go test ./...`, `go vet ./...`, CI |
