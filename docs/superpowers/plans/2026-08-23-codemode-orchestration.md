# CodeMode Orchestration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make go-agent CodeMode orchestration deterministic around planner JSON, canonical UTCP tools, mutation gating, and verification.

**Architecture:** Keep CodeMode as the sole planner/execution surface, but put a strict planner parser/validator and bounded repair loop in front of execution. Track inspection, mutation, and verification independently; reject read-only plans after inspection for mutation requests; validate every CodeMode CallTool/CallToolStream target against the canonical UTCP registry.

**Tech Stack:** Go, go-utcp CodeMode, standard library JSON/regexp testing, GitHub Actions.

**Spec:** CodeMode Orchestration Specification for go-agent from the preceding design discussion.

## Global Constraints

- CodeMode is the only planner/execution tool exposed to the planner.
- Canonical UTCP registry is the only source of valid tool names and capabilities.
- Invalid planner JSON never reaches tool execution.
- Mutation-required requests cannot complete before mutation and verification.
- After inspection is complete, mutation-required requests cannot execute another read-only CodeMode step.
- Planner and tool retries are bounded.

### Task 1: Lock the planner contract with failing unit tests

**Files:**
- Create: `agent_tool_orchestration_contract_test.go`

**Interfaces:**
- Consumes: existing `parseToolChoice`, `validateCodeModeCode`, `validatePlannedTool`, `orchestrationState`.
- Produces: regression coverage for strict planner schema and mutation gate behavior.

- [ ] **Step 1: Write failing tests** for empty planner objects, invalid `use_tool` combinations, and completion before required mutation/verification.
- [ ] **Step 2: Run the focused Go tests** and confirm the new schema assertions fail against the current permissive parser.
- [ ] **Step 3: Implement the minimal parser/state validation** required by the tests.
- [ ] **Step 4: Run the focused tests again** and confirm they pass.
- [ ] **Step 5: Commit** the contract tests and validation changes.

### Task 2: Add bounded planner JSON repair

**Files:**
- Modify: `agent_tool_orchestration.go`
- Test: `agent_tool_orchestration_contract_test.go`

**Interfaces:**
- Consumes: `parseToolChoice`, planner observations, model generation.
- Produces: bounded repair attempts with a terminal `planner_error=invalid_json` instead of silently consuming tool-loop steps.

- [ ] **Step 1: Write failing tests** for one successful repair and repair-limit exhaustion.
- [ ] **Step 2: Run focused tests** and confirm they fail because invalid planner output is currently just appended to observations.
- [ ] **Step 3: Implement a two-attempt planner repair loop** with an explicit repair prompt and no tool execution for invalid plans.
- [ ] **Step 4: Run focused tests** and confirm the repair behavior passes.
- [ ] **Step 5: Commit** the bounded repair implementation.

### Task 3: Make orchestration state explicit and enforce verification

**Files:**
- Modify: `agent_tool_orchestration.go`
- Test: `agent_tool_orchestration_contract_test.go`

**Interfaces:**
- Consumes: `orchestrationState`, CodeMode execution results.
- Produces: separate inspection/mutation/verification state and completion checks.

- [ ] **Step 1: Write failing tests** proving mutation is not equivalent to completion and that verification is required after mutation.
- [ ] **Step 2: Run focused tests** and confirm the current `completionAllowed` behavior is insufficient.
- [ ] **Step 3: Update completion and observation transitions** so mutation requests enter verification before completion.
- [ ] **Step 4: Run focused tests** and confirm they pass.
- [ ] **Step 5: Commit** the state-machine enforcement.

### Task 4: Verify the complete repository test suite

**Files:**
- No new production files.

**Interfaces:**
- Consumes: all previous tasks.
- Produces: verified CodeMode orchestration behavior.

- [ ] **Step 1: Run `go test ./...`.**
- [ ] **Step 2: If failures are unrelated, isolate and document them; otherwise fix regressions before completion.**
- [ ] **Step 3: Run formatting/lint checks required by the repository.**
- [ ] **Step 4: Inspect the final diff for accidental scope expansion.**
