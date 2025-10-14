# ADK Alignment Migration Guide

The Google Agent Development Kit (ADK) documentation was not accessible from the execution
environment. The refactor below follows the published high-level concepts (coordinator agents,
tool registries, sub-agent directories, and deterministic session management) based on prior
knowledge of ADK terminology. Any divergences should be reconciled once the official guidance
can be reviewed.

## What Changed

1. **Tool and Sub-agent registries**
   - `pkg/agent` now exposes `ToolCatalog` and `SubAgentDirectory` interfaces with default
     in-memory implementations.
   - The coordinator agent consumes these abstractions instead of managing raw maps.
   - Custom registries can be injected via `agent.Options` to integrate with external discovery
     services.

2. **Runtime session manager**
   - `pkg/runtime` introduced a dedicated session manager that keeps the active session set
     threadsafe, deterministic, and ADK-aligned.
   - Runtime helpers (`NewSession`, `GetSession`, `ActiveSessions`, `RemoveSession`) proxy through
     the manager.

3. **Public accessors**
   - `agent.Agent` now publishes `Tools()`, `ToolSpecs()`, and `SubAgents()` accessors so the
     runtime can reflect the registered capabilities without touching internals.

4. **Documentation updates**
   - The README highlights the new abstractions and how they map to ADK nomenclature.

## Migration Steps

Existing callers can continue to pass `Options{Tools: ..., SubAgents: ...}`. For a gradual
migration:

1. Swap any direct access to `agent.Agent` internals (e.g. `agent.tools`) with the new accessor
   methods.
2. When you need custom discovery logic, implement the `ToolCatalog`/`SubAgentDirectory`
   interfaces and supply them through `agent.Options` alongside (or instead of) the legacy slices.
3. If you relied on mutating `runtime.Tools()` slices, switch to calling `rt.Agent().Tools()` and
   treat the result as read-only; every call returns a defensive copy.
4. For session lifecycle integrations, prefer `runtime.ActiveSessions()` over direct map access
   and use `runtime.RemoveSession()` to prune idle sessions.

> **TODO:** Once the official ADK specification is available, audit the naming and lifecycle hooks
> introduced here and adjust any mismatches.
