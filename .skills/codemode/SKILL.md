---
name: codemode
description: Use CodeMode for explicit multi-step Go execution when the task benefits from verified tool orchestration.
---

Use this skill when the user explicitly asks to run Go code, use CodeMode, or perform a compact multi-step tool workflow.

- Treat the canonical UTCP tool registry as authoritative.
- Use only exact registered tool names.
- Never invent a tool name or argument schema.
- Keep mutations explicit and verifiable.
- After a mutation, inspect the result when practical.
- Do not report success unless CodeMode execution actually returned successfully.
