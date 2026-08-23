---
name: refactor-readme
description: Refactor README.md as a real repository mutation task: inspect the repository and README first, improve structure and technical accuracy, then write or patch README.md and verify the result.
---

For README refactoring requests:

1. Apply this skill before inspecting or modifying README.md.
2. All repository inspection and mutation MUST happen inside `codemode.run_code`.
3. Inspect the repository as needed to verify commands, architecture, APIs, examples, and file paths used by the README.
4. Read the current README.md before making changes.
5. The planner MUST NOT call `filesystem.*`, `shell.*`, `git.*`, or any other canonical UTCP tool directly; invoke them from CodeMode with exact registered tool names.
6. CodeMode source MUST call the runtime API as `codemode.CallTool("exact.tool.name", args)` or `codemode.CallToolStream("exact.tool.name", args)`. Never emit unqualified `CallTool(...)` or `CallToolStream(...)`; those identifiers do not exist in the CodeMode execution scope.
7. CodeMode source MUST be a statement-only snippet. NEVER include `package` declarations, `import` declarations, or import blocks. Do not define a separate `main` function. The CodeMode runtime supplies the execution wrapper and `codemode` receiver.
8. Use only the runtime APIs and canonical UTCP tool names already exposed by CodeMode. Do not invent helper packages or imports.
9. For a mutation request, after the relevant inspection has completed, the NEXT CodeMode program MUST contain a real mutation. Do not perform another read-only CodeMode step once the README has been inspected.
10. For README mutation, use one of these concrete patterns:
   - `updated, err := codemode.CallTool("filesystem.patch", map[string]any{"path": "README.md", ...})`
   - `updated, err := codemode.CallTool("filesystem.write", map[string]any{"path": "README.md", ...})`
   Use the exact registered input schema for the selected filesystem tool. Keep `updated` and `err` in the same lexical scope as all dependent values.
11. When practical, combine inspection and mutation in the SAME CodeMode program. If the current README contents are already available in the observation, do not re-read it; mutate it directly in the next CodeMode program.
12. After mutation, use CodeMode again to verify the modified README.md. Verification is not a substitute for the required mutation.
13. Preserve accurate technical information and remove stale, duplicated, or misleading documentation.
14. Never report completion unless the README mutation and verification actually occurred.
