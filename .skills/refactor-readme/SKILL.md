---
name: refactor-readme
description: Refactor README.md as a real repository mutation task: inspect the repository and README first, improve structure and technical accuracy, then write or patch README.md and verify the result.
---

For README refactoring requests:

1. Apply this skill before inspecting or modifying README.md.
2. All repository inspection and mutation MUST happen inside `codemode.run_code`.
3. Use CodeMode as the only execution boundary for repository work. Internal `filesystem.*`, `shell.*`, and other UTCP calls shown in the workflow are expected only when invoked from CodeMode.
4. The FIRST CodeMode program MUST inspect the current README.md and any relevant repository context, then perform the requested README mutation in that SAME program when the required information is available. Do not spend separate CodeMode steps on inspection-only work when you already have enough information to edit the file.
5. The planner MUST NOT call `filesystem.*`, `shell.*`, `git.*`, or any other canonical UTCP tool directly; invoke them from CodeMode with exact registered tool names.
6. CodeMode source MUST call the runtime API as `codemode.CallTool("exact.tool.name", args)` or `codemode.CallToolStream("exact.tool.name", args)`. Never emit unqualified `CallTool(...)` or `CallToolStream(...)`; those identifiers do not exist in the CodeMode execution scope.
7. CodeMode source MUST be a statement-only snippet. NEVER include `package` declarations, `import` declarations, import blocks, or a separate `main` function. The CodeMode runtime supplies the execution wrapper and `codemode` receiver.
8. Use only the runtime APIs and canonical UTCP tool names already exposed by CodeMode. Do not invent helper packages, imports, or helper functions.
9. CRITICAL RESULT-TYPE RULE: `codemode.CallTool(...)` and `codemode.CallToolStream(...)` return generic `any`/interface values. NEVER write type assertions such as `result.(string)`, `value.(map[string]any)`, or any other concrete assertion unless the exact runtime result type has been verified from the preceding observation. For rendering or passing a result onward, use it as `any` or convert safely with `fmt.Sprint(result)` when textual content is required.
10. HARD MUTATION-PHASE RULE: once the current execution state says `inspection_complete=true` and `mutation_complete=false`, the planner MUST NOT return a completion response. It MUST return `use_tool=true` with `tool_name="codemode.run_code"`, and the CodeMode program MUST contain a real `filesystem.patch` or `filesystem.write` call. `use_tool=false` is invalid during this phase even when the model believes the requested work is complete.
11. HARD MUTATION OUTPUT: during mutation phase, return exactly this shape and leave `final_answer` empty: `{"use_tool":true,"tool_name":"codemode.run_code","arguments":{"code":"..."},"reason":"perform the required README mutation","final_answer":""}`. Do not return prose, a plan, or a verification-only program.
12. For the first CodeMode program, prefer this sequence in one lexical scope: inspect README/context -> construct the improved README content or patch -> call `filesystem.patch` or `filesystem.write`.
13. A CodeMode program that only reads/list/searches is NOT sufficient for this skill unless the read result proves that a mutation cannot or should not be performed. For an ordinary refactor request, the first successful inspection should be followed by a real mutation in the same CodeMode program.
14. For README mutation, use one of these concrete forms with the exact registered input schema:
   - `updated, err := codemode.CallTool("filesystem.patch", map[string]any{"path": "README.md", ...})`
   - `updated, err := codemode.CallTool("filesystem.write", map[string]any{"path": "README.md", ...})`
   Keep `updated`, `err`, and all dependent values in the same lexical scope.
15. If the previous observation already contains the README contents or sufficient repository context, do NOT issue another read-only CodeMode program. Mutate from the available evidence.
16. After a successful mutation, use at most one additional CodeMode program to verify the modified README.md when practical. Verification never replaces the required mutation.
17. Preserve accurate technical information and remove stale, duplicated, or misleading documentation.
18. Never report completion unless the README mutation and verification actually occurred.
