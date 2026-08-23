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
6. For a mutation request, actually modify README.md using `filesystem.patch` or `filesystem.write` from inside CodeMode. Do not stop after reading or planning.
7. Preserve accurate technical information and remove stale, duplicated, or misleading documentation.
8. Verify the modified README.md after the mutation using CodeMode and, when practical, validate referenced commands or examples.
9. Never report completion unless the README mutation and verification actually occurred.
