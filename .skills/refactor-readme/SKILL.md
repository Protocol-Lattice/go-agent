---
name: refactor-readme
description: Refactor README.md as a real repository mutation task: inspect the repository and README first, improve structure and technical accuracy, then write or patch README.md and verify the result.
---

For README refactoring requests:

1. Apply this skill before inspecting or modifying README.md.
2. Inspect the repository as needed to verify commands, architecture, APIs, examples, and file paths used by the README.
3. Read the current README.md before making changes.
4. Use CodeMode with the canonical filesystem tools when the task requires repository inspection or mutation.
5. For a mutation request, actually modify README.md using filesystem.patch or filesystem.write. Do not stop after reading or planning.
6. Preserve accurate technical information and remove stale, duplicated, or misleading documentation.
7. Verify the modified README.md after the mutation and, when practical, validate referenced commands or examples.
8. Never report completion unless the README mutation and verification actually occurred.
