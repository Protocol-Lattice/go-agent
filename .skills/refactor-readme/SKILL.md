---
name: refactor-readme
description: Refactor README.md using verified filesystem inspection and a real mutation. Prefer CodeMode for multi-step edits and validation.
---

When the user asks to refactor README.md:

1. Inspect README.md before proposing changes.
2. Preserve accurate existing project information unless the user asks to change it.
3. Make a real mutation; reading or planning alone is never completion.
4. Prefer `codemode.run_code` for a multi-step refactor when it can safely call canonical filesystem tools.
5. For CodeMode, use only tool names that are present in the canonical UTCP registry.
6. After writing, re-read README.md and verify the requested outcome.
7. Never claim the refactor is complete without a successful mutation and verification.
