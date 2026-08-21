---
name: code-review
description: Review Go code for correctness, concurrency, API usage, tests, and concrete regressions before suggesting changes.
---

For code-review requests:

1. Inspect the relevant files and tests first.
2. Separate verified findings from hypotheses.
3. Prioritize correctness, races, error handling, API contracts, and regression risk.
4. If the user asks for fixes, perform a real mutation rather than returning only a plan.
5. Validate changed code with the narrowest useful test or build command.
6. Never claim tests passed unless they actually ran and returned success.
