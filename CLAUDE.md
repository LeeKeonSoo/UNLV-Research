# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with
project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial
tasks, use judgment.

## 1. Think Before Coding

**Do not assume or hide confusion. Surface tradeoffs.**

Before implementing:

- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them; do not pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop, name it, and ask.

## 2. Simplicity First

**Use the minimum code that solves the problem. Add nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No unrequested flexibility or configurability.
- No error handling for impossible scenarios.
- If 200 lines could be 50, rewrite them.

Ask whether a senior engineer would consider the change overcomplicated. If so,
simplify it.

## 3. Surgical Changes

**Touch only what is required. Clean up only the mess created by the change.**

When editing existing code:

- Do not improve adjacent code, comments, or formatting without a reason.
- Do not refactor unrelated code.
- Match the existing style.
- Mention unrelated dead code; do not delete it without authorization.

Remove imports, variables, and functions made obsolete by the current change.
Do not remove pre-existing dead code unless asked. Every changed line should
trace to the user's request.

## 4. Goal-Driven Execution

**Define success criteria and verify them.**

Turn tasks into verifiable goals:

- "Add validation" -> write tests for invalid inputs, then make them pass.
- "Fix the bug" -> write a test that reproduces it, then make it pass.
- "Refactor X" -> ensure tests pass before and after.

For multi-step tasks, state a brief plan:

```text
1. [Step] -> verify: [check]
2. [Step] -> verify: [check]
3. [Step] -> verify: [check]
```

These guidelines are working when diffs are smaller, unnecessary rewrites are
rarer, and ambiguity is resolved before implementation.
