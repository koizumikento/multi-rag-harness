---
name: "multi-rag-memory"
description: "Use when working with the multi-rag-harness MCP server to retrieve or persist durable local project memory. Do not use for general web research, one-off notes, or storing secrets and transient command noise."
---

# Multi RAG Memory

Use this skill to operate multi-rag-harness as Codex's local retrieval and memory layer. Search before relying on memory, inspect sources before treating excerpts as evidence, and store only durable context that should help future agents.

## Do Not Use For

- General web research or current external facts.
- Temporary scratch notes, raw command output, or large logs.
- Secrets, credentials, personal data, proprietary tokens, model weights, caches, databases, or build artifacts.
- Replacing normal repo inspection with `rg`, tests, or source reads when the current workspace is the source of truth.

## Workflow

1. Before non-trivial repo work, search prior context with the narrowest tool:
   - Use `decision_search` for architectural or implementation choices.
   - Use `error_search` for recurring error messages, failed commands, and known fixes.
   - Use `code_search` for previously ingested code context.
   - Use `tool_search` when choosing among stored MCP tool descriptions.
   - Use `rag_search` for general document or mixed-memory context.
2. Treat search results as leads. Call `rag_get_source` for important hits before relying on an excerpt in implementation or review.
3. Store durable outcomes only after they are useful beyond the current turn:
   - Use `memory_store_decision` for choices that affect future implementation.
   - Use `memory_store_failure` for reusable error signatures, causes, fixes, and verification.
   - Use `memory_store_tool` for stable tool descriptions, schemas, examples, constraints, and known failure modes.
   - Use `memory_store_trace` for substantial tasks where a future agent benefits from the summary, commands, files, tests, and outcome.
4. Keep stored memory compact and source-grounded. Prefer summaries, exact file paths, command names, test names, and durable IDs over raw transcripts.
5. When ingesting files with `rag_ingest_path`, scope the path deliberately and avoid generated output, caches, local databases, virtual environments, and secrets.
6. Validate important memory behavior with MCP calls when changing this workflow: list tools, store a sample record, search it back, and fetch source context when applicable.

## Output

- State which memory searches were used and whether useful context was found.
- State which memory records were stored, including record type and durable reason.
- If no memory was stored, say why the result was not durable enough.

## Guardrails

- Do not auto-store every search query, search result, command, or final answer.
- Do not store content that the user has not authorized for durable local memory.
- If retrieved memory conflicts with current repo files or explicit user instructions, prefer the current repo or user instruction and mention the conflict.
- If the multi-rag-harness MCP server is unavailable, continue with ordinary local repo inspection and report that memory retrieval or storage was skipped.
