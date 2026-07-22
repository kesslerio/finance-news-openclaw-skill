# Theory: Deterministic Kimi Briefing Migration

> **SUPERSEDED (2026-07-22).** This document is retained only as historical rationale
> for the earlier direct-writer migration. The Kimi/Ollama-Cloud writer described below
> has since been replaced by the local tailnet routes (kalliope Qwen primary; gx10
> DeepSeek-V4-Flash fallback; deep research on DS4-high) for the Ollama Cloud pause.
> The "call the writer directly from the skill" seam still holds; only the model routes
> changed. See `README.md` for the current routing.

## Problem

The briefing path is split across two layers that should have different responsibilities. The outer cron and workflow layer should be boring: invoke the container, capture JSON, validate structure, and send it. The writing itself belongs inside `finance-news`, where the prompt, formatting rules, disclaimers, and fallbacks are already defined. Right now the repo still carries briefing-generation paths that depend on MiniMax and `openclaw agent`, which makes the core writing path depend on a second orchestration layer and on model routing outside the skill. That is the wrong failure boundary. If Kimi is the desired writer, the skill should call Kimi directly.

## Operating Theory

The current wrapper path already calls `python3 scripts/briefing.py` inside Docker. That is the correct seam to preserve. The real migration is therefore not a cron rewrite; it is an internal change to what `briefing.py` and `summarize.py` do for briefing generation. The deterministic part should be the shell behavior and the request path, not the prose itself. In practice that means a direct HTTP call from `scripts/summarize.py` to the Kimi Coding API using `KIMI_API_KEY`, with a small, explicit model/config surface and hard failure behavior when credentials or the API are unavailable.

The repo already proves that this style works: `summarize.py` has a direct Anthropic-compatible MiniMax path using `urllib`. The cleanest migration is to replace the active briefing summary path with the same pattern for Kimi rather than introducing SDKs, background agents, or repo-wide abstractions.

## Strategy

Keep scope tight around the briefing-critical write path. Leave headline fetching, ranking, and non-briefing features alone unless they directly block the migration. Make the briefing default to the Kimi-backed summary path so the workflow remains dumb and unchanged at the call site. Update `summarize.py` so the active summary model order is Kimi-first and no longer routes briefing generation through MiniMax or `openclaw agent`. Preserve the JSON schema and message formatting, including disclaimer handling, and hard fail when Kimi credentials or the API are unavailable.

## Key Discoveries

The live workflow must validate that the writer is `kimi` and that the localized briefing structure is present. Another important discovery is that this machine already has a working `KIMI_API_KEY`, and the local OpenClaw provider config resolves Kimi Coding as an Anthropic-compatible endpoint at `https://api.kimi.com/coding/`, which makes a direct `.../v1/messages` call the smallest reliable implementation.

## Open Questions

The narrow migration leaves headline translation and some auxiliary features on their existing providers. That is intentional for scope control, but it means the repo will still contain MiniMax references outside the core briefing-writing path. If later work expands to all LLM usage in `finance-news`, those paths should be revisited separately. Product requirement is now explicit: no fallback behavior on missing or broken Kimi for the active summary path.
