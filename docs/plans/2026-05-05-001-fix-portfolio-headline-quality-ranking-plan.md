---
title: "fix: Improve portfolio headline quality ranking"
type: fix
status: active
date: 2026-05-05
---

# fix: Improve portfolio headline quality ranking

## Summary

Reduce low-value portfolio article picks (for example, repetitive "X vs Y value stock" templates) by adding deterministic quality scoring and configurable pattern penalties while keeping Yahoo enabled as a source.

---

## Problem Frame

Morning portfolio output can surface weak templated articles even when they add little catalyst value. Current portfolio evidence selection stops at the first passing item instead of ranking all candidates by quality.

---

## Requirements

- R1. Keep Yahoo available as a source; do not globally block the domain.
- R2. Demote templated low-information headlines (for example, "which stock is better value", "X vs Y") deterministically.
- R3. Select the best available per-ticker article using ranked quality signals instead of first-match behavior.
- R4. Keep behavior configurable via JSON, not hard-coded-only thresholds.
- R5. Add regression tests proving low-value templates are filtered/demoted and higher-value catalyst stories win.

---

## Scope Boundaries

- Do not redesign macro headline ranking architecture.
- Do not change cron scheduling or delivery transport behavior in this plan.
- Do not remove Yahoo from feed configuration.

---

## Context & Research

### Relevant Code and Patterns

- `scripts/portfolio_attribution.py` currently uses `_is_generic_article`, `_has_catalyst`, and `evidence_for_articles(...)` with first-match selection.
- `scripts/ranking.py` already has deterministic score composition and is suitable as a reference style for weighted scoring.
- `config/portfolio_benchmarks.json` controls source allow/block policy and should hold quality configuration.

### Institutional Learnings

- Existing tests already codify quality guardrails for portfolio evidence and should be extended in place.

### External References

- None required; repo-local behavior and fixtures are sufficient.

---

## Key Technical Decisions

- Add a two-stage deterministic portfolio evidence selector:
  1. Eligibility gate (title/source/domain/basic catalyst checks)
  2. Candidate scoring/ranking (quality score) and choose highest score
- Move low-value title-template controls into `config/portfolio_benchmarks.json` so tuning does not require code edits.
- Keep Yahoo allowed, but apply the same template penalties and quality threshold rules as other sources.

---

## Open Questions

### Resolved During Planning

- Should Yahoo be removed to avoid weak content? No; keep Yahoo and demote low-value templates via scoring and thresholds.

### Deferred to Implementation

- Exact penalty weights and minimum score cutoff values after test calibration.

---

## Implementation Units

- U1. **Add Configurable Portfolio Evidence Quality Signals**

**Goal:** Introduce config keys for low-value template patterns, source/domain bonuses, and minimum evidence score.

**Requirements:** R1, R2, R4

**Dependencies:** None

**Files:**
- Modify: `config/portfolio_benchmarks.json`
- Test: `tests/test_portfolio_attribution.py`

**Approach:**
- Add a `quality_scoring` subsection under `source_quality` with:
  - template penalty patterns (`vs`, `better value stock`, `outpacing peers`, etc.)
  - optional source/domain score adjustments
  - minimum score threshold for acceptance
- Keep `finance.yahoo.com` out of blocked domains.

**Patterns to follow:**
- Existing config-driven quality controls in `source_quality`.

**Test scenarios:**
- Happy path: Yahoo catalyst article with strong signal and no low-value template is accepted by config policy.
- Edge case: Unknown source with strong catalyst title remains eligible but must pass threshold.
- Error path: Missing optional scoring keys falls back to safe defaults.

**Verification:**
- Config loads without schema/runtime errors and tests pass with new keys.

---

- U2. **Implement Ranked Evidence Selection for Portfolio Articles**

**Goal:** Replace first-match evidence selection with deterministic best-candidate ranking.

**Requirements:** R2, R3, R5

**Dependencies:** U1

**Files:**
- Modify: `scripts/portfolio_attribution.py`
- Test: `tests/test_portfolio_attribution.py`

**Approach:**
- Add candidate scoring helper(s) for portfolio article quality:
  - positive signals: catalyst keywords, allowed source/domain, recency
  - negative signals: low-value template regex/pattern matches
- Evaluate all eligible articles, rank by score, and return highest-scoring candidate above threshold.
- Preserve fail-safe behavior: if no candidate passes, return `None` evidence.

**Patterns to follow:**
- Deterministic scoring composition style in `scripts/ranking.py`.

**Test scenarios:**
- Happy path: Higher-signal Reuters/Bloomberg article beats weaker candidate.
- Happy path: High-quality Yahoo article can win when it outranks alternatives.
- Edge case: Two Yahoo template titles with "X vs Y value stock" are rejected or demoted below threshold.
- Integration: Mixed article list returns exactly one `Evidence` object from the highest-scoring valid candidate.
- Error path: Articles with missing link/source/title do not crash and are skipped deterministically.

**Verification:**
- Portfolio attribution tests pass and selected evidence matches expected top candidate.

---

- U3. **Validate End-to-End Morning Portfolio Output Quality**

**Goal:** Confirm live JSON output no longer elevates low-value template stories after rebuild.

**Requirements:** R1, R2, R5

**Dependencies:** U2

**Files:**
- Modify: `README.md`

**Approach:**
- Rebuild `finance-news-briefing` image after changes.
- Run morning JSON generation with portfolio CSV mount and inspect `portfolio_message`.
- Capture source distribution and representative selected titles for sanity-check notes.
- Document image refresh expectation to prevent stale-image false negatives.

**Patterns to follow:**
- Existing docker-run based smoke workflow in cron wrappers.

**Test scenarios:**
- Integration: Morning JSON run completes and emits portfolio section without selected low-value template headlines.
- Integration: Yahoo remains present in candidate universe; only low-value templates are suppressed.

**Verification:**
- Smoke output and tests show improved headline mix with Yahoo still available.

---

## System-Wide Impact

- **Interaction graph:** Portfolio article selection impacts morning portfolio content quality and downstream localization.
- **Error propagation:** No change to fail-open behavior; missing evidence continues to produce non-crashing output.
- **State lifecycle risks:** None; logic is stateless and per-run deterministic.
- **API surface parity:** JSON output schema remains unchanged.
- **Integration coverage:** Requires both unit tests and docker-backed smoke checks to avoid stale-image mismatches.

---

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| Over-penalizing valid comparative analysis headlines | Keep penalties configurable and test with mixed fixtures |
| Stale Docker image hides code changes in smoke tests | Force image rebuild before validation |
| Threshold too strict causing no evidence coverage | Add tests for threshold calibration and adjust defaults conservatively |

---

## Documentation / Operational Notes

- Add a short note in `README.md` that morning smoke validation should rebuild image after ranking/evidence changes.

---

## Sources & References

- Related code: `scripts/portfolio_attribution.py`
- Related code: `scripts/ranking.py`
- Related tests: `tests/test_portfolio_attribution.py`
- Related tests: `tests/test_ranking.py`
