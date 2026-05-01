---
title: feat: Add morning portfolio attribution brief
type: feat
status: active
date: 2026-05-01
---

# feat: Add morning portfolio attribution brief

## Summary

Implement a deterministic attribution-first portfolio message in `finance-news`: notable movers are explained against market, sector/theme ETF, FX proxy, and residual context before any article evidence is shown. Seed a repo-owned global benchmark map and replace the current article-link portfolio output with compact German/English attribution rows.

---

## Problem Frame

The morning WhatsApp portfolio section currently behaves like a ticker-news digest: it ranks movers, prints article bullets, and can degrade into generic fallback text. The useful product shape is portfolio-manager attribution: "what moved, what benchmark explains it, what residual remains, and whether a credible catalyst exists."

---

## Requirements

- R1. The morning portfolio output must be an attribution brief, not an article digest.
- R2. Movers must rank by portfolio relevance and absolute move, not article count.
- R3. Each included mover must show move, price, and a compact classification: market-driven, sector/theme-driven, FX-driven, idiosyncratic, mixed, or unexplained.
- R4. Absence of credible evidence must render as "no confirmed catalyst" or localized equivalent, never as filler article text.
- R5. Generic "new message/article for portfolio position" bullets must not appear.
- R6. Routine source appendices and long source link dumps must be removed from portfolio messages.
- R7. Article evidence must pass source quality and catalyst gates before inclusion.
- R8. Yahoo-style aggregators, listicles, recycled price-action recaps, and generic opinion articles must not create catalyst bullets.
- R9. Catalyst evidence must explain why the move matters now.
- R10. Catalyst bullets must include a confidence or uncertainty signal.
- R11. Weak, conflicted, or missing evidence must be stated plainly.
- R12. Small and mid-cap positions must work with quantitative attribution alone.
- R13. Each notable mover must compare against broad market, sector/theme, and applicable currency context when data exists.
- R14. Attribution must expose residual/idiosyncratic movement.
- R15. Sector/theme benchmarks must come from a repo-owned global taxonomy map.
- R16. The taxonomy must support broad regions, currencies, sectors, industries, and themes such as technology, semiconductors, AI infrastructure, industrials, financials, Japan, and China.
- R17. Ticker and portfolio metadata must be able to override generic category mapping.
- R18. Missing or ambiguous benchmark mappings must be visible as review-worthy uncertainty, not hidden precision.
- R19. WhatsApp copy must stay compact.
- R20. Output must avoid investment advice, price targets, unsupported sentiment, and recommendations.

**Origin actors:** Portfolio owner, morning scheduler, attribution engine, evidence gate, taxonomy maintainer.
**Origin flows:** Morning attribution brief, credible catalyst inclusion, benchmark taxonomy maintenance.
**Origin acceptance examples:** Japanese industrial sector move without evidence; semiconductor residual move with credible catalyst; Yahoo/listicle suppression; small-cap residual with unknown catalyst; ticker-specific benchmark override.

---

## Scope Boundaries

- Do not add buy/sell/hold advice, price targets, fair value, or trade recommendations.
- Do not require credible article evidence for every mover.
- Do not build a perfect taxonomy on day one; seed a maintainable map and make unknowns visible.
- Do not preserve generic fallback article bullets for compatibility.
- Do not add real-time newswire integrations.
- Do not rewrite the macro market briefing.
- Do not summarize every source. Sources are evidence only.

### Deferred to Follow-Up Work

- Premium-source authentication and full-text article extraction: separate research/source-quality track.
- Moving taxonomy ownership into `equity-research`: later shared-data consolidation once the finance-news behavior proves useful.
- Live FX decomposition with base-currency P&L weighting: later iteration if quote-level FX proxies are insufficient.

---

## Context & Research

### Relevant Code and Patterns

- `scripts/summarize.py` currently builds the portfolio message in `build_portfolio_message()`, hardcodes `$` prices, translates all article titles, appends source references, and ranks by raw daily change.
- `scripts/summarize.py` already has watchpoint data classes and helper tests for mover context, headline matching, sector clusters, and localized watchpoint formatting. The new attribution code should follow this deterministic helper style rather than adding LLM summarization.
- `scripts/fetch_news.py` owns quote fetching via `fetch_market_data()` and ticker news via Yahoo RSS. The attribution path should reuse quote fetching but treat Yahoo ticker articles as low-quality evidence by default.
- `config/config.json` already stores market index tickers, source weights, and localized labels. It is the natural home for brief copy labels; a separate benchmark map keeps ETF/taxonomy data out of translation config.
- `tests/test_summarize.py` already covers portfolio message formatting and watchpoints. Add focused unit coverage there, and add a small dedicated test file if attribution helpers outgrow the current module.
- Existing LLQD taxonomy data in the wider workspace has broad sector, specific sector, and investment-theme concepts but no ETF benchmark map. Use it as conceptual input only; keep this repo self-contained for the first implementation.

### Institutional Learnings

- Portfolio state should have one clear owner. Keep this change in `finance-news` because the current WhatsApp portfolio message is generated here; do not introduce a second scheduler or duplicate portfolio source.
- Shared LLQD configuration remains canonical in the portfolio repo, but this first slice can work from the CSV already mounted into `finance-news`.

### External References

- State Street's Select Sector SPDR pages provide a stable U.S. sector ETF family: XLC, XLY, XLP, XLE, XLF, XLV, XLI, XLB, XLRE, XLK, and XLU.
- iShares official fund pages provide broad global technology and semiconductor ETFs such as IXN and SOXX.
- Japan and regional mappings should start conservative; if local sector ETFs are hard to fetch reliably, use broad market ETFs/indexes plus explicit mapping uncertainty rather than false precision.

---

## Key Technical Decisions

- Add a small attribution module instead of growing `summarize.py` further: this keeps new benchmark, evidence, and classification logic testable.
- Seed benchmarks in a repo-owned JSON config: this satisfies the global taxonomy requirement without coupling this repo to the portfolio repo's internal CSV layout.
- Prefer deterministic evidence gates over article summarization: low-quality feeds are the problem, so the first line of defense should be source/catalyst filtering.
- Use quote `change_percent` from existing quote fetchers for both stocks and benchmarks: the first pass can compute relative/residual movement without storing time series.
- Keep output formatter deterministic and localized through config labels: no LLM should be needed to produce the morning portfolio rows.

---

## Open Questions

### Resolved During Planning

- Should the first implementation be full article analysis or attribution-first without article dependence? Resolved: attribution-first, with optional gated evidence only.
- Should the benchmark map live only in LLQD/equity-research? Resolved for this PR: no. Seed a repo-owned map in `finance-news`; later consolidation can move it once behavior is proven.
- Should Yahoo ticker RSS remain a catalyst source? Resolved: not by default. It can supply raw candidates, but Yahoo-style domains are blocked unless a later allowlist decision changes that.

### Deferred to Implementation

- Exact threshold constants for "mixed" versus "idiosyncratic": start conservative in code and tune against tests.
- Exact label phrasing in German: implement terse config labels and adjust after snapshot output review.
- Which benchmark mappings are fetchable in the runtime container: seed conservative defaults and treat missing quote data as explicit uncertainty.

---

## Implementation Units

- U1. **Seed benchmark and evidence configuration**
  - **Goal:** Add repo-owned data for benchmark attribution and source-quality gates.
  - **Requirements:** R7, R8, R13, R15, R16, R17, R18.
  - **Dependencies:** None.
  - **Files:**
    - Create: `config/portfolio_benchmarks.json`
    - Modify: `config/config.json`
    - Test: `tests/test_portfolio_attribution.py`
  - **Approach:** Add a compact JSON map with broad market defaults, region defaults, category/sector mappings, theme mappings, ticker overrides, source allowlist/blocklist, and uncertainty defaults. Start with highly liquid U.S. sector ETFs, broad global/region ETFs, semiconductor/technology theme ETFs, and explicit unknown handling.
  - **Patterns to follow:** Existing `config/config.json` structure for readable operator-owned config.
  - **Test scenarios:**
    - Loading config returns broad market, sector/theme, and source-quality defaults.
    - A ticker override wins over a generic category mapping.
    - Missing category mapping returns an explicit unknown mapping instead of a fabricated benchmark.
    - Blocked domains include Yahoo-style aggregator/listicle sources by default.
  - **Verification:** The map can be loaded without network access and exposes uncertainty metadata for unmapped tickers.

- U2. **Add attribution model helpers**
  - **Goal:** Compute benchmark-relative explanations for portfolio movers.
  - **Requirements:** R2, R3, R11, R12, R13, R14, R18.
  - **Dependencies:** U1.
  - **Files:**
    - Create: `scripts/portfolio_attribution.py`
    - Test: `tests/test_portfolio_attribution.py`
  - **Approach:** Define small data classes for benchmark mapping, quote change, evidence, and attribution result. Given portfolio rows, mover quotes, benchmark quotes, and optional articles, classify moves as market-driven, sector/theme-driven, FX-driven, idiosyncratic, mixed, or unexplained. Compute residual as ticker change minus the best available sector/theme benchmark, with market fallback when no specific benchmark exists.
  - **Execution note:** Implement behavior test-first because threshold mistakes will directly affect user-facing classification.
  - **Patterns to follow:** `MoverContext`, `SectorCluster`, and `WatchpointsData` in `scripts/summarize.py`.
  - **Test scenarios:**
    - Covers AE1. A Japan industrial mover close to its benchmark renders sector/theme-driven with no confirmed catalyst.
    - Covers AE2. A semiconductor mover far away from its benchmark renders idiosyncratic and preserves credible catalyst evidence.
    - Covers AE4. A small-cap or unmapped ticker renders quantitative attribution with mapping uncertainty.
    - Market benchmark explains a move when sector data is unavailable.
    - Residual magnitude sorts ahead of article volume when selecting notable rows.
  - **Verification:** Attribution helpers are pure and unit-testable with injected quote/article fixtures.

- U3. **Gate article evidence**
  - **Goal:** Only include source evidence when it is credible and catalyst-bearing.
  - **Requirements:** R4, R5, R7, R8, R9, R10, R11, R12.
  - **Dependencies:** U1, U2.
  - **Files:**
    - Create or modify: `scripts/portfolio_attribution.py`
    - Test: `tests/test_portfolio_attribution.py`
  - **Approach:** Implement deterministic source gate helpers using source/domain blocklists, allowlists, and catalyst keywords for earnings, guidance, M&A, regulation, product, capex, supply chain, analyst action, and demand/pricing events. Block generic price-action/listicle language. Return a confidence label for included evidence and "no confirmed catalyst" for everything else.
  - **Patterns to follow:** `is_generic_headline()` in `scripts/fetch_news.py` and headline normalization helpers in `scripts/summarize.py`.
  - **Test scenarios:**
    - Covers AE3. Yahoo/listicle/price-action candidates are blocked and do not create bullets.
    - Reuters/WSJ/FT/Bloomberg/CNBC-style candidates with guidance or earnings catalysts pass with confidence.
    - A credible source without a concrete catalyst is still suppressed.
    - Blocked evidence never appears in formatted source lists.
  - **Verification:** Evidence output is deterministic and does not require live feeds or LLM calls.

- U4. **Replace portfolio message rendering**
  - **Goal:** Make `build_portfolio_message()` render attribution rows instead of article digest rows.
  - **Requirements:** R1, R2, R3, R4, R5, R6, R10, R11, R19, R20.
  - **Dependencies:** U1, U2, U3.
  - **Files:**
    - Modify: `scripts/summarize.py`
    - Modify: `config/config.json`
    - Test: `tests/test_summarize.py`
  - **Approach:** Load benchmark config, fetch benchmark quotes for unique mapping tickers, call attribution helpers, and format compact localized rows. Replace source appendix behavior with inline evidence only when evidence is included. Fix price formatting so non-USD tickers do not always print `$`.
  - **Patterns to follow:** Existing deterministic `build_briefing_summary()` and localized label lookups.
  - **Test scenarios:**
    - Existing portfolio message test updates from article links/source appendix to attribution text.
    - German output includes attribution labels and "kein bestätigter Auslöser" for weak/no evidence.
    - Source appendix is omitted when no gated evidence exists.
    - A credible catalyst appears inline with confidence and one source reference only.
    - Japanese ticker price does not render as `$27830.00`.
  - **Verification:** `build_portfolio_message()` can render from fixture data without network when benchmark quote fetcher is injected or monkeypatched.

- U5. **Wire benchmark fetching into briefing generation**
  - **Goal:** Provide benchmark quote changes to the portfolio formatter during normal briefing runs.
  - **Requirements:** R1, R3, R13, R14, R19.
  - **Dependencies:** U1, U2, U4.
  - **Files:**
    - Modify: `scripts/summarize.py`
    - Test: `tests/test_summarize.py`
  - **Approach:** Keep `get_portfolio_news()` as the portfolio stock quote source, but compute the benchmark ticker set from portfolio metadata and benchmark config, fetch those quotes via existing `fetch_market_data()`, and pass them into the formatter. If fetching fails or times out, render with explicit benchmark uncertainty rather than failing the briefing.
  - **Patterns to follow:** Existing fail-soft handling around `get_portfolio_news()` and `get_portfolio_movers()`.
  - **Test scenarios:**
    - Benchmark fetch failure still renders portfolio rows with uncertainty.
    - Unique benchmark ticker set is deduplicated before fetch.
    - Fast mode remains bounded and does not fetch unrelated benchmark families.
  - **Verification:** The deterministic briefing path remains fail-open for portfolio attribution.

- U6. **Add regression coverage and smoke checks**
  - **Goal:** Lock in the new product contract and prevent fallback/link-dump regression.
  - **Requirements:** All requirements, especially R4-R8 and R19-R20.
  - **Dependencies:** U1-U5.
  - **Files:**
    - Modify: `tests/test_summarize.py`
    - Modify: `tests/test_portfolio_attribution.py`
    - Optional: `README.md`
  - **Approach:** Add focused unit tests for the examples above and one end-to-end deterministic JSON generation smoke with mocked market/portfolio/benchmark data. Update README only if existing user-facing docs describe the old source-list portfolio digest.
  - **Test scenarios:**
    - End-to-end morning JSON contains attribution-first portfolio text and no `## Sources` block for weak evidence.
    - No test expects generic fallback article bullets.
    - No output contains buy/sell/hold advice or price targets.
  - **Verification:** Targeted tests pass, full summarize test file passes, and lint/diff checks are clean.

---

## Verification Plan

- `python3 -m pytest -q tests/test_portfolio_attribution.py tests/test_summarize.py`
- `python3 -m pytest -q`
- `git diff --check`

---

## Risk Analysis & Mitigation

- **Benchmark false precision:** A bad sector ETF is worse than no ETF. Mitigate by explicit unknown/mapping-uncertain output and conservative seed mappings.
- **Runtime cost:** Fetching many benchmark tickers can slow the morning cron. Mitigate through deduplication, broad defaults, and fail-soft behavior.
- **Source overblocking:** Blocking Yahoo-style feeds may remove some useful articles. Accept this in v1 because the current failure mode is noisy evidence; credible source expansion can be added later.
- **Cross-repo taxonomy drift:** This repo's seeded map may diverge from portfolio taxonomy. Keep the map small and operator-owned for now; consolidate later only after the attribution product shape proves useful.
