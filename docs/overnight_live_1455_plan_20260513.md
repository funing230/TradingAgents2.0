# 14:55 Live Overnight Trading Plan

Date: 2026-05-13

## Objective

Turn the current research/backtest-oriented overnight pipeline into a practical pre-close inference path that can produce Top-3/Top-5 candidates before the A-share close.

Target execution window:

- Decision run starts: `14:55:00`
- Deterministic Top-3/Top-5 available: ideally `14:55:20` to `14:56:30`
- Optional TradingAgents2.0 explanation/risk report: `14:56` to `14:58`
- Buy execution: before `15:00`
- Exit rule: sell selected positions at next trading day's open (`next_open_sell`)

## Current System Reality

The current overnight path is not yet a live pre-close engine.

### Offline ranking layer

The historical/backtest path is built by:

- `scripts/build_csi300_overnight_labels.py`
- `scripts/clean_csi300_overnight_labels.py`
- `scripts/build_overnight_feature_table.py`
- `scripts/run_overnight_full_pipeline.py`

`build_overnight_feature_table.py` currently owns the first-pass ranking logic:

- `add_derived_features(df)`
- `add_baseline_scores(df)`
- `build_topn_input(df, top_n)`

This is suitable for backtest/regression, but it is not a complete 14:55 live inference path.

### Current overnight provider layer

`tradingagents/dataflows/overnight_pipeline_provider.py` is intentionally read-only:

- `load_feature_table()` reads a prebuilt CSV feature table.
- `get_trade_date_candidates(trade_date, top_k)` filters one historical `trade_date`.
- `summarize_trade_date_candidates(...)` and `build_candidate_prompt_payload(...)` format the offline candidates for the graph.

It does not fetch 14:55 quotes or build live features.

### Current TradingAgents2.0 role

`tradingagents/graph/trading_graph.py::propagate_overnight(...)` currently:

1. fetches overnight candidate payload/summary/candidates,
2. injects them into the graph state,
3. runs the analyst/research/trader/risk graph,
4. logs the final state.

In overnight mode, analyst files such as:

- `tradingagents/agents/analysts/social_media_analyst.py`
- `tradingagents/agents/analysts/fundamentals_analyst.py`
- `tradingagents/agents/analysts/news_analyst.py`

set tools to `[]` and analyze the provided overnight candidate context directly. Therefore the agent layer is currently an explanation/review layer, not the real-time stock-picking engine.

Note: paths like `/agents/analysts/social_media_analyst.py`, `/agents/analysts/fundamentals_analyst.py`, and `/agents/analysts/news_analyst.py` do not exist at repo root; the actual paths are under `tradingagents/agents/analysts/`.

## Required Live Architecture

### Layer 1: Batch 14:55 market snapshot

Need a low-latency batch quote source that can collect the whole universe, e.g. CSI300, in one or a small number of calls.

Minimum required fields:

- `ts_code`
- `last_price`

Recommended fields:

- `trade_date`
- `run_ts`
- `open`
- `high`
- `low`
- `pre_close`
- `volume`
- `amount`
- `pct_change`

Latency target:

- quote snapshot collection: <= 10-30 seconds preferred
- hard timeout: <= 60 seconds

The first implemented skeleton expects the snapshot to be supplied as CSV via `--snapshot-csv`. This makes the ranking path testable and data-source agnostic.

### Layer 2: Deterministic live ranking

New skeleton:

- `tradingagents/dataflows/overnight_live_provider.py`
- `scripts/run_overnight_live_inference.py`

Design choices:

- Do not use `next_open` or any future label.
- Join the 14:55 snapshot to the latest historical feature row strictly before `trade_date`.
- Build live-observable features such as:
  - `live_return_vs_prev_close`
  - `live_return_vs_pre_close`
  - `live_range_pos`
  - `from_day_high`
  - `from_day_low`
  - `live_px_ma5_ratio`
  - `live_px_ma10_ratio`
  - `live_near_limit_move_like`
  - `live_soft_outlier`
  - `live_intraday_pullback`
- Score deterministically with `score_live_candidates(df)`.
- Apply simple live risk filters with `apply_live_risk_filters(df)`.
- Select:
  - candidate pool: Top-K, e.g. 20
  - selected: Top-N risk-passing candidates, e.g. 3 or 5

Latency target:

- feature join + scoring for CSI300: sub-second to a few seconds on local machine

### Layer 3: TradingAgents2.0 review/report layer

After deterministic Top-N is available, the agent graph should be used for:

- explanation,
- risk review,
- candidate rejection rationale,
- sector concentration checks,
- human-readable execution summary.

It should not block the initial Top-N signal. The practical flow should be:

1. `14:55:00` run deterministic live inference.
2. `14:55:20-14:56:30` output structured Top-3/Top-5.
3. `14:56-14:58` optionally run TradingAgents review on the Top-K candidate pool.

## New Files Added

### `tradingagents/dataflows/overnight_live_provider.py`

Main functions:

- `load_history_feature_table(path=None)`
- `latest_history_by_symbol(history, trade_date=None)`
- `load_snapshot_csv(path)`
- `build_live_feature_frame(snapshot, history_latest, trade_date)`
- `score_live_candidates(df)`
- `apply_live_risk_filters(df, config=None)`
- `select_live_topn(scored, top_n=5, candidate_pool_size=20)`
- `run_live_inference(snapshot_csv, trade_date, history_feature_table_path=None, top_n=5, candidate_pool_size=20)`

### `scripts/run_overnight_live_inference.py`

Example command:

```bash
python3 scripts/run_overnight_live_inference.py \
  --trade-date 2026-05-13 \
  --snapshot-csv data/live_snapshots/snapshot_20260513_1455.csv \
  --history-feature-table data/overnight_mvp/features/overnight_features_20260101_20260430.csv \
  --top-n 5 \
  --candidate-pool-size 20
```

Outputs under `data/overnight_live/<trade_date>/<run_ts>/`:

- `live_features_<date>_topN_poolK.csv`
- `live_scored_<date>_topN_poolK.csv`
- `live_candidate_pool_<date>_topN_poolK.csv`
- `live_selected_<date>_topN_poolK.csv`
- `live_summary_<date>_topN_poolK.md`
- `manifest_<date>_topN_poolK.json`

## Still Missing / Next Implementation Steps

### P0

1. Implement a robust batch realtime quote collector.
   - Candidate module: `tradingagents/dataflows/realtime_snapshot_provider.py`
   - Candidate script: `scripts/fetch_realtime_snapshot.py`
   - Must support batch universe fetch with timeout and completeness checks.
2. Add snapshot schema validation:
   - reject stale snapshot,
   - reject missing `last_price`,
   - require enough universe coverage, e.g. >= 95% of target universe.
3. Wire live candidates into the TradingAgents graph.
   - Option A: add `propagate_overnight_live(...)` in `tradingagents/graph/trading_graph.py`.
   - Option B: allow `create_overnight_candidate_builder()` to accept precomputed `candidate_snapshot` and skip `route_to_vendor(...)`.
4. Add CLI entry:
   - `cli/main.py analyze --strategy-mode overnight-live ...`
   - or keep deterministic path in `scripts/run_overnight_live_inference.py` and add graph review as a second command.

### P1

5. Backtest/shadow-evaluate live features.
   - Replay historical 14:55-like snapshots where available.
   - If no minute snapshots exist, approximate with close but mark it as optimistic.
6. Add live output contract:
   - `trade_date`
   - `run_ts`
   - `snapshot_ts`
   - `top3/top5`
   - `risk_flags`
   - `exit_rule=next_open_sell`
   - `data_completeness`
   - `stale_data_flags`
7. Add latency metrics to manifest.

### P2

8. Add event/news veto layer with strict time budget.
9. Add industry concentration constraints.
10. Add position sizing and execution guardrails.
11. Add next-day open result logging and automatic post-trade review.

## Practical Answer to "Can 14:55 Top3/Top5 Be Fast?"

Yes, if the architecture separates signal generation from agent explanation.

- Deterministic ranking for ~300 stocks: fast.
- Slow part: realtime data collection and LLM graph.
- Therefore Top-3/Top-5 should be generated by the deterministic live layer first.
- TradingAgents2.0 should review/explain after the signal exists, not be on the critical path for initial selection.

## Current Status

Implemented now:

- live deterministic provider skeleton
- live inference CLI skeleton
- Tushare realtime snapshot collector skeleton
- design/latency plan

New files:

- `tradingagents/dataflows/overnight_live_provider.py`
- `tradingagents/dataflows/overnight_live_review_provider.py`
- `tradingagents/dataflows/realtime_snapshot_provider.py`
- `scripts/run_overnight_live_inference.py`
- `scripts/fetch_realtime_snapshot.py`
- `scripts/run_overnight_live_1455.py`
- `scripts/run_overnight_live_agent_review.py`
- `docs/overnight_live_1455_plan_20260513.md`

Smoke-tested now:

```bash
python3 -m py_compile \
  tradingagents/dataflows/overnight_live_provider.py \
  tradingagents/dataflows/realtime_snapshot_provider.py \
  scripts/fetch_realtime_snapshot.py \
  scripts/run_overnight_live_inference.py

python3 scripts/fetch_realtime_snapshot.py \
  --trade-date 2026-05-13 \
  --symbols 600188.SH,688256.SH,002180.SZ \
  --out /tmp/tushare_realtime_snapshot_smoke.csv \
  --manifest /tmp/tushare_realtime_snapshot_smoke.manifest.json \
  --min-coverage 0.90

python3 scripts/run_overnight_live_inference.py \
  --trade-date 2026-05-13 \
  --snapshot-csv /tmp/tushare_realtime_snapshot_smoke.csv \
  --history-feature-table data/overnight_mvp/features/overnight_features_20260101_20260430.csv \
  --top-n 2 \
  --candidate-pool-size 3 \
  --out-root /tmp/overnight_live_tushare_smoke
```

Observed smoke result:

- realtime snapshot rows: `3`
- coverage: `3/3 (100.00%)`
- quote_time: `11:30:00..11:30:00`
- live features rows: `3`
- live scored rows: `3`
- live selected rows: `2`

### P0-1: full-universe realtime snapshot benchmark

A full-universe snapshot benchmark was run against the latest symbols in
`data/overnight_mvp/features/overnight_features_20260101_20260430.csv`:

```bash
/usr/bin/time -f 'elapsed=%E' python3 scripts/fetch_realtime_snapshot.py \
  --trade-date 2026-05-13 \
  --history-feature-table data/overnight_mvp/features/overnight_features_20260101_20260430.csv \
  --out /tmp/tushare_full_universe_snapshot_smoke.csv \
  --manifest /tmp/tushare_full_universe_snapshot_smoke.manifest.json \
  --min-coverage 0.95
```

Observed result:

- snapshot rows: `324`
- coverage: `324/324 (100.00%)`
- quote_time: `11:30:00..11:30:00`
- elapsed: `0:01.07`

This confirms the Tushare realtime path is fast enough for the target universe size in the current environment. The quote timestamp was lunch-close time because the run happened at 12:50, so it must not be treated as a valid 14:55 execution snapshot.

### P0-2: stale quote guard

`fetch_realtime_snapshot.py` now supports:

- `--min-quote-time HH:MM:SS`
- `--fail-stale`

Smoke command:

```bash
python3 scripts/fetch_realtime_snapshot.py \
  --trade-date 2026-05-13 \
  --symbols 600188.SH,688256.SH,002180.SZ \
  --out /tmp/tushare_stale_check_snapshot.csv \
  --manifest /tmp/tushare_stale_check_snapshot.manifest.json \
  --min-quote-time 14:54:00 \
  --fail-stale
```

Observed result:

- coverage: `3/3 (100.00%)`
- quote_time: `11:30:00..11:30:00`
- freshness_ok: `False`
- exit_code: `1`
- error: `stale quote_time: max_quote_time=11:30:00 < min_quote_time=14:54:00`

This is the desired behavior: at 14:55, stale or lunch-session quotes should block a formal Top-3/Top-5 signal.

### P0-3: one-command 14:55 wrapper

Implemented:

- `scripts/run_overnight_live_1455.py`

This operational wrapper performs the complete critical-path workflow:

1. load universe,
2. fetch Tushare realtime snapshot,
3. validate coverage,
4. validate quote freshness,
5. abort before inference if the snapshot is not executable-quality,
6. run deterministic live inference,
7. write snapshot, selected Top-N, candidate pool, summary, and manifest to one run directory.

Strict-mode smoke command:

```bash
python3 scripts/run_overnight_live_1455.py \
  --trade-date 2026-05-13 \
  --symbols 600188.SH,688256.SH,002180.SZ \
  --top-n 2 \
  --candidate-pool-size 3 \
  --out-root /tmp/overnight_live_1455_strict_smoke \
  --min-quote-time 14:54:00
```

Observed strict-mode result at 13:03:

- snapshot rows: `3`
- coverage: `3/3 (100.00%)`
- quote_time: `13:03:34..13:03:36`
- freshness_ok: `False`
- exit_code: `1`
- abort reason: `stale quote_time max=13:03:36 < min_quote_time 14:54:00`

Allow-stale smoke command, for non-trading test only:

```bash
python3 scripts/run_overnight_live_1455.py \
  --trade-date 2026-05-13 \
  --symbols 600188.SH,688256.SH,002180.SZ \
  --top-n 2 \
  --candidate-pool-size 3 \
  --out-root /tmp/overnight_live_1455_allow_stale_smoke \
  --min-quote-time 14:54:00 \
  --allow-stale
```

Observed allow-stale result:

- snapshot rows: `3`
- live features written
- live scored written
- live candidate pool written
- live selected written
- live summary written
- manifest written

This validates both behaviors: strict mode blocks stale pre-14:54 data, while smoke mode can still exercise the full pipeline.

Implemented:

- graph-level live candidate injection via `TradingAgentsGraph.propagate_overnight_live(...)`
- live pre-close review schema and parser: `tradingagents/dataflows/overnight_live_review_provider.py`
- review runner: `scripts/run_overnight_live_agent_review.py`
- final 14:55 fusion in `overnight_live_provider.run_live_inference(...)`
- final wrapper support for `--agent-review-scores`, `--live-weight`, and `--agent-weight`

The intended production clock is now:

1. `14:20~14:40` build a Top15/Top20 buffer from live deterministic ranking.
2. Immediately run TradingAgents2.0 graph on that buffer to produce `agent_review_scores.csv`.
3. `14:55` refresh quotes and fuse deterministic score with agent review scores.
4. Output final Top3/Top5.

Fusion contract:

```text
final_live_score = live_weight * overnight_live_score
                 + agent_weight * agent_score
                 + agent_adjustment
                 - risk_penalty(agent_risk_level)
```

Rules:

- `agent_veto=true` forces `final_live_score=-999` and excludes the stock from final selected TopN.
- if no review score exists for a stock, neutral defaults are used: `agent_score=0.5`, `agent_adjustment=0`, `agent_risk_level=medium`, `agent_veto=false`.

Smoke tests completed:

```bash
python3 -m py_compile \
  tradingagents/dataflows/overnight_live_review_provider.py \
  tradingagents/dataflows/overnight_live_provider.py \
  tradingagents/graph/setup.py \
  tradingagents/graph/trading_graph.py \
  scripts/run_overnight_live_agent_review.py \
  scripts/run_overnight_live_inference.py \
  scripts/run_overnight_live_1455.py
```

Dry-run review smoke:

```bash
python3 scripts/run_overnight_live_agent_review.py \
  --trade-date 2026-05-13 \
  --candidate-pool-csv data/overnight_live_1455_sim/2026-05-13/20260513_130608/live_candidate_pool_20260513_top5_pool20.csv \
  --selected-csv data/overnight_live_1455_sim/2026-05-13/20260513_130608/live_selected_20260513_top5_pool20.csv \
  --top-k 15 \
  --target-top-n 5 \
  --snapshot-time-hint 13:06:06 \
  --out-dir /tmp/overnight_live_agent_review_smoke \
  --dry-run-neutral
```

Fusion smoke:

```bash
python3 scripts/run_overnight_live_inference.py \
  --trade-date 2026-05-13 \
  --snapshot-csv data/overnight_live_1455_sim/2026-05-13/20260513_130608/snapshot_20260513_20260513_130608_tushare.csv \
  --history-feature-table data/overnight_mvp/features/overnight_features_20260101_20260430.csv \
  --top-n 5 \
  --candidate-pool-size 20 \
  --agent-review-scores /tmp/overnight_live_agent_review_smoke/agent_review_scores.csv \
  --out-root /tmp/overnight_live_fusion_smoke
```

One-command wrapper fusion smoke:

```bash
python3 scripts/run_overnight_live_1455.py \
  --trade-date 2026-05-13 \
  --symbols 600188.SH,688256.SH,002180.SZ \
  --top-n 2 \
  --candidate-pool-size 3 \
  --min-quote-time 14:54:00 \
  --allow-stale \
  --agent-review-scores /tmp/overnight_live_agent_review_smoke/agent_review_scores.csv \
  --out-root /tmp/overnight_live_1455_fusion_smoke
```

Not implemented yet:

- CLI integration into `cli/main.py`
- full real LLM graph run at pre-close scale and latency measurement
- tests/shadow backtest for live scorer and agent-fusion policy
