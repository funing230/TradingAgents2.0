# Overnight Minute Prefetch + Cache-First Flow

## Goal
Decouple minute-data collection from decision-time inference so Tushare minute rate limits are spread across the afternoon and the overnight strategy reads local cache at decision time.

## New components
- `scripts/prefetch_overnight_minute_cache.py`
  - batch minute prefetcher
  - supports candidate caps, sharding, and `--missing-only`
- `tradingagents/dataflows/overnight_minute_cache.py`
  - shared cache naming/loading/fetching/summarizing helpers
- `scripts/build_overnight_feature_table.py`
  - now supports `--minute-cache-only`
  - reads prefetched minute cache first

## Recommended operating pattern
### 0) One-command multistage recall + auto minute prefetch
If you want recall Top50 to immediately trigger minute prefetch, use:

```bash
cd /home/sun/.openclaw/workspace/TradingAgents2.0
python3 scripts/run_overnight_live_multistage.py \
  --trade-date 2026-05-14 \
  --prefilter-snapshot-csv <snapshot_csv> \
  --final-snapshot-csv <snapshot_csv_or_1455_snapshot> \
  --enable-minute-prefetch \
  --minute-prefetch-missing-only \
  --minute-prefetch-candidate-limit 50 \
  --minute-prefetch-start-time 14:30:00 \
  --minute-prefetch-end-time 15:00:00 \
  --minute-prefetch-freq 5min
```

This produces recall Top50 first, then automatically launches local minute-cache prefetch using the recall candidate pool.

### 1) Build early candidate pool
Use recall/top pool from snapshot or another deterministic prefilter.

Example candidate input:
- `live_candidate_pool_*.csv`
- symbol column: `ts_code`
- optional rank column: `rank_in_live_day`

### 2) Prefetch minute windows in batches between 14:00 and 15:00
Example:
```bash
cd /home/sun/.openclaw/workspace/TradingAgents2.0
python3 scripts/prefetch_overnight_minute_cache.py \
  --trade-date 2026-05-14 \
  --candidate-csv data/overnight_live_multistage/2026-05-14/.../live_candidate_pool_20260514_recall_top50_pool50.csv \
  --symbol-column ts_code \
  --rank-column rank_in_live_day \
  --candidate-limit 50 \
  --start-time 14:30:00 \
  --end-time 15:00:00 \
  --freq 5min \
  --missing-only
```

### 3) Spread load with shards
Example: split Top100 into 4 shards
```bash
python3 scripts/prefetch_overnight_minute_cache.py ... --candidate-limit 100 --shard-id 0 --shard-count 4 --missing-only
python3 scripts/prefetch_overnight_minute_cache.py ... --candidate-limit 100 --shard-id 1 --shard-count 4 --missing-only
python3 scripts/prefetch_overnight_minute_cache.py ... --candidate-limit 100 --shard-id 2 --shard-count 4 --missing-only
python3 scripts/prefetch_overnight_minute_cache.py ... --candidate-limit 100 --shard-id 3 --shard-count 4 --missing-only
```

### 4) Decision-time feature build reads cache only
```bash
python3 scripts/build_overnight_feature_table.py \
  --start-date 2026-04-01 \
  --end-date 2026-04-30 \
  --include-minute-features \
  --minute-cache-only \
  --minute-cache-dir data/overnight_mvp/cache/minute_1430_features
```

## Suggested staggered schedule
Example cadence for Top50 / Top100 overnight prefetch:

- 14:05: run recall Top50 and auto prefetch shard 0/4
- 14:15: rerun prefetch shard 1/4 with `--missing-only`
- 14:25: rerun prefetch shard 2/4 with `--missing-only`
- 14:35: rerun prefetch shard 3/4 with `--missing-only`
- 14:45: rerun full Top50 with `--missing-only` and no sharding
- 14:53: final补漏一次，仅 `--missing-only`

### Copy-paste commands
Shard example:
```bash
python3 scripts/run_overnight_live_multistage.py \
  --trade-date 2026-05-14 \
  --prefilter-snapshot-csv <snapshot_csv> \
  --final-snapshot-csv <snapshot_csv> \
  --enable-minute-prefetch \
  --minute-prefetch-missing-only \
  --minute-prefetch-candidate-limit 100 \
  --minute-prefetch-shard-id 0 \
  --minute-prefetch-shard-count 4
```

Final补漏 example:
```bash
python3 scripts/prefetch_overnight_minute_cache.py \
  --trade-date 2026-05-14 \
  --candidate-csv <recall_candidate_pool_csv> \
  --symbol-column ts_code \
  --rank-column rank_in_live_day \
  --candidate-limit 50 \
  --start-time 14:30:00 \
  --end-time 15:00:00 \
  --freq 5min \
  --missing-only
```

## Why this is safer
- minute API load is moved out of the decision-time critical path
- repeated runs reuse local cache
- failures are inspectable via per-run manifests and per-file meta JSON
- `--missing-only` avoids re-hitting already cached symbol-date windows

## Practical recommendation for overnight strategy
- first use `5min`
- first use only `14:30:00 -> 15:00:00`
- first use Top50 / Top100 candidate pools only
- keep final decision path cache-first
