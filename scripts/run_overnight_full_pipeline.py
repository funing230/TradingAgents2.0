#!/usr/bin/env python3
"""One-click long-window overnight MVP pipeline.

Default target window:
  2026-01-01 -> 2026-04-30

Pipeline steps:
1. Build unified feature table from existing clean overnight labels + Tushare features
2. Run Top-5 factor-weight / filter-rule batch experiments
3. Run Top-5 cost-parameter batch backtest on the baseline Top-5 input table
4. Write a manifest markdown with all generated outputs

This orchestrator is intentionally simple: it shells out to the existing
project scripts so each stage remains independently debuggable.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Iterable


DEFAULT_START = "2026-01-01"
DEFAULT_END = "2026-04-30"
DEFAULT_TOP_N = 5
DEFAULT_OUT_ROOT = Path("data/overnight_mvp")


def _fmt_date(value: str) -> str:
    return value.replace("-", "")


def run_cmd(cmd: list[str], workdir: Path) -> subprocess.CompletedProcess:
    print("RUN:", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(workdir), check=True, text=True)


def write_manifest(
    path: Path,
    start_date: str,
    end_date: str,
    top_n: int,
    feature_path: Path,
    baseline_input_path: Path,
    factor_exp_csv: Path,
    factor_exp_md: Path,
    cost_batch_csv: Path,
    cost_batch_md: Path,
    extra_notes: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = f"""# Overnight MVP Full Pipeline Manifest

- Start date: `{start_date}`
- End date: `{end_date}`
- Top-N: `{top_n}`

## Generated artifacts
- Unified feature table: `{feature_path}`
- Baseline Top-{top_n} input: `{baseline_input_path}`
- Factor/filter batch CSV: `{factor_exp_csv}`
- Factor/filter batch MD: `{factor_exp_md}`
- Cost batch CSV: `{cost_batch_csv}`
- Cost batch MD: `{cost_batch_md}`

## Notes
"""
    if extra_notes:
        text += "\n".join(f"- {note}" for note in extra_notes)
    else:
        text += "- none"
    path.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="One-click long-window overnight MVP pipeline")
    parser.add_argument("--start-date", default=DEFAULT_START, help="YYYY-MM-DD")
    parser.add_argument("--end-date", default=DEFAULT_END, help="YYYY-MM-DD")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--labels", default="data/overnight_labels/csi300_overnight_labels_clean_20240101_20260430.csv")
    parser.add_argument("--limit-symbols", type=int, default=0, help="Optional cap for smoke tests")
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--include-minute-features", action="store_true", help="Merge cached/prefetched minute-window features into the offline feature table")
    parser.add_argument("--minute-cache-only", action="store_true", help="Do not call remote minute API during feature build; only read prefetched minute cache")
    parser.add_argument("--minute-cache-dir", default="data/overnight_mvp/cache/minute_1430_features", help="Minute cache directory for prefetched 14:30-15:00 bars")
    parser.add_argument("--minute-start-time", default="14:30:00")
    parser.add_argument("--minute-end-time", default="15:00:00")
    parser.add_argument("--slippage-bps-list", default="0,5,10")
    parser.add_argument("--initial-capitals", default="1000000")
    parser.add_argument("--factor-fee-bps", type=float, default=10.0)
    parser.add_argument("--factor-slippage-bps", type=float, default=5.0)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    start_suffix = _fmt_date(args.start_date)
    end_suffix = _fmt_date(args.end_date)
    suffix = f"{start_suffix}_{end_suffix}"

    feature_path = DEFAULT_OUT_ROOT / "features" / f"overnight_features_{suffix}.csv"
    baseline_input_path = DEFAULT_OUT_ROOT / "backtest_inputs" / f"topn_baseline_input_{suffix}.csv"
    factor_exp_csv = DEFAULT_OUT_ROOT / "experiments" / f"top5_factor_filter_batch_{suffix}_top{args.top_n}.csv"
    factor_exp_md = DEFAULT_OUT_ROOT / "experiments" / f"top5_factor_filter_batch_{suffix}_top{args.top_n}.md"
    cost_batch_csv = DEFAULT_OUT_ROOT / "backtest_results" / "batch" / f"top5_batch_summary_{suffix}_top{args.top_n}.csv"
    cost_batch_md = DEFAULT_OUT_ROOT / "backtest_results" / "batch" / f"top5_batch_summary_{suffix}_top{args.top_n}.md"
    manifest_path = DEFAULT_OUT_ROOT / "pipeline_manifests" / f"full_pipeline_{suffix}_top{args.top_n}.md"

    extra_notes: list[str] = []
    if not os.getenv("TUSHARE_TOKEN"):
        extra_notes.append("TUSHARE_TOKEN not found in current process environment; scripts may rely on repo .env")
    if os.getenv("TUSHARE_API_URL"):
        extra_notes.append(f"Using TUSHARE_API_URL from environment: {os.getenv('TUSHARE_API_URL')}")

    # Step 1: unified feature table + baseline top-N input
    cmd1 = [
        "python3",
        "scripts/build_overnight_feature_table.py",
        "--labels", args.labels,
        "--start-date", args.start_date,
        "--end-date", args.end_date,
        "--top-n", str(args.top_n),
    ]
    if args.limit_symbols > 0:
        cmd1 += ["--limit-symbols", str(args.limit_symbols)]
    if args.force_refresh:
        cmd1 += ["--force-refresh"]
    if args.include_minute_features:
        cmd1 += [
            "--include-minute-features",
            "--minute-cache-dir", args.minute_cache_dir,
            "--minute-start-time", args.minute_start_time,
            "--minute-end-time", args.minute_end_time,
        ]
        if args.minute_cache_only:
            cmd1 += ["--minute-cache-only"]
    run_cmd(cmd1, repo)

    # Step 2: factor/filter batch experiments
    cmd2 = [
        "python3",
        "scripts/backtest_overnight_top5_factor_filter_batch.py",
        "--features", str(feature_path),
        "--top-n", str(args.top_n),
        "--initial-capital", "1000000",
        "--fee-bps", str(args.factor_fee_bps),
        "--slippage-bps", str(args.factor_slippage_bps),
    ]
    run_cmd(cmd2, repo)

    # Step 3: cost-parameter batch backtest on baseline top-N input
    cmd3 = [
        "python3",
        "scripts/backtest_overnight_top5_batch.py",
        "--input", str(baseline_input_path),
        "--top-n", str(args.top_n),
        "--initial-capitals", args.initial_capitals,
        "--fee-bps-list", args.fee_bps_list,
        "--slippage-bps-list", args.slippage_bps_list,
    ]
    run_cmd(cmd3, repo)

    write_manifest(
        path=manifest_path,
        start_date=args.start_date,
        end_date=args.end_date,
        top_n=args.top_n,
        feature_path=feature_path,
        baseline_input_path=baseline_input_path,
        factor_exp_csv=factor_exp_csv,
        factor_exp_md=factor_exp_md,
        cost_batch_csv=cost_batch_csv,
        cost_batch_md=cost_batch_md,
        extra_notes=extra_notes,
    )

    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
