#!/usr/bin/env python3
"""Run 14:55 live overnight inference from a batch quote snapshot CSV.

This is the fast deterministic path for real-money timing:
- Input: one batch snapshot collected around 14:55.
- Join: latest historical overnight feature rows strictly before trade_date.
- Output: Top pool and Top-N picks without using next_open / future labels.

The quote collector is intentionally not implemented here yet.  In production,
wire --snapshot-csv to a low-latency batch source such as a broker quote API,
Level-1 feed, or a validated AkShare/Eastmoney snapshot export.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from tradingagents.dataflows.overnight_live_provider import run_live_inference


DEFAULT_OUT_ROOT = Path("data/overnight_live")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _fmt_date(value: str) -> str:
    return str(value).replace("-", "")


def _write_outputs(result: dict, out_root: Path, top_n: int, candidate_pool_size: int) -> dict[str, Path]:
    suffix = f"{_fmt_date(result['trade_date'])}_top{top_n}_pool{candidate_pool_size}"
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / result["trade_date"] / run_ts
    _ensure_dir(run_dir)

    paths = {
        "features": run_dir / f"live_features_{suffix}.csv",
        "scored": run_dir / f"live_scored_{suffix}.csv",
        "candidate_pool": run_dir / f"live_candidate_pool_{suffix}.csv",
        "selected": run_dir / f"live_selected_{suffix}.csv",
        "summary": run_dir / f"live_summary_{suffix}.md",
        "manifest": run_dir / f"manifest_{suffix}.json",
    }

    result["features"].to_csv(paths["features"], index=False)
    result["scored"].to_csv(paths["scored"], index=False)
    result["candidate_pool"].to_csv(paths["candidate_pool"], index=False)
    result["selected"].to_csv(paths["selected"], index=False)

    selected_cols = [
        c for c in [
            "rank_in_live_day", "ts_code", "name", "industry", "market",
            "overnight_live_score", "last_price", "live_return_vs_pre_close",
            "live_return_vs_prev_close", "from_day_high", "live_range_pos",
            "hist_overnight_prev_1d", "hist_overnight_prev_3d_mean",
            "hist_overnight_positive_rate_5d", "live_reject_reasons",
        ] if c in result["selected"].columns
    ]
    pool_cols = [c for c in selected_cols if c in result["candidate_pool"].columns]

    summary = [
        "# Live Overnight Inference Summary",
        "",
        f"- trade_date: `{result['trade_date']}`",
        f"- snapshot_csv: `{result['snapshot_csv']}`",
        f"- history_feature_table_path: `{result['history_feature_table_path']}`",
        f"- candidate_pool_size: `{candidate_pool_size}`",
        f"- top_n: `{top_n}`",
        f"- generated_at: `{datetime.now().isoformat(timespec='seconds')}`",
        "- exit_rule: `next_open_sell`",
        "- future_label_usage: `none`",
        "",
        "## Selected Top-N",
        "",
    ]
    if result["selected"].empty:
        summary.append("- No candidates passed live risk filters.")
    else:
        summary.append(result["selected"][selected_cols].to_markdown(index=False))
    summary.extend(["", "## Candidate Pool Preview", ""])
    summary.append(result["candidate_pool"].head(candidate_pool_size)[pool_cols].to_markdown(index=False))
    paths["summary"].write_text("\n".join(summary) + "\n", encoding="utf-8")

    manifest = {
        "trade_date": result["trade_date"],
        "snapshot_csv": result["snapshot_csv"],
        "history_feature_table_path": result["history_feature_table_path"],
        "candidate_pool_size": candidate_pool_size,
        "top_n": top_n,
        "exit_rule": "next_open_sell",
        "future_label_usage": "none",
        "rows": {
            "features": int(len(result["features"])),
            "scored": int(len(result["scored"])),
            "candidate_pool": int(len(result["candidate_pool"])),
            "selected": int(len(result["selected"])),
        },
        "paths": {k: str(v) for k, v in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic 14:55 live overnight Top-N inference")
    parser.add_argument("--trade-date", required=True, help="Decision date, YYYY-MM-DD")
    parser.add_argument("--snapshot-csv", required=True, help="Batch realtime quote snapshot CSV collected around 14:55")
    parser.add_argument("--history-feature-table", default=None, help="Historical overnight feature table CSV")
    parser.add_argument("--top-n", type=int, default=5, help="Number of final picks")
    parser.add_argument("--candidate-pool-size", type=int, default=20, help="Candidate pool size to expose to agents")
    parser.add_argument("--agent-review-scores", default=None, help="Optional TradingAgents2.0 pre-close review scores CSV")
    parser.add_argument("--heavy-review-scores", default=None, help="Optional heavy Top50->Top15 review scores CSV")
    parser.add_argument("--light-review-scores", default=None, help="Optional light/fast Top15 review scores CSV")
    parser.add_argument("--live-weight", type=float, default=0.75, help="Final fusion weight for deterministic live score")
    parser.add_argument("--agent-weight", type=float, default=0.25, help="Legacy light-only fusion weight for agent_score")
    parser.add_argument("--heavy-weight", type=float, default=0.25, help="Multi-stage fusion weight for heavy_score")
    parser.add_argument("--light-weight", type=float, default=0.15, help="Multi-stage fusion weight for light agent_score")
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT), help="Output root")
    args = parser.parse_args()

    result = run_live_inference(
        snapshot_csv=args.snapshot_csv,
        trade_date=args.trade_date,
        history_feature_table_path=args.history_feature_table,
        top_n=args.top_n,
        candidate_pool_size=args.candidate_pool_size,
        review_scores_path=args.agent_review_scores,
        heavy_review_scores_path=args.heavy_review_scores,
        light_review_scores_path=args.light_review_scores,
        live_weight=args.live_weight,
        agent_weight=args.agent_weight,
        heavy_weight=args.heavy_weight,
        light_weight=args.light_weight,
    )
    paths = _write_outputs(result, Path(args.out_root), args.top_n, args.candidate_pool_size)

    print(f"Wrote live features: {paths['features']} rows={len(result['features'])}")
    print(f"Wrote live scored: {paths['scored']} rows={len(result['scored'])}")
    print(f"Wrote live candidate pool: {paths['candidate_pool']} rows={len(result['candidate_pool'])}")
    print(f"Wrote live selected: {paths['selected']} rows={len(result['selected'])}")
    print(f"Wrote summary: {paths['summary']}")
    print(f"Wrote manifest: {paths['manifest']}")


if __name__ == "__main__":
    main()
