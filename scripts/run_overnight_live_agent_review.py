#!/usr/bin/env python3
"""Run TradingAgents2.0 review on a pre-close live overnight candidate buffer.

Intended schedule:
- 14:20-14:40: run deterministic live prefilter to produce Top15/Top20 buffer.
- immediately after: run this script so TradingAgents2.0 can review candidates
  before the 14:55 final quote refresh.
- 14:55: run final wrapper with --agent-review-scores.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.dataflows.overnight_live_review_provider import (
    build_live_review_payload,
    load_live_candidate_pool,
    summarize_live_candidates,
    write_review_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TradingAgents2.0 review for live overnight candidate buffer")
    parser.add_argument("--trade-date", required=True, help="Decision date YYYY-MM-DD")
    parser.add_argument("--candidate-pool-csv", required=True, help="CSV produced by live prefilter/inference")
    parser.add_argument("--selected-csv", default=None, help="Optional selected CSV; defaults to candidate head target_top_n")
    parser.add_argument("--top-k", type=int, default=15, help="How many buffer candidates to review")
    parser.add_argument("--target-top-n", type=int, default=5, help="Final target TopN at 14:55")
    parser.add_argument("--snapshot-time-hint", default=None, help="Optional quote time hint, e.g. 14:30:00")
    parser.add_argument("--out-dir", required=True, help="Output directory for report/scores/state")
    parser.add_argument("--analysts", default="market,news,fundamentals,social", help="Comma-separated analysts")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--skip-probe", action="store_true", help="Skip LLM live probing for faster startup")
    parser.add_argument("--dry-run-neutral", action="store_true", help="Do not call graph; write neutral review scores for pipeline smoke tests")
    args = parser.parse_args()

    pool = load_live_candidate_pool(args.candidate_pool_csv).head(args.top_k).copy()
    selected = load_live_candidate_pool(args.selected_csv).head(args.target_top_n).copy() if args.selected_csv else pool.head(args.target_top_n).copy()
    payload = build_live_review_payload(
        pool,
        trade_date=args.trade_date,
        top_k=args.top_k,
        target_top_n=args.target_top_n,
        snapshot_time_hint=args.snapshot_time_hint,
    )
    summary = summarize_live_candidates(pool, top_k=args.top_k)
    constraints = {
        "candidate_source": "live_preclose_buffer",
        "review_role": "pre_1455_agent_review",
        "top_n": args.target_top_n,
        "candidate_pool_size": args.top_k,
        "exit_rule": "next_open_sell",
        "final_fusion_required": True,
        "agent_review_schema": [
            "ts_code", "agent_score", "agent_risk_level", "agent_veto", "agent_adjustment", "agent_reason"
        ],
    }

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "agent_review_prompt.md").write_text(payload + "\n", encoding="utf-8")

    if args.dry_run_neutral:
        decision_text = (
            "Dry-run neutral review. TradingAgents2.0 graph was not called.\n\n"
            "AGENT_REVIEW_JSON_START\n"
            + json.dumps(
                {
                    "trade_date": args.trade_date,
                    "target_top_n": args.target_top_n,
                    "reviews": [
                        {
                            "ts_code": str(row["ts_code"]),
                            "agent_score": 0.5,
                            "agent_risk_level": "medium",
                            "agent_veto": False,
                            "agent_adjustment": 0.0,
                            "agent_reason": "dry_run_neutral",
                        }
                        for _, row in pool.iterrows()
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\nAGENT_REVIEW_JSON_END"
        )
        final_state = {"dry_run_neutral": True, "candidate_count": int(len(pool))}
    else:
        analysts = [x.strip() for x in args.analysts.split(",") if x.strip()]
        graph = TradingAgentsGraph(
            selected_analysts=analysts,
            debug=args.debug,
            config=DEFAULT_CONFIG.copy(),
            run_probe=not args.skip_probe,
        )
        final_state, decision_text = graph.propagate_overnight_live(
            trade_date=args.trade_date,
            payload=payload,
            candidate_summary=summary,
            candidates_json=pool.to_json(orient="records", force_ascii=False),
            selected_json=selected.to_json(orient="records", force_ascii=False),
            constraints=constraints,
        )

    paths = write_review_artifacts(out, final_state, pool, decision_text)
    manifest = {
        "trade_date": args.trade_date,
        "candidate_pool_csv": args.candidate_pool_csv,
        "selected_csv": args.selected_csv,
        "top_k": args.top_k,
        "target_top_n": args.target_top_n,
        "dry_run_neutral": args.dry_run_neutral,
        "paths": paths,
    }
    manifest_path = out / "agent_review_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote agent review prompt: {out / 'agent_review_prompt.md'}")
    for k, v in paths.items():
        print(f"Wrote {k}: {v}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
