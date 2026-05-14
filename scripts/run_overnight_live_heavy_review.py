#!/usr/bin/env python3
"""Run heavy TradingAgents2.0 review for Top50 -> Top15 overnight live funnel."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.llm_clients.pool import LLMPool
from tradingagents.dataflows.overnight_live_heavy_review_provider import (
    build_heavy_review_payload,
    load_live_candidate_pool,
    summarize_live_candidates,
    write_heavy_review_artifacts,
)


HEAVY_SYSTEM = """你是 TradingAgents2.0 的重度 pre-close research review agent。
你的任务是把 deterministic Top50 一夜持股法候选池压缩为 Top15 研究池，而不是直接给最终 Top5。

硬性要求：
- 只基于用户提供的候选池字段，不调用外部数据，不编造新闻/公告/财报。
- 从候选池整体质量、行业分布、短期趋势、隔夜动能、尾盘结构、风险排雷、组合构造角度做研究筛选。
- 必须先输出 HEAVY_REVIEW_JSON_START / HEAVY_REVIEW_JSON_END 包裹的 JSON；JSON 前不要输出任何文字。
- 禁止输出 <thinking>、推理过程、草稿、分析步骤，禁止复述全部 50 只的逐条长文点评。
- JSON 使用新 schema：top_picks + rejects + summary。
- top_picks 最多 target_top_n 条；未出现在 top_picks/rejects 的股票默认视为 watch，不要全量枚举 50 只。
- JSON 中每个 heavy_reason 控制在 20 个中文字符内，heavy_risk_flags 最多 2 个短标签。
- JSON 后的人类报告控制在 120 中文字以内；如果 JSON 已较长，可只输出 1-2 句极简总结。
"""


def _content_text(result) -> str:
    content = getattr(result, "content", result)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join([p for p in parts if p])
    return str(content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run heavy TradingAgents2.0 Top50 -> Top15 review")
    parser.add_argument("--trade-date", required=True)
    parser.add_argument("--candidate-pool-csv", required=True)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--target-top-n", type=int, default=15)
    parser.add_argument("--snapshot-time-hint", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--role", default="research_manager", help="LLM pool role to use; default research_manager")
    parser.add_argument("--model-key", default=None, help="Optional direct llm_pool model key")
    parser.add_argument("--mode", default="chat", help="LLM mode when --model-key is used")
    parser.add_argument("--dry-run-neutral", action="store_true", help="Skip LLM and emit neutral heavy scores for smoke tests")
    args = parser.parse_args()

    started = time.time()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    pool = load_live_candidate_pool(args.candidate_pool_csv).head(args.top_k).copy()
    payload = build_heavy_review_payload(
        pool,
        trade_date=args.trade_date,
        top_k=args.top_k,
        target_top_n=args.target_top_n,
        snapshot_time_hint=args.snapshot_time_hint,
    )
    (out / "heavy_review_prompt.md").write_text(payload + "\n", encoding="utf-8")

    if args.dry_run_neutral:
        top_picks = []
        for i, row in pool.reset_index(drop=True).head(args.target_top_n).iterrows():
            top_picks.append(
                {
                    "ts_code": row["ts_code"],
                    "heavy_score": 0.5,
                    "heavy_tier": "watch",
                    "heavy_veto": False,
                    "heavy_adjustment": 0.0,
                    "heavy_keep_rank": i + 1,
                    "heavy_reason": "dry_run_neutral",
                    "heavy_risk_flags": [],
                }
            )
        decision_text = (
            "HEAVY_REVIEW_JSON_START\n"
            + json.dumps(
                {
                    "trade_date": args.trade_date,
                    "target_top_n": args.target_top_n,
                    "top_picks": top_picks,
                    "rejects": [],
                    "summary": {
                        "core_count": 0,
                        "watch_top15_count": len(top_picks),
                        "reject_count": 0,
                        "notes": "dry_run_neutral; others default watch",
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\nHEAVY_REVIEW_JSON_END"
        )
        llm_label = "dry_run_neutral"
    else:
        config = DEFAULT_CONFIG.copy()
        llm_pool = LLMPool(config)
        if args.model_key:
            llm = llm_pool.get_llm_by_key(args.model_key, mode=args.mode)
            llm_label = f"{args.model_key}:{args.mode}"
        else:
            llm = llm_pool.get_llm(args.role)
            llm_label = f"role:{args.role}"
        result = llm.invoke([SystemMessage(content=HEAVY_SYSTEM), HumanMessage(content=payload)])
        decision_text = _content_text(result)

    final_state = {
        "mode": "heavy_top50_to_top15_review",
        "trade_date": args.trade_date,
        "llm": llm_label,
        "candidate_count": int(len(pool)),
        "target_top_n": args.target_top_n,
        "snapshot_time_hint": args.snapshot_time_hint,
        "summary": summarize_live_candidates(pool, top_k=args.top_k),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    paths = write_heavy_review_artifacts(out, final_state, pool, decision_text, target_top_n=args.target_top_n)
    manifest = {
        **final_state,
        "candidate_pool_csv": args.candidate_pool_csv,
        "paths": paths,
    }
    manifest_path = out / "heavy_review_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    print(f"LLM: {llm_label}")
    print(f"Elapsed seconds: {manifest['elapsed_seconds']}")
    print(f"Wrote prompt: {out / 'heavy_review_prompt.md'}")
    for k, v in paths.items():
        print(f"Wrote {k}: {v}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
