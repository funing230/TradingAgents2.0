#!/usr/bin/env python3
"""Fast one-shot TradingAgents2.0 review for pre-close live overnight buffer.

This is the lightweight production path for the 14:30 -> 14:55 window.
It intentionally does NOT run the full LangGraph debate pipeline and does NOT
collect Yahoo/global context.  It uses TradingAgents2.0's LLM pool/role config
for a single structured review call, then writes agent_review_scores.csv for
14:55 fusion.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.llm_clients.pool import LLMPool
from tradingagents.dataflows.overnight_live_review_provider import (
    build_live_review_payload,
    load_live_candidate_pool,
    summarize_live_candidates,
    write_review_artifacts,
)


FAST_SYSTEM = """你是 TradingAgents2.0 的轻量化 pre-close review agent。
你的任务不是长篇聊天，而是在 A 股收盘前快速审查 TopK 一夜持股法候选池，输出可机器解析的 agent_review_scores。

硬性要求：
- 不要调用外部数据；只基于用户提供的候选池字段判断。
- 不要编造新闻、公告、财报或未提供事实。
- 如果证据不足，给 medium 风险和中性分。
- 必须输出 AGENT_REVIEW_JSON_START / AGENT_REVIEW_JSON_END 包裹的 JSON。
- 每个输入候选都必须有 reviews 记录。
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
    parser = argparse.ArgumentParser(description="Run fast one-shot TradingAgents2.0 review for live overnight buffer")
    parser.add_argument("--trade-date", required=True)
    parser.add_argument("--candidate-pool-csv", required=True)
    parser.add_argument("--selected-csv", default=None)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--target-top-n", type=int, default=5)
    parser.add_argument("--snapshot-time-hint", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--role", default="market_analyst", help="LLM pool role to use; default is fast Gemini chat role")
    parser.add_argument("--model-key", default=None, help="Optional direct llm_pool model key, e.g. gpt/gemini/claude")
    parser.add_argument("--mode", default="chat", help="LLM mode when --model-key is used")
    args = parser.parse_args()

    started = time.time()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    pool = load_live_candidate_pool(args.candidate_pool_csv).head(args.top_k).copy()
    selected = load_live_candidate_pool(args.selected_csv).head(args.target_top_n).copy() if args.selected_csv else pool.head(args.target_top_n).copy()
    payload = build_live_review_payload(
        pool,
        trade_date=args.trade_date,
        top_k=args.top_k,
        target_top_n=args.target_top_n,
        snapshot_time_hint=args.snapshot_time_hint,
    )
    (out / "fast_agent_review_prompt.md").write_text(payload + "\n", encoding="utf-8")

    config = DEFAULT_CONFIG.copy()
    llm_pool = LLMPool(config)
    if args.model_key:
        llm = llm_pool.get_llm_by_key(args.model_key, mode=args.mode)
        llm_label = f"{args.model_key}:{args.mode}"
    else:
        llm = llm_pool.get_llm(args.role)
        llm_label = f"role:{args.role}"

    result = llm.invoke([SystemMessage(content=FAST_SYSTEM), HumanMessage(content=payload)])
    decision_text = _content_text(result)
    final_state = {
        "mode": "fast_one_shot_review",
        "trade_date": args.trade_date,
        "llm": llm_label,
        "candidate_count": int(len(pool)),
        "target_top_n": args.target_top_n,
        "snapshot_time_hint": args.snapshot_time_hint,
        "selected_reference": json.loads(selected.to_json(orient="records", force_ascii=False)),
        "summary": summarize_live_candidates(pool, top_k=args.top_k),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    paths = write_review_artifacts(out, final_state, pool, decision_text)
    manifest = {
        **final_state,
        "candidate_pool_csv": args.candidate_pool_csv,
        "selected_csv": args.selected_csv,
        "paths": paths,
    }
    manifest_path = out / "fast_agent_review_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    print(f"LLM: {llm_label}")
    print(f"Elapsed seconds: {manifest['elapsed_seconds']}")
    print(f"Wrote prompt: {out / 'fast_agent_review_prompt.md'}")
    for k, v in paths.items():
        print(f"Wrote {k}: {v}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
