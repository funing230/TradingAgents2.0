from __future__ import annotations

"""Lightweight news + social-sentiment context builder for overnight live review.

Goal:
- build a structured News Top10 and Social-sentiment Top10 context block
- use only currently available repository vendors/tools
- avoid claiming a dedicated social-media feed when only news aggregation exists
"""

import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from tradingagents.dataflows.interface import route_to_vendor


@dataclass
class NewsContextResult:
    news_top10: list[dict[str, Any]]
    social_top10: list[dict[str, Any]]
    global_news_top10: list[dict[str, Any]]
    vendor: str
    notes: list[str]


def _normalize_ts_code_for_vendor(ts_code: str) -> str:
    code = str(ts_code).strip().upper()
    m = re.match(r"^(\d{6})\.(SZ|SH|BJ)$", code)
    return m.group(1) if m else code


def _safe_json_loads(text: str) -> Any | None:
    try:
        return json.loads(text)
    except Exception:
        return None


def _parse_markdown_news(text: str, ticker: str | None = None) -> list[dict[str, Any]]:
    if not text or text.startswith("No news found") or text.startswith("Error fetching") or text.startswith("No global news"):
        return []

    articles: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("### "):
            if current:
                articles.append(current)
            title_line = line[4:].strip()
            source = ""
            m = re.match(r"^(.*?)\s*\(source:\s*(.*?)\)\s*$", title_line)
            if m:
                title = m.group(1).strip()
                source = m.group(2).strip()
            else:
                title = title_line
            current = {
                "ticker": ticker,
                "title": title,
                "summary": "",
                "source": source,
                "link": "",
                "published_at": "",
                "sentiment": None,
                "relevance": None,
                "channel": "news",
                "raw": [],
            }
            continue
        if current is None:
            continue
        if line.startswith("Published:"):
            current["published_at"] = line.split(":", 1)[1].strip()
        elif line.startswith("Link:"):
            current["link"] = line.split(":", 1)[1].strip()
        elif line:
            current["raw"].append(line)
    if current:
        articles.append(current)

    for item in articles:
        raw = item.pop("raw", [])
        item["summary"] = " ".join(raw).strip()
    return articles


def _parse_alpha_vantage_news(payload: Any, ticker: str | None = None) -> list[dict[str, Any]]:
    if isinstance(payload, str):
        payload = _safe_json_loads(payload)
    if not isinstance(payload, dict):
        return []
    feed = payload.get("feed", []) or []
    out: list[dict[str, Any]] = []
    for article in feed:
        ticker_sentiments = article.get("ticker_sentiment", []) or []
        relevance = None
        sentiment = None
        for row in ticker_sentiments:
            if ticker and str(row.get("ticker", "")).upper() == str(ticker).upper():
                relevance = _to_float(row.get("relevance_score"))
                sentiment = _to_float(row.get("ticker_sentiment_score"))
                break
        if sentiment is None:
            sentiment = _to_float(article.get("overall_sentiment_score"))
        out.append(
            {
                "ticker": ticker,
                "title": article.get("title", ""),
                "summary": article.get("summary", ""),
                "source": article.get("source", ""),
                "link": article.get("url", ""),
                "published_at": article.get("time_published", ""),
                "sentiment": sentiment,
                "relevance": relevance,
                "channel": "news",
            }
        )
    return out


def _to_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _score_news_item(item: dict[str, Any]) -> float:
    relevance = item.get("relevance")
    sentiment = item.get("sentiment")
    summary_len = len(str(item.get("summary", "")))
    freshness_bonus = 0.05 if item.get("published_at") else 0.0
    score = 0.0
    if relevance is not None:
        score += float(relevance)
    if sentiment is not None:
        score += 0.15 * abs(float(sentiment))
    score += min(summary_len / 400.0, 0.10)
    score += freshness_bonus
    return score


def _score_social_item(item: dict[str, Any]) -> float:
    sentiment = item.get("sentiment")
    news_score = _score_news_item(item)
    source = str(item.get("source", "")).lower()
    socialish_bonus = 0.15 if any(k in source for k in ["social", "forum", "stocktwits", "x", "twitter", "reddit", "snowball", "eastmoney", "股吧"]) else 0.0
    sentiment_mag = abs(float(sentiment)) if sentiment is not None else 0.0
    return news_score + 0.35 * sentiment_mag + socialish_bonus


def _dedupe_articles(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        key = (
            re.sub(r"\s+", " ", str(item.get("title", "")).strip().lower()),
            str(item.get("ticker", "") or "GLOBAL").upper(),
        )
        if not key[0] or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _detect_vendor_name(raw: Any) -> str:
    if isinstance(raw, dict) and "feed" in raw:
        return "alpha_vantage"
    if isinstance(raw, str):
        head = raw[:200].lower()
        if "source: akshare" in head:
            return "akshare"
        if "global market news" in head or "news, from" in head:
            return "yfinance_or_markdown_vendor"
    return "configured_news_vendor"


def _fetch_ticker_news(ts_code: str, start_date: str, end_date: str) -> tuple[list[dict[str, Any]], str, str | None]:
    vendor_input = _normalize_ts_code_for_vendor(ts_code)
    raw = route_to_vendor("get_news", vendor_input, start_date, end_date)
    vendor_name = _detect_vendor_name(raw)
    parsed = _parse_alpha_vantage_news(raw, ticker=vendor_input)
    if not parsed:
        parsed = _parse_markdown_news(str(raw), ticker=ts_code)
    for item in parsed:
        item["ticker"] = ts_code
    note = None
    if not parsed:
        note = f"no_news:{ts_code}"
    return parsed, vendor_name, note


def _fetch_global_news(trade_date: str, look_back_days: int, limit: int) -> tuple[list[dict[str, Any]], str | None]:
    raw = route_to_vendor("get_global_news", trade_date, look_back_days, limit)
    parsed = _parse_alpha_vantage_news(raw, ticker=None)
    if not parsed:
        parsed = _parse_markdown_news(str(raw), ticker="GLOBAL")
        for item in parsed:
            item["ticker"] = "GLOBAL"
    note = None if parsed else "no_global_news"
    return parsed, note


def build_news_social_context(
    candidate_pool: pd.DataFrame,
    trade_date: str,
    top_k_candidates: int = 15,
    news_top_n: int = 10,
    social_top_n: int = 10,
    look_back_days: int = 3,
    global_news_limit: int = 10,
) -> NewsContextResult:
    pool = candidate_pool.head(top_k_candidates).copy()
    start_date = (datetime.strptime(trade_date, "%Y-%m-%d") - timedelta(days=look_back_days)).strftime("%Y-%m-%d")

    all_items: list[dict[str, Any]] = []
    global_items: list[dict[str, Any]] = []
    vendor = "configured_news_vendor"
    notes: list[str] = []

    for _, row in pool.iterrows():
        ts_code = str(row.get("ts_code", "")).strip()
        if not ts_code:
            continue
        try:
            items, vendor_name, note = _fetch_ticker_news(ts_code, start_date, trade_date)
        except Exception as exc:
            items, vendor_name, note = [], vendor, f"ticker_news_fetch_failed:{ts_code}:{type(exc).__name__}:{exc}"
        if vendor == "configured_news_vendor" and vendor_name:
            vendor = vendor_name
        if note:
            notes.append(note)
        for item in items:
            item["rank_in_pool"] = int(row.get("rank_in_live_day", len(all_items) + 1)) if pd.notna(row.get("rank_in_live_day", None)) else None
            item["candidate_score"] = _to_float(row.get("overnight_live_score"))
        all_items.extend(items)

    try:
        global_items, global_note = _fetch_global_news(trade_date, look_back_days=look_back_days, limit=global_news_limit)
    except Exception as exc:
        global_items, global_note = [], f"global_news_fetch_failed:{type(exc).__name__}:{exc}"
    if global_note:
        notes.append(global_note)

    all_items = _dedupe_articles(all_items)
    global_items = _dedupe_articles(global_items)

    news_ranked = sorted(all_items, key=_score_news_item, reverse=True)[:news_top_n]
    social_ranked = sorted(all_items, key=_score_social_item, reverse=True)[:social_top_n]
    if len(news_ranked) < news_top_n and global_items:
        needed = news_top_n - len(news_ranked)
        news_ranked.extend(global_items[:needed])

    for i, item in enumerate(news_ranked, start=1):
        item["news_rank"] = i
    for i, item in enumerate(social_ranked, start=1):
        item["social_rank"] = i
    for i, item in enumerate(global_items[:global_news_limit], start=1):
        item["global_rank"] = i

    return NewsContextResult(
        news_top10=news_ranked,
        social_top10=social_ranked,
        global_news_top10=global_items[:global_news_limit],
        vendor=vendor,
        notes=notes,
    )


def summarize_news_social_context(ctx: NewsContextResult) -> dict[str, Any]:
    ticker_failure_notes = [n for n in ctx.notes if str(n).startswith("ticker_news_fetch_failed:")]
    global_failure_notes = [n for n in ctx.notes if str(n).startswith("global_news_fetch_failed:")]
    no_news_notes = [n for n in ctx.notes if str(n).startswith("no_news:")]
    no_global_news = any(str(n) == "no_global_news" for n in ctx.notes)

    failure_reason_counts: Counter[str] = Counter()
    for note in ticker_failure_notes + global_failure_notes:
        parts = str(note).split(":", 3)
        if len(parts) >= 3:
            failure_reason_counts[parts[2]] += 1
        else:
            failure_reason_counts["unknown"] += 1

    degraded = bool(ticker_failure_notes or global_failure_notes or no_news_notes or no_global_news)

    return {
        "vendor": ctx.vendor,
        "degraded": degraded,
        "degrade_reasons": sorted(set(
            (["ticker_news_fetch_failed"] if ticker_failure_notes else [])
            + (["global_news_fetch_failed"] if global_failure_notes else [])
            + (["no_ticker_news"] if no_news_notes else [])
            + (["no_global_news"] if no_global_news else [])
        )),
        "ticker_news_success_count": len(dict(Counter([str(x.get("ticker", "")) for x in ctx.news_top10 if str(x.get("ticker", "")).upper() != "GLOBAL"]))),
        "ticker_news_failure_count": len(ticker_failure_notes),
        "global_news_success_count": len(ctx.global_news_top10),
        "global_news_failure_count": len(global_failure_notes) + int(no_global_news),
        "failure_reason_counts": dict(failure_reason_counts),
        "news_top10_count": len(ctx.news_top10),
        "social_top10_count": len(ctx.social_top10),
        "global_news_top10_count": len(ctx.global_news_top10),
        "news_ticker_counts": dict(Counter([str(x.get("ticker", "")) for x in ctx.news_top10])),
        "social_ticker_counts": dict(Counter([str(x.get("ticker", "")) for x in ctx.social_top10])),
        "notes": ctx.notes,
    }


def build_news_social_context_block(ctx: NewsContextResult) -> str:
    payload = {
        "vendor": ctx.vendor,
        "notes": ctx.notes + [
            "social_top10 is sentiment-style ranking derived from currently available news/aggregated sources; it is not a dedicated standalone social-media feed unless such a provider is later added."
        ],
        "news_top10": ctx.news_top10,
        "social_sentiment_top10": ctx.social_top10,
        "global_news_top10": ctx.global_news_top10,
    }
    return (
        "\n\n附加上下文：以下是程序预抓取并排序后的 News Top10 / Social-sentiment Top10。\n"
        "使用规则：\n"
        "- 可以引用这些条目作为风险/催化/情绪依据。\n"
        "- 不要声称 social_sentiment_top10 来自独立 Twitter/Reddit/雪球 专用接口，除非条目 source 明确显示。\n"
        "- 如条目不足、为空或 source 含糊，宁可保守表述。\n"
        "```json\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n```\n"
    )
