"""Global market context collector.

Runs before all analysts to gather US market indices, macro news,
and cross-market sentiment. Stores the result in
state["global_market_context"] so every analyst can reference it.

This node does NOT use an LLM — it fetches data directly and formats
a structured text summary. Fast and zero-cost.
"""

import logging
from datetime import datetime, timedelta

from tradingagents.dataflows.interface import route_to_vendor, detect_market

logger = logging.getLogger(__name__)

# US indices to track for cross-market context
_US_INDICES = [
    ("^GSPC", "S&P 500"),
    ("^IXIC", "NASDAQ Composite"),
    ("^DJI", "Dow Jones"),
]

# Look-back window for index data (trading days)
_INDEX_LOOKBACK_DAYS = 5
_NEWS_LOOKBACK_DAYS = 3
_NEWS_LIMIT = 10


def _fetch_index_summary(trade_date: str) -> str:
    """Fetch recent price data for US indices and format a summary."""
    lines = []
    end_date = trade_date
    try:
        start_dt = datetime.strptime(trade_date, "%Y-%m-%d") - timedelta(days=_INDEX_LOOKBACK_DAYS * 2)
        start_date = start_dt.strftime("%Y-%m-%d")
    except ValueError:
        start_date = trade_date

    for symbol, name in _US_INDICES:
        try:
            data = route_to_vendor("get_stock_data", symbol, start_date, end_date)
            if data and len(data.strip()) > 0:
                # Extract last few lines of CSV for recent prices
                csv_lines = [l for l in data.strip().split("\n") if l and not l.startswith("#")]
                if len(csv_lines) > 1:
                    header = csv_lines[0]
                    recent = csv_lines[-min(3, len(csv_lines) - 1):]
                    lines.append(f"### {name} ({symbol})")
                    lines.append(header)
                    lines.extend(recent)
                    lines.append("")
        except Exception as e:
            logger.debug("Failed to fetch %s: %s", symbol, e)
            lines.append(f"### {name} ({symbol}): data unavailable")
            lines.append("")

    return "\n".join(lines) if lines else "US index data unavailable."


def _fetch_global_news(trade_date: str) -> str:
    """Fetch recent global macro news."""
    try:
        news = route_to_vendor("get_global_news", trade_date, _NEWS_LOOKBACK_DAYS, _NEWS_LIMIT)
        if news and len(news.strip()) > 20:
            return news
    except Exception as e:
        logger.debug("Failed to fetch global news: %s", e)
    return "Global news data unavailable."


def create_global_context_collector():
    """Create a graph node that collects global market context.

    Returns a function compatible with LangGraph's node interface.
    The node reads state["trade_date"] and writes state["global_market_context"].
    """

    def global_context_node(state):
        trade_date = state.get("trade_date", "")
        ticker = state.get("company_of_interest", "")
        market = detect_market(ticker)

        # Only collect US context when analyzing A-share stocks
        if market != "cn":
            return {"global_market_context": ""}

        sections = []
        sections.append("# Global Market Context (Auto-collected)")
        sections.append(f"Trade date: {trade_date}\n")

        # 1. US index summary
        sections.append("## US Market Indices (Recent)")
        sections.append(_fetch_index_summary(trade_date))

        # 2. Global macro news
        sections.append("## Global Macro News")
        sections.append(_fetch_global_news(trade_date))

        context = "\n".join(sections)
        logger.info("Global context collected: %d chars", len(context))

        return {"global_market_context": context}

    return global_context_node
