from tradingagents.dataflows.overnight_news_social_context import (
    _dedupe_articles,
    _parse_markdown_news,
    summarize_news_social_context,
    NewsContextResult,
)


def test_parse_markdown_news_basic():
    text = """## 000001.SZ News\n\n### Title A (source: Yahoo Finance)\nSummary line 1\nLink: https://example.com/a\n\n### Title B (source: EastMoney)\nPublished: 2026-05-14 10:00:00\nLink: https://example.com/b\n"""
    items = _parse_markdown_news(text, ticker="000001.SZ")
    assert len(items) == 2
    assert items[0]["title"] == "Title A"
    assert items[0]["ticker"] == "000001.SZ"
    assert items[0]["link"] == "https://example.com/a"
    assert items[1]["source"] == "EastMoney"


def test_dedupe_articles_by_title_and_ticker():
    items = [
        {"title": "Same Title", "ticker": "000001.SZ"},
        {"title": "Same Title", "ticker": "000001.SZ"},
        {"title": "Same Title", "ticker": "000002.SZ"},
    ]
    deduped = _dedupe_articles(items)
    assert len(deduped) == 2


def test_summarize_news_social_context_counts():
    ctx = NewsContextResult(
        news_top10=[{"ticker": "000001.SZ"}, {"ticker": "000002.SZ"}],
        social_top10=[{"ticker": "000001.SZ"}],
        global_news_top10=[{"ticker": "GLOBAL"}],
        vendor="akshare",
        notes=["ok"],
    )
    summary = summarize_news_social_context(ctx)
    assert summary["vendor"] == "akshare"
    assert summary["news_top10_count"] == 2
    assert summary["social_top10_count"] == 1
    assert summary["news_ticker_counts"]["000001.SZ"] == 1
