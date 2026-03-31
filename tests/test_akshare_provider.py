"""
Unit tests for akshare_provider.py

Run unit tests only:   pytest tests/test_akshare_provider.py -m "not integration"
Run all tests:         pytest tests/test_akshare_provider.py
"""

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

from tradingagents.dataflows.akshare_provider import (
    _to_ak_symbol,
    _to_ak_market,
    _fmt_date,
    _safe_call,
    AkShareRateLimitError,
    get_stock_data,
    get_indicators,
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement,
    get_news,
    get_global_news,
    get_insider_transactions,
)


# ===================================================================
# 1. Pure unit tests — no network
# ===================================================================


class TestSymbolConversion:
    def test_pure_digits(self):
        assert _to_ak_symbol("000001") == "000001"

    def test_tushare_format_sz(self):
        assert _to_ak_symbol("000001.SZ") == "000001"

    def test_tushare_format_sh(self):
        assert _to_ak_symbol("600000.SH") == "600000"

    def test_tushare_format_bj(self):
        assert _to_ak_symbol("430047.BJ") == "430047"

    def test_case_insensitive(self):
        assert _to_ak_symbol("000001.sz") == "000001"

    def test_whitespace(self):
        assert _to_ak_symbol("  000001  ") == "000001"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Cannot convert"):
            _to_ak_symbol("AAPL")


class TestMarketDetection:
    def test_explicit_sz(self):
        assert _to_ak_market("000001.SZ") == "sz"

    def test_explicit_sh(self):
        assert _to_ak_market("600000.SH") == "sh"

    def test_guess_sz(self):
        assert _to_ak_market("000001") == "sz"
        assert _to_ak_market("300750") == "sz"

    def test_guess_sh(self):
        assert _to_ak_market("600000") == "sh"


class TestDateHelper:
    def test_fmt_date(self):
        assert _fmt_date("2024-01-15") == "20240115"


class TestSafeCall:
    def test_normal(self):
        assert _safe_call(lambda: 42) == 42

    def test_rate_limit(self):
        with pytest.raises(AkShareRateLimitError):
            _safe_call(lambda: (_ for _ in ()).throw(Exception("访问过于频繁")))

    def test_other_error(self):
        with pytest.raises(ValueError):
            _safe_call(lambda: (_ for _ in ()).throw(ValueError("bad")))


class TestGetStockDataMocked:
    @patch("tradingagents.dataflows.akshare_provider.ak")
    def test_returns_csv(self, mock_ak):
        mock_ak.stock_zh_a_hist.return_value = pd.DataFrame({
            "日期": ["2024-01-02", "2024-01-03"],
            "股票代码": ["000001", "000001"],
            "开盘": [9.39, 9.19],
            "收盘": [9.21, 9.20],
            "最高": [9.42, 9.22],
            "最低": [9.21, 9.15],
            "成交量": [1158366, 733610],
            "成交额": [1075742000, 673673000],
            "振幅": [2.68, 0.92],
            "涨跌幅": [-1.92, -0.11],
            "涨跌额": [-0.18, -0.01],
            "换手率": [0.60, 0.38],
        })

        result = get_stock_data("000001", "2024-01-02", "2024-01-03")

        assert "000001" in result
        assert "AkShare" in result
        assert "9.21" in result

    @patch("tradingagents.dataflows.akshare_provider.ak")
    def test_empty(self, mock_ak):
        mock_ak.stock_zh_a_hist.return_value = pd.DataFrame()
        result = get_stock_data("000001", "2024-01-02", "2024-01-03")
        assert "No data found" in result


class TestGetFundamentalsMocked:
    @patch("tradingagents.dataflows.akshare_provider.ak")
    def test_returns_info(self, mock_ak):
        mock_ak.stock_individual_info_em.return_value = pd.DataFrame({
            "item": ["股票代码", "股票简称", "行业"],
            "value": ["000001", "平安银行", "银行"],
        })
        mock_ak.stock_financial_abstract_ths.side_effect = Exception("skip")
        mock_ak.stock_individual_fund_flow.side_effect = Exception("skip")

        result = get_fundamentals("000001")
        assert "平安银行" in result
        assert "AkShare" in result


class TestFinancialStatementsMocked:
    def _mock_sina_report(self, mock_ak):
        mock_ak.stock_financial_report_sina.return_value = pd.DataFrame({
            "报告日": ["20231231", "20230930"],
            "类型": ["合并期末", "合并期末"],
            "total": [1000000, 900000],
        })

    @patch("tradingagents.dataflows.akshare_provider.ak")
    def test_balance_sheet(self, mock_ak):
        self._mock_sina_report(mock_ak)
        result = get_balance_sheet("000001", "quarterly")
        assert "Balance Sheet" in result
        assert "AkShare" in result

    @patch("tradingagents.dataflows.akshare_provider.ak")
    def test_cashflow(self, mock_ak):
        self._mock_sina_report(mock_ak)
        result = get_cashflow("000001", "quarterly")
        assert "Cash Flow" in result

    @patch("tradingagents.dataflows.akshare_provider.ak")
    def test_income_statement(self, mock_ak):
        self._mock_sina_report(mock_ak)
        result = get_income_statement("000001", "quarterly")
        assert "Income Statement" in result

    @patch("tradingagents.dataflows.akshare_provider.ak")
    def test_annual_filter(self, mock_ak):
        self._mock_sina_report(mock_ak)
        result = get_balance_sheet("000001", "annual")
        assert "20231231" in result
        assert "20230930" not in result


class TestNewsMocked:
    @patch("tradingagents.dataflows.akshare_provider.ak")
    def test_get_news(self, mock_ak):
        mock_ak.stock_news_em.return_value = pd.DataFrame({
            "关键词": ["000001"],
            "新闻标题": ["平安银行发布年报"],
            "新闻内容": ["内容摘要"],
            "发布时间": ["2024-01-05 10:00:00"],
            "文章来源": ["东方财富"],
            "新闻链接": ["http://example.com"],
        })

        result = get_news("000001", "2024-01-01", "2024-01-10")
        assert "平安银行" in result

    @patch("tradingagents.dataflows.akshare_provider.ak")
    def test_get_news_empty(self, mock_ak):
        mock_ak.stock_news_em.return_value = pd.DataFrame()
        result = get_news("000001", "2024-01-01", "2024-01-10")
        assert "No news" in result

    @patch("tradingagents.dataflows.akshare_provider.ak")
    def test_global_news(self, mock_ak):
        mock_ak.stock_info_global_em.return_value = pd.DataFrame({
            "标题": ["Fed raises rates"],
            "摘要": ["Summary"],
            "发布时间": ["2024-01-05"],
            "链接": ["http://example.com"],
        })

        result = get_global_news("2024-01-10")
        assert "Global Market News" in result
        assert "Fed raises rates" in result


class TestInsiderMocked:
    @patch("tradingagents.dataflows.akshare_provider.ak")
    def test_block_trade(self, mock_ak):
        mock_ak.stock_dzjy_mrtj.return_value = pd.DataFrame({
            "证券代码": ["000001", "600000"],
            "证券简称": ["平安银行", "浦发银行"],
            "交易日期": ["2023-11-22", "2023-11-22"],
        })

        result = get_insider_transactions("000001")
        assert "000001" in result
        assert "平安银行" in result

    @patch("tradingagents.dataflows.akshare_provider.ak")
    def test_empty(self, mock_ak):
        mock_ak.stock_dzjy_mrtj.return_value = pd.DataFrame()
        result = get_insider_transactions("000001")
        assert "No insider" in result


# ===================================================================
# 2. Interface routing tests
# ===================================================================


class TestInterfaceRouting:
    def test_akshare_in_vendor_list(self):
        from tradingagents.dataflows.interface import VENDOR_LIST
        assert "akshare" in VENDOR_LIST

    def test_akshare_in_all_methods(self):
        from tradingagents.dataflows.interface import VENDOR_METHODS
        for method, vendors in VENDOR_METHODS.items():
            assert "akshare" in vendors, f"akshare missing from {method}"


# ===================================================================
# 3. Integration tests — real API calls
# ===================================================================


@pytest.mark.integration
class TestAkShareIntegration:
    def test_get_stock_data_real(self):
        result = get_stock_data("000001", "2024-01-01", "2024-01-10")
        assert "000001" in result
        assert "Open" in result
        lines = result.strip().split("\n")
        assert len(lines) > 3

    def test_get_fundamentals_real(self):
        result = get_fundamentals("000001")
        assert "平安银行" in result

    def test_get_balance_sheet_real(self):
        result = get_balance_sheet("000001", "quarterly")
        assert "Balance Sheet" in result

    def test_get_income_statement_real(self):
        result = get_income_statement("000001", "quarterly")
        assert "Income Statement" in result

    def test_get_news_real(self):
        result = get_news("000001", "2024-01-01", "2026-12-31")
        assert isinstance(result, str)

    def test_get_global_news_real(self):
        result = get_global_news("2026-03-31")
        assert "Global Market News" in result

    def test_symbol_with_exchange(self):
        result = get_stock_data("000001.SZ", "2024-01-01", "2024-01-10")
        assert "000001" in result
