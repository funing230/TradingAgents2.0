"""
A-share overnight return labels for close-to-next-open research.

This module now supports multiple next-day exit labels for overnight-holding
research:

    buy at T close -> sell at T+1 open / 09:35 / 09:45 / 10:00

The implementation stays deterministic and inspectable:
- daily bars define the T close and the next trade date
- 5-minute ``stk_mins`` bars define intraday exit prices on T+1
- per-row errors are surfaced instead of silently dropped
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import calendar
import time
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd

from .tushare_provider import TushareRateLimitError, _fmt_date, _get_pro, _parse_date, _safe_call, _to_ts_code


_DATE_FMT = "%Y-%m-%d"
DEFAULT_EXIT_TIMES = ("09:35", "09:45", "10:00")
DEFAULT_MINUTE_CACHE_DIR = Path("data/tushare_minute_cache")


@dataclass(frozen=True)
class OvernightLabel:
    """Close-to-next-open label for one symbol on one trade date."""

    symbol: str
    ts_code: str
    trade_date: str
    close: float
    next_trade_date: str
    next_open: float
    overnight_return_open: float
    source: str = "tushare.daily+trade_cal"

    def to_dict(self) -> dict:
        return asdict(self)


def _normalize_date(date: str) -> str:
    """Return YYYY-MM-DD for YYYY-MM-DD or YYYYMMDD input."""
    return _parse_date(str(date).strip())


def _date_window(start_date: str, end_date: str) -> tuple[str, str]:
    start = datetime.strptime(_normalize_date(start_date), _DATE_FMT)
    end = datetime.strptime(_normalize_date(end_date), _DATE_FMT)
    if end < start:
        raise ValueError(f"end_date {end_date!r} is earlier than start_date {start_date!r}")
    return start.strftime(_DATE_FMT), end.strftime(_DATE_FMT)


def _normalize_exit_times(exit_times: Sequence[str] | None) -> list[str]:
    values = list(exit_times or DEFAULT_EXIT_TIMES)
    normalized = []
    for value in values:
        item = str(value).strip()
        if len(item) == 5:
            item = f"{item}:00"
        try:
            datetime.strptime(item, "%H:%M:%S")
        except ValueError as exc:
            raise ValueError(f"Invalid exit time {value!r}; expected HH:MM or HH:MM:SS") from exc
        normalized.append(item)
    return normalized


def _exit_field_suffix(exit_time: str) -> str:
    return exit_time.replace(":", "")[:4]


def get_trade_calendar(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch open A-share trade dates.

    Prefer Tushare ``trade_cal`` when permission exists. Some valid tokens have
    ``daily`` / ``stk_mins`` but not ``trade_cal`` permission; in that case we
    derive the open-date list from the SH index daily bars so overnight labels
    still work with the user's existing token.
    """
    start_date, end_date = _date_window(start_date, end_date)
    pro = _get_pro()

    try:
        df = _safe_call(
            pro.trade_cal,
            exchange="SSE",
            start_date=_fmt_date(start_date),
            end_date=_fmt_date(end_date),
            is_open="1",
        )
        if df is not None and not df.empty:
            df = df.copy()
            df["trade_date"] = df["cal_date"].map(_parse_date)
            return df[["trade_date"]].sort_values("trade_date").reset_index(drop=True)
    except TushareRateLimitError as exc:
        if "trade_cal" not in str(exc) and "权限" not in str(exc):
            raise

    proxy_df = _safe_call(
        pro.daily,
        ts_code="000001.SZ",
        start_date=_fmt_date(start_date),
        end_date=_fmt_date(end_date),
    )
    if proxy_df is None or proxy_df.empty:
        return pd.DataFrame(columns=["trade_date"])
    proxy_df = proxy_df.copy()
    proxy_df["trade_date"] = proxy_df["trade_date"].map(_parse_date)
    return proxy_df[["trade_date"]].drop_duplicates().sort_values("trade_date").reset_index(drop=True)


def _future_month_window(start: datetime, months: int = 3) -> str:
    """Return the last day of the month ``months`` after start."""
    month_index = start.month - 1 + months
    year = start.year + month_index // 12
    month = month_index % 12 + 1
    day = calendar.monthrange(year, month)[1]
    return datetime(year, month, day).strftime(_DATE_FMT)


def get_next_trade_date(trade_date: str, lookahead_days: int = 20) -> str:
    """Return the next open A-share trading date after ``trade_date``."""
    trade_date = _normalize_date(trade_date)
    start = datetime.strptime(trade_date, _DATE_FMT)
    end = start + timedelta(days=lookahead_days)
    cal = get_trade_calendar(trade_date, end.strftime(_DATE_FMT))
    future_dates = cal.loc[cal["trade_date"] > trade_date, "trade_date"]
    if not future_dates.empty:
        return str(future_dates.iloc[0])

    month_end = _future_month_window(start, months=3)
    if month_end != end.strftime(_DATE_FMT):
        cal = get_trade_calendar(trade_date, month_end)
        future_dates = cal.loc[cal["trade_date"] > trade_date, "trade_date"]
        if not future_dates.empty:
            return str(future_dates.iloc[0])

    raise ValueError(f"No next trade date found after {trade_date} within {lookahead_days} days")


def get_daily_bars(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch Tushare daily bars with normalized dates and OHLC columns."""
    start_date, end_date = _date_window(start_date, end_date)
    pro = _get_pro()
    ts_code = _to_ts_code(symbol)
    df = _safe_call(
        pro.daily,
        ts_code=ts_code,
        start_date=_fmt_date(start_date),
        end_date=_fmt_date(end_date),
    )
    if df is None or df.empty:
        return pd.DataFrame(columns=["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"])

    df = df.copy()
    df["trade_date"] = df["trade_date"].map(_parse_date)
    keep = [c for c in ["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"] if c in df.columns]
    return df[keep].sort_values("trade_date").reset_index(drop=True)


def _intraday_cache_path(cache_dir: Path, ts_code: str, trade_date: str, freq: str = "5min") -> Path:
    safe_code = str(ts_code).upper().replace("/", "_")
    safe_date = _normalize_date(trade_date).replace("-", "")
    safe_freq = str(freq).replace("/", "_")
    return cache_dir / f"{safe_code}_{safe_date}_{safe_freq}.csv"


def get_intraday_bars(
    symbol: str,
    trade_date: str,
    freq: str = "5min",
    retries: int = 1,
    rate_limit_sleep_s: float = 31.0,
    cache_dir: Path | None = DEFAULT_MINUTE_CACHE_DIR,
    use_cache: bool = True,
    write_cache: bool = True,
) -> pd.DataFrame:
    """Fetch next-day minute bars used for multi-exit labels.

    Tushare ``stk_mins`` is rate-limited quite aggressively (often 2/minute and
    sometimes 2/day on low-tier tokens), so this helper supports on-disk caching
    and resumable enrichment flows.
    """
    trade_date = _normalize_date(trade_date)
    pro = _get_pro()
    ts_code = _to_ts_code(symbol)
    start_dt = f"{trade_date} 09:30:00"
    end_dt = f"{trade_date} 10:05:00"

    cache_path: Path | None = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = _intraday_cache_path(cache_dir, ts_code, trade_date, freq=freq)
        if use_cache and cache_path.exists():
            try:
                cached = pd.read_csv(cache_path)
                if cached is not None and not cached.empty:
                    cached = cached.copy()
                    cached["trade_time"] = pd.to_datetime(cached["trade_time"], errors="coerce")
                    keep = [c for c in ["ts_code", "trade_time", "open", "high", "low", "close", "vol", "amount"] if c in cached.columns]
                    return cached[keep].sort_values("trade_time").reset_index(drop=True)
            except Exception:
                pass

    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            df = _safe_call(
                pro.stk_mins,
                ts_code=ts_code,
                start_date=start_dt,
                end_date=end_dt,
                freq=freq,
            )
            if df is None or df.empty:
                return pd.DataFrame(columns=["ts_code", "trade_time", "open", "high", "low", "close", "vol", "amount"])
            out = df.copy()
            out["trade_time"] = pd.to_datetime(out["trade_time"], errors="coerce")
            keep = [c for c in ["ts_code", "trade_time", "open", "high", "low", "close", "vol", "amount"] if c in out.columns]
            out = out[keep].sort_values("trade_time").reset_index(drop=True)
            if cache_path is not None and write_cache:
                out.to_csv(cache_path, index=False)
            return out
        except Exception as exc:
            last_exc = exc
            msg = str(exc)
            if any(kw in msg for kw in ("频率超限", "每分钟", "rate limit", "2次/分钟", "2次/天")) and attempt < retries:
                time.sleep(rate_limit_sleep_s * (attempt + 1))
                continue
            raise

    raise RuntimeError(f"stk_mins fetch failed for {ts_code} {trade_date}: {last_exc}")


def extract_intraday_exit_prices(mins: pd.DataFrame, trade_date: str, exit_times: Sequence[str] | None = None) -> dict[str, float | None]:
    """Extract exact 5-minute close prices for requested exit times."""
    exit_times = _normalize_exit_times(exit_times)
    prices: dict[str, float | None] = {}
    if mins is None or mins.empty:
        for exit_time in exit_times:
            prices[exit_time] = None
        return prices

    ts = mins.copy()
    ts["trade_time"] = pd.to_datetime(ts["trade_time"], errors="coerce")
    for exit_time in exit_times:
        target = pd.Timestamp(f"{_normalize_date(trade_date)} {exit_time}")
        matched = ts.loc[ts["trade_time"] == target]
        if matched.empty:
            prices[exit_time] = None
        else:
            prices[exit_time] = float(pd.to_numeric(matched.iloc[-1]["close"], errors="coerce"))
    return prices


def enrich_overnight_row_with_intraday_exits(
    row: dict | pd.Series,
    exit_times: Sequence[str] | None = None,
    cache_dir: Path | None = DEFAULT_MINUTE_CACHE_DIR,
    use_cache: bool = True,
    write_cache: bool = True,
    retries: int = 0,
    rate_limit_sleep_s: float = 31.0,
) -> dict:
    """Enrich one existing overnight row with T+1 intraday exit prices/returns."""
    exit_times = _normalize_exit_times(exit_times)
    record = dict(row)
    close = pd.to_numeric(record.get("close"), errors="coerce")
    next_trade_date = record.get("next_trade_date")
    ts_code = record.get("ts_code") or record.get("symbol")

    if pd.isna(close) or float(close) <= 0:
        record["minute_source"] = None
        record["minute_error"] = "invalid_close"
        for exit_time in exit_times:
            suffix = _exit_field_suffix(exit_time)
            record[f"next_close_{suffix}"] = None
            record[f"overnight_return_{suffix}"] = None
        return record

    try:
        mins = get_intraday_bars(
            str(ts_code),
            str(next_trade_date),
            cache_dir=cache_dir,
            use_cache=use_cache,
            write_cache=write_cache,
            retries=retries,
            rate_limit_sleep_s=rate_limit_sleep_s,
        )
        exit_prices = extract_intraday_exit_prices(mins, str(next_trade_date), exit_times=exit_times)
        record["minute_source"] = "tushare.stk_mins:5min"
        record["minute_error"] = None
    except Exception as exc:
        exit_prices = {exit_time: None for exit_time in exit_times}
        record["minute_source"] = None
        record["minute_error"] = f"{type(exc).__name__}: {exc}"

    close_value = float(close)
    for exit_time, price in exit_prices.items():
        suffix = _exit_field_suffix(exit_time)
        record[f"next_close_{suffix}"] = price
        record[f"overnight_return_{suffix}"] = None if price is None else (float(price) / close_value) - 1.0
    return record


def enrich_overnight_labels_with_intraday_exits(
    df: pd.DataFrame,
    exit_times: Sequence[str] | None = None,
    cache_dir: Path | None = DEFAULT_MINUTE_CACHE_DIR,
    use_cache: bool = True,
    write_cache: bool = True,
    retries: int = 0,
    rate_limit_sleep_s: float = 31.0,
    skip_completed: bool = True,
) -> pd.DataFrame:
    """Enrich an existing overnight label table with cached/resumable intraday exits."""
    exit_times = _normalize_exit_times(exit_times)
    rows: list[dict] = []
    for row in df.to_dict(orient="records"):
        already_complete = True
        for exit_time in exit_times:
            suffix = _exit_field_suffix(exit_time)
            if pd.isna(row.get(f"next_close_{suffix}")):
                already_complete = False
                break
        if skip_completed and already_complete:
            rows.append(dict(row))
            continue
        rows.append(enrich_overnight_row_with_intraday_exits(
            row,
            exit_times=exit_times,
            cache_dir=cache_dir,
            use_cache=use_cache,
            write_cache=write_cache,
            retries=retries,
            rate_limit_sleep_s=rate_limit_sleep_s,
        ))
    return pd.DataFrame(rows)


def get_overnight_label(symbol: str, trade_date: str, exit_times: Sequence[str] | None = None) -> dict:
    """Build one close-to-next-open + multi-exit overnight label."""
    exit_times = _normalize_exit_times(exit_times)
    trade_date = _normalize_date(trade_date)
    next_trade_date = get_next_trade_date(trade_date)
    ts_code = _to_ts_code(symbol)
    bars = get_daily_bars(ts_code, trade_date, next_trade_date)

    if bars.empty:
        raise ValueError(f"No daily bars found for {ts_code} between {trade_date} and {next_trade_date}")

    by_date = bars.set_index("trade_date")
    if trade_date not in by_date.index:
        raise ValueError(f"No T-day close found for {ts_code} on {trade_date}")
    if next_trade_date not in by_date.index:
        raise ValueError(f"No T+1 open found for {ts_code} on {next_trade_date}")

    close = float(by_date.loc[trade_date, "close"])
    next_open = float(by_date.loc[next_trade_date, "open"])
    if close == 0:
        raise ValueError(f"Close price is zero for {ts_code} on {trade_date}")

    label = OvernightLabel(
        symbol=str(symbol).strip().upper(),
        ts_code=ts_code,
        trade_date=trade_date,
        close=close,
        next_trade_date=next_trade_date,
        next_open=next_open,
        overnight_return_open=(next_open / close) - 1.0,
        source="tushare.daily+stk_mins",
    ).to_dict()

    try:
        mins = get_intraday_bars(ts_code, next_trade_date, cache_dir=DEFAULT_MINUTE_CACHE_DIR, use_cache=True, write_cache=True)
        exit_prices = extract_intraday_exit_prices(mins, next_trade_date, exit_times=exit_times)
        label["minute_source"] = "tushare.stk_mins:5min"
        label["minute_error"] = None
    except Exception as exc:
        exit_prices = {exit_time: None for exit_time in exit_times}
        label["minute_source"] = None
        label["minute_error"] = f"{type(exc).__name__}: {exc}"

    for exit_time, price in exit_prices.items():
        suffix = _exit_field_suffix(exit_time)
        label[f"next_close_{suffix}"] = price
        label[f"overnight_return_{suffix}"] = None if price is None else (float(price) / close) - 1.0

    return label


def build_overnight_labels(symbols: Iterable[str], trade_dates: Iterable[str], exit_times: Sequence[str] | None = None) -> pd.DataFrame:
    """Build overnight labels for a small symbol/date grid.

    Errors are captured per row so exploratory demos can continue and show
    exactly which symbol/date failed.
    """
    rows: list[dict] = []
    for symbol in symbols:
        for trade_date in trade_dates:
            try:
                rows.append(get_overnight_label(symbol, trade_date, exit_times=exit_times))
            except Exception as exc:
                rows.append({
                    "symbol": str(symbol).strip().upper(),
                    "trade_date": _normalize_date(trade_date),
                    "error": f"{type(exc).__name__}: {exc}",
                })
    return pd.DataFrame(rows)


__all__ = [
    "DEFAULT_EXIT_TIMES",
    "OvernightLabel",
    "get_trade_calendar",
    "get_next_trade_date",
    "get_daily_bars",
    "get_intraday_bars",
    "extract_intraday_exit_prices",
    "enrich_overnight_row_with_intraday_exits",
    "enrich_overnight_labels_with_intraday_exits",
    "get_overnight_label",
    "build_overnight_labels",
]
