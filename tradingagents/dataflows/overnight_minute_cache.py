from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from tradingagents.dataflows.tushare_provider import _fmt_date, _get_pro, _parse_date, _safe_call, _to_ts_code


DEFAULT_MINUTE_CACHE_DIR = Path("data/overnight_mvp/cache/minute_1430_features")


def normalize_date(value: str) -> str:
    return _parse_date(str(value).strip())


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def is_rate_limit_error(message: str) -> bool:
    msg = str(message)
    return any(kw in msg for kw in ["频率超限", "每分钟", "每小时", "rate limit", "2次/秒", "2次/分钟", "2次/天"])


def minute_cache_path(
    cache_dir: Path,
    ts_code: str,
    trade_date: str,
    freq: str,
    start_time: str,
    end_time: str,
) -> Path:
    safe = str(ts_code).replace("/", "_")
    day = _fmt_date(normalize_date(trade_date))
    start_label = str(start_time).replace(":", "")[:4]
    end_label = str(end_time).replace(":", "")[:4]
    freq_label = str(freq).replace("/", "_")
    return Path(cache_dir) / f"minute_{safe}_{day}_{start_label}_{end_label}_{freq_label}.csv"


def minute_meta_path(
    cache_dir: Path,
    ts_code: str,
    trade_date: str,
    freq: str,
    start_time: str,
    end_time: str,
) -> Path:
    return minute_cache_path(cache_dir, ts_code, trade_date, freq, start_time, end_time).with_suffix(".meta.json")


def load_cached_minute_window_frame(
    ts_code: str,
    trade_date: str,
    cache_dir: Path,
    start_time: str = "14:30:00",
    end_time: str = "15:00:00",
    freq: str = "5min",
) -> tuple[pd.DataFrame, str | None]:
    cache_path = minute_cache_path(cache_dir, ts_code, trade_date, freq, start_time, end_time)
    if not cache_path.exists():
        return pd.DataFrame(), "cache_miss"
    try:
        df = pd.read_csv(cache_path)
        if "trade_time" in df.columns:
            df["trade_time"] = pd.to_datetime(df["trade_time"], errors="coerce")
        return df, None
    except Exception as exc:
        return pd.DataFrame(), f"cache_read_error: {type(exc).__name__}: {exc}"


def _write_meta(path: Path, payload: dict[str, object]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def fetch_minute_window_frame(
    ts_code: str,
    trade_date: str,
    cache_dir: Path,
    start_time: str = "14:30:00",
    end_time: str = "15:00:00",
    freq: str = "5min",
    force_refresh: bool = False,
    max_retries: int = 2,
    allow_remote: bool = True,
    write_meta: bool = True,
) -> tuple[pd.DataFrame, str | None]:
    cache_dir = Path(cache_dir)
    ensure_dir(cache_dir)
    trade_date = normalize_date(trade_date)
    cache_path = minute_cache_path(cache_dir, ts_code, trade_date, freq, start_time, end_time)
    meta_path = minute_meta_path(cache_dir, ts_code, trade_date, freq, start_time, end_time)

    if cache_path.exists() and not force_refresh:
        df, err = load_cached_minute_window_frame(ts_code, trade_date, cache_dir, start_time=start_time, end_time=end_time, freq=freq)
        if err is None:
            return df, None

    if not allow_remote:
        return pd.DataFrame(), "cache_miss"

    pro = _get_pro()
    start_dt = f"{trade_date} {start_time}"
    end_dt = f"{trade_date} {end_time}"
    last_err: str | None = None
    started_at = time.time()

    for attempt in range(max_retries + 1):
        try:
            df = _safe_call(
                pro.stk_mins,
                ts_code=_to_ts_code(ts_code),
                start_date=start_dt,
                end_date=end_dt,
                freq=freq,
            )
            break
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            if is_rate_limit_error(exc) and attempt < max_retries:
                sleep_s = 31.0 * (attempt + 1)
                print(f"  rate-limited on minute {ts_code} {trade_date}; retry {attempt + 1}/{max_retries} after {sleep_s:.1f}s", flush=True)
                time.sleep(sleep_s)
                continue
            if write_meta:
                _write_meta(meta_path, {
                    "ts_code": ts_code,
                    "trade_date": trade_date,
                    "freq": freq,
                    "start_time": start_time,
                    "end_time": end_time,
                    "status": "error",
                    "error": last_err,
                    "elapsed_seconds": round(time.time() - started_at, 3),
                    "cache_path": str(cache_path),
                })
            return pd.DataFrame(), last_err

    if df is None or df.empty:
        if write_meta:
            _write_meta(meta_path, {
                "ts_code": ts_code,
                "trade_date": trade_date,
                "freq": freq,
                "start_time": start_time,
                "end_time": end_time,
                "status": "empty",
                "elapsed_seconds": round(time.time() - started_at, 3),
                "cache_path": str(cache_path),
            })
        return pd.DataFrame(), "empty"

    df = df.copy()
    if "trade_time" in df.columns:
        df["trade_time"] = pd.to_datetime(df["trade_time"], errors="coerce")
    keep = [c for c in ["ts_code", "trade_time", "open", "high", "low", "close", "vol", "amount"] if c in df.columns]
    df = df[keep].sort_values("trade_time").reset_index(drop=True)
    df.to_csv(cache_path, index=False)
    if write_meta:
        _write_meta(meta_path, {
            "ts_code": ts_code,
            "trade_date": trade_date,
            "freq": freq,
            "start_time": start_time,
            "end_time": end_time,
            "status": "ok",
            "row_count": int(len(df)),
            "elapsed_seconds": round(time.time() - started_at, 3),
            "cache_path": str(cache_path),
        })
    return df, None


def summarize_minute_features(mins: pd.DataFrame, ts_code: str, trade_date: str) -> dict[str, object]:
    record: dict[str, object] = {"ts_code": ts_code, "trade_date": normalize_date(trade_date)}
    if mins is None or mins.empty:
        record.update({
            "minute_bar_count_30m": 0,
            "minute_last30_return": pd.NA,
            "minute_last15_return": pd.NA,
            "minute_range_pos_30m": pd.NA,
            "minute_vwap_gap_30m": pd.NA,
            "minute_vol_30m": pd.NA,
            "minute_amount_30m": pd.NA,
        })
        return record

    data = mins.copy().sort_values("trade_time").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "vol", "amount"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    first_open = pd.to_numeric(data.iloc[0].get("open"), errors="coerce")
    last_close = pd.to_numeric(data.iloc[-1].get("close"), errors="coerce")
    low_30m = pd.to_numeric(data.get("low"), errors="coerce").min()
    high_30m = pd.to_numeric(data.get("high"), errors="coerce").max()
    vol_30m = pd.to_numeric(data.get("vol"), errors="coerce").sum(min_count=1)
    amount_30m = pd.to_numeric(data.get("amount"), errors="coerce").sum(min_count=1)

    last15 = data.loc[data["trade_time"].dt.strftime("%H:%M:%S") >= "14:45:00"].copy() if "trade_time" in data.columns else pd.DataFrame()
    last15_open = pd.to_numeric(last15.iloc[0].get("open"), errors="coerce") if not last15.empty else pd.NA

    vwap_30m = pd.NA
    if pd.notna(vol_30m) and float(vol_30m) > 0 and pd.notna(amount_30m):
        vwap_30m = float(amount_30m) / float(vol_30m)

    range_pos = pd.NA
    if pd.notna(high_30m) and pd.notna(low_30m) and float(high_30m) != float(low_30m):
        range_pos = (float(last_close) - float(low_30m)) / (float(high_30m) - float(low_30m))

    record.update({
        "minute_bar_count_30m": int(len(data)),
        "minute_last30_return": None if pd.isna(first_open) or float(first_open) == 0 or pd.isna(last_close) else float(last_close) / float(first_open) - 1.0,
        "minute_last15_return": None if pd.isna(last15_open) or float(last15_open) == 0 or pd.isna(last_close) else float(last_close) / float(last15_open) - 1.0,
        "minute_range_pos_30m": range_pos,
        "minute_vwap_gap_30m": None if pd.isna(vwap_30m) or float(vwap_30m) == 0 or pd.isna(last_close) else float(last_close) / float(vwap_30m) - 1.0,
        "minute_vol_30m": None if pd.isna(vol_30m) else float(vol_30m),
        "minute_amount_30m": None if pd.isna(amount_30m) else float(amount_30m),
    })
    return record
