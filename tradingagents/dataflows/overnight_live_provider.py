from __future__ import annotations

"""Live overnight candidate inference helpers.

This module is intentionally deterministic and data-source agnostic.  The 14:55
runtime should feed it a batch market snapshot that is already collected by a
separate realtime quote adapter.  Keeping quote collection outside this module
lets the ranking path stay fast, testable, and replayable.

Expected snapshot columns (minimum):
- ts_code
- last_price

Recommended snapshot columns:
- trade_date, open, high, low, pre_close, volume, amount, pct_change, run_ts

The live scorer does not use next_open / future labels.  It reuses yesterday-and-
earlier historical features from the offline overnight feature table, then swaps
in 14:55-observable live price/range fields for today's decision.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from tradingagents.default_config import DEFAULT_CONFIG


@dataclass
class LiveOvernightConfig:
    history_feature_table_path: Path
    candidate_pool_size: int = 20
    top_n: int = 5
    min_price: float = 3.0
    max_abs_live_return: float = 0.095
    max_from_day_high: float = 0.08
    require_snapshot_trade_date: bool = True


LIVE_SCORE_SPECS = [
    # Historical close/overnight tendency available before today's buy decision.
    ("hist_ret_close_1d", 0.10, True),
    ("hist_ret_close_3d", 0.08, True),
    ("hist_ret_close_5d", 0.05, True),
    ("live_px_ma5_ratio", 0.08, False),
    ("live_px_ma10_ratio", 0.05, False),
    ("hist_close_range_pos_5d", 0.08, True),
    ("hist_close_drawdown_10d", 0.07, True),
    ("hist_close_vol_5d", 0.07, False),
    ("hist_overnight_prev_1d", 0.07, False),
    ("hist_overnight_prev_3d_mean", 0.07, False),
    ("hist_overnight_prev_5d_mean", 0.06, False),
    ("hist_overnight_prev_5d_std", 0.03, False),
    ("hist_overnight_positive_rate_5d", 0.05, False),
    # 14:55-observable intraday state.
    ("live_return_vs_prev_close", 0.10, True),
    ("live_range_pos", 0.07, True),
    ("from_day_high", 0.05, True),
    ("gap_days", 0.02, False),
    ("is_new_listing_180d", 0.02, False),
    ("prev_limit_move_like_1d", 0.02, True),
    ("prev_soft_outlier_1d", 0.01, True),
]


HIST_RENAME = {
    "ret_close_1d": "hist_ret_close_1d",
    "ret_close_3d": "hist_ret_close_3d",
    "ret_close_5d": "hist_ret_close_5d",
    "close_range_pos_5d": "hist_close_range_pos_5d",
    "close_drawdown_10d": "hist_close_drawdown_10d",
    "close_vol_5d": "hist_close_vol_5d",
    "overnight_prev_1d": "hist_overnight_prev_1d",
    "overnight_prev_3d_mean": "hist_overnight_prev_3d_mean",
    "overnight_prev_5d_mean": "hist_overnight_prev_5d_mean",
    "overnight_prev_5d_std": "hist_overnight_prev_5d_std",
    "overnight_positive_rate_5d": "hist_overnight_positive_rate_5d",
}


def _resolve_repo_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return Path(DEFAULT_CONFIG["project_dir"]).parent / p


def _as_boolish(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(float)
    return series.astype(str).str.lower().map({"true": 1.0, "false": 0.0}).fillna(0.0)


def load_history_feature_table(path: str | Path | None = None) -> pd.DataFrame:
    feature_path = _resolve_repo_path(path or DEFAULT_CONFIG["overnight_feature_table_path"])
    if not feature_path.exists():
        raise FileNotFoundError(f"Historical overnight feature table not found: {feature_path}")
    df = pd.read_csv(feature_path)
    required = {"trade_date", "ts_code", "close"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Historical feature table missing required columns: {missing}")
    df = df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    return df.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def latest_history_by_symbol(history: pd.DataFrame, trade_date: str | None = None) -> pd.DataFrame:
    hist = history.copy()
    if trade_date:
        cutoff = pd.to_datetime(trade_date, errors="coerce")
        hist = hist.loc[hist["trade_date"] < cutoff].copy()
    if hist.empty:
        raise ValueError("No historical feature rows available before requested live trade_date")
    idx = hist.groupby("ts_code")["trade_date"].idxmax()
    latest = hist.loc[idx].copy().reset_index(drop=True)
    latest = latest.rename(columns=HIST_RENAME)
    latest = latest.rename(columns={"trade_date": "history_trade_date", "close": "history_close"})
    return latest


def load_snapshot_csv(path: str | Path) -> pd.DataFrame:
    snapshot_path = Path(path)
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Live snapshot CSV not found: {snapshot_path}")
    snap = pd.read_csv(snapshot_path)
    if "ts_code" not in snap.columns:
        raise ValueError("Snapshot CSV missing required column: ts_code")
    if "last_price" not in snap.columns:
        # Accept common aliases from quote vendors.
        for alias in ["price", "last", "close", "最新价"]:
            if alias in snap.columns:
                snap = snap.rename(columns={alias: "last_price"})
                break
    if "last_price" not in snap.columns:
        raise ValueError("Snapshot CSV missing required column: last_price")
    return snap.copy()


def build_live_feature_frame(
    snapshot: pd.DataFrame,
    history_latest: pd.DataFrame,
    trade_date: str,
) -> pd.DataFrame:
    snap = snapshot.copy()
    snap["trade_date"] = str(trade_date)
    for col in ["last_price", "open", "high", "low", "pre_close", "volume", "amount", "pct_change"]:
        if col in snap.columns:
            snap[col] = pd.to_numeric(snap[col], errors="coerce")

    hist_keep = [
        c for c in [
            "ts_code", "history_trade_date", "history_close", "name", "industry", "market", "gap_days",
            "hist_ret_close_1d", "hist_ret_close_3d", "hist_ret_close_5d",
            "close_ma5_ratio", "close_ma10_ratio", "hist_close_range_pos_5d", "hist_close_drawdown_10d",
            "hist_close_vol_5d", "hist_overnight_prev_1d", "hist_overnight_prev_3d_mean",
            "hist_overnight_prev_5d_mean", "hist_overnight_prev_5d_std",
            "hist_overnight_positive_rate_5d", "is_new_listing_180d",
            "prev_limit_move_like_1d", "prev_soft_outlier_1d",
        ] if c in history_latest.columns
    ]
    out = snap.merge(history_latest[hist_keep], on="ts_code", how="inner")
    if out.empty:
        raise ValueError("Snapshot and historical feature table have no overlapping ts_code values")

    out["live_return_vs_prev_close"] = out["last_price"] / out["history_close"].replace(0, pd.NA) - 1.0
    if "pre_close" in out.columns:
        out["live_return_vs_pre_close"] = out["last_price"] / out["pre_close"].replace(0, pd.NA) - 1.0
    else:
        out["live_return_vs_pre_close"] = out["live_return_vs_prev_close"]

    high = out["high"] if "high" in out.columns else out["last_price"]
    low = out["low"] if "low" in out.columns else out["last_price"]
    intraday_range = (high - low).replace(0, pd.NA)
    out["live_range_pos"] = (out["last_price"] - low) / intraday_range
    out["from_day_high"] = out["last_price"] / high.replace(0, pd.NA) - 1.0
    out["from_day_low"] = out["last_price"] / low.replace(0, pd.NA) - 1.0

    # Reconstruct historical MA levels from historical close ratios when possible.
    if "close_ma5_ratio" in out.columns:
        ma5 = out["history_close"] / (1.0 + pd.to_numeric(out["close_ma5_ratio"], errors="coerce"))
        out["live_px_ma5_ratio"] = out["last_price"] / ma5.replace(0, pd.NA) - 1.0
    if "close_ma10_ratio" in out.columns:
        ma10 = out["history_close"] / (1.0 + pd.to_numeric(out["close_ma10_ratio"], errors="coerce"))
        out["live_px_ma10_ratio"] = out["last_price"] / ma10.replace(0, pd.NA) - 1.0

    out["live_near_limit_move_like"] = out["live_return_vs_pre_close"].abs() >= 0.095
    out["live_soft_outlier"] = out["live_return_vs_prev_close"].abs() >= 0.075
    out["live_intraday_pullback"] = out["from_day_high"] <= -0.05
    return out


def score_live_candidates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["overnight_live_score"] = 0.0
    for col, weight, ascending in LIVE_SCORE_SPECS:
        if col not in out.columns:
            continue
        series = _as_boolish(out[col]) if out[col].dtype == bool else pd.to_numeric(out[col], errors="coerce")
        ranks = series.rank(pct=True, ascending=ascending)
        out[f"live_score_component__{col}"] = ranks
        out["overnight_live_score"] += weight * ranks.fillna(0.5)

    penalty = pd.Series(0.0, index=out.index)
    for flag, pen in [
        ("live_near_limit_move_like", 0.20),
        ("live_soft_outlier", 0.08),
        ("live_intraday_pullback", 0.10),
        ("prev_limit_move_like_1d", 0.05),
        ("prev_soft_outlier_1d", 0.03),
    ]:
        if flag in out.columns:
            penalty += _as_boolish(out[flag]) * pen
    out["overnight_live_score"] = out["overnight_live_score"] - penalty
    out["rank_in_live_day"] = out["overnight_live_score"].rank(method="first", ascending=False)
    return out.sort_values(["rank_in_live_day", "ts_code"]).reset_index(drop=True)


def apply_live_risk_filters(df: pd.DataFrame, config: LiveOvernightConfig | None = None) -> pd.DataFrame:
    cfg = config or LiveOvernightConfig(history_feature_table_path=Path(DEFAULT_CONFIG["overnight_feature_table_path"]))
    out = df.copy()
    reasons: list[list[str]] = []
    for _, row in out.iterrows():
        r: list[str] = []
        price = row.get("last_price")
        live_ret = row.get("live_return_vs_pre_close", row.get("live_return_vs_prev_close"))
        from_high = row.get("from_day_high")
        if pd.isna(price) or float(price) < cfg.min_price:
            r.append("price_below_min_or_missing")
        if pd.notna(live_ret) and abs(float(live_ret)) > cfg.max_abs_live_return:
            r.append("near_or_beyond_daily_limit_move")
        if pd.notna(from_high) and float(from_high) < -cfg.max_from_day_high:
            r.append("large_intraday_pullback_from_high")
        if bool(row.get("live_near_limit_move_like", False)):
            r.append("live_near_limit_move_like")
        reasons.append(r)
    out["live_reject_reasons"] = [";".join(r) for r in reasons]
    out["live_pass_risk_filter"] = out["live_reject_reasons"].eq("")
    return out


def _boolish_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def _load_light_review_scores(review_scores_path: str | Path) -> pd.DataFrame:
    review_path = Path(review_scores_path)
    if not review_path.exists():
        raise FileNotFoundError(f"Agent review scores not found: {review_path}")
    review = pd.read_csv(review_path)
    if "ts_code" not in review.columns:
        raise ValueError(f"Agent review scores missing ts_code: {review_path}")
    for col, default in [
        ("agent_score", 0.5),
        ("agent_adjustment", 0.0),
        ("agent_veto", False),
        ("agent_risk_level", ""),
        ("agent_reason", ""),
    ]:
        if col not in review.columns:
            review[col] = default
    review = review[["ts_code", "agent_score", "agent_adjustment", "agent_veto", "agent_risk_level", "agent_reason"]].copy()
    review["agent_score"] = pd.to_numeric(review["agent_score"], errors="coerce").clip(0, 1).fillna(0.5)
    review["agent_adjustment"] = pd.to_numeric(review["agent_adjustment"], errors="coerce").clip(-0.2, 0.2).fillna(0.0)
    review["agent_veto"] = _boolish_series(review["agent_veto"])
    return review


def _load_heavy_review_scores(heavy_review_scores_path: str | Path) -> pd.DataFrame:
    review_path = Path(heavy_review_scores_path)
    if not review_path.exists():
        raise FileNotFoundError(f"Heavy review scores not found: {review_path}")
    review = pd.read_csv(review_path)
    if "ts_code" not in review.columns:
        raise ValueError(f"Heavy review scores missing ts_code: {review_path}")
    for col, default in [
        ("heavy_score", 0.5),
        ("heavy_tier", "watch"),
        ("heavy_veto", False),
        ("heavy_adjustment", 0.0),
        ("heavy_keep_rank", pd.NA),
        ("heavy_reason", ""),
        ("heavy_risk_flags", ""),
    ]:
        if col not in review.columns:
            review[col] = default
    review = review[["ts_code", "heavy_score", "heavy_tier", "heavy_veto", "heavy_adjustment", "heavy_keep_rank", "heavy_reason", "heavy_risk_flags"]].copy()
    review["heavy_score"] = pd.to_numeric(review["heavy_score"], errors="coerce").clip(0, 1).fillna(0.5)
    review["heavy_adjustment"] = pd.to_numeric(review["heavy_adjustment"], errors="coerce").clip(-0.2, 0.2).fillna(0.0)
    review["heavy_keep_rank"] = pd.to_numeric(review["heavy_keep_rank"], errors="coerce")
    review["heavy_veto"] = _boolish_series(review["heavy_veto"])
    review["heavy_tier"] = review["heavy_tier"].astype(str).str.lower().where(
        review["heavy_tier"].astype(str).str.lower().isin(["core", "watch", "reject"]),
        "watch",
    )
    return review


def apply_agent_review_fusion(
    scored: pd.DataFrame,
    review_scores_path: str | Path | None = None,
    live_weight: float = 0.75,
    agent_weight: float = 0.25,
) -> pd.DataFrame:
    """Fuse deterministic live score with pre-14:55 TradingAgents review scores.

    Expected review CSV columns:
    - ts_code
    - agent_score in [0, 1]
    - agent_adjustment in [-0.2, 0.2]
    - agent_veto boolean-ish
    - agent_risk_level
    - agent_reason

    Vetoed candidates are not deleted here; they receive a very low
    final_live_score and are excluded by select_live_topn via
    live_pass_risk_filter=False.
    """
    out = scored.copy()
    out["final_live_score"] = out["overnight_live_score"]
    out["agent_score"] = pd.NA
    out["agent_adjustment"] = 0.0
    out["agent_veto"] = False
    out["agent_risk_level"] = ""
    out["agent_reason"] = ""
    if not review_scores_path:
        out["rank_in_final_live_day"] = out["final_live_score"].rank(method="first", ascending=False)
        out["rank_in_live_day"] = out["rank_in_final_live_day"]
        return out.sort_values(["rank_in_live_day", "ts_code"]).reset_index(drop=True)

    review = _load_light_review_scores(review_scores_path)

    out = out.merge(review, on="ts_code", how="left", suffixes=("", "_review"))
    for col in ["agent_score", "agent_adjustment", "agent_veto", "agent_risk_level", "agent_reason"]:
        review_col = f"{col}_review"
        if review_col in out.columns:
            out[col] = out[review_col].combine_first(out[col])
            out = out.drop(columns=[review_col])
    out["agent_score"] = pd.to_numeric(out["agent_score"], errors="coerce").fillna(0.5)
    out["agent_adjustment"] = pd.to_numeric(out["agent_adjustment"], errors="coerce").fillna(0.0)
    out["agent_veto"] = _boolish_series(out["agent_veto"])
    risk_penalty = out["agent_risk_level"].astype(str).str.lower().map({"low": 0.0, "medium": 0.03, "high": 0.12}).fillna(0.03)
    out["final_live_score"] = live_weight * out["overnight_live_score"] + agent_weight * out["agent_score"] + out["agent_adjustment"] - risk_penalty
    out.loc[out["agent_veto"], "final_live_score"] = -999.0
    if "live_reject_reasons" not in out.columns:
        out["live_reject_reasons"] = ""
    out.loc[out["agent_veto"], "live_reject_reasons"] = out.loc[out["agent_veto"], "live_reject_reasons"].astype(str).where(
        out.loc[out["agent_veto"], "live_reject_reasons"].astype(str).eq(""),
        out.loc[out["agent_veto"], "live_reject_reasons"].astype(str) + ";",
    ) + "agent_veto"
    if "live_pass_risk_filter" not in out.columns:
        out["live_pass_risk_filter"] = True
    out.loc[out["agent_veto"], "live_pass_risk_filter"] = False
    out["rank_in_final_live_day"] = out["final_live_score"].rank(method="first", ascending=False)
    out["rank_in_live_day"] = out["rank_in_final_live_day"]
    return out.sort_values(["rank_in_live_day", "ts_code"]).reset_index(drop=True)


def apply_multi_stage_review_fusion(
    scored: pd.DataFrame,
    heavy_review_scores_path: str | Path | None = None,
    light_review_scores_path: str | Path | None = None,
    live_weight: float = 0.60,
    heavy_weight: float = 0.25,
    light_weight: float = 0.15,
) -> pd.DataFrame:
    """Fuse deterministic live score with heavy Top50 review and light Top15 review.

    Heavy review is the earlier Top50 -> Top15 research stage.  Light review is
    the later fast pre-close stage.  Either input may be absent; missing per-row
    scores fall back to neutral values.
    """
    out = scored.copy()
    out["final_live_score"] = out["overnight_live_score"]

    # Initialize all review columns so downstream summaries stay stable.
    for col, default in [
        ("heavy_score", pd.NA),
        ("heavy_tier", ""),
        ("heavy_veto", False),
        ("heavy_adjustment", 0.0),
        ("heavy_keep_rank", pd.NA),
        ("heavy_reason", ""),
        ("heavy_risk_flags", ""),
        ("agent_score", pd.NA),
        ("agent_adjustment", 0.0),
        ("agent_veto", False),
        ("agent_risk_level", ""),
        ("agent_reason", ""),
    ]:
        out[col] = default

    if heavy_review_scores_path:
        heavy = _load_heavy_review_scores(heavy_review_scores_path)
        out = out.merge(heavy, on="ts_code", how="left", suffixes=("", "_heavy_review"))
        for col in ["heavy_score", "heavy_tier", "heavy_veto", "heavy_adjustment", "heavy_keep_rank", "heavy_reason", "heavy_risk_flags"]:
            review_col = f"{col}_heavy_review"
            if review_col in out.columns:
                out[col] = out[review_col].combine_first(out[col])
                out = out.drop(columns=[review_col])

    if light_review_scores_path:
        light = _load_light_review_scores(light_review_scores_path)
        out = out.merge(light, on="ts_code", how="left", suffixes=("", "_light_review"))
        for col in ["agent_score", "agent_adjustment", "agent_veto", "agent_risk_level", "agent_reason"]:
            review_col = f"{col}_light_review"
            if review_col in out.columns:
                out[col] = out[review_col].combine_first(out[col])
                out = out.drop(columns=[review_col])

    out["heavy_score"] = pd.to_numeric(out["heavy_score"], errors="coerce").fillna(0.5)
    out["heavy_adjustment"] = pd.to_numeric(out["heavy_adjustment"], errors="coerce").fillna(0.0)
    out["heavy_veto"] = _boolish_series(out["heavy_veto"])
    out["heavy_tier"] = out["heavy_tier"].astype(str).str.lower().replace({"": "watch", "<na>": "watch", "nan": "watch"})
    out["agent_score"] = pd.to_numeric(out["agent_score"], errors="coerce").fillna(0.5)
    out["agent_adjustment"] = pd.to_numeric(out["agent_adjustment"], errors="coerce").fillna(0.0)
    out["agent_veto"] = _boolish_series(out["agent_veto"])

    heavy_tier_penalty = out["heavy_tier"].map({"core": 0.0, "watch": 0.05, "reject": 0.20}).fillna(0.05)
    light_risk_penalty = out["agent_risk_level"].astype(str).str.lower().map({"low": 0.0, "medium": 0.03, "high": 0.12}).fillna(0.03)
    out["final_live_score"] = (
        live_weight * out["overnight_live_score"]
        + heavy_weight * out["heavy_score"]
        + light_weight * out["agent_score"]
        + out["heavy_adjustment"]
        + out["agent_adjustment"]
        - heavy_tier_penalty
        - light_risk_penalty
    )

    veto_mask = out["heavy_veto"] | out["agent_veto"] | out["heavy_tier"].eq("reject")
    out.loc[veto_mask, "final_live_score"] = -999.0
    if "live_reject_reasons" not in out.columns:
        out["live_reject_reasons"] = ""
    if "live_pass_risk_filter" not in out.columns:
        out["live_pass_risk_filter"] = True
    reason = pd.Series("", index=out.index)
    reason = reason.where(~out["heavy_veto"], reason + ";heavy_veto")
    reason = reason.where(~out["heavy_tier"].eq("reject"), reason + ";heavy_reject")
    reason = reason.where(~out["agent_veto"], reason + ";agent_veto")
    reason = reason.str.strip(";")
    has_reason = reason.ne("")
    out.loc[has_reason, "live_reject_reasons"] = out.loc[has_reason, "live_reject_reasons"].astype(str).where(
        out.loc[has_reason, "live_reject_reasons"].astype(str).eq(""),
        out.loc[has_reason, "live_reject_reasons"].astype(str) + ";",
    ) + reason.loc[has_reason]
    out.loc[veto_mask, "live_pass_risk_filter"] = False
    out["rank_in_final_live_day"] = out["final_live_score"].rank(method="first", ascending=False)
    out["rank_in_live_day"] = out["rank_in_final_live_day"]
    return out.sort_values(["rank_in_live_day", "ts_code"]).reset_index(drop=True)


def select_live_topn(scored: pd.DataFrame, top_n: int = 5, candidate_pool_size: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranked = scored.sort_values(["rank_in_live_day", "ts_code"]).copy()
    pool = ranked.head(candidate_pool_size).copy()
    selected = ranked.loc[ranked["live_pass_risk_filter"]].head(top_n).copy()
    return pool.reset_index(drop=True), selected.reset_index(drop=True)


def run_live_inference(
    snapshot_csv: str | Path,
    trade_date: str,
    history_feature_table_path: str | Path | None = None,
    top_n: int = 5,
    candidate_pool_size: int = 20,
    review_scores_path: str | Path | None = None,
    heavy_review_scores_path: str | Path | None = None,
    light_review_scores_path: str | Path | None = None,
    live_weight: float = 0.75,
    agent_weight: float = 0.25,
    heavy_weight: float = 0.25,
    light_weight: float = 0.15,
) -> dict[str, Any]:
    cfg = LiveOvernightConfig(
        history_feature_table_path=Path(history_feature_table_path or DEFAULT_CONFIG["overnight_feature_table_path"]),
        top_n=top_n,
        candidate_pool_size=candidate_pool_size,
    )
    history = load_history_feature_table(cfg.history_feature_table_path)
    latest = latest_history_by_symbol(history, trade_date=trade_date)
    snapshot = load_snapshot_csv(snapshot_csv)
    features = build_live_feature_frame(snapshot, latest, trade_date=trade_date)
    scored = score_live_candidates(features)
    scored = apply_live_risk_filters(scored, cfg)
    if heavy_review_scores_path or light_review_scores_path:
        scored = apply_multi_stage_review_fusion(
            scored,
            heavy_review_scores_path=heavy_review_scores_path,
            light_review_scores_path=light_review_scores_path,
            live_weight=live_weight,
            heavy_weight=heavy_weight,
            light_weight=light_weight,
        )
    else:
        scored = apply_agent_review_fusion(
            scored,
            review_scores_path=review_scores_path,
            live_weight=live_weight,
            agent_weight=agent_weight,
        )
    pool, selected = select_live_topn(scored, top_n=top_n, candidate_pool_size=candidate_pool_size)
    return {
        "trade_date": str(trade_date),
        "snapshot_csv": str(snapshot_csv),
        "history_feature_table_path": str(cfg.history_feature_table_path),
        "features": features,
        "scored": scored,
        "candidate_pool": pool,
        "selected": selected,
    }
