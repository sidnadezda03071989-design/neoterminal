"""Backfill level-reach outcomes for the calibration history."""
import argparse
import importlib
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

_BASE = Path(__file__).resolve().parent.parent
if str(_BASE) not in sys.path:
    sys.path.insert(0, str(_BASE))

from app_pkg import config, db
from app_pkg.ai.apply_rules import (
    LEVEL_HORIZON_BARS,
    LEVEL_REACH,
    apply_all_rules,
    invalidate_level_history_cache,
)
from app_pkg.data import market_snapshot as ms
from app_pkg.data.fetch import get_series_df

DEFAULT_BARS = 6000
MIN_LEVEL_SAMPLES = 50
TARGET_LEVEL_SAMPLES = 500
EXPECTED_LEVEL_TYPES = ("VAH", "VAL", "POC", "STRUCT_HIGH", "STRUCT_LOW")
_CACHE: dict[tuple[str, str], pd.DataFrame] = {}


def _preload(symbol: str, tf: str, limit: int) -> None:
    key = (symbol, tf)
    if key in _CACHE:
        return
    df = get_series_df(symbol, tf, limit=limit,
                       history_limit=config.MAX_DATA_LIMIT)
    if df is None or df.empty:
        _CACHE[key] = pd.DataFrame()
        return
    _CACHE[key] = df.sort_values("timestamp").reset_index(drop=True)


def _slice(symbol: str, tf: str, to_sec) -> pd.DataFrame:
    df = _CACHE.get((symbol, tf))
    if df is None or df.empty:
        return pd.DataFrame()
    mask = df["timestamp"] <= pd.to_datetime(int(to_sec), unit="s", utc=True)
    return df[mask].tail(300).reset_index(drop=True)


def _fake_replay(symbol, tf, from_sec=None, to_sec=None, limit=1200):
    key = (str(symbol).upper(), str(tf))
    if key not in _CACHE:
        return None
    if to_sec is not None:
        return _slice(key[0], key[1], to_sec)
    return _CACHE[key].sort_values("timestamp").reset_index(drop=True)


def _bar_ts_sec(value) -> int:
    ts = pd.to_datetime(value)
    if getattr(ts, "tz", None) is None:
        ts = ts.tz_localize("UTC")
    return int(ts.timestamp())


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _probability(value):
    result = _number(value)
    if result is None:
        return None
    if result > 1.0:
        result /= 100.0
    if result < 0.0 or result > 1.0:
        return None
    return result


def level_hit(level: dict, high, low, entry_index: int,
              horizon: int = LEVEL_HORIZON_BARS):
    price = _number(level.get("price")) if isinstance(level, dict) else None
    side = str(level.get("side") or "").upper() if isinstance(level, dict) else ""
    if price is None or price <= 0 or side not in ("UP", "DOWN"):
        return None
    start = entry_index + 1
    end = start + int(horizon)
    if start >= len(high) or end > len(high):
        return None
    if side == "UP":
        return int(any(float(high[i]) >= price for i in range(start, end)))
    return int(any(float(low[i]) <= price for i in range(start, end)))


def _context(level: dict, raw: float, hit: int) -> str:
    return json.dumps({
        "calibration_family": LEVEL_REACH,
        "level_type": str(level.get("type") or "").upper(),
        "raw_prob": raw,
        "hit": int(hit),
        "horizon_bars": LEVEL_HORIZON_BARS,
        "side": str(level.get("side") or "").upper(),
    }, ensure_ascii=False, separators=(",", ":"))


def _db_level_stats(symbol: str, tf: str) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    try:
        conn = sqlite3.connect(str(config.DB_PATH))
        rows = conn.execute(
            "SELECT side, hit, context_json FROM charon_calibration_history "
            "WHERE symbol=? AND timeframe=?", (symbol, tf),
        ).fetchall()
        conn.close()
    except Exception:
        return result
    for side, hit, context_json in rows:
        try:
            context = json.loads(context_json or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(context, dict):
            continue
        family = str(context.get("calibration_family") or "").upper()
        if str(side or "").upper() != LEVEL_REACH and family != LEVEL_REACH:
            continue
        level_type = str(context.get("level_type") or "").upper()
        if not level_type:
            continue
        try:
            horizon = int(context.get("horizon_bars", LEVEL_HORIZON_BARS))
        except (TypeError, ValueError):
            horizon = LEVEL_HORIZON_BARS
        if horizon != LEVEL_HORIZON_BARS:
            continue
        item = result.setdefault(level_type, {"rows": 0, "hits": 0})
        item["rows"] += 1
        item["hits"] += 1 if hit else 0
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Level-reach calibration backfill")
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--tf", default="15m")
    parser.add_argument("--bars", type=int, default=DEFAULT_BARS)
    parser.add_argument("--no-derivatives", action="store_true")
    args = parser.parse_args()

    symbol = str(args.symbol).upper()
    tf = str(args.tf)
    requested_bars = max(0, int(args.bars))
    if args.no_derivatives:
        ms.DERIVATIVES_ENABLED = False
        print("[config] derivatives disabled via --no-derivatives")

    _preload(symbol, tf, max(config.HISTORY_LIMITS.get(tf, 20000),
                              340 + LEVEL_HORIZON_BARS + requested_bars))
    for other_tf in ("1H", "4H", "1D"):
        _preload(symbol, other_tf, 300)

    module = importlib.import_module("app_pkg.data.market_snapshot")
    module.get_replay_df = _fake_replay
    real_scanner = getattr(module, "_scanner_edge", None)
    real_crowd = getattr(module, "_resolve_crowd", None)
    real_extra = getattr(module, "_build_extra_blocks", None)
    cache = {}

    def scanner_cached(symbol_inner, timeframe_inner):
        key = ("scanner", symbol_inner, timeframe_inner)
        if key not in cache and callable(real_scanner):
            cache[key] = real_scanner(symbol_inner, timeframe_inner)
        return cache.get(key)

    def crowd_cached(symbol_inner):
        key = ("crowd", symbol_inner)
        if key not in cache and callable(real_crowd):
            cache[key] = real_crowd(symbol_inner)
        return cache.get(key)

    def extra_cached(symbol_inner, clock_ts, names=None):
        key = ("extra", symbol_inner, clock_ts,
               tuple(names) if names is not None else "all")
        if key not in cache and callable(real_extra):
            cache[key] = real_extra(symbol_inner, clock_ts, names)
        return cache.get(key)

    if callable(real_scanner):
        module._scanner_edge = scanner_cached
    if callable(real_crowd):
        module._resolve_crowd = crowd_cached
    if callable(real_extra):
        module._build_extra_blocks = extra_cached

    df = _CACHE[(symbol, tf)]
    if df is None or df.empty:
        print("No market data")
        return
    high = df["high"].astype(float).to_numpy(dtype="float64")
    low = df["low"].astype(float).to_numpy(dtype="float64")
    timestamps = [_bar_ts_sec(value) for value in df["timestamp"]]
    start = 340
    end = len(df) - LEVEL_HORIZON_BARS
    stop = min(end, start + requested_bars)
    print(f"[preload] {symbol}/{tf} bars={len(df)} analyzable={max(0, stop - start)}")
    if start >= stop:
        print("Not enough bars")
        return

    written = 0
    analyzed = 0
    run_stats: dict[str, dict[str, int]] = {}
    for index in range(start, stop):
        timestamp = timestamps[index]
        try:
            snapshot = ms.compact_snapshot(symbol, tf, upto_sec=float(timestamp))
            result = apply_all_rules(snapshot)
        except Exception as exc:
            print(f"bar {index} ts={timestamp} snapshot fail: {exc!r}")
            continue
        analyzed += 1
        levels = result.get("levels", []) if isinstance(result, dict) else []
        for level in levels:
            if not isinstance(level, dict):
                continue
            level_type = str(level.get("type") or "").strip().upper()
            raw = _probability(level.get("raw_prob"))
            if not level_type or raw is None:
                continue
            hit = level_hit(level, high, low, index)
            if hit is None:
                continue
            db.db_save_calibration_entry(
                timestamp, symbol, tf, raw, LEVEL_REACH, hit,
                _context(level, raw, hit),
            )
            item = run_stats.setdefault(level_type, {"rows": 0, "hits": 0})
            item["rows"] += 1
            item["hits"] += hit
            written += 1

    invalidate_level_history_cache()
    try:
        from app_pkg.ai.calibration import _load_curves
        _load_curves(force=True)
    except Exception:
        pass

    stored_stats = _db_level_stats(symbol, tf)
    print()
    print("=" * 60)
    print("LEVEL-REACH BACKFILL REPORT")
    print("=" * 60)
    print(f"Symbol/TF: {symbol} {tf}")
    print(f"requested_bars={requested_bars} analyzed={analyzed} written={written}")
    print(f"horizon_bars={LEVEL_HORIZON_BARS} min_samples={MIN_LEVEL_SAMPLES}")
    all_types = sorted(set(EXPECTED_LEVEL_TYPES) | set(run_stats) | set(stored_stats))
    for level_type in all_types:
        current = run_stats.get(level_type, {"rows": 0, "hits": 0})
        stored = stored_stats.get(level_type, {"rows": 0, "hits": 0})
        rate = current["hits"] / current["rows"] if current["rows"] else 0.0
        print(f"{level_type}: run_rows={current['rows']} run_hits={current['hits']} "
              f"hit_rate={rate:.3f} stored_rows={stored['rows']} "
              f"stored_hits={stored['hits']}")
    target_met = all(
        stored_stats.get(level_type, {}).get("rows", 0) >= TARGET_LEVEL_SAMPLES
        for level_type in EXPECTED_LEVEL_TYPES
    )
    print(f"target >= {TARGET_LEVEL_SAMPLES} samples per level type: "
          f"{'MET' if target_met else 'NOT MET'}")
    if not target_met:
        print("Increase --bars to accumulate at least 500 samples per level type.")


if __name__ == "__main__":
    main()
