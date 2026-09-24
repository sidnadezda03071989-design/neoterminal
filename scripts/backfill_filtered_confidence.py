#!/usr/bin/env python3
"""Filtered calibration backtest: only bars where |pu - pd| >= min_gap.

Чисто read-only анализ: НЕ пишет в БД. Измеряет hit-rate на подмножестве
high-confidence сигналов и сравнивает с unfiltered baseline (46%).

Usage:
    python scripts/backfill_filtered_confidence.py --bars 5000
    python scripts/backfill_filtered_confidence.py --bars 5000 --min-gap 0.25
    python scripts/backfill_filtered_confidence.py --bars 5000 --no-derivatives
"""
import argparse
import importlib
import sys
from pathlib import Path

import pandas as pd

_BASE = Path(__file__).resolve().parent.parent
if str(_BASE) not in sys.path:
    sys.path.insert(0, str(_BASE))

from app_pkg import config                      # noqa: E402
from app_pkg.ai.apply_rules import apply_all_rules, SIG_THRESHOLD  # noqa: E402
from app_pkg.data import market_snapshot as ms      # noqa: E402
from app_pkg.data.fetch import get_series_df        # noqa: E402

ATR_K = 1.5
HORIZON = 20

_CACHE: dict[tuple, pd.DataFrame] = {}
SYMBOL = None


def _preload(tf, limit):
    if (SYMBOL, tf) in _CACHE:
        return
    df = get_series_df(SYMBOL, tf, limit=limit, history_limit=config.MAX_DATA_LIMIT)
    df = df.sort_values("timestamp").reset_index(drop=True)
    _CACHE[(SYMBOL, tf)] = df


def _slice(tf, to_sec):
    df = _CACHE.get((SYMBOL, tf))
    if df is None or df.empty:
        return df
    mask = df["timestamp"] <= pd.to_datetime(int(to_sec), unit="s", utc=True)
    out = df[mask]
    return out.tail(300).reset_index(drop=True)


def _fake_replay(symbol, tf, from_sec=None, to_sec=None, limit=1200):
    if (SYMBOL, tf) not in _CACHE:
        return None
    df = _slice(tf, to_sec) if to_sec is not None else _CACHE[(SYMBOL, tf)]
    if df is None or df.empty:
        return df
    return df.sort_values("timestamp").reset_index(drop=True)


def _bar_ts_sec(ts):
    ts = pd.to_datetime(ts)
    if getattr(ts, "tz", None) is None:
        ts = ts.tz_localize("UTC")
    return int(ts.timestamp())


def triple_barrier(high, low, close, entry, atr_pct, entry_idx, horizon, atr_k):
    """Outcome (+1/-1/0/None) для бара entry_idx со входом (entry, atr_pct)."""
    if entry <= 0 or atr_pct is None or atr_pct <= 0:
        return None
    upper = entry * (1.0 + atr_k * atr_pct)
    lower = entry * (1.0 - atr_k * atr_pct)
    j0 = entry_idx + 1
    jend = j0 + horizon
    if jend > len(close):
        return None
    for k in range(j0, jend):
        if low[k] <= lower:
            return -1  # SL-приоритет
        if high[k] >= upper:
            return 1
    return 0  # таймаут


def main():
    global SYMBOL

    parser = argparse.ArgumentParser(description="Filtered confidence calibration backtest")
    parser.add_argument("--symbol", default="ETHUSDT", help="Trading pair (default ETHUSDT)")
    parser.add_argument("--tf", default="15m", help="Timeframe (default 15m)")
    parser.add_argument("--bars", type=int, default=5000, help="Number of bars (default 5000)")
    parser.add_argument("--min-gap", type=float, default=0.20, help="Minimum |pu-pd| gap (default 0.20)")
    parser.add_argument("--no-derivatives", action="store_true",
                        help="Disable derivatives fetch (rules 21-24)")
    args = parser.parse_args()

    SYMBOL = args.symbol.upper()
    tf = args.tf
    n_bars = args.bars
    min_gap = args.min_gap

    if args.no_derivatives:
        print("[config] derivatives disabled via --no-derivatives")
        ms.DERIVATIVES_ENABLED = False

    _preload(tf, config.HISTORY_LIMITS.get(tf, 20000))
    for _tf in ("1H", "4H", "1D"):
        _preload(_tf, 300)

    # Подмена фетча: компакт-снимок для реплея читает из памяти (без сети).
    mod = importlib.import_module("app_pkg.data.market_snapshot")
    mod.get_replay_df = _fake_replay

    _real_scanner = mod._scanner_edge
    _real_crowd = mod._resolve_crowd

    # Кэшируем медленные внешние блоки
    _S = {}
    def _scan_edge_cached(symbol_inner, timeframe_inner):
        nonlocal _S
        if "scanner" not in _S:
            _S["scanner"] = _real_scanner(symbol_inner, timeframe_inner)
        return _S["scanner"]

    def _crowd_cached(symbol_inner):
        nonlocal _S
        if "crowd" not in _S:
            _S["crowd"] = _real_crowd(symbol_inner)
        return _S["crowd"]

    _real_extra = mod._build_extra_blocks
    _S_extra = {}
    def _extra_cached(symbol_inner, clock_ts, names=None):
        nonlocal _S_extra
        key = tuple(names) if names is not None else "all"
        if key not in _S_extra:
            _S_extra[key] = _real_extra(symbol_inner, clock_ts, names)
        return _S_extra[key]

    mod._scanner_edge = _scan_edge_cached
    mod._resolve_crowd = _crowd_cached
    mod._build_extra_blocks = _extra_cached

    df = _CACHE[(SYMBOL, tf)]
    n = len(df)
    high = df["high"].astype(float).to_numpy(dtype="float64")
    low = df["low"].astype(float).to_numpy(dtype="float64")
    close = df["close"].astype(float).to_numpy(dtype="float64")
    ts = [_bar_ts_sec(x) for x in df["timestamp"]]

    # ATR(14)/close
    from app_pkg.indicators import _atr
    atr_s = _atr(pd.Series(high), pd.Series(low), pd.Series(close), 14)
    atr_pct_arr = (atr_s.to_numpy(dtype="float64") / close)
    print(f"[preload] {SYMBOL}/{tf} bars={n} (analyzable window up to {n - HORIZON})")

    start = 340
    end = n - HORIZON
    if start >= end:
        print(f"мало баров: n={n}")
        return

    total_bars = 0
    filtered_bars = 0
    skipped_bars = 0

    up_hits = 0
    down_hits = 0
    filtered_long_signals = 0
    filtered_short_signals = 0
    gaps_filtered: list[float] = []

    all_up_hits = 0
    all_down_hits = 0
    all_total = 0

    for i in range(start, min(end, start + n_bars)):
        bar_ts = ts[i]
        try:
            snap = ms.compact_snapshot(SYMBOL, tf, upto_sec=float(bar_ts))
            verdict = apply_all_rules(snap)
        except Exception as exc:
            print(f"bar {i} ts={bar_ts} snapshot fail: {exc!r}")
            continue

        pu_v = verdict.get("pu")
        pd_v = verdict.get("pd")
        if pu_v is None or pd_v is None:
            continue

        gap = abs(pu_v - pd_v)
        total_bars += 1

        # Определяем сигнал
        diff = pu_v - pd_v
        if diff > SIG_THRESHOLD:
            sig_dir = "LONG"
        elif diff < -SIG_THRESHOLD:
            sig_dir = "SHORT"
        else:
            sig_dir = "FLAT"

        entry_price = float(close[i])
        atr_pct_v = float(atr_pct_arr[i]) if atr_pct_arr[i] is not None else None
        outcome = triple_barrier(high, low, close, entry_price, atr_pct_v,
                                 i, HORIZON, ATR_K)
        if outcome is None:
            continue

        # Unfiltered stats
        if outcome == 1:
            all_up_hits += 1
        elif outcome == -1:
            all_down_hits += 1
        all_total += 1

        # FILTER: gap < min_gap -> skip
        if gap < min_gap:
            skipped_bars += 1
            continue

        filtered_bars += 1
        gaps_filtered.append(gap)

        if sig_dir == "LONG":
            filtered_long_signals += 1
            if outcome == 1:
                up_hits += 1
        elif sig_dir == "SHORT":
            filtered_short_signals += 1
            if outcome == -1:
                down_hits += 1
        # FLAT signals counted in filtered_bars but not directional

    # --- Compute metrics ---
    filtered_hit_rate_up = up_hits / max(1, filtered_long_signals)
    filtered_hit_rate_down = down_hits / max(1, filtered_short_signals)

    if filtered_bars > 0 and gaps_filtered:
        avg_gap = sum(gaps_filtered) / len(gaps_filtered)
        combined_filtered_hit = (up_hits + down_hits) / filtered_bars
    else:
        avg_gap = 0.0
        combined_filtered_hit = 0.0

    filter_pass_rate = filtered_bars / max(1, total_bars)

    baseline_up_rate = all_up_hits / max(1, all_total)
    baseline_down_rate = all_down_hits / max(1, all_total)
    baseline_combined = (baseline_up_rate + baseline_down_rate) / 2.0

    delta_up = filtered_hit_rate_up - 0.46
    delta_down = filtered_hit_rate_down - 0.466

    # --- REPORT ---
    print()
    print("=" * 60)
    print("FILTERED CALIBRATION REPORT")
    print("=" * 60)
    print(f"Symbol: {SYMBOL} | TF: {tf} | Bars: {total_bars}")
    print(f"Min confidence gap: {min_gap} ({min_gap*100:.0f}%)")
    print()
    print("FILTER STATS:")
    print(f"  Total bars processed: {total_bars}")
    print(f"  Signals passing filter: {filtered_bars} ({filter_pass_rate:.1%})")
    print(f"  Signals skipped (gap < {min_gap}): {skipped_bars}")
    print(f"  Average gap (filtered): {avg_gap:.3f}")
    print()
    print("HIT-RATE (FILTERED ONLY):")
    print(f"  UP signals: {filtered_long_signals} → hit rate {filtered_hit_rate_up:.1%}")
    print(f"  DOWN signals: {filtered_short_signals} → hit rate {filtered_hit_rate_down:.1%}")
    print(f"  Combined: {combined_filtered_hit:.1%}")
    print()
    print("COMPARISON vs UNFILTERED BASELINE:")
    print(f"  Baseline UP:   46.0% → Filtered: {filtered_hit_rate_up:.1%} (Δ {delta_up:+.1%})")
    print(f"  Baseline DOWN: 46.6% → Filtered: {filtered_hit_rate_down:.1%} (Δ {delta_down:+.1%})")
    print(f"  Baseline Combined: ~46.3% → Filtered: {combined_filtered_hit:.1%}")
    print()
    print("VERDICT:")

    sufficient_data = filtered_bars >= 100
    threshold_55 = combined_filtered_hit >= 0.55
    threshold_50_55 = 0.50 <= combined_filtered_hit < 0.55
    threshold_below_50 = combined_filtered_hit < 0.50

    if threshold_55 and sufficient_data:
        print(f"  Filter improves hit-rate: YES")
        print(f"  Sufficient sample size (≥100 filtered signals): YES")
        print(f"  STRONG EDGE at high confidence. Use min-gap={min_gap} as entry gate.")
    elif threshold_50_55 and sufficient_data:
        print(f"  Filter improves hit-rate: MARGINAL")
        print(f"  Sufficient sample size (≥100 filtered signals): YES")
        print(f"  MARGINAL improvement. Consider raising gap to {min_gap + 0.05}.")
    elif threshold_below_50 and sufficient_data:
        print(f"  Filter improves hit-rate: NO")
        print(f"  Sufficient sample size (≥100 filtered signals): YES")
        print(f"  No edge even at high confidence. Rules produce biased but wrong signals.")
    else:
        print(f"  Filter improves hit-rate: INCONCLUSIVE")
        print(f"  Sufficient sample size (≥100 filtered signals): NO")
        print(f"  INSUFFICIENT DATA. Lower min-gap or increase --bars.")

    print()
    print("--- Computed baseline (from this run) ---")
    print(f"  Unfiltered UP rate: {baseline_up_rate:.1%} ({all_up_hits}/{all_total})")
    print(f"  Unfiltered DOWN rate: {baseline_down_rate:.1%} ({all_down_hits}/{all_total})")
    print(f"  Unfiltered Combined: {baseline_combined:.1%}")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    from io import StringIO
    _old_stdout = sys.stdout
    _buf = StringIO()
    sys.stdout = _buf
    try:
        main()
    finally:
        sys.stdout = _old_stdout
    output = _buf.getvalue()
    print(output, end="")
    # Также пишем в файл для чтения после выполнения
    _out_path = Path(__file__).resolve().parent / "filtered_output.txt"
    with open(str(_out_path), "w", encoding="utf-8") as _f:
        _f.write(output)