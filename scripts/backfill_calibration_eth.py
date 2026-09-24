"""Offline backfill: apply_all_rules на N барах ETHUSDT/TF -> charon_calibration_history.

Записывает 2 строки на бар (side=UP raw_prob=pu_raw, side=DOWN raw_prob=pd_raw),
hit по triple-barrier исходу (конвенция labels/backfill: SL-приоритет, atr_k, horizon).

Правила 21-24 (дерривативы) активны: _derivatives_block вызывается из
compact_snapshot. Статистика срабатываний правил печатается в отчёте.

Фетч один раз в память; market_snapshot.get_replay_df подменяется, чтобы
compact_snapshot(upto_sec=ts) работал по одному df без сети.
"""
import argparse
import importlib
import sqlite3
import sys
from pathlib import Path

import pandas as pd

_BASE = Path(__file__).resolve().parent.parent
if str(_BASE) not in sys.path:
    sys.path.insert(0, str(_BASE))

from app_pkg import config, db                      # noqa: E402
from app_pkg.ai.apply_rules import apply_all_rules  # noqa: E402
from app_pkg.data import market_snapshot as ms      # noqa: E402
from app_pkg.data.fetch import get_series_df        # noqa: E402

ATR_K = 1.5          # конвенция CBR_BACKFILL_ATR_K
HORIZON = 20         # конвенция CBR_BACKFILL_HORIZON

# Один сетевой фетч на ТФ в память.
_CACHE: dict[tuple, pd.DataFrame] = {}


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
        return None  # барьер вырожден
    upper = entry * (1.0 + atr_k * atr_pct)
    lower = entry * (1.0 - atr_k * atr_pct)
    j0 = entry_idx + 1
    jend = j0 + horizon
    if jend > len(close):
        return None  # не хватает будущих баров
    for k in range(j0, jend):
        if low[k] <= lower:
            return -1  # SL-приоритет
        if high[k] >= upper:
            return 1
    return 0  # таймаут


def main():
    parser = argparse.ArgumentParser(description="Charon calibration backfill")
    parser.add_argument("--symbol", default="ETHUSDT", help="Trading pair (default ETHUSDT)")
    parser.add_argument("--tf", default="15m", help="Timeframe (default 15m)")
    parser.add_argument("--bars", type=int, default=1000, help="Number of bars (default 1000)")
    parser.add_argument("--no-derivatives", action="store_true",
                        help="Disable derivatives fetch (rules 21-24)")
    args = parser.parse_args()

    symbol = args.symbol.upper()
    tf = args.tf
    n_bars = args.bars

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

    # Кэшируем медленные внешние блоки: они почти не меняются бар-к-бару
    # и делают compact_snapshot основным источником лагов.
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

    df = _CACHE[(symbol, tf)]
    n = len(df)
    high = df["high"].astype(float).to_numpy(dtype="float64")
    low = df["low"].astype(float).to_numpy(dtype="float64")
    close = df["close"].astype(float).to_numpy(dtype="float64")
    ts = [_bar_ts_sec(x) for x in df["timestamp"]]

    # Точный atr_pct = ATR(14)/close по каждому бару.
    from app_pkg.indicators import _atr
    atr_s = _atr(pd.Series(high), pd.Series(low), pd.Series(close), 14)
    atr_pct_arr = (atr_s.to_numpy(dtype="float64") / close)
    print(f"[preload] {symbol}/{tf} bars={n} (analyzable window up to {n - HORIZON})")

    # Прогрев: компакт-снимку нужны ~300 баров до барьера (_LOOKBACK).
    start = 340
    end = n - HORIZON
    if start >= end:
        print(f"мало баров: n={n}")
        return

    entries, outcomes = [], []
    written = 0
    # Статистика правил 21-24
    rules_activity: dict[int, int] = {21: 0, 22: 0, 23: 0, 24: 0}
    none_counts: dict[str, int] = {"funding": 0, "oi": 0, "cvd": 0}

    for i in range(start, min(end, start + n_bars)):
        bar_ts = ts[i]
        try:
            snap = ms.compact_snapshot(symbol, tf, upto_sec=float(bar_ts))
            verdict = apply_all_rules(snap)
        except Exception as exc:  # noqa: BLE001
            print(f"bar {i} ts={bar_ts} snapshot fail: {exc!r}")
            continue

        # Статистика правил 21-24
        fired = verdict.get("fired", [])
        for rn in (21, 22, 23, 24):
            if rn in fired:
                rules_activity[rn] = rules_activity.get(rn, 0) + 1

        # Статистика None в d-блоке
        d_block = snap.get("d", {}) if isinstance(snap.get("d"), dict) else {}
        if d_block.get("funding_rate") is None:
            none_counts["funding"] += 1
        if d_block.get("oi") is None:
            none_counts["oi"] += 1
        if d_block.get("cvd") is None:
            none_counts["cvd"] += 1

        raw = verdict.get("raw") or {}
        pu_raw = raw.get("pu")
        pd_raw = raw.get("pd")
        if pu_raw is None and pd_raw is None:
            continue
        entry_price = float(close[i])
        atr_pct_v = float(atr_pct_arr[i]) if atr_pct_arr[i] is not None \
            else None
        outcome = triple_barrier(high, low, close, entry_price, atr_pct_v,
                                 i, HORIZON, ATR_K)
        if outcome is None:
            continue
        hit_up = 1 if outcome == 1 else 0
        hit_down = 1 if outcome == -1 else 0
        if pu_raw is not None:
            db.db_save_calibration_entry(bar_ts, symbol, tf, float(pu_raw),
                                         "UP", hit_up, "{}")
            written += 1
        if pd_raw is not None:
            db.db_save_calibration_entry(bar_ts, symbol, tf, float(pd_raw),
                                         "DOWN", hit_down, "{}")
            written += 1
        entries.append(bar_ts)
        outcomes.append(outcome)

    # Перечитываем кривые (сброс кэша TTL).
    from app_pkg.ai.calibration import _load_curves
    _load_curves(force=True)

    c = sqlite3.connect(str(config.DB_PATH))
    row = c.execute(
        "SELECT side, COUNT(*) n, SUM(hit) h FROM charon_calibration_history "
        "WHERE symbol=? AND timeframe=? GROUP BY side", (symbol, tf)).fetchall()
    c.close()
    counts = {r[0]: (r[1], r[2]) for r in row}

    from app_pkg.ai.calibration import CALIBRATOR
    curves = CALIBRATOR.export_curves()

    # --- REPORT ---
    total_bars = len(entries)
    up_hit_rate = (counts.get("UP", (0, 0))[1] or 0) / max(1, counts.get("UP", (0, 0))[0])
    down_hit_rate = (counts.get("DOWN", (0, 0))[1] or 0) / max(1, counts.get("DOWN", (0, 0))[0])

    print("\n" + "=" * 60)
    print("=== BACKFILL REPORT ===")
    print("=" * 60)
    print(f"Symbol/TF: {symbol} {tf}  bars_analyzed={total_bars}  written={written}")
    print(f"outcome balance: +1:{outcomes.count(1)} -1:{outcomes.count(-1)} "
          f"0:{outcomes.count(0)}")
    for side in ("UP", "DOWN"):
        n_rows, hits = counts.get(side, (0, 0))
        print(f"side={side}: rows={n_rows} hits={hits} "
              f"hit_rate={(hits or 0) / max(1, n_rows):.3f}")
    active = {s: bool(v) for s, v in curves.items()}
    print("calibration active (UP/DOWN):", active)
    print("curve nodes:", {s: len(v) for s, v in curves.items()})
    print()
    print("--- Derivatives Rules 21-24 Activity ---")
    print(f"  Rule 21 (funding extreme): fired {rules_activity.get(21, 0)}/{total_bars} bars")
    print(f"  Rule 22 (OI divergence):   fired {rules_activity.get(22, 0)}/{total_bars} bars")
    print(f"  Rule 23 (CVD trend):       fired {rules_activity.get(23, 0)}/{total_bars} bars")
    print(f"  Rule 24 (taker imbalance): fired {rules_activity.get(24, 0)}/{total_bars} bars")
    print(f"  Fields None count: funding={none_counts['funding']}, "
          f"OI={none_counts['oi']}, CVD={none_counts['cvd']} (out of {total_bars} bars)")
    print()
    print(f"Hit-rate UP: {up_hit_rate:.4f} (baseline 46.0%)")
    print(f"Hit-rate DOWN: {down_hit_rate:.4f} (baseline 46.6%)")
    delta_up = up_hit_rate - 0.46
    delta_down = down_hit_rate - 0.466
    print(f"Delta UP: {delta_up:+.4f}")
    print(f"Delta DOWN: {delta_down:+.4f}")
    combined_hit = (up_hit_rate + down_hit_rate) / 2.0
    print(f"Combined hit rate: {combined_hit:.4f} (baseline {0.463:.4f})")
    success = combined_hit >= 0.52
    print(f"Success criterion (>=52%): {'MET' if success else 'NOT MET'}")
    if success:
        print("Recommendation: Proceed to ensemble")
    else:
        print("Recommendation: Try different TF / Investigate data gaps")


if __name__ == "__main__":
    main()