"""Offline backfill: apply_all_rules на 1000 барах ETHUSDT/15m -> charon_calibration_history.

Записывает 2 строки на бар (side=UP raw_prob=pu_raw, side=DOWN raw_prob=pd_raw),
hit по triple-barrier исходу (конвенция labels/backfill: SL-приоритет, atr_k, horizon).

Фетч один раз в память; market_snapshot.get_replay_df подменяется, чтобы
compact_snapshot(rules_only=True, upto_sec=ts) работал по одному df без сети.
"""
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

SYMBOL = "ETHUSDT"
TF = "15m"
ATR_K = 1.5          # конвенция CBR_BACKFILL_ATR_K
HORIZON = 20         # конвенция CBR_BACKFILL_HORIZON
N_BARS = 1000        # сколько баров размечаем

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

    _preload(TF, config.HISTORY_LIMITS.get(TF, 20000))
    for tf in ("1H", "4H", "1D"):
        _preload(tf, 300)

    # Подмена фетча: компакт-снимок для реплея читает из памяти (без сети).
    mod = importlib.import_module("app_pkg.data.market_snapshot")
    mod.get_replay_df = _fake_replay

    _real_scanner = mod._scanner_edge
    _real_crowd = mod._resolve_crowd

    # Кэшируем медленные внешние блоки: они почти не меняются бар-к-бару
    # и делают compact_snapshot(rules_only) основным источником лагов.
    _S = {}
    def _scan_edge_cached(symbol, timeframe):
        if "scanner" not in _S:
            _S["scanner"] = _real_scanner(symbol, timeframe)
        return _S["scanner"]

    def _crowd_cached(symbol):
        if "crowd" not in _S:
            _S["crowd"] = _real_crowd(symbol)
        return _S["crowd"]

    _real_extra = mod._build_extra_blocks
    _S_extra = {}
    def _extra_cached(symbol, clock_ts, names=None):
        key = tuple(names) if names is not None else "all"
        if key not in _S_extra:
            _S_extra[key] = _real_extra(symbol, clock_ts, names)
        return _S_extra[key]

    mod._scanner_edge = _scan_edge_cached
    mod._resolve_crowd = _crowd_cached
    mod._build_extra_blocks = _extra_cached

    df = _CACHE[(SYMBOL, TF)]
    n = len(df)
    high = df["high"].astype(float).to_numpy(dtype="float64")
    low = df["low"].astype(float).to_numpy(dtype="float64")
    close = df["close"].astype(float).to_numpy(dtype="float64")
    ts = [_bar_ts_sec(x) for x in df["timestamp"]]

    # Точный atr_pct = ATR(14)/close по каждому бару (НЕ 2dp-округлённый из
    # компакт-снимка): барьеры triple-barrier должны строиться по той же
    # точности, что и CBR backfill (convention labels/backfill).
    from app_pkg.indicators import _atr
    atr_s = _atr(pd.Series(high), pd.Series(low), pd.Series(close), 14)
    atr_pct_arr = (atr_s.to_numpy(dtype="float64") / close)
    print(f"[preload] ETHUSDT/15m bars={n} (analyzable window up to {n - HORIZON})")

    # Прогрев: компакт-снимку нужны ~300 баров до барьера (_LOOKBACK).
    start = 340
    end = n - HORIZON
    if start >= end:
        print(f"мало баров: n={n}")
        return

    entries, outcomes = [], []
    written = 0
    for i in range(start, min(end, start + N_BARS)):
        bar_ts = ts[i]
        try:
            snap = ms.compact_snapshot(SYMBOL, TF, upto_sec=float(bar_ts),
                                       rules_only=True)
            verdict = apply_all_rules(snap)
        except Exception as exc:  # noqa: BLE001
            print(f"bar {i} ts={bar_ts} snapshot fail: {exc!r}")
            continue
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
            db.db_save_calibration_entry(bar_ts, SYMBOL, TF, float(pu_raw),
                                         "UP", hit_up, "{}")
            written += 1
        if pd_raw is not None:
            db.db_save_calibration_entry(bar_ts, SYMBOL, TF, float(pd_raw),
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
        "WHERE symbol=? AND timeframe=? GROUP BY side", (SYMBOL, TF)).fetchall()
    c.close()
    counts = {r[0]: (r[1], r[2]) for r in row}

    from app_pkg.ai.calibration import CALIBRATOR
    curves = CALIBRATOR.export_curves()

    print("\n=== BACKFILL REPORT ===")
    print(f"Symbol/TF: {SYMBOL} {TF}  bars_analyzed={len(entries)}  written={written}")
    print(f"outcome balance: +1:{outcomes.count(1)} -1:{outcomes.count(-1)} "
          f"0:{outcomes.count(0)}")
    for side in ("UP", "DOWN"):
        n_rows, hits = counts.get(side, (0, 0))
        print(f"side={side}: rows={n_rows} hits={hits} "
              f"hit_rate={(hits or 0) / max(1, n_rows):.3f}")
    active = {s: bool(v) for s, v in curves.items()}
    print("calibration active (UP/DOWN):", active)
    print("curve nodes:", {s: len(v) for s, v in curves.items()})


if __name__ == "__main__":
    main()