"""Выбор 3 баров для smoke-теста правила 17 (Phase 2).

Критерии (BTCUSDT 1h, последние N дней):
  A_trend  — явный тренд:      adx>30 AND adx_slope>=0 AND adx_max_50<=35
                                -> правило 17 НЕ срабатывает;
  B_flat   — флэт:             adx<20 AND adx_max_50<25
                                -> правило 17 НЕ срабатывает;
  C_dying  — умирающий тренд:  adx_max_50>35 AND 20<adx<28 AND adx_slope<-1.0
                                -> правило 17 СРАБАТЫВАЕТ.

Для каждого бара берётся ПЕРВЫЙ подходящий по времени. Если C не найден за
90 дней — окно расширяется до 180, затем пороги ослабляются до
(adx_max_50>30, adx_slope<-0.7); причина печатается в отчёт.

Результат: data/phase2_bars.json
Запуск:
    python scripts/pick_3_bars.py
    python scripts/pick_3_bars.py --symbol BTCUSDT --timeframe 1h --days 90
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app_pkg import config
from app_pkg.data import market_snapshot as ms

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(REPO, "data", "phase2_bars.json")


def load_df(symbol: str, tf: str, days: int) -> pd.DataFrame:
    from app_pkg.data.fetch import get_replay_df
    limit = config.HISTORY_LIMITS.get(tf, 20000)
    from_sec = int(time.time()) - days * 86400
    print(f"[fetch] {symbol}/{tf} ~{days}d limit={limit} ...")
    df = get_replay_df(symbol, tf, from_sec, None, limit=limit)
    if df is None or df.empty:
        raise SystemExit(f"нет данных {symbol}/{tf} (сеть?)")
    return df.sort_values("timestamp").reset_index(drop=True)


def bar_rows(df: pd.DataFrame) -> pd.DataFrame:
    """По каждому бару: adx, adx_slope (10), adx_max_50 — как в tr{}."""
    high = pd.Series(df["high"], dtype="float64")
    low = pd.Series(df["low"], dtype="float64")
    close = pd.Series(df["close"], dtype="float64")
    _pdi, _mdi, adx_s = ms._di(high, low, close, 14)
    adx_arr = (pd.Series(adx_s, dtype="float64")
               .replace([np.inf, -np.inf], np.nan)
               .to_numpy(dtype="float64"))
    frame = pd.DataFrame({"timestamp": df["timestamp"], "adx": adx_arr}).dropna()
    valid = frame["adx"].to_numpy(dtype="float64")
    rows = []
    for i in range(50, len(valid) + 1):
        vals = valid[:i]
        rows.append({
            "timestamp": int(pd.Timestamp(frame["timestamp"].iloc[i - 1]).timestamp()),
            "adx": round(float(vals[-1]), 2),
            "adx_slope": ms._slope(vals, 10),
            "adx_max_50": round(float(max(vals[-50:])), 2),
        })
    return pd.DataFrame(rows)


def rule17_fires(row) -> bool:
    return (row["adx_max_50"] > 30 and row["adx_slope"] is not None
            and row["adx_slope"] < -0.5 and row["adx"] > 20)


def pick(rows: pd.DataFrame, name: str, cond, require_fire: bool):
    """Первый по времени бар, удовлетворяющий cond (+ правило 17 как задано)."""
    for _, row in rows.iterrows():
        if cond(row) and rule17_fires(row) == require_fire:
            return row
    return None


def find_bars(rows: pd.DataFrame) -> dict:
    a = pick(rows, "A",
             lambda r: r["adx"] > 30 and (r["adx_slope"] or 0) >= 0
             and r["adx_max_50"] <= 35, require_fire=False)
    b = pick(rows, "B",
             lambda r: r["adx"] < 20 and r["adx_max_50"] < 25,
             require_fire=False)
    c = pick(rows, "C",
             lambda r: r["adx_max_50"] > 35 and 20 < r["adx"] < 28
             and r["adx_slope"] is not None and r["adx_slope"] < -1.0,
             require_fire=True)
    return {"A_trend": a, "B_flat": b, "C_dying_trend": c}


def _row_dict(row) -> dict | None:
    if row is None:
        return None
    return {"ts": int(row["timestamp"]), "adx": float(row["adx"]),
            "adx_slope": (None if row["adx_slope"] is None
                          else float(row["adx_slope"])),
            "adx_max_50": float(row["adx_max_50"]),
            "rule17": bool(rule17_fires(row))}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--days", type=int, default=90)
    args = p.parse_args()

    note = ""
    bars = {}
    for days in (args.days, 180):
        df = load_df(args.symbol, args.timeframe, days)
        rows = bar_rows(df)
        print(f"[scan] {len(rows)} баров с >=50 ADX за {days}d")
        bars = find_bars(rows)
        if bars["C_dying_trend"] is not None:
            note = f"окно {days}d"
            break
        note = f"C не найден за {days}d"
    if bars.get("C_dying_trend") is None:
        # ослабление порогов (сообщаем в отчёте)
        df = load_df(args.symbol, args.timeframe, 180)
        rows = bar_rows(df)
        c = pick(rows, "C",
                 lambda r: r["adx_max_50"] > 30 and 20 < r["adx"] < 28
                 and r["adx_slope"] is not None and r["adx_slope"] < -0.7,
                 require_fire=True)
        bars["C_dying_trend"] = c
        note = "C найден с ослабленными порогами (max50>30, slope<-0.7) за 180d"

    result = {"symbol": args.symbol, "timeframe": args.timeframe,
              "window_days": args.days, "note": note,
              "bars": {k: _row_dict(v) for k, v in bars.items()}}
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(f"[note] {note}")
    for k, v in result["bars"].items():
        if v is None:
            print(f"  {k}: НЕ НАЙДЕН")
        else:
            print(f"  {k}: ts={v['ts']} adx={v['adx']} "
                  f"slope={v['adx_slope']} max50={v['adx_max_50']} "
                  f"rule17={v['rule17']}")
    print(f"[saved] {OUT_PATH}")
    missing = [k for k, v in result["bars"].items() if v is None]
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
