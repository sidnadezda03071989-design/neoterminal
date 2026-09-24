"""Активность правила 17 (trend mortality) на истории.

Правило 17: tr.adx_max_50 > 30 AND tr.adx_slope < -0.5 AND tr.adx > 20.

Прогон ~700 дней BTCUSDT 1h. Для каждого бара, где доступно >=50 значений
ADX, пересчитываем ровно те числа, что попадают в tr{} (adx, adx_slope,
adx_max_50), и проверяем условие. ADX-ряд причинный, поэтому расчёт на
полном ряду совпадает с оконным расчётом снимка.

Ориентиры: 5-15% (фильтр), >30% (агрессивно), <2% (мёртвое правило).

Запуск:
    python scripts/rule17_activity.py
    python scripts/rule17_activity.py --symbol BTCUSDT --timeframe 1h --days 700
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app_pkg import config
from app_pkg.data import market_snapshot as ms


def load_df(symbol: str, tf: str, days: int) -> pd.DataFrame:
    """OHLCV через get_replay_df (без live-мержа) за последние `days` суток."""
    from app_pkg.data.fetch import get_replay_df
    limit = config.HISTORY_LIMITS.get(tf, 20000)
    from_sec = int(time.time()) - days * 86400
    print(f"[fetch] {symbol}/{tf} ~{days}d limit={limit} ...")
    df = get_replay_df(symbol, tf, from_sec, None, limit=limit)
    if df is None or df.empty:
        raise SystemExit(f"нет данных {symbol}/{tf} (сеть?)")
    return df.sort_values("timestamp").reset_index(drop=True)


def rule17_fires(adx: float, adx_slope: float | None,
                 adx_max_50: float | None) -> bool:
    """Три условия правила 17 на готовых числах tr{}."""
    if adx is None or adx_slope is None or adx_max_50 is None:
        return False
    return adx_max_50 > 30 and adx_slope < -0.5 and adx > 20


def scan(df: pd.DataFrame) -> dict:
    """Прогон правила 17 по всем барам с >=50 значениями ADX."""
    high = pd.Series(df["high"], dtype="float64")
    low = pd.Series(df["low"], dtype="float64")
    close = pd.Series(df["close"], dtype="float64")
    _pdi, _mdi, adx_s = ms._di(high, low, close, 14)
    adx_arr = (pd.Series(adx_s, dtype="float64")
               .replace([np.inf, -np.inf], np.nan)
               .to_numpy(dtype="float64"))
    valid = adx_arr[~np.isnan(adx_arr)]

    total = fired = strong_peak = slope_dn = alive = 0
    for i in range(50, len(valid) + 1):
        vals = valid[:i]
        adx = float(vals[-1])
        adx_slope = ms._slope(vals, 10)
        adx_max_50 = float(max(vals[-50:]))
        total += 1
        peak_ok = adx_max_50 > 30
        slope_ok = adx_slope is not None and adx_slope < -0.5
        alive_ok = adx > 20
        strong_peak += int(peak_ok)
        slope_dn += int(slope_ok)
        alive += int(alive_ok)
        if peak_ok and slope_ok and alive_ok:
            fired += 1
    return {"total": total, "fired": fired, "strong_peak": strong_peak,
            "slope_dn": slope_dn, "alive": alive}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--days", type=int, default=700)
    args = p.parse_args()

    df = load_df(args.symbol, args.timeframe, args.days)
    res = scan(df)
    total, fired = res["total"], res["fired"]
    if total == 0:
        print("нет баров с >=50 значениями ADX — мало данных")
        return 1
    pct = 100.0 * fired / total

    print(f"Баров всего (ADX>=50): {total}")
    print(f"Сработало правило 17:  {fired} ({pct:.2f}%)")
    print(f"  adx_max_50>30:       {res['strong_peak']} "
          f"({100.0 * res['strong_peak'] / total:.1f}%)")
    print(f"  adx_slope<-0.5:      {res['slope_dn']} "
          f"({100.0 * res['slope_dn'] / total:.1f}%)")
    print(f"  adx>20:              {res['alive']} "
          f"({100.0 * res['alive'] / total:.1f}%)")

    if pct < 2.0:
        verdict = "МЁРТВОЕ (<2%) — крутить пороги"
    elif pct > 30.0:
        verdict = "АГРЕССИВНО (>30%) — крутить пороги"
    elif 5.0 <= pct <= 15.0:
        verdict = "ФИЛЬТР (5-15%) — ок"
    else:
        verdict = "ПРИЕМЛЕМО (2-30%) — ок"
    print(f"Вердикт: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
