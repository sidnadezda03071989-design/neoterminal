"""Volume Profile: POC/VAH/VAL для адаптивных TP/SL барьеров.

Чистая функция `calculate_vp` над OHLCV-DataFrame: биннинг по типичной
цене (high+low+close)/3. Никаких сетевых вызовов и записи в БД.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app_pkg.indicators import _atr

_MIN_BARS = 50


def _last_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
              period: int = 14) -> float | None:
    """Последнее значение ATR(period) (Wilder) или None."""
    atr = _atr(pd.Series(high), pd.Series(low), pd.Series(close), period)
    if atr is None or len(atr) == 0:
        return None
    value = atr.iloc[-1]
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value) or value <= 0.0:
        return None
    return value


def calculate_vp(df: pd.DataFrame, num_bins: int = 50) -> dict:
    """Volume Profile по типичной цене.

    - POC  — бин максимального накопленного объёма (цена центра бина);
    - VAH/VAL — границы 70% value area, расширяемой от POC наружу к
      соседу с БОЛЬШИМ объёмом (стандартная процедура value area);
    - HVN — бины с объёмом >= 1.5x среднего, LVN — 0 < объём <= 0.5x
      среднего (пустые бины не считаются узлами);
    - hvn_count_above/lvn_count_below — счётчики узлов выше/ниже POC;
    - dist_to_poc_atr = (close - poc) / ATR(14);
    - in_value_area = val <= close <= vah.

    Пустой {} при <50 барах, нулевом диапазоне цены или нулевом объёме.
    Возвращает только float/int/bool/None (JSON-чисто).
    """
    if df is None or len(df) < _MIN_BARS:
        return {}
    try:
        high = pd.to_numeric(df["high"], errors="coerce").to_numpy(dtype=float)
        low = pd.to_numeric(df["low"], errors="coerce").to_numpy(dtype=float)
        close = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=float)
        volume = pd.to_numeric(df["volume"], errors="coerce").to_numpy(dtype=float)
    except (KeyError, TypeError, ValueError):
        return {}

    mask = ~(np.isnan(high) | np.isnan(low) | np.isnan(close)
             | np.isnan(volume))
    if int(mask.sum()) < _MIN_BARS:
        return {}
    high, low, close, volume = high[mask], low[mask], close[mask], volume[mask]

    tp = (high + low + close) / 3.0
    price_min = float(tp.min())
    price_max = float(tp.max())
    price_range = price_max - price_min
    total_volume = float(volume.sum())
    if price_range <= 0.0 or total_volume <= 0.0 or num_bins < 2:
        return {}

    edges = np.linspace(price_min, price_max, num_bins + 1)
    idx = np.clip(np.searchsorted(edges, tp, side="right") - 1, 0,
                  num_bins - 1)
    bins = np.zeros(num_bins, dtype=float)
    np.add.at(bins, idx, volume)
    centers = (edges[:-1] + edges[1:]) / 2.0

    poc_idx = int(np.argmax(bins))
    poc = float(centers[poc_idx])

    # 70% value area: расширяемся от POC к соседу с большим объёмом
    target = total_volume * 0.70
    lo = hi = poc_idx
    acc = float(bins[poc_idx])
    while acc < target and (lo > 0 or hi < num_bins - 1):
        left = float(bins[lo - 1]) if lo > 0 else -1.0
        right = float(bins[hi + 1]) if hi < num_bins - 1 else -1.0
        if right > left:
            hi += 1
            acc += float(bins[hi])
        elif left >= 0.0:
            lo -= 1
            acc += float(bins[lo])
        else:
            hi += 1
            acc += float(bins[hi])
    vah = float(centers[hi])
    val = float(centers[lo])

    # HVN/LVN узлы (пустые бины — не узлы)
    mean_vol = total_volume / num_bins
    hvn = bins >= 1.5 * mean_vol
    lvn = (bins > 0.0) & (bins <= 0.5 * mean_vol)
    hvn_count_above = int(np.sum(hvn & (centers > poc)))
    lvn_count_below = int(np.sum(lvn & (centers < poc)))

    last_close = float(close[-1])
    atr = _last_atr(high, low, close)
    dist_to_poc_atr = (last_close - poc) / atr if atr is not None else None

    return {
        "poc": poc,
        "vah": vah,
        "val": val,
        "hvn_count_above": hvn_count_above,
        "lvn_count_below": lvn_count_below,
        "dist_to_poc_atr": dist_to_poc_atr,
        "in_value_area": bool(val <= last_close <= vah),
    }