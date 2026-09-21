# -*- coding: utf-8 -*-
"""Сырые рыночные данные для ИИ (Market Snapshot).

Единственный источник правды для нейросети: НИКАКОГО текста, только числа.
Нейросеть не получает «словесное описание рынка» — она получает этот JSON
и делает выводы по формулам системного промпта.

get_raw_market_data(symbol, timeframe) -> dict:
    technicals   — RSI(14), ATR(14), BB %B, SMA20 diff % по последним свечам;
    scanner_edge — лучшая комбинация сканера (winrate/sharpe/max_dd/params);
    sentiment    — заглушка (нет API) {ls_ratio, long_pct, fear_greed}.
"""

import logging

import pandas as pd

from app_pkg import config, db, utils
from app_pkg.data.fetch import get_series_df
from app_pkg.indicators import _atr, rsi_wilder

log = logging.getLogger(__name__)

# Периоды индикаторов (фиксированные — единый контракт для ИИ).
_RSI_PERIOD = 14
_ATR_PERIOD = 14
_BB_PERIOD = 20
_BB_STD = 2.0
_SMA_PERIOD = 20
# Свечей на расчёт: ~стабильный хвост RSI/BB/SMA без "прогрева" первых баров.
_LOOKBACK = 300


def _round(v, digits=6):
    """float-значение или None (NaN/Inf/нечисло не должны попасть в JSON)."""
    v = utils._clean(v)
    return round(v, digits) if v is not None else None


def _technicals(symbol, timeframe, upto_sec=None):
    """Индикаторы по последним свечам: RSI, ATR, BB %B, SMA20 diff %.

    upto_sec — барьер реплея: ряд обрезается по времени, чтобы сырой JSON
    НИКОГДА не содержал свечей из будущего (никакого look-ahead для ИИ).
    Все значения — числа (None, если данных меньше необходимого минимума).
    """
    df = get_series_df(symbol, timeframe, limit=_LOOKBACK)
    if df is None or df.empty:
        return {
            "rsi": None, "atr": None, "bb_pct_b": None,
            "sma20_diff_pct": None, "close": None,
        }
    if upto_sec is not None:
        mask = df["timestamp"] <= pd.to_datetime(int(upto_sec), unit="s", utc=True)
        df = df[mask]
        if df.empty:
            return {
                "rsi": None, "atr": None, "bb_pct_b": None,
                "sma20_diff_pct": None, "close": None,
            }
    if len(df) < max(_RSI_PERIOD, _BB_PERIOD, _SMA_PERIOD) + 5:
        return {
            "rsi": None, "atr": None, "bb_pct_b": None,
            "sma20_diff_pct": None, "close": None,
        }
    close = pd.Series(df["close"], dtype="float64")
    last_close = _round(close.iloc[-1], 8)

    rsi = _round(rsi_wilder(close, _RSI_PERIOD).iloc[-1], 4)
    atr = _round(_atr(df["high"], df["low"], df["close"],
                      _ATR_PERIOD).iloc[-1], 8)

    sma20 = close.rolling(_SMA_PERIOD).mean()
    bb_mid = sma20
    bb_std = close.rolling(_BB_PERIOD).std()
    bb_up = bb_mid + _BB_STD * bb_std
    bb_low = bb_mid - _BB_STD * bb_std
    bb_range = utils._clean((bb_up - bb_low).iloc[-1])
    bb_pct_b = None
    if bb_range:
        bb_pct_b = _round((last_close - bb_low.iloc[-1]) / bb_range, 4)

    sma20_last = utils._clean(sma20.iloc[-1])
    sma20_diff_pct = None
    if sma20_last and last_close:
        sma20_diff_pct = _round((last_close - sma20_last) / sma20_last * 100.0,
                                4)

    return {
        "rsi": rsi, "atr": atr, "bb_pct_b": bb_pct_b,
        "sma20_diff_pct": sma20_diff_pct, "close": last_close,
    }


def _scanner_edge(symbol, timeframe):
    """Лучшая комбинация сканера: только цифры (как в Reference Data).

    winrate/max_dd — out-of-sample (test) окно; None, если скана не было.
    """
    s = db.db_get_best_scan_stats(symbol, timeframe)
    if not s:
        return {"winrate": None, "sharpe": None, "max_dd": None, "params": {}}
    test = s.get("test") or {}
    params = {}
    for k, v in (s.get("params") or {}).items():
        v = utils._clean(v)
        if v is None:
            params[str(k)] = None
        else:
            params[str(k)] = v
    return {
        "winrate": _round(test.get("winrate"), 4),
        "sharpe": _round(s.get("sharpe"), 4),
        "max_dd": _round(test.get("max_dd"), 4),
        "params": params,
    }


def get_raw_market_data(symbol, timeframe, upto_sec=None):
    """Чистый словарь с цифрами для ИИ (без единого текстового пояснения).

    upto_sec — барьер реплея (срез без взгляда в будущее); None в live.
    Поддерживает любой symbol/timeframe из config; при неизвестной паре
    технические показатели уходят в None, а не в ошибку — ИИ получает
    структуру всегда.
    """
    symbol = str(symbol or "").upper()
    timeframe = str(timeframe or "").strip()
    return {
        "technicals": _technicals(symbol, timeframe, upto_sec),
        "scanner_edge": _scanner_edge(symbol, timeframe),
        "sentiment": {"ls_ratio": 0, "long_pct": 0, "fear_greed": 50},
    }