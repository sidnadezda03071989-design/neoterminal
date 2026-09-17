# -*- coding: utf-8 -*-
"""Формирование контекста для ИИ: свечи, индикаторы, мульти-ТФ тренды.

Секции маркируются заголовками:
    === Candles <tf> [*** MAIN ***] ===
    === Indicators <tf> ===
    === Trends ===
Агенты в agents.py используют эти маркеры для разделения входных данных.
"""

import logging
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from app_pkg import config, utils
from app_pkg.data.fetch import get_series_df
from app_pkg.indicators import compute_indicators

log = logging.getLogger(__name__)

_IND_COLS = ("sma20", "sma50", "ema50", "rsi", "macd", "macd_signal",
             "bb_up", "bb_mid", "bb_low")


def _compact_price(x, symbol):
    """Округление цены по config.PRICE_PRECISION."""
    if x is None:
        return None
    prec = config.PRICE_PRECISION.get(symbol, 2)
    try:
        return round(float(x), prec)
    except (TypeError, ValueError):
        return None


def _compact_candle(c, symbol):
    """Свеча -> компактная строка С ЗАПЯТЫМИ: [t,o,h,l,c,v]."""
    def _fmt(v):
        p = _compact_price(v, symbol)
        return "" if p is None else f"{p}"

    try:
        vol = float(c.get("volume"))
    except (TypeError, ValueError):
        vol = 0.0
    return (f"[{int(c['time'])},{_fmt(c['open'])},{_fmt(c['high'])},"
            f"{_fmt(c['low'])},{_fmt(c['close'])},{vol:.1f}]")


def format_tf_section(symbol, tf, main_tf, limit=None, upto_sec=None):
    """Секция одного таймфрейма (свечи + индикаторы).

    ВСЕГДА возвращает (str, upto_sec). Если данных нет — ("", upto_sec).
    """
    max_candles = config.CONTEXT_TF_LIMITS.get(tf, 60)
    if limit is None:
        limit = max_candles * 2
    df = get_series_df(symbol, tf, limit=limit,
                       history_limit=config.TRENDS_HISTORY)
    if df is None or df.empty:
        return "", upto_sec
    if upto_sec is not None:
        mask = df["timestamp"] <= pd.to_datetime(int(upto_sec), unit="s", utc=True)
        df = df[mask]
        if df.empty:
            return "", upto_sec
    if len(df) > max_candles:
        df = df.iloc[-max_candles:]

    candles = []
    for _, r in df.iterrows():
        ts = pd.Timestamp(r["timestamp"])
        candles.append({
            "time": int(ts.timestamp()),
            "open": utils._clean(r["open"]),
            "high": utils._clean(r["high"]),
            "low": utils._clean(r["low"]),
            "close": utils._clean(r["close"]),
            "volume": utils._clean(r["volume"]),
        })
    new_upto = candles[-1]["time"]

    if tf == main_tf:
        header = f"=== Candles {tf} *** MAIN *** ==="
    else:
        header = f"=== Candles {tf} ==="
    lines = [header]
    lines.extend(_compact_candle(c, symbol) for c in candles)

    ind = compute_indicators(df)
    last = ind.iloc[-1]
    vals = []
    for col in _IND_COLS:
        v = last.get(col)
        if v is not None and not (isinstance(v, float) and pd.isna(v)):
            vals.append(f"{col}={v:.4g}")
    if vals:
        lines.append(f"=== Indicators {tf} ===")
        lines.extend(vals)

    return "\n".join(lines), new_upto


def _compute_multi_tf_trends(symbol, main_tf, upto_sec=None):
    """dict {tf: {trend, last_price, sma20, sma50, rsi14}} по всем ТФ.

    Параллельно через ThreadPoolExecutor(max_workers=config.CONTEXT_WORKERS).
    upto_sec: если задан (replay), каждый ТФ обрезается по времени
    timestamp <= upto_sec ДО compute_indicators — тренды не видят будущее.
    """
    out = {}

    def _calc(tf):
        df = get_series_df(symbol, tf, limit=100,
                           history_limit=config.TRENDS_HISTORY)
        if upto_sec is not None and df is not None and not df.empty:
            mask = df["timestamp"] <= pd.to_datetime(int(upto_sec), unit="s", utc=True)
            df = df[mask]
        if df is None or len(df) < 20:
            return tf, None
        ind = compute_indicators(df)
        last_close = utils._clean(df["close"].iloc[-1])
        sma20 = utils._clean(ind["sma20"].iloc[-1])
        sma50 = utils._clean(ind["sma50"].iloc[-1])
        rsi14 = utils._clean(ind["rsi"].iloc[-1]) if "rsi" in ind.columns else None
        if last_close is None or sma20 is None or last_close == sma20:
            trend = "sideways"
        elif last_close > sma20:
            trend = "up"
            if sma50 is not None and sma20 < sma50:
                trend = "sideways"
        else:
            trend = "down"
            if sma50 is not None and sma20 > sma50:
                trend = "sideways"
        return tf, {
            "trend": trend,
            "last_price": last_close,
            "sma20": sma20,
            "sma50": sma50,
            "rsi14": rsi14,
        }

    with ThreadPoolExecutor(max_workers=config.CONTEXT_WORKERS) as pool:
        futures = [pool.submit(_calc, tf) for tf in config.TIMEFRAMES]
        for fut in futures:
            tf, res = fut.result()
            if res:
                out[tf] = res
    return out


def build_multi_tf_context(symbol, main_tf, upto_sec=None) -> str:
    """Собирает секции по всем таймфреймам, main помечает *** MAIN ***.

    Возвращает строку (пустую, если данных вообще нет).
    """
    sections = []
    for tf in config.TIMEFRAMES:
        section, _ = format_tf_section(
            symbol, tf, main_tf, limit=None, upto_sec=upto_sec)
        if section:
            sections.append(section)

    trends = _compute_multi_tf_trends(symbol, main_tf, upto_sec)
    if trends:
        t_lines = ["=== Trends ==="]
        for tf, t in trends.items():
            t_lines.append(
                f"{tf}: trend={t['trend']} last_price={t['last_price']} "
                f"sma20={t['sma20']} sma50={t['sma50']} rsi14={t['rsi14']}")
        sections.append("\n".join(t_lines))

    return "\n\n".join(sections)