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

from app_pkg import config, db, utils
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
    """Свеча -> компактная строка БЕЗ запятых: [t o h l c v].

    Пробел вместо запятых — экономия ~15% символов (запятая+пробел
    часто дают отдельный токен у BPE-токенизаторов).
    """
    def _fmt(v):
        p = _compact_price(v, symbol)
        return "" if p is None else f"{p}"

    try:
        vol = float(c.get("volume"))
    except (TypeError, ValueError):
        vol = 0.0
    return (f"[{int(c['time'])} {_fmt(c['open'])} {_fmt(c['high'])} "
            f"{_fmt(c['low'])} {_fmt(c['close'])} {vol:g}]")


def _older_summary(df_older, symbol):
    """Строка-сводка по старым барам: high/low/trend вместо полного ряда."""
    if df_older is None or df_older.empty:
        return ""
    high = _compact_price(df_older["high"].max(), symbol)
    low = _compact_price(df_older["low"].min(), symbol)
    first = utils._clean(df_older["close"].iloc[0])
    last = utils._clean(df_older["close"].iloc[-1])
    trend = "flat"
    if first is not None and last is not None and last != first:
        trend = "up" if last > first else "down"
    return (f"older {len(df_older)} bars: high={high} low={low} "
            f"trend={trend}")


def format_tf_section(symbol, tf, main_tf, limit=None, upto_sec=None):
    """Секция одного таймфрейма (свечи + индикаторы).

    Главный ТФ: последние config.CONTEXT_CANDLES_MAIN свечей полностью,
    старшие бары — одна строка-сводка (_older_summary). Остальные ТФ:
    только последние config.CONTEXT_CANDLES_OTHER свечей. Индикаторы:
    последние config.CONTEXT_INDICATOR_TAIL значений каждого ряда.
    ВСЕГДА возвращает (str, upto_sec). Если данных нет — ("", upto_sec).
    """
    if tf == main_tf:
        max_candles = config.CONTEXT_CANDLES_MAIN
    else:
        max_candles = config.CONTEXT_CANDLES_OTHER
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

    summary = ""
    if len(df) > max_candles:
        # Старые бары (вне последних max_candles) — в одну строку-сводку
        # для главного ТФ; для остальных ТФ сводка не нужна.
        if tf == main_tf:
            summary = _older_summary(df.iloc[:-max_candles], symbol)
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
    if summary:
        lines.append(summary)
    lines.extend(_compact_candle(c, symbol) for c in candles)

    ind = compute_indicators(df)
    tail = config.CONTEXT_INDICATOR_TAIL
    vals = []
    for col in _IND_COLS:
        if col not in ind.columns:
            continue
        series = ind[col].dropna()
        if series.empty:
            continue
        pieces = " ".join(f"{v:.4g}" for v in series.iloc[-tail:])
        vals.append(f"{col}={pieces}")
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


def format_scanner_stats(symbol, timeframe) -> str:
    """Секция «=== Scanner stats ===» для промпта ИИ (symbol+timeframe).

    Единая точка правды по статистике сканера для агентов: лучший результат
    по combined_sharpe из scan_results + ЯВНАЯ вербальная рекомендация
    (доверять / не доверять сигналам стратегии). Пустая строка, если скана
    по этой паре ещё не было — секция просто не попадает в промпт.

    Используется build_multi_tf_context (=> Харон, чат, AI Backtest).
    """
    s = db.db_get_best_scan_stats(symbol, timeframe)
    if not s:
        return ""
    params = s.get("params") or {}
    params_str = (", ".join(f"{k}={v}" for k, v in sorted(params.items()))
                  if params else "default")
    combined = utils._clean(s.get("combined_sharpe")) or 0.0
    test_sharpe = utils._clean(s.get("sharpe")) or 0.0
    winrate = utils._clean(s.get("winrate"))
    if combined > 3:
        advice = (f"Доверяй сигналам {s.get('strategy')} "
                  f"(исторически очень надежная стратегия на этом активе).")
    elif test_sharpe < 0:
        advice = (f"НЕ доверяй сигналам {s.get('strategy')} — ищи уровень "
                  f"поддержки/сопротивления (стратегия убыточна на истории).")
    else:
        advice = (f"Сигналы {s.get('strategy')} учитывай как второстепенные: "
                  f"подтверждай структурой рынка.")
    lines = [f"=== Scanner stats ({symbol} {timeframe}) ==="]
    lines.append(
        f"Статистика сканера для {symbol} {timeframe}: "
        f"{s.get('strategy')} ({params_str}) даёт combined Sharpe "
        f"{round(float(combined), 2)}")
    if winrate is not None:
        lines.append(f"out-of-sample winrate={round(float(winrate) * 100, 1)}%")
    lines.append(advice)
    return "\n".join(lines)


def build_multi_tf_context(symbol, main_tf, upto_sec=None) -> str:
    """Собирает секции по всем таймфреймам, main помечает *** MAIN ***.

    Возвращает строку (пустую, если данных вообще нет).
    """
    sections = []
    for tf in config.TIMEFRAMES:
        # Токен-диета: младшие ТФ не тащим, если main_tf их «перекрывает»
        # (при main=15m свечи 1m/5m — шум). В Trends они остаются.
        if tf != main_tf and tf in config.CONTEXT_SKIP_TF:
            continue
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

    # Статистика сканера (лучший edge для symbol+main_tf): traders и ИИ
    # должны видеть, какие стратегии реально работают на этом активе.
    # Ровно те же данные, что в карточке «Статистика стратегий» во вкладке
    # «Новости» (GET /api/scanner/stats).
    scan_stats = format_scanner_stats(symbol, main_tf)
    if scan_stats:
        sections.append(scan_stats)

    return "\n\n".join(sections)