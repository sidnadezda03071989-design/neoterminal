# -*- coding: utf-8 -*-
"""Blueprint: /api/data, /api/last-bar, /api/replay-data, /api/watchlist."""

import logging
from concurrent.futures import ThreadPoolExecutor

from flask import Blueprint, jsonify, request

from app_pkg import config, utils
from app_pkg.data.fetch import get_series_df, get_replay_df
from app_pkg.indicators import compute_indicators, df_to_payload

bp = Blueprint("data", __name__)
log = logging.getLogger(__name__)


@bp.route("/api/data")
def api_data():
    symbol = request.args.get("symbol", "BTCUSDT").upper()
    tf = request.args.get("timeframe", config.DEFAULT_TIMEFRAME)
    try:
        limit = int(request.args.get("limit", config.INITIAL_LOAD_CANDLES))
    except (TypeError, ValueError):
        limit = config.INITIAL_LOAD_CANDLES

    if symbol not in config.SYMBOLS:
        return jsonify({"error": "Invalid symbol"}), 400
    if tf not in config.TF_SECONDS:
        return jsonify({"error": "Invalid timeframe"}), 400
    if limit < 1 or limit > config.MAX_DATA_LIMIT:
        return jsonify({"error": f"limit must be 1..{config.MAX_DATA_LIMIT}"}), 400

    # history_limit=limit: холодный старт тянет только запрошенное (1 страница
    # Binance на limit=1000), а повторный запрос с бОльшим limit ДОзагружает
    # недостающую историю в тот же кеш вместо полной перезагрузки.
    df = get_series_df(symbol, tf, limit=limit, history_limit=limit)
    candles, indicators, last_price = df_to_payload(df)
    return jsonify({
        "symbol": symbol,
        "timeframe": tf,
        "candles": candles,
        "indicators": indicators,
        "last_price": last_price,
        "ts": utils.now_sec(),
    })


@bp.route("/api/last-bar")
def api_last_bar():
    """Лёгкий поллинг: последний бар + свежие индикаторы последнего бара.

    Внутри get_series_df(symbol, tf, limit=LAST_BAR_HISTORY) (tail из кеша),
    по нему compute_indicators() -> значения последней строки (NaN -> None).
    Полный пересчёт всего графика не делается — только скаляры для фронта.
    """
    symbol = request.args.get("symbol", "BTCUSDT").upper()
    tf = request.args.get("timeframe", config.DEFAULT_TIMEFRAME)

    if symbol not in config.SYMBOLS:
        return jsonify({"error": "Invalid symbol"}), 400
    if tf not in config.TF_SECONDS:
        return jsonify({"error": "Invalid timeframe"}), 400

    df = get_series_df(symbol, tf, limit=config.LAST_BAR_HISTORY)
    candle = None
    indicators = None
    if df is not None and not df.empty:
        ts = utils.epoch_secs(df["timestamp"].iloc[-1:])
        last_row = df.iloc[-1]
        candle = {
            "time": int(ts[0]),
            "open": utils._clean(last_row["open"]),
            "high": utils._clean(last_row["high"]),
            "low": utils._clean(last_row["low"]),
            "close": utils._clean(last_row["close"]),
            "volume": utils._clean(last_row["volume"]),
        }
        # Индикаторы последнего бара: скаляры, NaN -> None. Если данных мало
        # (все NaN, например SMA50 на 3 барах) — отдаём то, что есть, не падаем.
        try:
            ind = compute_indicators(df).iloc[-1]
            indicators = {
                str(k): utils._clean(v) for k, v in ind.items()
            }
        except Exception as exc:  # noqa: BLE001
            log.debug("last-bar indicators %s %s error: %s", symbol, tf, exc)
            indicators = None
    return jsonify({
        "candle": candle,
        "indicators": indicators,
        "ts": utils.now_sec(),
    })


@bp.route("/api/replay-data")
def api_replay_data():
    symbol = request.args.get("symbol", "BTCUSDT").upper()
    tf = request.args.get("timeframe", config.DEFAULT_TIMEFRAME)
    try:
        from_sec = int(request.args.get("from", 0) or 0) or None
        to_sec = int(request.args.get("to", 0) or 0) or None
        limit = int(request.args.get("limit", config.REPLAY_LIMIT))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid range params"}), 400

    if symbol not in config.SYMBOLS:
        return jsonify({"error": "Invalid symbol"}), 400
    if tf not in config.TF_SECONDS:
        return jsonify({"error": "Invalid timeframe"}), 400

    df = get_replay_df(symbol, tf, from_sec, to_sec, limit=limit)
    candles, indicators, last_price = df_to_payload(df)
    return jsonify({
        "symbol": symbol,
        "timeframe": tf,
        "from": from_sec,
        "to": to_sec,
        "candles": candles,
        "indicators": indicators,
        "last_price": last_price,
        "total": len(candles),
    })


@bp.route("/api/watchlist")
def api_watchlist():
    items = []

    def _item(sym):
        try:
            # history_limit=WATCHLIST_HISTORY: тонкая дозагрузка (100 баров),
            # а не полная история 1H; limit=300 достаточно для расчёта change.
            df = get_series_df(sym, "1H", limit=300,
                               history_limit=config.WATCHLIST_HISTORY)
        except Exception as exc:  # noqa: BLE001
            log.debug("watchlist %s error: %s", sym, exc)
            return None
        if df is None or df.empty:
            return None
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else df.iloc[-1]
        prev_close = utils._clean(prev["close"])
        last_close = utils._clean(last["close"])
        change = None
        if prev_close and last_close is not None:
            change = round((last_close - prev_close) / prev_close * 100.0, 2)
        return {
            "symbol": sym,
            "price": last_close,
            "change": change,
            "volume": utils._clean(last["volume"]),
        }

    # Параллельная загрузка всех символов: холодный кеш 10 символов
    # укладывается в ~длительность одной (самой медленной) загрузки.
    with ThreadPoolExecutor(max_workers=config.CONTEXT_WORKERS) as pool:
        for item in pool.map(_item, config.SYMBOLS):
            if item:
                items.append(item)
    return jsonify({"watchlist": items})