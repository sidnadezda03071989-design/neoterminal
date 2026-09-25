"""Blueprint: /api/data, /api/last-bar, /api/replay-data, /api/watchlist."""

import logging
from concurrent.futures import ThreadPoolExecutor

from flask import Blueprint, jsonify, request

from app_pkg import config, utils
from app_pkg.data.fetch import get_replay_df, get_series_df
from app_pkg.indicators import compute_indicators, df_last_candle, df_to_payload

bp = Blueprint("data", __name__)
log = logging.getLogger(__name__)


def _source_now_sec(symbol):
    """«Сейчас» в эпохе источника данных (для синка таймера закрытия свечи).

    Живой канал через app_pkg.data.live.data_now_sec (Binance = эпоха биржи),
    иначе fallback на локальные часы сервера. Импорт ленивый: live.py тянет
    тяжёлые зависимости (yfinance/ws), а роуты не должны грузить их на старте.
    """
    try:
        from app_pkg.data.live import data_now_sec
        return data_now_sec(symbol)
    except Exception:  # noqa: BLE001 — live-модуль может быть недоступен
        return utils.now_sec()


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
        "now": _source_now_sec(symbol),
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
    candle = df_last_candle(df)
    indicators = None
    if candle is not None:
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
        "now": _source_now_sec(symbol),
    })


@bp.route("/api/live/subscribe", methods=["POST"])
def api_live_subscribe():
    """Регистрирует открытую в UI пару (symbol, timeframe) для SSE-пушей.

    Тело: {"symbol": "BTCUSDT", "timeframe": "1m"}.
    Без этого активного списка сервер шлёт candle_update по ВСЕМ 18 парам,
    что переполняет очередь клиента: свечи на графике начинают отставать и
    «пропадать». Фронт вызывает при загрузке и при смене symbol/ТФ.
    """
    body = request.get_json(silent=True) or {}
    symbol = (body.get("symbol") or request.args.get("symbol") or "").upper()
    tf = body.get("timeframe") or request.args.get("timeframe")
    if not symbol or not tf:
        return jsonify({"error": "symbol and timeframe required"}), 400
    try:
        from app_pkg.data.live import set_active_pairs
        pairs = set_active_pairs([(symbol, tf)])
    except Exception as exc:  # noqa: BLE001 — live-модуль может быть не поднят
        logger.debug("live subscribe failed: %s", exc)
        return jsonify({"error": "live module unavailable"}), 503
    return jsonify({"ok": True, "active": sorted(f"{s}|{t}" for s, t in pairs)})


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
    # upto_sec (replay): индикаторы считаются ТОЛЬКО до барьера replay_time.
    # Свечи остаются полными (df не режется), а indicators получаются по
    # префиксу <= upto_sec — окна (SMA/EMA/BB/RSI...) не «видят» будущее.
    upto_sec = request.args.get("upto_sec")
    if upto_sec:
        try:
            upto_sec = int(float(upto_sec))
        except (TypeError, ValueError):
            upto_sec = None
        if upto_sec is not None and df is not None and not df.empty:
            ts_arr = utils.epoch_secs(df["timestamp"])
            if (ts_arr <= upto_sec).any():
                df = df.iloc[ts_arr <= upto_sec]
            else:
                df = df.iloc[:0]
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