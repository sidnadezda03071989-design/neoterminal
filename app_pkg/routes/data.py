"""Blueprint: /api/data, /api/last-bar, /api/replay-data, /api/watchlist."""

import logging

import pandas as pd
from flask import Blueprint, jsonify, request

from app_pkg import config, utils
from app_pkg.data.fetch import get_cached_series_df, get_replay_df, get_series_df
from app_pkg.data.live import get_live_bar
from app_pkg.data.watchlist import get_forex_quotes
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
        log.debug("live subscribe failed: %s", exc)
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


def _daily_change(df):
    """Изменение последней цены от начала текущего торгового дня."""
    if df is None or df.empty:
        return None
    try:
        stamps = pd.to_datetime(df["timestamp"], utc=True)
        last_close = utils._clean(df.iloc[-1]["close"])
        day_start = stamps.iloc[-1].normalize()
        previous = df.loc[stamps < day_start, "close"].dropna()
        if not previous.empty:
            base_close = utils._clean(previous.iloc[-1])
        else:
            today_open = df.loc[stamps >= day_start, "open"].dropna()
            base_close = utils._clean(today_open.iloc[0]) if not today_open.empty else None
        if base_close in (None, 0) or last_close is None:
            return None
        return round((last_close - base_close) / base_close * 100.0, 2)
    except Exception as exc:  # noqa: BLE001
        log.debug("daily change error: %s", exc)
        return None


def _number(value):
    value = utils._clean(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def _bar_change(bar):
    if not bar:
        return None
    close = _number(bar.get("close"))
    base = _number(bar.get("open"))
    if close is None or base in (None, 0):
        return None
    return round((close - base) / base * 100.0, 2)


def _watchlist_item(sym, forex_quotes=None):
    """Быстрый снимок котировки: live-бар → валидный кэш → пустое значение.

    Здесь нельзя запускать историческую загрузку: Yahoo/MT5 может не ответить
    несколько минут, а список на UI должен появляться сразу целиком.
    """
    base = {
        "symbol": sym,
        "name": config.ASSET_NAMES.get(sym, sym),
        "price": None,
        "change": None,
        "volume": None,
        "source": None,
    }

    daily_bar = get_live_bar(sym, "1D")
    price_bar = daily_bar or get_live_bar(sym, "1H")
    change = _bar_change(daily_bar)

    df = None
    if price_bar is None or change is None:
        for tf in ("1H", "1D"):
            candidate = get_cached_series_df(sym, tf)
            if candidate is not None and not candidate.empty:
                df = candidate
                break

    if price_bar is not None:
        price = _number(price_bar.get("close"))
        volume = _number(price_bar.get("volume"))
        if price is not None:
            base["source"] = "live"
    elif df is not None and not df.empty:
        last = df.iloc[-1]
        price = _number(last.get("close"))
        volume = _number(last.get("volume"))
        base["source"] = "cache"
    else:
        price = None
        volume = None

    if change is None and df is not None and not df.empty:
        change = _daily_change(df)

    if price is None and sym in config.FOREX_SYMBOLS:
        quote = (forex_quotes or {}).get(sym)
        if quote:
            price = _number(quote.get("price"))
            change = _number(quote.get("change"))
            if price is not None:
                base["source"] = "fallback"

    return {**base, "price": price, "change": change, "volume": volume}


@bp.route("/api/watchlist")
def api_watchlist():
    # Историческая загрузка здесь запрещена. Для недоступных live-кешей
    # используется один batch-запрос с жёстким timeout и общим кэшем.
    forex_quotes = get_forex_quotes()
    items = [_watchlist_item(sym, forex_quotes) for sym in config.SYMBOLS]
    return jsonify({"watchlist": items})
