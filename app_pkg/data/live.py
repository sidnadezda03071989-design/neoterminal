"""Фоновые потоки live-баров.

- Binance WS: один большой stream "символ×таймфрейм" для всей крипты.
  При обрыве — переподключение с паузой LIVE_RECONNECT_DELAY (8с).
- Форекс: поллинг yfinance КАЖДЫЕ 90 СЕКУНД (не 30!). Каждый символ
  обёрнут в try/except, чтобы один упавший не рвал весь цикл.

start_background_threads() запускает оба потока (daemon=True).
"""

import contextlib
import json
import logging
import threading
import time

import pandas as pd

from app_pkg import config
from app_pkg.data import yfinance
from app_pkg.ws import _ws_push

logger = logging.getLogger(__name__)

# Если WS не присылает ВООБЩЕ ничего дольше этого (в т.ч. в момент, когда
# Binance молчит на всех 18 стримах), считаем соединение мёртвым и пересоздаём.
_WS_SILENCE_TIMEOUT = 90  # сек; > ping_interval + ping_timeout + запас

# (symbol, tf) -> {"ts": int (unix secs, открытие бара),
#                  "open"/"high"/"low"/"close": float, "volume": float}
live_bars = {}
_live_bars_lock = threading.Lock()

# heartbeat watchdog: [0] = time.monotonic() последнего WS-сообщения
_live_ws_last_msg = [time.monotonic()]

# Последний event-time (E, мс) Binance WS — эпоха БИРЖИ. Локальные часы машины
# могут отставать от биржи (замер: ~16.6с!), поэтому таймер закрытия свечи
# должен тикать в эпохе источника: крипта — Binance epoch, форекс — time.time().
_binance_epoch_ms = 0

# Пары, реально открытые в UI (symbol, tf). Заполняется фронтом через
# /api/live/subscribe. Нужен, чтобы SSE-пуши не заливали клиента событиями
# по всем 18 парам (3 символа × 6 ТФ), а только по открытой — иначе очередь
# клиента переполняется и свечи на графике отстают/пропадают.
_active_pairs = set()
_active_lock = threading.Lock()


def set_active_pairs(pairs):
    """Заменить набор активных пар (symbol, tf), приходящих из UI."""
    global _active_pairs
    clean = set()
    for item in (pairs or []):
        if not item:
            continue  # None или пустая пара — пропускаем
        try:
            s, t = item
        except (TypeError, ValueError):
            continue  # нераспаковываемый элемент (строка, число) — пропускаем
        if s and t:
            clean.add((str(s), str(t)))
    with _active_lock:
        _active_pairs = clean
    return clean


def active_pairs():
    """Копия набора активных пар (потокобезопасно)."""
    with _active_lock:
        return set(_active_pairs)


def data_now_sec(symbol):
    """«Сейчас» в эпохе ИСТОЧНИКА данных для symbol (сек).

    Крипта: бары Binance в эпохе биржевых часов — локальные часы машины могут
    отставать (замер: ~16.6с), из-за чего таймер закрытия свечи «не совпадает»
    с реальным закрытием. Берём event-time (E) из Binance WS.
    Форекс: бары MT5 привязаны к локальным часам сервера — time.time().
    Fallback для крипты (WS ещё не поднялся): time.time().
    """
    if symbol in config.CRYPTO_SYMBOLS and _binance_epoch_ms:
        return _binance_epoch_ms / 1000.0
    return time.time()


def _binance_ws_message(_ws, raw):
    try:
        msg = json.loads(raw)
        # Combined stream отдаёт payload ВНУТРИ "data":
        #   {"stream": "btcusdt@kline_1m", "data": {"e": "kline", ..., "k": {...}}}
        # Раньше читали msg["k"] напрямую -> всегда None -> сообщение молча
        # игнорировалось, live_bars оставался пустым и 1m-бар «застывал».
        payload = msg.get("data") if isinstance(msg.get("data"), dict) else msg
        k = payload.get("k") or {}
        if not k:
            return
        # event-time биржи (мс): для data_now_sec() — «эпоха Binance»,
        # в которой живут таймстампы свечей крипты (см. комментарий выше).
        evt = payload.get("E")
        if isinstance(evt, (int, float)) and evt > 0:
            global _binance_epoch_ms
            _binance_epoch_ms = int(evt)
        # heartbeat: обновляем метку «последнее сообщение» для watchdog
        _live_ws_last_msg[0] = time.monotonic()
        symbol = str(k.get("s") or "").upper()
        interval = str(k.get("i") or "").lower()
        if not symbol or not interval:
            return
        tf = {v: key for key, v in config.BINANCE_TF.items()}.get(interval)
        if tf is None:
            return
        if symbol not in config.CRYPTO_SYMBOLS:
            return
        bar = {
            "ts": int(k["t"] // 1000),
            "open": float(k["o"]),
            "high": float(k["h"]),
            "low": float(k["l"]),
            "close": float(k["c"]),
            "volume": float(k["v"]),
            "closed": bool(k.get("x")),
        }
        with _live_bars_lock:
            live_bars[(symbol, tf)] = bar
    except Exception as exc:  # noqa: BLE001
        logger.debug("binance ws message error: %s", exc)


def _binance_ws_loop():
    """Один большой stream: все крипто-символы × все таймфреймы (18 стримов).

    websocket-client.run_forever — блокирующий бесконечный цикл, живущий ровно
    столько, сколько живо TCP/WS-соединение. Раньше реконнект был неочевидным:
      - ws.run_forever(ping_interval=20) в основном потоке мог повиснуть навечь
        при обрыве через REST-пул (RemoteDisconnected), reconnect не срабатывал,
        live_bars оставался пустым, и в UI оставалась только последняя свеча из
        /api/data с разрывом в несколько минут («свечи лагают/пропадают»);
      - затем watchdog по «таймауту join» рвал и ЗДОРОВОЕ соединение каждые
        35с -> live_bars «то полон, то пуст», свечи приходили рывками.

    Теперь правильно: run_forever живёт в daemon-потоке; watchdog закрывает
    сокет ТОЛЬКО если сообщения молчат дольше _WS_SILENCE_TIMEOUT. Пока данные
    идут — соединение держится без реконнектов.
    """
    import websocket  # websocket-client (лениво)

    streams = [f"{sym.lower()}@kline_{interval}"
               for sym in sorted(config.CRYPTO_SYMBOLS)
               for interval in config.BINANCE_TF.values()]
    url = config.BINANCE_WS_URL + "/".join(streams)
    while True:
        try:
            _live_ws_last_msg[0] = time.monotonic()
            ws = websocket.WebSocketApp(
                url,
                on_message=_binance_ws_message,
                on_error=lambda _w, e: logger.warning("binance ws error: %s", e),
                on_close=lambda _w, code=None, msg=None: logger.info(
                    "binance ws closed (code=%s)", code),
            )
            logger.info("binance ws connected (%d streams)", len(streams))
            wf = threading.Thread(target=ws.run_forever,
                                  kwargs={"ping_interval": 10,
                                          "ping_timeout": 8},
                                  daemon=True)
            wf.start()
            # Даём соединению короткую возможность открыться, прежде чем
            # проверять его живучесть.
            time.sleep(3)
            while wf.is_alive():
                stale = time.monotonic() - _live_ws_last_msg[0]
                if stale > _WS_SILENCE_TIMEOUT:
                    logger.warning("binance ws silent %.0fs — closing", stale)
                    with contextlib.suppress(Exception):
                        ws.close()
                    with contextlib.suppress(Exception):
                        wf.join(timeout=3)
                    break
                time.sleep(2)
        except Exception as exc:  # noqa: BLE001
            logger.warning("binance ws loop error: %s", exc)
        logger.info("binance ws reconnect in %ss", config.LIVE_RECONNECT_DELAY)
        time.sleep(config.LIVE_RECONNECT_DELAY)


def _forex_bar_from_df(symbol: str, tf: str, df) -> dict | None:
    """Собирает live-бар из хвоста готового OHLCV-ответа (как Binance WS).

    У крипты формирующийся бар приходит из WS (все таймфреймы).
    У форекса WS нет — формирующийся бар берём из хвоста источника
    (MT5 primary, yfinance fallback): последняя строка — текущий бар.
    """
    if df is None or df.empty:
        return None
    last = df.iloc[-1]
    ts = int(pd.Timestamp(last["timestamp"]).timestamp())
    if ts <= 0:
        return None
    return {
        "ts": ts,
        "open": float(last["open"]),
        "high": float(last["high"]),
        "low": float(last["low"]),
        "close": float(last["close"]),
        "volume": float(last["volume"]),
        "closed": False,
    }


def _forex_live_tick():
    """Один такт форекс-поллинга: заполнить live_bars формирующимися барами."""
    from app_pkg.data.fetch import (  # лениво: fetch тянет mt5/yfinance
        fetch_ohlcv,
    )

    active = active_pairs()
    fx_active = sorted({(s, t) for s, t in active
                        if s in config.FOREX_SYMBOLS})

    if fx_active:
        for symbol, tf in fx_active:
            try:
                df = fetch_ohlcv(symbol, tf, limit=3)
                bar = _forex_bar_from_df(symbol, tf, df)
                if bar is not None:
                    with _live_bars_lock:
                        live_bars[(symbol, tf)] = bar
            except Exception as exc:  # noqa: BLE001
                logger.debug("forex live %s %s error: %s", symbol, tf, exc)
    else:
        for symbol in sorted(config.FOREX_SYMBOLS):
            try:
                df = yfinance.fetch_yfinance(symbol, "1D", period="10d")
                bar = _forex_bar_from_df(symbol, "1D", df)
                if bar is not None:
                    with _live_bars_lock:
                        live_bars[(symbol, "1D")] = bar
            except Exception as exc:  # noqa: BLE001
                logger.debug("forex live poll %s error: %s", symbol, exc)
                continue


def _forex_poll_interval() -> float:
    """Интервал форекс-поллинга, привязанный к наименьшему активному ТФ.

    Раньше поллинг шёл раз в FOREX_LIVE_POLL_INTERVAL (90с) независимо от ТФ.
    Для 1m-пары это МЕДЛЕННЕЕ длины свечи (60с): новая свеча приходила не
    при открытии, а с опозданием и иногда целая минута пропускалась (90с не
    кратны 60с) — «на 1m графике не добавляет свечи новые». Крипта не
    страдает: Binance WS шлёт события при КАЖДОЙ сделке и на границе бара.

    Теперь интервал = доля наименьшего активного ТФ: 1m -> ~3с, 5m -> ~15с,
    15m -> ~45с, 1H/4H/1D -> потолок FOREX_LIVE_POLL_INTERVAL. Так новая
    свеча (и её live-обновления) доходит до SSE почти сразу после открытия.
    Без активных форекс-пар — прежние 90с (греем только дневки).

    Быстрый поллинг разрешён ТОЛЬКО при работающем MT5 (локальная функция,
    копейки по CPU). Иначе источником служит yfinance, и частые запросы
    упираются в rate-limit: тогда держим исходные 90с.
    """
    from app_pkg.data.mt5 import mt5_state

    active = active_pairs()
    steps = [config.TF_SECONDS.get(tf, 60)
             for sym, tf in active if sym in config.FOREX_SYMBOLS]
    if not steps:
        return config.FOREX_LIVE_POLL_INTERVAL
    smallest = min(steps)
    adaptive = max(1.0, min(config.FOREX_LIVE_POLL_INTERVAL, smallest / 20.0))
    if not mt5_state.get("ok"):
        return config.FOREX_LIVE_POLL_INTERVAL
    return adaptive


def _forex_live_loop():
    """Поллинг формирующихся баров форекса.

    Раньше заполнялся ТОЛЬКО (symbol, "1D") — поэтому суб-дневные форекс-ТФ
    (5m, 1H, ...) жили без live-бара: apply_live_merge не накладывал свежие
    данные, _extend_newer не дозаполнял закрытые бары, а формирующийся бар из
    cold-загрузки запекался в кэш (TTL 120с) и после закрытия «менял размер»
    или пропадал. Теперь заполняем формирующийся бар для ОТКРЫТЫХ в UI пар
    (как фильтр SSE-пушей): подписанная форекс-пара ведёт себя как крипта.

    Интервал между тиками — адаптивный (см. _forex_poll_interval): для мелких
    ТФ (1m) опрашиваем источник в несколько раз чаще, чтобы новая свеча не
    «пропускалась» между поллингами.

    Когда открытых форекс-пар нет — греем только дневные бары всех символов
    (как раньше), чтобы не дёргать yfinance лишний раз.
    """
    while True:
        try:
            _forex_live_tick()
        except Exception as exc:  # noqa: BLE001
            logger.warning("forex live loop error: %s", exc)
        time.sleep(_forex_poll_interval())


def _sse_candle_push_loop():
    """Каждые SSE_CANDLE_PUSH_INTERVAL (по умолчанию 1с) читает live_bars
    (заполняется Binance WS / forex-поллингом) и шлёт _ws_push("candle_update")
    только для пар, чьё содержимое ИЗМЕНИЛОСЬ с прошлого тика — дедуп.

    Фильтр по активным подпискам (set_active_pairs): шлём только то, что
    реально открыто в UI. Если подписок нет (active пуст) — шлём все пары
    (fallback, чтобы свечи не «зависали», пока фронт ещё не подписался).

    Полезная нагрузка: {symbol, timeframe, candle:{...}, closed}.
    """
    last_sent = {}
    first = True
    while True:
        if not first:
            time.sleep(config.SSE_CANDLE_PUSH_INTERVAL)
        first = False
        try:
            with _live_bars_lock:
                snapshot = dict(live_bars)
            active = active_pairs()
            for (symbol, tf), bar in snapshot.items():
                if not bar:
                    continue
                if active and (symbol, tf) not in active:
                    continue
                sig = (
                    int(bar.get("ts", 0)),
                    float(bar.get("open") or 0),
                    float(bar.get("high") or 0),
                    float(bar.get("low") or 0),
                    float(bar.get("close") or 0),
                    float(bar.get("volume") or 0),
                    bool(bar.get("closed", False)),
                )
                if last_sent.get((symbol, tf)) == sig:
                    continue
                last_sent[(symbol, tf)] = sig
                candle = {
                    "time": int(bar.get("ts", 0)),
                    "open": bar.get("open"),
                    "high": bar.get("high"),
                    "low": bar.get("low"),
                    "close": bar.get("close"),
                    "volume": bar.get("volume"),
                }
                try:
                    _ws_push("candle_update", {
                        "symbol": symbol,
                        "timeframe": tf,
                        "candle": candle,
                        "closed": bool(bar.get("closed", False)),
                        # «Сейчас» в эпохе источника — фронт синхронизирует
                        # по нему таймер закрытия свечи (см. candle_timer.js).
                        "now": data_now_sec(symbol),
                    })
                except Exception as exc:  # noqa: BLE001
                    logger.debug("candle_update push error: %s", exc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("sse candle push tick error: %s", exc)


def start_background_threads():
    """Запускает Binance WS, forex-поллинг и SSE candle-push (daemon)."""
    t1 = threading.Thread(target=_binance_ws_loop, name="binance-ws", daemon=True)
    t2 = threading.Thread(target=_forex_live_loop, name="forex-live", daemon=True)
    t3 = threading.Thread(target=_sse_candle_push_loop, name="sse-candle-push", daemon=True)
    t1.start()
    t2.start()
    t3.start()
    try:
        from app_pkg.alerts import start_alert_thread
        start_alert_thread()
    except Exception as exc:  # noqa: BLE001
        logger.warning("alerts thread start failed: %s", exc)
    logger.info("live background threads started (binance-ws, forex-live, sse-candle-push, alerts)")
    return t1, t2, t3