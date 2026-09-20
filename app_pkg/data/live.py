"""Фоновые потоки live-баров.

- Binance WS: один большой stream "символ×таймфрейм" для всей крипты.
  При обрыве — переподключение с паузой LIVE_RECONNECT_DELAY (8с).
- Форекс: поллинг yfinance КАЖДЫЕ 90 СЕКУНД (не 30!). Каждый символ
  обёрнут в try/except, чтобы один упавший не рвал весь цикл.

start_background_threads() запускает оба потока (daemon=True).
"""

import json
import logging
import threading
import time

import pandas as pd

from app_pkg import config
from app_pkg.data import yfinance

logger = logging.getLogger(__name__)

# (symbol, tf) -> {"ts": int (unix secs, открытие бара),
#                  "open"/"high"/"low"/"close": float, "volume": float}
live_bars = {}
_live_bars_lock = threading.Lock()


def _binance_ws_message(_ws, raw):
    try:
        msg = json.loads(raw)
        k = msg.get("k") or {}
        symbol = str(k.get("s") or "").upper()
        interval = str(k.get("i") or "").lower()
        if not symbol or not interval:
            return
        tf = {v: key for key, v in config.BINANCE_TF.items()}.get(interval)
        if tf is None:
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
    """Один большой stream: все крипто-символы × все таймфреймы."""
    import websocket  # websocket-client (лениво)

    while True:
        streams = []
        for sym in sorted(config.CRYPTO_SYMBOLS):
            for interval in config.BINANCE_TF.values():
                streams.append(f"{sym.lower()}@kline_{interval}")
        url = config.BINANCE_WS_URL + "/".join(streams)
        try:
            ws = websocket.WebSocketApp(
                url,
                on_message=_binance_ws_message,
                on_error=lambda _w, e: logger.warning("binance ws error: %s", e),
            )
            logger.info("binance ws connected (%d streams)", len(streams))
            ws.run_forever(ping_interval=20, ping_timeout=10)
        except Exception as exc:  # noqa: BLE001
            logger.warning("binance ws loop error: %s", exc)
        logger.info("binance ws reconnect in %ss", config.LIVE_RECONNECT_DELAY)
        time.sleep(config.LIVE_RECONNECT_DELAY)


def _forex_live_loop():
    """Поллинг yfinance каждые FOREX_LIVE_POLL_INTERVAL секунд."""
    while True:
        for symbol in sorted(config.FOREX_SYMBOLS):
            try:
                df = yfinance.fetch_yfinance(symbol, "1D", period="10d")
                if df is not None and not df.empty:
                    last = df.iloc[-1]
                    bar = {
                        "ts": int(pd.Timestamp(last["timestamp"]).timestamp()),
                        "open": float(last["open"]),
                        "high": float(last["high"]),
                        "low": float(last["low"]),
                        "close": float(last["close"]),
                        "volume": float(last["volume"]),
                        "closed": False,
                    }
                    with _live_bars_lock:
                        live_bars[(symbol, "1D")] = bar
            except Exception as exc:  # noqa: BLE001
                logger.debug("forex live poll %s error: %s", symbol, exc)
                continue
        time.sleep(config.FOREX_LIVE_POLL_INTERVAL)


def start_background_threads():
    """Запускает Binance WS, поллинг форекса и цикл алертов (daemon-потоки)."""
    t1 = threading.Thread(target=_binance_ws_loop, name="binance-ws", daemon=True)
    t2 = threading.Thread(target=_forex_live_loop, name="forex-live", daemon=True)
    t1.start()
    t2.start()
    # Цикл алертов — ленивый импорт, чтобы не создавать цикл импортов.
    try:
        from app_pkg.alerts import start_alert_thread
        start_alert_thread()
    except Exception as exc:  # noqa: BLE001
        logger.warning("alerts thread start failed: %s", exc)
    logger.info("live background threads started (binance-ws, forex-live, alerts)")
    return t1, t2