"""Быстрые live-котировки форекса/металлов для правого watchlist.

Ограниченный внешний fallback: два запроса идут параллельно, каждый имеет
жёсткий timeout, а результат кэшируется на 25с. Историческая загрузка OHLCV
сюда не попадает.
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from app_pkg import config

logger = logging.getLogger(__name__)
_CACHE_TTL_SEC = 25.0
_REQUEST_TIMEOUT_SEC = 2.0
_SCANNER_URL = "https://scanner.tradingview.com/{market}/scan"

_cache = {"ts": 0.0, "values": {}}
_cache_lock = threading.Lock()
_refresh_lock = threading.Lock()


def _payload(symbols):
    return {
        "symbols": {
            "tickers": [f"OANDA:{symbol}" for symbol in symbols],
            "query": {"types": []},
        },
        "columns": ["name", "close", "change"],
    }


def _fetch_market(market, symbols):
    response = requests.post(
        _SCANNER_URL.format(market=market),
        json=_payload(symbols),
        timeout=_REQUEST_TIMEOUT_SEC,
        headers={"User-Agent": "NeoTerminal/1.0"},
    )
    response.raise_for_status()
    return response.json().get("data") or []


def _fetch_quotes():
    forex_symbols = sorted(config.FOREX_SYMBOLS - {"XAUUSD"})
    markets = (("forex", forex_symbols), ("cfd", ["XAUUSD"]))
    values = {}

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="watchlist") as pool:
        futures = {
            pool.submit(_fetch_market, market, symbols): symbols
            for market, symbols in markets
        }
        for future, requested in futures.items():
            try:
                rows = future.result(timeout=_REQUEST_TIMEOUT_SEC + 0.5)
            except Exception as exc:  # noqa: BLE001 — источник опционален
                logger.debug("watchlist market %s failed: %s", ",".join(requested), exc)
                rows = []
            for row in rows or []:
                values_for_row = row.get("d") or []
                if len(values_for_row) < 3:
                    continue
                symbol = str(values_for_row[0]).replace("OANDA:", "").upper()
                try:
                    price = float(values_for_row[1])
                    change = float(values_for_row[2])
                except (TypeError, ValueError):
                    continue
                if symbol in requested and price > 0:
                    values[symbol] = {"price": price, "change": round(change, 2)}

    return values


def get_forex_quotes():
    """Вернуть кэш; при протухании обновить с жёстким общим таймаутом."""
    now = time.monotonic()
    with _cache_lock:
        if now - _cache["ts"] < _CACHE_TTL_SEC and _cache["values"]:
            return dict(_cache["values"])

    # Один refresh одновременно; остальные HTTP-потоки ждут не более 2.5с
    # и затем получают stale cache, если источник временно недоступен.
    with _refresh_lock:
        now = time.monotonic()
        with _cache_lock:
            if now - _cache["ts"] < _CACHE_TTL_SEC and _cache["values"]:
                return dict(_cache["values"])
            stale = dict(_cache["values"])

        try:
            fresh = _fetch_quotes()
        except Exception:  # noqa: BLE001
            fresh = {}

        with _cache_lock:
            if fresh:
                _cache["values"] = fresh
                _cache["ts"] = now
                return dict(fresh)
            # Даже неудачный запрос немного сдвигает timestamp, чтобы не
            # запускать сетевую ошибку на каждом 30-секундном UI refresh.
            _cache["ts"] = now
            return dict(stale)
