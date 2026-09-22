# -*- coding: utf-8 -*-
"""Толпа (crowd sentiment): long/short-соотношения Binance Futures + Fear&Greed.

get_crowd_snapshot(symbol) -> dict
    Только крипта с суффиксом USDT/BUSD/USDC — иначе {} (не применимо).
    Схема: {"ls_ratio", "long_pct", "short_pct", "taker_buy_sell",
            "fear_greed", "ok", "ts"} — только числа/None/bool.
    Ошибка сети -> поля null + "ok": false, БЕЗ exception.
    Кэш 5 минут внутри геттера (snapshot сам не кэширует).

Источники:
    fapi.binance.com/futures/data/globalLongShortAccountRatio (period=5m&limit=1)
    fapi.binance.com/futures/data/takerlongshortRatio      (period=5m&limit=1)
    api.alternative.me/fng/?limit=1                        (Fear & Greed 0-100)
"""

import logging
import threading
import time

import requests

from app_pkg.utils import _clean

log = logging.getLogger(__name__)

FAPI_BASE = "https://fapi.binance.com"
FNG_URL = "https://api.alternative.me/fng/"

REQUEST_TIMEOUT = 5.0
CACHE_TTL = 300.0  # 5 минут

# Крипта: суффиксы спот-пар Binance. Остальное (форекс/металлы) не применимо.
_CRYPTO_SUFFIXES = ("USDT", "BUSD", "USDC")


def _is_crypto(symbol):
    s = str(symbol or "").upper()
    return s.endswith(_CRYPTO_SUFFIXES)


def _num(value, digits=4):
    """float (с округлением) или None (NaN/нечисло)."""
    v = _clean(value)
    return round(v, digits) if v is not None else None


def _blank():
    """Null-схема при ошибке источника (без 0/50-заглушек)."""
    return {
        "ls_ratio": None, "long_pct": None, "short_pct": None,
        "taker_buy_sell": None, "fear_greed": None, "ok": False, "ts": None,
    }


def _fapi_row(url, params):
    """Первый элемент списка (или сам объект) от fapi.binance.com."""
    resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list) and data:
        return data[0]
    return data if isinstance(data, dict) else {}


def _fetch_fear_greed():
    """Fear & Greed Index (0-100, int), None при ошибке/пустом ответе."""
    try:
        resp = requests.get(FNG_URL, params={"limit": 1},
                            timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        value = None
        if isinstance(data.get("data"), list) and data["data"]:
            value = _num(data["data"][0].get("value"), digits=0)
            if value is not None:
                value = int(value)
        return value
    except Exception as exc:  # noqa: BLE001 — сеть/JSON: fng не критичен
        log.warning("fear&greed fetch failed: %s", exc)
        return None


def _fetch_crowd(symbol):
    """Живой срез толпы для торговой пары (без кеша)."""
    long_short = _fapi_row(
        f"{FAPI_BASE}/futures/data/globalLongShortAccountRatio",
        {"symbol": symbol, "period": "5m", "limit": 1})
    taker = _fapi_row(
        f"{FAPI_BASE}/futures/data/takerlongshortRatio",
        {"symbol": symbol, "period": "5m", "limit": 1})

    return {
        "ls_ratio": _num(long_short.get("longShortRatio")),
        "long_pct": _num(long_short.get("longAccount")),
        "short_pct": _num(long_short.get("shortAccount")),
        "taker_buy_sell": _num(taker.get("buySellRatio")),
        "fear_greed": _fetch_fear_greed(),
        "ok": True,
        "ts": int(time.time()),
    }


# Кэш геттера: отдельно от снимка, чтобы AI Backtest не дёргал сеть на клик.
_CACHE = {}
_CACHE_LOCK = threading.Lock()


def get_crowd_snapshot(symbol):
    """Серентимент толпы по крипте: TTL-кеш 5 мин -> сеть (без exception).

    Не крипта -> {} (снимок ставит "applicable": false).
    """
    symbol = str(symbol or "").upper()
    if not _is_crypto(symbol):
        return {}

    now = time.time()
    with _CACHE_LOCK:
        item = _CACHE.get(symbol)
        if item and (now - item["ts"]) < CACHE_TTL:
            return item["value"]

    try:
        payload = _fetch_crowd(symbol)
    except Exception as exc:  # noqa: BLE001 — сеть/HTTP/JSON: деградация
        log.warning("crowd snapshot %s failed: %s", symbol, exc)
        payload = _blank()

    with _CACHE_LOCK:
        _CACHE[symbol] = {"value": payload, "ts": time.time()}
    return payload