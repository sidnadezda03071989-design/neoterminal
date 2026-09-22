"""Деривативы крипты: open interest, funding, basis, top long/short (Binance).

get_derivatives_snapshot(symbol) -> dict
    Только крипта (суффиксы USDT/BUSD/USDC) — иначе {} (не применимо).
    Схема:
      {"open_interest", "oi_change_1h_pct", "funding_rate", "basis_pct",
       "top_ls_ratio", "ok", "ts"} — только числа/None/bool.
    Ошибка сети -> поля null + "ok": false, БЕЗ exception. Кэш 5 минут.

Источники:
    fapi /fapi/v1/openInterest                        — текущий OI
    fapi /futures/data/openInterestHist (period=1h, limit=2) — Δ за час
    fapi /fapi/v1/premiumIndex                        — lastFundingRate, markPrice
    spot api.binance.com/api/v3/ticker/price          — цена для basis_pct
    fapi /futures/data/topLongShortPositionRatio (period=5m, limit=1)
"""

import logging
import threading
import time

import requests

from app_pkg.utils import _clean

log = logging.getLogger(__name__)

FAPI_BASE = "https://fapi.binance.com"
SPOT_BASE = "https://api.binance.com"

REQUEST_TIMEOUT = 5.0
CACHE_TTL = 300.0  # 5 минут

_CRYPTO_SUFFIXES = ("USDT", "BUSD", "USDC")


def _is_crypto(symbol):
    s = str(symbol or "").upper()
    return s.endswith(_CRYPTO_SUFFIXES)


def _num(value, digits=8):
    v = _clean(value)
    return round(v, digits) if v is not None else None


def _blank():
    return {
        "open_interest": None, "oi_change_1h_pct": None, "funding_rate": None,
        "basis_pct": None, "top_ls_ratio": None, "ok": False, "ts": None,
    }


def _get_json(url, params=None):
    """Ответ-объект (dict) от Binance fapi/spot без exception наружу."""
    resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict):
        return data
    if isinstance(data, list) and data:
        return data[0]
    return {}


def _get_list(url, params=None):
    """Ответ-список от Binance (fallback на []) без exception наружу."""
    resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, list) else []


def _fetch_derivatives(symbol):
    """Живой срез деривативов для торговой пары (без кеша)."""
    oi = _get_json(f"{FAPI_BASE}/fapi/v1/openInterest",
                   {"symbol": symbol})
    hist = _get_list(f"{FAPI_BASE}/futures/data/openInterestHist",
                     {"symbol": symbol, "period": "1h", "limit": 2})
    prem = _get_json(f"{FAPI_BASE}/fapi/v1/premiumIndex",
                     {"symbol": symbol})
    spot = _get_json(f"{SPOT_BASE}/api/v3/ticker/price",
                     {"symbol": symbol})
    top = _get_json(f"{FAPI_BASE}/futures/data/topLongShortPositionRatio",
                    {"symbol": symbol, "period": "5m", "limit": 1})

    open_interest = _num(oi.get("openInterest"))

    oi_change = None
    if len(hist) >= 2:
        cur = _clean(hist[0].get("sumOpenInterest"))
        prev = _clean(hist[1].get("sumOpenInterest"))
        if cur is not None and prev:
            oi_change = _num((cur - prev) / prev * 100.0, 4)

    funding_rate = _num(prem.get("lastFundingRate"), 8)

    basis_pct = None
    mark = _clean(prem.get("markPrice"))
    price = _clean(spot.get("price"))
    if mark and price:
        basis_pct = _num((mark - price) / price * 100.0, 4)

    top_ls_ratio = _num(top.get("longShortRatio"), 4)

    return {
        "open_interest": open_interest,
        "oi_change_1h_pct": oi_change,
        "funding_rate": funding_rate,
        "basis_pct": basis_pct,
        "top_ls_ratio": top_ls_ratio,
        "ok": True,
        "ts": int(time.time()),
    }


# Кэш геттера: снимок сам не кэширует.
_CACHE = {}
_CACHE_LOCK = threading.Lock()


def get_derivatives_snapshot(symbol):
    """Срез деривативов по крипте: TTL-кеш 5 мин -> сеть (без exception).

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
        payload = _fetch_derivatives(symbol)
    except Exception as exc:  # noqa: BLE001 — сеть/HTTP/JSON: деградация
        log.warning("derivatives snapshot %s failed: %s", symbol, exc)
        payload = _blank()

    with _CACHE_LOCK:
        _CACHE[symbol] = {"value": payload, "ts": time.time()}
    return payload