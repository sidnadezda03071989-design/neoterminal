"""Binance/Bybit Futures API client for Funding Rate, Open Interest, Taker Ratio.

LRU cache with TTL=300s, timeout=5s, retry-once, graceful degradation.
Fallback to Bybit when Binance returns 4xx/5xx or connection error.

All public methods return list[dict] (possibly empty) — never raise.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from typing import Any

import requests

log = logging.getLogger(__name__)

BINANCE_FAPI = "https://fapi.binance.com"
BYBIT_PUBLIC = "https://api.bybit.com"

REQUEST_TIMEOUT = 5.0
CACHE_TTL = 300.0  # 5 минут
MAX_CACHE_ENTRIES = 100


class _LRUCache:
    """Thread-safe LRU cache with per-key TTL."""

    def __init__(self, ttl: float, maxsize: int = MAX_CACHE_ENTRIES):
        self._ttl = ttl
        self._maxsize = maxsize
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        now = time.time()
        with self._lock:
            if key not in self._data:
                return None
            stamp, value = self._data[key]
            if now - stamp > self._ttl:
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return value

    def put(self, key: str, value: Any) -> None:
        now = time.time()
        with self._lock:
            self._data[key] = (now, value)
            self._data.move_to_end(key)
            while len(self._data) > self._maxsize:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


class DerivativesClient:
    """Client for Binance Futures (fapi) with Bybit fallback.

    All public methods return list[dict] — [] on total failure.
    """

    def __init__(
        self,
        base_url: str = BINANCE_FAPI,
        fallback_url: str = BYBIT_PUBLIC,
    ):
        self._base = base_url.rstrip("/")
        self._fallback = fallback_url.rstrip("/")
        self._cache = _LRUCache(CACHE_TTL)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "NeoTerminal/1.0"})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_funding_rate(
        self, symbol: str, limit: int = 3, upto_sec: int | None = None
    ) -> list[dict]:
        """Last N funding rates from Binance futures.

        Returns list of dicts with keys: symbol, fundingRate, fundingTime.
        Filters out entries after upto_sec if provided.
        """
        cache_key = f"funding:{symbol}:{limit}"
        cached = self._cache.get(cache_key)
        raw = cached if cached is not None else self._fetch_with_fallback(
            f"{self._base}/fapi/v1/fundingRate",
            {"symbol": symbol, "limit": limit},
        )
        if not raw:
            return []
        if cached is None:
            self._cache.put(cache_key, raw)
        return self._filter_upto(raw, upto_sec)

    def get_open_interest(
        self, symbol: str, period: str = "15m", limit: int = 500,
        upto_sec: int | None = None,
    ) -> list[dict]:
        """Historical Open Interest data from Binance futures.

        Returns list of dicts with keys: symbol, sumOpenInterest, sumOpenInterestValue,
        timestamp. Filters out entries after upto_sec if provided.
        """
        cache_key = f"oi_hist:{symbol}:{period}:{limit}"
        cached = self._cache.get(cache_key)
        raw = cached if cached is not None else self._fetch_with_fallback(
            f"{self._base}/futures/data/openInterestHist",
            {"symbol": symbol, "period": period, "limit": limit},
        )
        if not raw:
            return []
        if cached is None:
            self._cache.put(cache_key, raw)
        return self._filter_upto(raw, upto_sec)

    def get_taker_ratio(
        self, symbol: str, period: str = "15m", limit: int = 500,
        upto_sec: int | None = None,
    ) -> list[dict]:
        """Taker long/short ratio from Binance futures.

        Returns list of dicts with keys: symbol, buySellRatio, sellVol, buyVol, timestamp.
        Filters out entries after upto_sec if provided.
        """
        cache_key = f"taker:{symbol}:{period}:{limit}"
        cached = self._cache.get(cache_key)
        raw = cached if cached is not None else self._fetch_with_fallback(
            f"{self._base}/futures/data/takerlongshortRatio",
            {"symbol": symbol, "period": period, "limit": limit},
        )
        if not raw:
            return []
        if cached is None:
            self._cache.put(cache_key, raw)
        return self._filter_upto(raw, upto_sec)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_with_fallback(self, url: str, params: dict) -> list[dict]:
        """Try Binance, fallback to Bybit on 4xx/5xx or connection error.

        Returns [] on total failure (graceful degradation).
        """
        # Try primary (Binance fapi)
        data = self._try_fetch(url, params)
        if data is not None:
            return data
        # Fallback: Bybit
        bybit_url = self._bybit_url_for(url)
        bybit_params = self._bybit_params_for(url, params)
        if bybit_url:
            log.info("falling back to Bybit for %s", url)
            data = self._try_fetch(bybit_url, bybit_params)
            if data is not None:
                return data
        return []

    def _try_fetch(self, url: str, params: dict) -> list[dict] | None:
        """Single attempt; returns parsed list or None on failure."""
        try:
            resp = self._session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code >= 400:
                log.warning("HTTP %d from %s", resp.status_code, url)
                return None
            data = resp.json()
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                # Some endpoints return a dict with a 'data' key (Bybit)
                inner = data.get("result") or data.get("data") or data
                if isinstance(inner, list):
                    return inner
                # Single dict wrapped in list
                if isinstance(inner, dict) and inner:
                    return [inner]
            return []
        except (requests.RequestException, ValueError, TypeError) as exc:
            log.debug("fetch failed %s: %s", url, exc)
            return None

    def _filter_upto(
        self, data: list[dict], upto_sec: int | None
    ) -> list[dict]:
        """Filter entries whose timestamp is > upto_sec (future data leak prevention)."""
        if upto_sec is None:
            return data
        out = []
        for entry in data:
            ts = entry.get("fundingTime") or entry.get("timestamp") or 0
            # Handle string timestamps
            try:
                ts_val = int(ts) // 1000  # Binance ms timestamps -> sec
            except (TypeError, ValueError):
                ts_val = 0
            if ts_val <= upto_sec:
                out.append(entry)
        return out

    # ------------------------------------------------------------------
    # Bybit URL/param mapping (simplified)
    # ------------------------------------------------------------------

    def _bybit_url_for(self, url: str) -> str | None:
        """Map Binance endpoint URL to Bybit URL or None."""
        if "/fapi/v1/fundingRate" in url:
            return f"{self._fallback}/v5/market/funding/history"
        if "/futures/data/openInterestHist" in url:
            return f"{self._fallback}/v5/market/open-interest"
        if "/futures/data/takerlongshortRatio" in url:
            return f"{self._fallback}/v5/market/taker-volume"
        return None

    def _bybit_params_for(self, url: str, params: dict) -> dict:
        """Translate Binance query params to Bybit format."""
        symbol = params.get("symbol", "")
        bybit_symbol = symbol.replace("USDT", "USDT")  # Same format on Bybit
        if "/fapi/v1/fundingRate" in url:
            return {
                "category": "linear",
                "symbol": bybit_symbol,
                "limit": min(params.get("limit", 3), 200),
            }
        if "/futures/data/openInterestHist" in url:
            period_map = {"15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d"}
            interval = period_map.get(params.get("period", "15m"), "15m")
            return {
                "category": "linear",
                "symbol": bybit_symbol,
                "intervalTime": interval,
                "limit": min(params.get("limit", 500), 200),
            }
        if "/futures/data/takerlongshortRatio" in url:
            return {
                "category": "linear",
                "symbol": bybit_symbol,
                "period": params.get("period", "15m"),
                "limit": min(params.get("limit", 500), 200),
            }
        return {}


# Singleton for application use
_CLIENT = DerivativesClient()
_CLIENT_LOCK = threading.Lock()


def get_client() -> DerivativesClient:
    """Return the singleton DerivativesClient."""
    return _CLIENT


def reset_client_for_testing(client: DerivativesClient | None = None) -> DerivativesClient:
    """Replace singleton with a fresh client (testing only)."""
    global _CLIENT
    with _CLIENT_LOCK:
        _CLIENT = client or DerivativesClient()
    return _CLIENT