"""Кеши результатов ИИ: контекст (TTL 45с) и вердикт (TTL 10с).

Внутренние _get/_set/_invalidate работают с любой парой cache+lock;
публичные функции разделены на context и verdict.
"""

import threading
import time

from app_pkg import config

_CONTEXT_CACHE = {}
_CONTEXT_CACHE_LOCK = threading.Lock()

_VERDICT_CACHE = {}
_VERDICT_CACHE_LOCK = threading.Lock()

# Кеш ответов LLM (токен-диета): повторный analysis-запрос с тем же
# system/messages/последней свечой в пределах TTL идёт без похода в сеть.
_LLM_RESPONSE_CACHE = {}
_LLM_RESPONSE_CACHE_LOCK = threading.Lock()


def _record(mtype, value, labels=None):
    """Записать метрику без хардкода констант (всё в config/metrics)."""
    try:
        from app_pkg.metrics import _record_metric
        _record_metric(mtype, value, labels)
    except Exception:  # noqa: BLE001
        pass


def _get(cache: dict, lock: threading.Lock, key, ttl: float, cache_name: str = ""):
    with lock:
        item = cache.get(key)
        if item and (time.time() - item["ts"]) < ttl:
            _record("cache_hits", 1, {"name": cache_name})
            return item["value"]
    _record("cache_misses", 1, {"name": cache_name})
    return None


def _set(cache: dict, lock: threading.Lock, key, value) -> None:
    with lock:
        cache[key] = {"value": value, "ts": time.time()}


def _invalidate(cache: dict, lock: threading.Lock, key) -> None:
    with lock:
        cache.pop(key, None)


# ------------------------------------------------------------- контекст
def get_cached_context(key):
    return _get(_CONTEXT_CACHE, _CONTEXT_CACHE_LOCK, key,
                config.CONTEXT_CACHE_TTL, cache_name="context")


def set_cached_context(key, value) -> None:
    _set(_CONTEXT_CACHE, _CONTEXT_CACHE_LOCK, key, value)


def invalidate_context(key) -> None:
    _invalidate(_CONTEXT_CACHE, _CONTEXT_CACHE_LOCK, key)


# -------------------------------------------------------------- вердикт
def get_cached_verdict(key):
    return _get(_VERDICT_CACHE, _VERDICT_CACHE_LOCK, key,
                config.VERDICT_CACHE_TTL, cache_name="verdict")


def set_cached_verdict(key, value) -> None:
    _set(_VERDICT_CACHE, _VERDICT_CACHE_LOCK, key, value)


def invalidate_verdict(key) -> None:
    _invalidate(_VERDICT_CACHE, _VERDICT_CACHE_LOCK, key)


def invalidate_verdicts_for(symbol: str, timeframe: str) -> int:
    """Сбросить ВСЕ verdict-записи по паре (symbol, timeframe).

    Ключ вердикта — (symbol, timeframe, mode, replay_index, upto_sec):
    mode входит в ключ, поэтому replay/live не пересекаются. Хелпер нужен
    для явной инвалидации при смене режима/данных (mode = 3-й элемент).
    """
    with _VERDICT_CACHE_LOCK:
        stale = [k for k in _VERDICT_CACHE
                 if len(k) >= 3 and k[0] == symbol and k[1] == timeframe]
        for k in stale:
            _VERDICT_CACHE.pop(k, None)
    return len(stale)


# ---------------------------------------------------- ответы LLM (dieta)
def get_cached_llm_response(key):
    """Ответ LLM по ключу (sha1 system+messages+last_candle) или None."""
    return _get(_LLM_RESPONSE_CACHE, _LLM_RESPONSE_CACHE_LOCK, key,
                config.LLM_RESPONSE_CACHE_TTL, cache_name="llm_response")


def set_cached_llm_response(key, value) -> None:
    _set(_LLM_RESPONSE_CACHE, _LLM_RESPONSE_CACHE_LOCK, key, value)