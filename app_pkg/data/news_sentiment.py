"""Сентимент новостей по активу: lexicon-скоринг заголовков (без LLM).

get_news_sentiment(symbol, limit=30) -> dict
    Лента берётся из существующего геттера app_pkg/data/news.get_symbol_news
    (read-only импорт; сеть и кеш внутри него). Заголовки скорим по словарям
    BULLISH/BEARISH (по 30-40 токенов): поиск подстрок, регистронезависимо,
    итог клип [−1, 1].

    Схема:
      {"avg_sentiment", "bullish_count", "bearish_count", "neutral_count",
       "sample_size", "worst_headline_score", "best_headline_score",
       "ok", "ts"} — только числа/None/bool.

    Пустая лента/ошибка -> "ok": false с null/0. Кэш 5 минут.
    LLM тут НЕ вызывается.
"""

import logging
import threading
import time

from app_pkg.data.news import get_symbol_news

log = logging.getLogger(__name__)

CACHE_TTL = 300.0  # 5 минут

# Лексикон: токены ищутся подстрокой в .lower() заголовке. Токены подобраны
# так, чтобы не ловить ложные срабатывания в нейтральных словах
# («red»/«up»/«down»/«beat» намеренно НЕ включены).
BULLISH = (
    "surge", "surges", "surged", "soar", "soars", "soared",
    "rally", "rallies", "rallied", "rallying",
    "gains", "gained", "gaining",
    "record high", "all-time high",
    "breakout", "bullish", "uptrend",
    "rebound", "rebounds", "rebounded", "recovery",
    "outperform", "beats expectations", "beat expectations",
    "strong demand", "upgraded", "upgrade",
    "boosts", "positive", "pump", "milestone",
    "profits", "inflows", "accumulation",
    "adoption", "optimism", "momentum",
)

BEARISH = (
    "crash", "crashes", "crashed",
    "plunge", "plunges", "plunged",
    "plummet", "plummets", "plummeted",
    "drop", "drops", "dropped",
    "fall", "falls", "fell", "falling",
    "slump", "slumps", "slumped",
    "slips", "slipped",
    "decline", "declines", "declined",
    "sell-off", "selling",
    "downtrend", "bearish", "worst",
    "recession", "panic", "fear",
    "loss", "loses", "weak",
    "downgrade", "downgraded", "negative",
)


def _blank():
    return {
        "avg_sentiment": None, "bullish_count": 0, "bearish_count": 0,
        "neutral_count": 0, "sample_size": 0, "worst_headline_score": None,
        "best_headline_score": None, "ok": False, "ts": None,
    }


def _score_headline(title):
    """Один заголовок -> score в [−1, 1] (0, если словаря не зацепило)."""
    low = (title or "").lower()
    bulls = sum(low.count(t) for t in BULLISH)
    bears = sum(low.count(t) for t in BEARISH)
    if bulls == 0 and bears == 0:
        return 0.0
    raw = (bulls - bears) / (bulls + bears)
    return max(-1.0, min(1.0, raw))


def _score_news(news_items):
    """Список новостей -> агрегат lexicon-скоринга заголовков."""
    scores = [_score_headline(n.get("title")) for n in news_items]
    bullish = sum(1 for s in scores if s > 0)
    bearish = sum(1 for s in scores if s < 0)
    neutral = sum(1 for s in scores if s == 0)
    return {
        "avg_sentiment": round(sum(scores) / len(scores), 4) if scores else None,
        "bullish_count": bullish,
        "bearish_count": bearish,
        "neutral_count": neutral,
        "sample_size": len(scores),
        "worst_headline_score": round(min(scores), 4) if scores else None,
        "best_headline_score": round(max(scores), 4) if scores else None,
        "ok": bool(scores),
        "ts": int(time.time()),
    }


# Кэш геттера: снимок сам не кэширует.
_CACHE = {}
_CACHE_LOCK = threading.Lock()


def get_news_sentiment(symbol, limit=30):
    """Сентимент ленты новостей по активу: TTL-кеш 5 мин -> get_symbol_news.

    Пустая лента/ошибка геттера -> null-схема с "ok": false, без exception.
    """
    symbol = str(symbol or "").upper()
    limit = max(1, int(limit or 30))
    key = (symbol, limit)

    now = time.time()
    with _CACHE_LOCK:
        item = _CACHE.get(key)
        if item and (now - item["ts"]) < CACHE_TTL:
            return item["value"]

    try:
        feed = get_symbol_news(symbol)
        news = (feed or {}).get("news") or []
        payload = _score_news(list(news)[:limit])
    except Exception as exc:  # noqa: BLE001 — лента недоступна: деградация
        log.warning("news sentiment %s failed: %s", symbol, exc)
        payload = _blank()

    with _CACHE_LOCK:
        _CACHE[key] = {"value": payload, "ts": time.time()}
    return payload