# -*- coding: utf-8 -*-
"""Новости по активу: Finnhub (основной) + CoinGecko (вторичный).

Finnhub /news отдаёт категориальные ленты (crypto/forex/general) — они
покрывают весь список символов терминала; company-news работает только для
акций, поэтому для форекса и крипты используется лента соответствующей
категории. CoinGecko /news доступен только на Pro-плане (demo-ключ даёт
HTTP 401) — провайдер подключён как вторичный и молча деградирует.

Все ответы кешируются в data_cache (общий кеш проекта), TTL задаёт
NEWS_CACHE_TTL — бесплатный тариф Finnhub лимитен, ходить в сеть на каждый
клик нельзя.

Формат новости наружу (единый для обоих провайдеров):
  {id, title, summary, source, url, image, datetime}
"""

import logging
import re
import threading
import time
from datetime import datetime, timezone

import requests

from app_pkg import config

log = logging.getLogger(__name__)

REQUEST_TIMEOUT = 15.0
MAX_NEWS_ITEMS = 30          # сколько новостей отдавать наружу (после слияния)

# Кеш новостей (по образцу app_pkg/cache.py): отдельный от кеша свечей
# data_cache, потому что там OrderedDict {"df": DataFrame, ...} под бары.
_NEWS_CACHE = {}
_NEWS_CACHE_LOCK = threading.Lock()
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _clean_text(raw, limit=300):
    """HTML-теги и лишние пробелы из summary (Forexlive шлёт <p>-абзацы)."""
    if not raw or not isinstance(raw, str):
        return ""
    text = _TAG_RE.sub(" ", raw)
    text = _WS_RE.sub(" ", text).strip()
    if len(text) > limit:
        text = text[:limit].rstrip() + "…"
    return text


def _norm_epoch(value):
    """Finnhub шлёт int-epoch, CoinGecko — ISO-строку/seconds: -> int epoch."""
    if isinstance(value, (int, float)) and value > 0:
        return int(value)
    if isinstance(value, str):
        try:
            iso = value.replace("Z", "+00:00")
            return int(datetime.fromisoformat(iso).timestamp())
        except ValueError:
            return 0
    return 0


def _fetch_finnhub_category(category):
    """Лента Finnhub /news?category=... -> список нормализованных новостей."""
    if not config.FINNHUB_API_KEY:
        return []
    try:
        resp = requests.get(
            f"{config.FINNHUB_BASE_URL}/news",
            params={"category": category, "token": config.FINNHUB_API_KEY},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        raw = resp.json()
    except Exception as exc:  # noqa: BLE001 — сеть/JSON: деградация, не падение
        log.warning("finnhub news fetch failed: %s", exc)
        return []
    out = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        headline = (item.get("headline") or "").strip()
        url = (item.get("url") or "").strip()
        if not headline or not url:
            continue
        out.append({
            "id": f"fh-{item.get('id', headline[:40])}",
            "title": headline,
            "summary": _clean_text(item.get("summary")),
            "source": (item.get("source") or "").strip() or "Finnhub",
            "url": url,
            "image": (item.get("image") or "").strip(),
            "datetime": _norm_epoch(item.get("datetime")),
            "provider": "finnhub",
            "category": category,
        })
    return out


def _fetch_coingecko():
    """Лента CoinGecko /news -> нормализованный список (Pro-only, молча)."""
    params = {}
    if config.COINGECKO_API_KEY:
        params["x_cg_demo_api_key"] = config.COINGECKO_API_KEY
    try:
        resp = requests.get(f"{config.COINGECKO_BASE_URL}/news",
                            params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        raw = resp.json()
    except Exception as exc:  # noqa: BLE001 — 401 на free-плане это норма
        log.info("coingecko news unavailable (это штатно на free-плане): %s", exc)
        return []
    if not isinstance(raw, dict):
        return []
    out = []
    for item in raw.get("data", []):
        if not isinstance(item, dict):
            continue
        title = (item.get("title") or "").strip()
        url = (item.get("url") or "").strip()
        if not title or not url:
            continue
        out.append({
            "id": f"cg-{item.get('id', title[:40])}",
            "title": title,
            "summary": _clean_text(item.get("description")),
            "source": (item.get("author") or "").strip() or "CoinGecko",
            "url": url,
            "image": (item.get("thumb_w_full") or item.get("thumb_w_350")
                      or item.get("image") or "").strip(),
            "datetime": _norm_epoch(item.get("updated_at")),
            "provider": "coingecko",
            "category": "crypto",
        })
    return out


def _merge_by_time(lists):
    """Слить ленты провайдеров, отсортировать по свежести, убрать дубли URL."""
    seen_urls = set()
    merged = []
    for items in lists:
        for item in items:
            key = item.get("url", "")
            if not key or key in seen_urls:
                continue
            seen_urls.add(key)
            merged.append(item)
    merged.sort(key=lambda n: n.get("datetime") or 0, reverse=True)
    return merged[:MAX_NEWS_ITEMS]


def _news_for_symbol(symbol):
    """Свежие новости для символа: подбор категорий Finnhub под актив.

    Категория `forex` у Finnhub стабильно отвечает таймаутом (проверено
    дважды при 10с и 15с), поэтому форекс берём из `general` — там те же
    макро/валютные ленты (Forexlive и др.). Крипта и золото — `crypto`.
    """
    is_crypto = symbol in config.CRYPTO_SYMBOLS
    is_gold = symbol == "XAUUSD"
    finnhub_cats = ["crypto"] if (is_crypto or is_gold) else ["general"]
    coingecko = _fetch_coingecko() if (is_crypto or is_gold) else []
    news = _merge_by_time(
        [_fetch_finnhub_category(c) for c in finnhub_cats] + [coingecko])
    return {
        "symbol": symbol,
        "news": news,
        "count": len(news),
        "sources": sorted({n["provider"] for n in news}),
        "fetched_at": int(time.time()),
    }


def get_symbol_news(symbol, refresh=False):
    """Публичный вход роута: TTL-кеш NEWS_CACHE_TTL -> сеть.

    refresh=True игнорирует кеш (кнопка ⟳ в UI).
    """
    key = symbol.upper()
    now = time.time()
    if not refresh:
        with _NEWS_CACHE_LOCK:
            item = _NEWS_CACHE.get(key)
            if item and (now - item["ts"]) < config.NEWS_CACHE_TTL:
                return item["value"]
    payload = _news_for_symbol(key)
    with _NEWS_CACHE_LOCK:
        _NEWS_CACHE[key] = {"value": payload, "ts": time.time()}
    return payload
