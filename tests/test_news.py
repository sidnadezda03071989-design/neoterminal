"""Тесты новостей по активу (app_pkg/data/news.py + routes/news.py).

Сеть не трогаем: провайдеры _fetch_finnhub_category/_fetch_coingecko
монкипатчатся. Проверяем нормализацию текста/времени, слияние лент,
кеш+refresh и поведение роута (error при пустой ленте).
"""

import time

import pytest

from app_pkg import config, create_app
from app_pkg.data import news as news_mod
from app_pkg.data.news import _clean_text, _merge_by_time, _norm_epoch, get_symbol_news

app = create_app()


@pytest.fixture(autouse=True)
def _clear_cache():
    news_mod._NEWS_CACHE.clear()
    yield
    news_mod._NEWS_CACHE.clear()


# ------------------------------------------------------------ нормализация
def test_clean_text_strips_html_and_limits():
    raw = "<p>Привет<br>мир</p>   с  пробелами "
    out = _clean_text(raw, limit=100)
    assert "<" not in out and ">" not in out
    assert "Привет" in out and "мир" in out
    long = "x" * 500
    assert len(_clean_text(long, limit=300)) <= 301  # 300 + "…"


def test_clean_text_non_string():
    assert _clean_text(None) == ""
    assert _clean_text(123) == ""


def test_norm_epoch_int_and_iso():
    assert _norm_epoch(1789964999) == 1789964999
    assert _norm_epoch("2026-09-21T12:00:00Z") > 0
    assert _norm_epoch("мусор") == 0
    assert _norm_epoch(None) == 0
    assert _norm_epoch(0) == 0


# ------------------------------------------------------------ слияние лент
def _mk(pid, n, dt, url):
    return {"id": f"{pid}-{n}", "title": f"t{n}", "summary": "",
            "source": pid, "url": url, "image": "", "datetime": dt,
            "provider": pid, "category": "crypto"}


def test_merge_sorts_and_dedups():
    a = [_mk("fh", 1, 100, "https://a/1"), _mk("fh", 2, 300, "https://a/2")]
    b = [_mk("cg", 3, 200, "https://a/1"),  # дубль URL — отбросить
         _mk("cg", 4, 400, "https://b/1")]
    merged = _merge_by_time([a, b])
    urls = [n["url"] for n in merged]
    assert urls == ["https://b/1", "https://a/2", "https://a/1"]  # по свежести
    assert len(urls) == len(set(urls))


def test_merge_caps_items():
    many = [_mk("fh", i, i, f"https://a/{i}") for i in range(50)]
    assert len(_merge_by_time([many])) == news_mod.MAX_NEWS_ITEMS


# ------------------------------------------------------------ кеш + refresh
def test_cache_and_refresh(monkeypatch):
    calls = {"n": 0}

    def fake_payload(symbol):
        calls["n"] += 1
        return {"symbol": symbol, "news": [_mk("fh", 1, 1, "https://a/1")],
                "count": 1, "sources": ["finnhub"], "fetched_at": 0}

    monkeypatch.setattr(news_mod, "_news_for_symbol", fake_payload)
    first = get_symbol_news("BTCUSDT")
    second = get_symbol_news("BTCUSDT")          # из кеша, без похода в сеть
    assert first is second and calls["n"] == 1
    get_symbol_news("BTCUSDT", refresh=True)     # refresh игнорирует кеш
    assert calls["n"] == 2


def test_cache_ttl_expires(monkeypatch):
    monkeypatch.setattr(config, "NEWS_CACHE_TTL", 0.0)  # всё просрочено
    calls = {"n": 0}

    def fake_payload(symbol):
        calls["n"] += 1
        return {"symbol": symbol, "news": [], "count": 0, "sources": [],
                "fetched_at": 0}

    monkeypatch.setattr(news_mod, "_news_for_symbol", fake_payload)
    get_symbol_news("BTCUSDT")
    get_symbol_news("BTCUSDT")
    assert calls["n"] == 2


# --------------------------------------------------- подбор категорий
def test_categories_crypto_vs_forex(monkeypatch):
    seen = {}

    def fake_fh(category):
        seen.setdefault(category, 0)
        seen[category] += 1
        return []

    monkeypatch.setattr(news_mod, "_fetch_finnhub_category", fake_fh)
    monkeypatch.setattr(news_mod, "_fetch_coingecko", list)
    news_mod._news_for_symbol("BTCUSDT")
    assert seen == {"crypto": 1}
    seen.clear()
    news_mod._news_for_symbol("EURUSD")
    assert seen == {"general": 1}  # forex-категория Finnhub не используется


def test_provider_failure_degrades(monkeypatch):
    """Сеть отвалилась -> обе ленты пустые, но payload валиден (без исключения)."""
    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(news_mod.requests, "get", boom)
    p = news_mod._news_for_symbol("BTCUSDT")
    assert p["news"] == [] and p["count"] == 0
    assert p["sources"] == []


# ------------------------------------------------------------ роут
def test_route_error_when_empty(monkeypatch):
    monkeypatch.setattr(news_mod, "_news_for_symbol",
                        lambda s: {"symbol": s, "news": [], "count": 0,
                                   "sources": [], "fetched_at": 0})
    client = app.test_client()
    r = client.get("/api/news/BTCUSDT")
    assert r.status_code == 200
    assert r.get_json()["error"]


def test_route_ok(monkeypatch):
    monkeypatch.setattr(news_mod, "_news_for_symbol",
                        lambda s: {"symbol": s,
                                   "news": [_mk("fh", 1, 1, "https://a/1")],
                                   "count": 1, "sources": ["finnhub"],
                                   "fetched_at": int(time.time())})
    client = app.test_client()
    r = client.get("/api/news/BTCUSDT")
    data = r.get_json()
    assert r.status_code == 200 and data["count"] == 1
    assert "error" not in data
