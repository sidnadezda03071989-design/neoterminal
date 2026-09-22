# -*- coding: utf-8 -*-
"""Тесты lexicon-сентимента новостей (app_pkg/data/news_sentiment.py).

Лента подменяется (get_symbol_news монкипатчится) — сети нет. Ветки: успех /
пустая лента / ошибка геттера / лимит / кэш / клип скоринга ([-1, 1]).
"""

import pytest

from app_pkg.data import news_sentiment as ns
from app_pkg.data.news_sentiment import (_score_headline,
                                         get_news_sentiment)


@pytest.fixture(autouse=True)
def _clear_cache():
    ns._CACHE.clear()
    yield
    ns._CACHE.clear()


def _feed(titles):
    return {"news": [{"id": f"t{i}", "title": t, "url": f"https://a/{i}", }
                     for i, t in enumerate(titles)],
            "count": len(titles), "sources": ["finnhub"], "fetched_at": 0}


# ------------------------------------------------------------- успех
def test_news_sentiment_success(monkeypatch):
    monkeypatch.setattr(ns, "get_symbol_news", lambda s: _feed([
        "Bitcoin surges to a record high",   # +1
        "Crypto crash: fear spreads as prices plunge",  # -1
        "Rates unchanged as expected",       # 0
    ]))
    out = get_news_sentiment("BTCUSDT")
    assert out["ok"] is True
    assert out["sample_size"] == 3
    assert out["bullish_count"] == 1
    assert out["bearish_count"] == 1
    assert out["neutral_count"] == 1
    assert out["avg_sentiment"] == 0.0
    assert out["worst_headline_score"] == -1.0
    assert out["best_headline_score"] == 1.0
    assert isinstance(out["ts"], int)


def test_news_sentiment_case_insensitive(monkeypatch):
    monkeypatch.setattr(ns, "get_symbol_news", lambda s: _feed([
        "SURGES and RALLY", "BEARISH DOWNTrend"]))
    out = get_news_sentiment("BTCUSDT")
    assert out["bullish_count"] == 1
    assert out["bearish_count"] == 1


# ------------------------------------------------- пустая лента / ошибка
def test_news_sentiment_empty_feed(monkeypatch):
    monkeypatch.setattr(ns, "get_symbol_news", lambda s: _feed([]))
    out = get_news_sentiment("EURUSD")
    assert out["ok"] is False
    assert out["sample_size"] == 0
    assert out["avg_sentiment"] is None
    assert out["best_headline_score"] is None


def test_news_sentiment_getter_error(monkeypatch):
    def boom(symbol):
        raise RuntimeError("news down")

    monkeypatch.setattr(ns, "get_symbol_news", boom)
    out = get_news_sentiment("BTCUSDT")
    assert out["ok"] is False
    assert out["sample_size"] == 0
    assert out["avg_sentiment"] is None


# -------------------------------------------------------------- лимит
def test_news_sentiment_limit(monkeypatch):
    titles = [f"neutral headline #{i}" for i in range(50)]
    monkeypatch.setattr(ns, "get_symbol_news", lambda s: _feed(titles))
    out = get_news_sentiment("BTCUSDT", limit=5)
    assert out["sample_size"] == 5
    assert out["neutral_count"] == 5


# -------------------------------------------------------------- кэш
def test_news_sentiment_cache(monkeypatch):
    calls = {"n": 0}

    def fake_news(symbol):
        calls["n"] += 1
        return _feed(["Bitcoin surges"])

    monkeypatch.setattr(ns, "get_symbol_news", fake_news)
    a = get_news_sentiment("BTCUSDT")
    b = get_news_sentiment("BTCUSDT")
    assert a is b and calls["n"] == 1


# ---------------------------------------------- клип скоринга [-1; 1]
def test_score_headline_clips_and_counts():
    assert _score_headline("surge") == 1.0
    assert _score_headline("crash") == -1.0
    assert _score_headline("neutral phrasing") == 0.0
    # 5 бычьих против 2 медвежьих -> (5-2)/7, строго в [-1, 1]
    s = _score_headline("surge surge surge surge surge crash crash")
    assert -1.0 <= s <= 1.0
    assert s == pytest.approx(3 / 7, abs=1e-9)