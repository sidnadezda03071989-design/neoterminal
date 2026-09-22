# -*- coding: utf-8 -*-
"""Интеграция доп. блоков в Market Snapshot (get_raw_market_data).

Dev-БД и сеть не трогаются: get_series_df подменён синтетическим срезом,
db_get_best_scan_stats -> None, все 5 внешних геттеров монкипатчатся.
Проверяем: проброс данных при успехе, деградацию при {} и exception,
время календаря из среза (replay-safe), JSON-сериализуемость снимка.
"""

import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from app_pkg.data import market_snapshot as ms

_UTC = timezone.utc
# Пятница 22:00 UTC — последняя свеча среза (для проверки replay-времени).
_CANDLE_TS = int(datetime(2026, 2, 20, 22, 0, tzinfo=_UTC).timestamp())
_MON_TS = int(datetime(2026, 2, 16, 14, 0, tzinfo=_UTC).timestamp())

_NULL_CROWD = {"ls_ratio": None, "long_pct": None, "fear_greed": None,
               "ok": False}
_NOT_APPLICABLE = {"applicable": False, "ok": False}


def _mk_df(n=120, end_ts=_CANDLE_TS, step=900):
    ts = pd.to_datetime([end_ts - i * step for i in range(n)][::-1],
                        unit="s", utc=True)
    return pd.DataFrame({
        "timestamp": ts,
        "open": [100.0] * n, "high": [102.0] * n, "low": [98.0] * n,
        "close": [100.0] * n, "volume": [10.0] * n,
    })


@pytest.fixture()
def hermetic(monkeypatch):
    """Снимок герметичен: ни БД, ни сети."""
    monkeypatch.setattr(ms, "get_series_df", lambda *a, **k: _mk_df())
    monkeypatch.setattr(ms.db, "db_get_best_scan_stats", lambda s, tf: None)


def _crowd_ok():
    """Крауд как от провайдера: fear_greed в исходной шкале 0-100."""
    return {"ls_ratio": 2.2, "long_pct": 0.689, "short_pct": 0.311,
            "taker_buy_sell": 1.05, "fear_greed": 52, "ok": True, "ts": 1}


def _crowd_norm():
    """Крауд после унификации шкалы: fear_greed -> 0-1, taker_buy_sell ratio
    -> доля покупок r/(1+r)=1.05/2.05 (контракт снимка)."""
    return {**_crowd_ok(), "fear_greed": 0.52, "taker_buy_sell": 0.5122}


def _macro_ok():
    return {"us10y": {"value": 4.21, "prev_value": 4.18, "change": 0.03,
                      "date": "2026-09-18"},
            "fed_rate": {"value": 4.25, "prev_value": 4.50, "change": -0.25,
                         "date": "2026-09-18"},
            "dxy": {"value": 123.45, "prev_value": 122.0, "change": 1.45,
                    "date": "2026-09-15"},
            "ok": True, "ts": 1}


def _calendar_ok():
    return {"events": [{"time_unix": 10, "event": "ECB", "country": "EUR",
                        "impact": "high", "estimate": 3.75, "actual": 3.75}],
            "high_impact_in_2h": 1, "next_event_in_min": 75,
            "ok": True, "ts": _CANDLE_TS}


def _derivatives_ok():
    return {"open_interest": 123456.0, "oi_change_1h_pct": 1.69,
            "funding_rate": 0.00012, "basis_pct": 0.1, "top_ls_ratio": 1.35,
            "ok": True, "ts": 1}


def _news_ok():
    return {"avg_sentiment": 0.5, "bullish_count": 3, "bearish_count": 1,
            "neutral_count": 2, "sample_size": 6, "worst_headline_score": -1.0,
            "best_headline_score": 1.0, "ok": True, "ts": 1}


def _patch_ok_getters(monkeypatch, calendar=None):
    """Все геттеры отвечают «успех» (calendar — опциональный спай)."""
    monkeypatch.setattr(ms, "get_crowd_snapshot", lambda s: _crowd_ok())
    monkeypatch.setattr(ms, "get_macro_snapshot", lambda s: _macro_ok())
    monkeypatch.setattr(ms, "get_derivatives_snapshot",
                        lambda s: _derivatives_ok())
    monkeypatch.setattr(ms, "get_news_sentiment",
                        lambda s, **k: _news_ok())
    if calendar is None:
        calendar = lambda c, *a, **k: _calendar_ok()  # noqa: E731
    monkeypatch.setattr(ms, "get_econ_calendar", calendar)


# ------------------------------------------------------- успех: крипта
def test_crypto_blocks_success(hermetic, monkeypatch):
    """BTCUSDT: crowd/derivatives/news_sentiment/macro — живые данные."""
    _patch_ok_getters(monkeypatch)
    raw = ms.get_raw_market_data("BTCUSDT", "15m")
    assert raw["sentiment"] == _crowd_norm()
    assert raw["derivatives"] == _derivatives_ok()
    assert raw["news_sentiment"] == _news_ok()
    assert raw["macro"] == _macro_ok()
    # крипте календарь не применим (нет валюты) — блок без вызова геттера
    assert raw["calendar"] == _NOT_APPLICABLE


# ------------------------------------------------------- успех: форекс
def test_forex_blocks_success_and_replay_time(hermetic, monkeypatch):
    """EURUSD: macro/calendar живые, время календаря — из среза (не системные)."""
    seen = {}

    def spy_calendar(ccy, ts_override=None):
        seen["ccy"] = ccy
        seen["ts"] = ts_override
        return _calendar_ok()

    _patch_ok_getters(monkeypatch, calendar=spy_calendar)
    raw = ms.get_raw_market_data("EURUSD", "15m")
    assert raw["macro"] == _macro_ok()
    assert raw["calendar"] == _calendar_ok()
    assert seen["ccy"] == "EUR"
    assert seen["ts"] == _CANDLE_TS  # replay-safe: последняя свеча среза

    # ts_override форсирует и calendar
    seen.clear()
    ms.get_raw_market_data("EURUSD", "15m", ts_override=_MON_TS)
    assert seen["ts"] == _MON_TS


# --------------------------------------- {} от геттеров -> не применимо
def test_empty_getters_degrade_to_not_applicable(hermetic, monkeypatch):
    """{} от геттера: sentiment -> null-схема, остальные -> applicable:false."""
    monkeypatch.setattr(ms, "get_crowd_snapshot", lambda s: {})
    monkeypatch.setattr(ms, "get_macro_snapshot", lambda s: {})
    monkeypatch.setattr(ms, "get_derivatives_snapshot", lambda s: {})
    monkeypatch.setattr(ms, "get_news_sentiment", lambda s, **k: {})
    monkeypatch.setattr(ms, "get_econ_calendar", lambda c, *a, **k: {})

    raw = ms.get_raw_market_data("EURUSD", "15m")
    assert raw["sentiment"] == _NULL_CROWD
    for name in ("macro", "calendar", "derivatives", "news_sentiment"):
        assert raw[name] == _NOT_APPLICABLE, name


# ---------------------------------------------- exception у геттеров
def test_getter_exceptions_produce_null_schemas(hermetic, monkeypatch):
    """Исключение в любом геттере -> null-схема блока с ok:false."""
    def boom(*a, **k):
        raise RuntimeError("provider down")

    monkeypatch.setattr(ms, "get_crowd_snapshot", boom)
    monkeypatch.setattr(ms, "get_macro_snapshot", boom)
    monkeypatch.setattr(ms, "get_derivatives_snapshot", boom)
    monkeypatch.setattr(ms, "get_news_sentiment", boom)
    monkeypatch.setattr(ms, "get_econ_calendar", boom)

    raw = ms.get_raw_market_data("EURUSD", "15m")
    assert raw["sentiment"] == _NULL_CROWD
    assert raw["macro"]["ok"] is False
    for series in ("us10y", "fed_rate", "dxy"):
        assert raw["macro"][series]["value"] is None
    assert raw["derivatives"]["open_interest"] is None
    assert raw["derivatives"]["ok"] is False
    assert raw["news_sentiment"]["sample_size"] == 0
    assert raw["news_sentiment"]["ok"] is False
    assert raw["calendar"]["events"] == []
    assert raw["calendar"]["ok"] is False


# -------------------------------------------------- JSON-сериализуемость
@pytest.mark.parametrize("symbol", ["BTCUSDT", "EURUSD"])
def test_snapshot_always_json_serializable(symbol, hermetic, monkeypatch):
    """Снимок в любом сценарии — валидный JSON из примитивов/вложенных dict."""
    _patch_ok_getters(monkeypatch)
    raw = ms.get_raw_market_data(symbol, "15m")
    _assert_json_contract(raw)

    # деградация ({} от всех геттеров) тоже обязана быть JSON-чистой
    monkeypatch.setattr(ms, "get_crowd_snapshot", lambda s: {})
    monkeypatch.setattr(ms, "get_macro_snapshot", lambda s: {})
    monkeypatch.setattr(ms, "get_derivatives_snapshot", lambda s: {})
    monkeypatch.setattr(ms, "get_news_sentiment", lambda s, **k: {})
    monkeypatch.setattr(ms, "get_econ_calendar", lambda c, *a, **k: {})
    _assert_json_contract(ms.get_raw_market_data(symbol, "15m"))


def _assert_json_contract(raw):
    dumped = json.dumps(raw)
    assert json.loads(dumped) == raw  # round-trip идентичен (нет NaN/Inf)

    def check(node):
        if isinstance(node, dict):
            for value in node.values():
                check(value)
        elif isinstance(node, list):
            for item in node:
                check(item)
        else:
            assert node is None or isinstance(
                node, (int, float, bool, str)), \
                f"запрещённый тип в JSON: {type(node)} ({node!r})"

    check(raw)
    # блоки на месте всегда (структуру ИИ получает даже без данных)
    for name in ("technicals", "scanner_edge", "sentiment", "clock",
                 "macro", "calendar", "derivatives", "news_sentiment"):
        assert name in raw, name


# --------------------------------------------------------- роуты наружу
def test_raw_route_returns_all_blocks(monkeypatch):
    """GET /api/ai-data/raw: снимок со всеми блоками, 200, без сети/БД."""
    from app_pkg import create_app

    app = create_app()
    monkeypatch.setattr(ms, "get_series_df", lambda *a, **k: _mk_df())
    monkeypatch.setattr(ms.db, "db_get_best_scan_stats", lambda s, tf: None)
    monkeypatch.setattr(ms, "get_crowd_snapshot", lambda s: _crowd_ok())
    monkeypatch.setattr(ms, "get_macro_snapshot", lambda s: _macro_ok())
    monkeypatch.setattr(ms, "get_derivatives_snapshot",
                        lambda s: _derivatives_ok())
    monkeypatch.setattr(ms, "get_news_sentiment", lambda s, **k: _news_ok())
    monkeypatch.setattr(ms, "get_econ_calendar",
                        lambda c, *a, **k: _calendar_ok())

    resp = app.test_client().get(
        "/api/ai-data/raw?symbol=BTCUSDT&timeframe=15m")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["sentiment"] == _crowd_norm()
    assert data["derivatives"] == _derivatives_ok()
    assert data["news_sentiment"] == _news_ok()
    assert data["calendar"] == _NOT_APPLICABLE
    _assert_json_contract(data)


def test_extra_routes_register_and_validate(monkeypatch):
    """4 новых blueprint'а: работают, без symbol -> 400."""
    from app_pkg import create_app

    app = create_app()
    monkeypatch.setattr(ms, "get_crowd_snapshot", lambda s: {})
    monkeypatch.setattr(ms.db, "db_get_best_scan_stats", lambda s, tf: None)

    payload = {
        "/api/sentiment": _crowd_ok(),
        "/api/macro": _macro_ok(),
        "/api/calendar": _calendar_ok(),
        "/api/derivatives": _derivatives_ok(),
        "/api/news-sentiment": _news_ok(),
    }
    import app_pkg.routes.extra_data as extra
    monkeypatch.setattr(extra, "get_crowd_snapshot", lambda s: payload["/api/sentiment"])
    monkeypatch.setattr(extra, "get_macro_snapshot", lambda s: payload["/api/macro"])
    monkeypatch.setattr(extra, "get_derivatives_snapshot", lambda s: payload["/api/derivatives"])
    monkeypatch.setattr(extra, "get_news_sentiment", lambda s, **k: payload["/api/news-sentiment"])
    monkeypatch.setattr(extra, "get_econ_calendar", lambda c, **k: payload["/api/calendar"])

    client = app.test_client()
    for url, expected in payload.items():
        symbol = "EURUSD" if url == "/api/calendar" else "BTCUSDT"
        resp = client.get(f"{url}?symbol={symbol}")
        assert resp.status_code == 200, url
        assert resp.get_json() == expected, url
        assert client.get(url).status_code == 400, url

    # не-крипта: геттер вернул {} -> {"applicable": false, "ok": false}
    monkeypatch.setattr(extra, "get_crowd_snapshot", lambda s: {})
    resp = client.get("/api/sentiment?symbol=EURUSD")
    assert resp.status_code == 200
    assert resp.get_json() == {"applicable": False, "ok": False}