"""Тесты макро (app_pkg/data/macro.py): FRED-срез + экономический календарь.

Сеть не трогаем: requests.get монкипатчится по URL. Ветки: успех / нет ключа /
ошибка сети / не-валюта / кэш / replay-safe ts_override.
"""

from datetime import datetime, timezone

import pytest

from app_pkg import config
from app_pkg.data import macro as macro_mod
from app_pkg.data.macro import get_econ_calendar, get_macro_snapshot

_UTC = timezone.utc
_BASE = int(datetime(2026, 9, 21, 12, 0, tzinfo=_UTC).timestamp())


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None


@pytest.fixture(autouse=True)
def _clear_cache():
    macro_mod._CACHE.clear()
    yield
    macro_mod._CACHE.clear()


def _fake_ok_get(url, params=None, **kwargs):
    if "series/observations" in url:
        rows = {
            "DGS10": [("2026-09-18", "4.21"), ("2026-09-17", "4.18")],
            "FEDFUNDS": [("2026-09-18", "4.25"), ("2026-08-01", "4.50")],
            "DTWEXBGS": [("2026-09-15", "123.45"), ("2026-08-15", "122.00")],
        }
        sid = params["series_id"]
        return _Resp({"observations": [{"date": d, "value": v}
                                       for d, v in rows[sid]]})
    if "calendar/economic" in url:
        assert params["from"] == "2026-09-21"
        assert params["to"] == "2026-09-22"
        return _Resp({"economicCalendar": [
            {"event": "ECB Main Rate", "country": "EUR", "impact": "high",
             "time": "2026-09-21 13:15:00", "estimate": "3.75",
             "actual": "3.75"},
            {"event": "US CPI y/y", "country": "US", "impact": "medium",
             "time": "2026-09-21 10:00:00", "estimate": "2.6", "actual": ""},
            {"event": "China PMI", "country": "CN", "impact": "high",
             "time": "2026-09-21 14:00:00", "estimate": "50.1", "actual": ""},
            {"event": "Housing Starts", "country": "US", "impact": "low",
             "time": "2026-09-21 15:00:00", "estimate": "1.2", "actual": ""},
        ]})
    raise AssertionError(f"неожиданный url: {url}")


# ------------------------------------------------------------- макро: успех
def test_macro_success(monkeypatch):
    monkeypatch.setattr(config, "FRED_API_KEY", "test-key")
    monkeypatch.setattr(macro_mod.requests, "get", _fake_ok_get)
    out = get_macro_snapshot("EURUSD")
    assert out["ok"] is True
    assert out["us10y"]["value"] == 4.21
    assert out["us10y"]["prev_value"] == 4.18
    assert out["us10y"]["change"] == 0.03
    assert out["us10y"]["date"] == "2026-09-18"
    assert out["fed_rate"]["change"] == -0.25
    assert out["dxy"]["value"] == 123.45
    assert out["dxy"]["prev_value"] == 122.00
    assert isinstance(out["ts"], int)


def test_macro_no_key_no_network(monkeypatch):
    monkeypatch.setattr(config, "FRED_API_KEY", "")
    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        raise AssertionError("сеть не должна вызываться")

    monkeypatch.setattr(macro_mod.requests, "get", boom)
    out = get_macro_snapshot("BTCUSDT")
    assert out["ok"] is False
    for series in ("us10y", "fed_rate", "dxy"):
        assert out[series] == {"value": None, "prev_value": None,
                               "change": None, "date": None}
    assert calls["n"] == 0


def test_macro_network_error_null(monkeypatch):
    monkeypatch.setattr(config, "FRED_API_KEY", "test-key")

    def boom(*a, **k):
        raise RuntimeError("fred down")

    monkeypatch.setattr(macro_mod.requests, "get", boom)
    out = get_macro_snapshot("EURUSD")
    assert out["ok"] is False
    assert out["us10y"]["value"] is None
    assert out["ts"] is None


def test_macro_cache(monkeypatch):
    monkeypatch.setattr(config, "FRED_API_KEY", "test-key")
    calls = {"n": 0}

    def fake_fetch(symbol):
        calls["n"] += 1
        return {"us10y": {"value": 4.21}, "ok": True, "ts": 1}

    monkeypatch.setattr(macro_mod, "_fetch_macro", fake_fetch)
    a = get_macro_snapshot("EURUSD")
    b = get_macro_snapshot("EURUSD")
    assert a is b and calls["n"] == 1


# --------------------------------------------------- календарь: успех
def test_calendar_success_filters_and_flags(monkeypatch):
    monkeypatch.setattr(config, "FINNHUB_API_KEY", "test-key")
    monkeypatch.setattr(macro_mod.requests, "get", _fake_ok_get)
    out = get_econ_calendar("EUR", ts_override=_BASE)
    assert out["ok"] is True
    # CN (не EUR/US) и impact=low отфильтрованы
    assert [e["event"] for e in out["events"]] == ["US CPI y/y", "ECB Main Rate"]
    assert out["events"][1]["impact"] == "high"
    # время запроса — от среза: from/to = (дата _BASE .. +1 день)
    # upcoming: ECB через 75 мин (high) -> оба флага вычислены от _BASE
    assert out["high_impact_in_2h"] == 1
    assert out["next_event_in_min"] == 75
    assert out["ts"] == _BASE


def test_calendar_no_key_empty_schema(monkeypatch):
    monkeypatch.setattr(config, "FINNHUB_API_KEY", "")
    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        raise AssertionError("сеть не должна вызываться")

    monkeypatch.setattr(macro_mod.requests, "get", boom)
    out = get_econ_calendar("EUR", ts_override=_BASE)
    assert out["events"] == []
    assert out["high_impact_in_2h"] == 0
    assert out["next_event_in_min"] is None
    assert out["ok"] is False
    assert calls["n"] == 0


def test_calendar_invalid_ccy_is_empty(monkeypatch):
    monkeypatch.setattr(config, "FINNHUB_API_KEY", "test-key")

    def boom(*a, **k):
        raise AssertionError("сеть не должна вызываться")

    monkeypatch.setattr(macro_mod.requests, "get", boom)
    assert get_econ_calendar("BTCUSDT") == {}
    assert get_econ_calendar("") == {}
    assert get_econ_calendar(None) == {}


def test_calendar_network_error_blank(monkeypatch):
    monkeypatch.setattr(config, "FINNHUB_API_KEY", "test-key")

    def boom(*a, **k):
        raise RuntimeError("finnhub down")

    monkeypatch.setattr(macro_mod.requests, "get", boom)
    out = get_econ_calendar("EUR", ts_override=_BASE)
    assert out["events"] == []
    assert out["ok"] is False
    assert out["ts"] == _BASE


def test_calendar_cache(monkeypatch):
    monkeypatch.setattr(config, "FINNHUB_API_KEY", "test-key")
    calls = {"n": 0}

    def fake_fetch(ccy, ts):
        calls["n"] += 1
        return {"events": [], "high_impact_in_2h": 0,
                "next_event_in_min": None, "ok": True, "ts": ts}

    monkeypatch.setattr(macro_mod, "_fetch_calendar", fake_fetch)
    a = get_econ_calendar("EUR", ts_override=_BASE)
    b = get_econ_calendar("EUR", ts_override=_BASE)
    assert a is b and calls["n"] == 1