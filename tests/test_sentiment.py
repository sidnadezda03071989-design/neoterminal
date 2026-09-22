"""Тесты crowd-сентимента (app_pkg/data/sentiment.py).

Сеть не трогаем: requests.get монкипатчится по URL. Ветки: успех / ошибка
сети / не-крипта / кэш.
"""

import pytest

from app_pkg.data import sentiment as sent_mod
from app_pkg.data.sentiment import get_crowd_snapshot


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None


@pytest.fixture(autouse=True)
def _clear_cache():
    sent_mod._CACHE.clear()
    yield
    sent_mod._CACHE.clear()


def _fake_ok_get(url, params=None, **kwargs):
    if "globalLongShortAccountRatio" in url:
        return _Resp([{"symbol": params.get("symbol"),
                       "longAccount": "68.93", "shortAccount": "31.07",
                       "longShortRatio": "2.22", "timestamp": 1}])
    if "takerlongshortRatio" in url:
        return _Resp([{"symbol": params.get("symbol"),
                       "buySellRatio": "1.05", "timestamp": 1}])
    if "alternative.me" in url:
        return _Resp({"data": [{"value": "52",
                                "value_classification": "Neutral"}]})
    raise AssertionError(f"неожиданный url: {url}")


# ------------------------------------------------------------- успех
def test_crowd_success(monkeypatch):
    monkeypatch.setattr(sent_mod.requests, "get", _fake_ok_get)
    out = get_crowd_snapshot("BTCUSDT")
    assert out["ok"] is True
    assert out["ls_ratio"] == 2.22
    assert out["long_pct"] == 68.93
    assert out["short_pct"] == 31.07
    assert out["taker_buy_sell"] == 1.05
    assert out["fear_greed"] == 52
    assert isinstance(out["ts"], int)


def test_crowd_accepts_other_crypto_suffixes(monkeypatch):
    monkeypatch.setattr(sent_mod.requests, "get", _fake_ok_get)
    for symbol in ("ETHUSDT", "SOLBUSD", "FOOUSDC"):
        out = get_crowd_snapshot(symbol)
        assert out["ok"] is True


# --------------------------------------------------------- не-крипта
def test_crowd_non_crypto_is_empty(monkeypatch):
    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        raise AssertionError("сеть не должна вызываться")

    monkeypatch.setattr(sent_mod.requests, "get", boom)
    assert get_crowd_snapshot("EURUSD") == {}
    assert get_crowd_snapshot("XAUUSD") == {}
    assert get_crowd_snapshot("") == {}
    assert calls["n"] == 0


# ----------------------------------------------------- ошибка сети
def test_crowd_network_error_null_schema(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(sent_mod.requests, "get", boom)
    out = get_crowd_snapshot("BTCUSDT")
    assert out["ok"] is False
    for key in ("ls_ratio", "long_pct", "short_pct", "taker_buy_sell",
                "fear_greed", "ts"):
        assert out[key] is None


# -------------------------------------------------------------- кэш
def test_crowd_cache(monkeypatch):
    calls = {"n": 0}

    def fake_fetch(symbol):
        calls["n"] += 1
        return {"ok": True, "ts": 1}

    monkeypatch.setattr(sent_mod, "_fetch_crowd", fake_fetch)
    a = get_crowd_snapshot("BTCUSDT")
    b = get_crowd_snapshot("BTCUSDT")
    assert a is b and calls["n"] == 1


def test_crowd_cache_expires(monkeypatch):
    monkeypatch.setattr(sent_mod, "CACHE_TTL", 0.0)
    calls = {"n": 0}
    monkeypatch.setattr(sent_mod, "_fetch_crowd",
                        lambda s: (calls.__setitem__("n", calls["n"] + 1)
                                   or {"ok": True, "ts": 1}))
    get_crowd_snapshot("BTCUSDT")
    get_crowd_snapshot("BTCUSDT")
    assert calls["n"] == 2