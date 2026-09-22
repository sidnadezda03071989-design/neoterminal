"""Тесты деривативов (app_pkg/data/derivatives.py).

Сеть не трогаем: requests.get монкипатчится по URL. Ветки: успех / ошибка
сети / не-крипта / кэш.
"""

import pytest

from app_pkg.data import derivatives as deriv_mod
from app_pkg.data.derivatives import get_derivatives_snapshot


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None


@pytest.fixture(autouse=True)
def _clear_cache():
    deriv_mod._CACHE.clear()
    yield
    deriv_mod._CACHE.clear()


def _fake_ok_get(url, params=None, **kwargs):
    if url.endswith("/fapi/v1/openInterest"):
        return _Resp({"symbol": params["symbol"], "openInterest": "123456.789",
                      "time": 1})
    if "openInterestHist" in url:
        assert params["period"] == "1h" and params["limit"] == 2
        return _Resp([
            {"symbol": params["symbol"], "sumOpenInterest": "120000.0",
             "timestamp": 2},
            {"symbol": params["symbol"], "sumOpenInterest": "118000.0",
             "timestamp": 1},
        ])
    if url.endswith("/fapi/v1/premiumIndex"):
        return _Resp({"symbol": params["symbol"], "markPrice": "50000",
                      "lastFundingRate": "0.00012000", "time": 1})
    if url.endswith("/api/v3/ticker/price"):
        return _Resp({"symbol": params["symbol"], "price": "49950"})
    if "topLongShortPositionRatio" in url:
        assert params["period"] == "5m"
        return _Resp([{"symbol": params["symbol"], "longShortRatio": "1.35",
                       "timestamp": 1}])
    raise AssertionError(f"неожиданный url: {url}")


# ------------------------------------------------------------- успех
def test_derivatives_success(monkeypatch):
    monkeypatch.setattr(deriv_mod.requests, "get", _fake_ok_get)
    out = get_derivatives_snapshot("BTCUSDT")
    assert out["ok"] is True
    assert out["open_interest"] == 123456.789
    # (120000 - 118000) / 118000 * 100 = 1.694915... -> округлено до 4
    assert out["oi_change_1h_pct"] == pytest.approx(1.6949, abs=0.0001)
    assert out["funding_rate"] == 0.00012000
    # (50000 - 49950) / 49950 * 100 = 0.10010...
    assert out["basis_pct"] == pytest.approx(0.1001, abs=0.0001)
    assert out["top_ls_ratio"] == 1.35
    assert isinstance(out["ts"], int)


# --------------------------------------------------------- не-крипта
def test_derivatives_non_crypto_is_empty(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("сеть не должна вызываться")

    monkeypatch.setattr(deriv_mod.requests, "get", boom)
    assert get_derivatives_snapshot("EURUSD") == {}
    assert get_derivatives_snapshot("XAUUSD") == {}
    assert get_derivatives_snapshot("") == {}


# ----------------------------------------------------- ошибка сети
def test_derivatives_network_error_null_schema(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(deriv_mod.requests, "get", boom)
    out = get_derivatives_snapshot("BTCUSDT")
    assert out["ok"] is False
    for key in ("open_interest", "oi_change_1h_pct", "funding_rate",
                "basis_pct", "top_ls_ratio", "ts"):
        assert out[key] is None


# -------------------------------------------------------------- кэш
def test_derivatives_cache(monkeypatch):
    calls = {"n": 0}

    def fake_fetch(symbol):
        calls["n"] += 1
        return {"ok": True, "ts": 1}

    monkeypatch.setattr(deriv_mod, "_fetch_derivatives", fake_fetch)
    a = get_derivatives_snapshot("BTCUSDT")
    b = get_derivatives_snapshot("BTCUSDT")
    assert a is b and calls["n"] == 1


def test_derivatives_cache_expires(monkeypatch):
    monkeypatch.setattr(deriv_mod, "CACHE_TTL", 0.0)
    calls = {"n": 0}
    monkeypatch.setattr(deriv_mod, "_fetch_derivatives",
                        lambda s: (calls.__setitem__("n", calls["n"] + 1)
                                   or {"ok": True, "ts": 1}))
    get_derivatives_snapshot("SOLUSDT")
    get_derivatives_snapshot("SOLUSDT")
    assert calls["n"] == 2