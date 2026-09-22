"""Тесты микроструктуры стакана (app_pkg/data/micro.py) — без сети.

Проверяем:
  - _compute: spread_norm / ob_imb_10 / bid_sum_10 / ask_sum_10 с клипом;
  - large_trades_ratio (порог 3× медиана нотионала, дробь 0-1);
  - get_micro_snapshot: кеш 30с, деградация в null-схему, не-крипта -> {};
  - JSON-чистота (нет NaN/Inf) и ts.
"""

import pytest

from app_pkg.data import micro


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def _depth(bids, asks):
    return {"bids": [[str(p), str(q)] for p, q in bids],
            "asks": [[str(p), str(q)] for p, q in asks]}


# ---------------------------------------------------------------- расчёты
def test_compute_spread_and_imbalance():
    bids = [(100.0, 100), (99.9, 90), (99.8, 80), (99.7, 70), (99.6, 60),
            (99.5, 50), (99.4, 40), (99.3, 30), (99.2, 20), (99.1, 10),
            (99.0, 5), (98.9, 5)]
    asks = [(100.1, 80), (100.2, 70), (100.3, 60), (100.4, 50), (100.5, 40),
            (100.6, 30), (100.7, 20), (100.8, 15), (100.9, 10), (101.0, 5),
            (101.1, 5), (101.2, 5)]
    out = micro._compute(bids, asks, [])

    assert out["spread_norm"] == pytest.approx(0.1 / 100.05, abs=1e-6)

    def _notional(rows):
        return sum(p * q for p, q in rows)

    bid_sum = _notional(bids[:10])
    ask_sum = _notional(asks[:10])
    assert out["bid_sum_10"] == pytest.approx(bid_sum)
    assert out["ask_sum_10"] == pytest.approx(ask_sum)
    imb = (bid_sum - ask_sum) / (bid_sum + ask_sum)
    assert out["ob_imb_10"] == pytest.approx(imb, abs=1e-4)


def test_compute_imbalance_clipped():
    """«Стакан на 90% на одной стороне» не выходит за ±0.95 клипа."""
    bids = [(100.0, 1000)] * 10
    asks = [(100.1, 1)] * 10
    out = micro._compute(bids, asks, [])
    assert out["ob_imb_10"] == 0.95
    assert out["spread_norm"] == pytest.approx(0.1 / 100.05, abs=1e-6)


def test_compute_spread_none_when_crossed():
    out = micro._compute([(100.0, 1)], [(99.9, 1)], [])
    assert out["spread_norm"] is None  # crossed book -> нет валидного спреда
    assert out["ob_imb_10"] == pytest.approx((100.0 - 99.9) / 199.9, abs=1e-4)


def test_compute_large_trades_ratio():
    trades = [
        {"p": "100", "q": "1"}, {"p": "100", "q": "1"}, {"p": "100", "q": "1"},
        {"p": "100", "q": "1.2"}, {"p": "100", "q": "1.3"},
        {"p": "100", "q": "1.5"}, {"p": "100", "q": "2"}, {"p": "100", "q": "2"},
        {"p": "100", "q": "3"}, {"p": "100", "q": "3"},
        {"p": "100", "q": "20"}, {"p": "100", "q": "15"},
    ]
    notional = sorted(float(t["p"]) * float(t["q"]) for t in trades)
    total = sum(notional)
    med = notional[len(notional) // 2]
    large = sum(n for n in notional if n >= 3.0 * med)
    out = micro._compute([(100.0, 1)], [(100.1, 1)], trades)
    assert out["large_trades_ratio"] == pytest.approx(large / total, abs=1e-4)
    assert 0.0 <= out["large_trades_ratio"] <= 1.0


def test_compute_single_level_depth():
    out = micro._compute([(100.0, 5)], [(100.2, 5)], [])
    assert out["spread_norm"] == pytest.approx(0.2 / 100.1, abs=1e-6)
    assert out["bid_sum_10"] == pytest.approx(500.0)
    assert out["ob_imb_10"] == pytest.approx(-1.0 / 1001.0, abs=1e-4)


def test_compute_empty_book():
    out = micro._compute([], [], [])
    assert out["spread_norm"] is None
    assert out["bid_sum_10"] is None
    assert out["ob_imb_10"] is None


# ------------------------------------------------------------ геттер (кеш)
def test_get_micro_snapshot_fetches_and_caches(monkeypatch):
    depth = _depth([(100.0, 100)], [(100.1, 80)])
    calls = []

    def fake_get(path, params=None):
        calls.append((path, params["symbol"]))
        if path == "/api/v3/depth":
            return _Resp(depth)
        return _Resp([{"p": "100", "q": "1"}])

    monkeypatch.setattr(micro, "_binance_get", fake_get)
    micro._CACHE.clear()
    r1 = micro.get_micro_snapshot("BTCUSDT")
    r2 = micro.get_micro_snapshot("BTCUSDT")
    assert r1["ok"] is True and r2["ok"] is True
    assert len(calls) == 2  # 1 запрос на снимок, второй из кеша
    assert r1["spread_norm"] == pytest.approx(0.1 / 100.05, abs=1e-6)
    assert isinstance(r1["ts"], int)


def test_get_micro_snapshot_cache_ttl(monkeypatch):
    monkeypatch.setattr(micro, "CACHE_TTL", 0.0)  # TTL=0 -> всегда фетч
    calls = []

    def fake_get(path, params=None):
        calls.append(path)
        if path == "/api/v3/depth":
            return _Resp({"bids": [], "asks": []})
        return _Resp([])

    monkeypatch.setattr(micro, "_binance_get", fake_get)
    micro._CACHE.clear()
    micro.get_micro_snapshot("ETHUSDT")
    micro.get_micro_snapshot("ETHUSDT")
    assert len(calls) == 4


def test_get_micro_snapshot_network_failure_blank(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("binance down")

    monkeypatch.setattr(micro, "_binance_get", boom)
    micro._CACHE.clear()
    out = micro.get_micro_snapshot("SOLUSDT")
    assert out["ok"] is False
    for key in ("spread_norm", "ob_imb_10", "bid_sum_10", "ask_sum_10",
                "large_trades_ratio"):
        assert out[key] is None


def test_get_micro_snapshot_non_crypto():
    micro._CACHE.clear()
    assert micro.get_micro_snapshot("EURUSD") == {}
    assert micro.get_micro_snapshot("") == {}


def test_get_micro_snapshot_json_clean(monkeypatch):
    def fake_get(path, params=None):
        if path == "/api/v3/depth":
            return _Resp(_depth([(100.0, 100)], [(100.1, 80)]))
        return _Resp([])

    monkeypatch.setattr(micro, "_binance_get", fake_get)
    micro._CACHE.clear()
    out = micro.get_micro_snapshot("BTCUSDT")
    import json
    assert json.loads(json.dumps(out)) == out
    assert out["ok"] is True