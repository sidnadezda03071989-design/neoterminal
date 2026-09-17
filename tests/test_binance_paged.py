"""Тесты постраничной загрузки свечей Binance (без сети, _binance_get мокается).

Проверяем: объём до 20000 свечей, сдвиг endTime на 1 секунду назад,
обрыв на короткой пачке, обрезку по start_sec и проброс total=limit
из app_pkg.data.fetch.fetch_ohlcv.
"""

import pandas as pd
import pytest

from app_pkg import config
from app_pkg.data import binance, fetch

MS = 1000
PAGE = binance._KLINES_PAGE_LIMIT


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def _kline(ts_ms):
    """Одна свеча Binance: [openTime, o, h, l, c, v, ...]."""
    return [ts_ms, "1.0", "2.0", "0.5", "1.5", "10.0", ts_ms + 59999,
            "0", 0, "0", "0", "0"]


class _FakeBinance:
    """Псевдо-сервер klines: отдаёт до PAGE свечей с endTime (или до now_ms)."""

    def __init__(self, now_ms, step_sec=300, earliest_ms=0):
        self.now_ms = now_ms
        self.step_ms = step_sec * MS
        self.earliest_ms = earliest_ms
        self.calls = []   # params каждого запроса
        self.pages = []   # таймстемпы, реально отданные в каждом ответе

    def get(self, path, params):
        assert path == "/api/v3/klines"
        self.calls.append(dict(params))
        end_ms = params.get("endTime", self.now_ms)
        limit = params["limit"]
        base = self.earliest_ms + ((end_ms - self.earliest_ms) // self.step_ms) * self.step_ms
        ts = [base - i * self.step_ms for i in range(limit)]
        ts = sorted(t for t in ts if t >= self.earliest_ms)  # как Binance: от старых к новым
        self.pages.append(list(ts))
        return _FakeResp([_kline(t) for t in ts])


def _patch(monkeypatch, server):
    monkeypatch.setattr(binance, "_binance_get", server.get)
    monkeypatch.setattr(binance, "_PAGE_SLEEP_SECONDS", 0.0)


def test_paged_two_and_half_thousand_rows(monkeypatch):
    server = _FakeBinance(1_700_000_000_000, step_sec=300)
    _patch(monkeypatch, server)

    df = binance.fetch_binance_paged("BTCUSDT", "5m", total=2500)

    assert len(server.calls) == 3
    assert len(df) == 2500
    assert list(df.columns) == binance._EMPTY_COLUMNS
    assert df["timestamp"].is_unique
    assert df["timestamp"].is_monotonic_increasing
    assert str(df["timestamp"].dt.tz) == "UTC"
    assert all(c["limit"] == PAGE for c in server.calls)
    assert all(c["symbol"] == "BTCUSDT" and c["interval"] == "5m"
               for c in server.calls)
    assert "endTime" not in server.calls[0]  # первая пачка — самые свежие бары
    # каждая следующая страница: endTime = min(предыдущей пачки) - 1 секунда
    for i in range(len(server.pages) - 1):
        assert server.calls[i + 1]["endTime"] == min(server.pages[i]) - MS


def test_paged_twenty_thousand_rows(monkeypatch):
    server = _FakeBinance(1_700_000_000_000, step_sec=60)
    _patch(monkeypatch, server)

    df = binance.fetch_binance_paged("ETHUSDT", "1m", total=20000)

    assert len(server.calls) == 20  # 20 страниц по 1000
    assert len(df) == 20000
    assert df["timestamp"].is_unique
    assert df["timestamp"].is_monotonic_increasing


def test_paged_stops_on_short_page(monkeypatch):
    now_ms = 1_700_000_000_000
    # в истории всего 1200 свечей: первая пачка полная, вторая короткая (200)
    server = _FakeBinance(now_ms, step_sec=300,
                          earliest_ms=now_ms - 1199 * 300 * MS)
    _patch(monkeypatch, server)

    df = binance.fetch_binance_paged("SOLUSDT", "5m", total=20000)

    assert len(server.calls) == 2            # третьего запроса нет
    assert len(server.pages[1]) == 200       # короткая пачка -> обрыв цикла
    assert len(df) == 1200                   # вся доступная история
    assert server.calls[1]["endTime"] == min(server.pages[0]) - MS


def test_paged_respects_start_sec(monkeypatch):
    now_ms = 1_700_000_000_000
    server = _FakeBinance(now_ms, step_sec=300)
    _patch(monkeypatch, server)
    start_sec = 1_699_700_000.0  # ровно на границе бара
    start_ts = pd.to_datetime(start_sec, unit="s", utc=True)

    df = binance.fetch_binance_paged("BTCUSDT", "5m", total=5000,
                                     start_sec=start_sec, end_sec=now_ms / MS)

    # пагинация назад прекращается, как только курсор ушёл левее start_sec
    assert len(server.calls) == 2
    assert server.calls[1]["endTime"] == min(server.pages[0]) - MS
    assert min(server.pages[0]) - MS >= int(start_sec * MS)
    assert min(server.pages[1]) - MS < int(start_sec * MS)
    # строки левее start_sec отброшены, окно не превышает total
    assert (df["timestamp"] >= start_ts).all()
    assert len(df) == 1000
    assert df["timestamp"].is_monotonic_increasing
def test_paged_empty_response(monkeypatch):
    server = _FakeBinance(1_000_000_000, step_sec=300,
                          earliest_ms=2_000_000_000)  # данных нет
    _patch(monkeypatch, server)

    df = binance.fetch_binance_paged("BTCUSDT", "1m", total=5000)

    assert len(server.calls) == 1
    assert df.empty
    assert list(df.columns) == binance._EMPTY_COLUMNS


def test_paged_tolerates_repeated_page(monkeypatch):
    """Источник игнорирует endTime — цикл не должен зацикливаться."""
    server = _FakeBinance(1_700_000_000_000, step_sec=300)
    _patch(monkeypatch, server)

    def same_page(_path, _params, _s=server):
        return _FakeResp([_kline(1_699_700_100_000 + i * 300 * MS)
                          for i in range(PAGE)])

    monkeypatch.setattr(binance, "_binance_get", same_page)

    df = binance.fetch_binance_paged("BTCUSDT", "5m", total=20000)

    assert len(df) == PAGE  # одна успешная пачка, дальше прогресса нет


def test_paged_partial_on_late_error(monkeypatch):
    server = _FakeBinance(1_700_000_000_000, step_sec=300)
    calls = {"n": 0}

    def flaky(path, params):
        calls["n"] += 1
        if calls["n"] > 2:
            raise RuntimeError("boom")
        return server.get(path, params)

    monkeypatch.setattr(binance, "_binance_get", flaky)
    monkeypatch.setattr(binance, "_PAGE_SLEEP_SECONDS", 0.0)

    df = binance.fetch_binance_paged("BTCUSDT", "5m", total=20000)

    assert calls["n"] == 3
    assert len(df) == 2000  # две успешные пачки вместо падения


def test_paged_first_page_error_raises(monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("boom")

    monkeypatch.setattr(binance, "_binance_get", boom)

    with pytest.raises(RuntimeError):
        binance.fetch_binance_paged("BTCUSDT", "5m", total=2000)


def test_fetch_binance_signature_unchanged(monkeypatch):
    server = _FakeBinance(1_700_000_000_000, step_sec=300)
    _patch(monkeypatch, server)

    df = binance.fetch_binance("BTCUSDT", "5m", limit=10,
                               start_sec=1_699_000_000, end_sec=1_699_999_000)

    assert len(server.calls) == 1  # без пагинации: ровно один запрос
    assert server.calls[0] == {"symbol": "BTCUSDT", "interval": "5m", "limit": 10,
                              "startTime": 1_699_000_000_000,
                              "endTime": 1_699_999_000_000}
    assert len(df) == 10


def test_fetch_ohlcv_passes_limit_as_total(monkeypatch):
    seen = {}

    def fake_paged(symbol, interval, total=None, start_sec=None, end_sec=None):
        seen.update(symbol=symbol, interval=interval, total=total,
                    start_sec=start_sec, end_sec=end_sec)
        return pd.DataFrame(columns=fetch._EMPTY_COLUMNS)

    monkeypatch.setattr(binance, "fetch_binance_paged", fake_paged)

    df = fetch.fetch_ohlcv("BTCUSDT", "5m", limit=12345,
                           start_sec=10.0, end_sec=20.0)

    assert seen == {"symbol": "BTCUSDT", "interval": config.BINANCE_TF["5m"],
                    "total": 12345, "start_sec": 10.0, "end_sec": 20.0}
    assert df.empty
    assert list(df.columns) == fetch._EMPTY_COLUMNS