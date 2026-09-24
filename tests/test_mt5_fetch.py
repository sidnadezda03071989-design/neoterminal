"""Тесты fetch_mt5 (MetaTrader5 мокается, реального терминала не нужно).

Проверяем: переименование tick_volume → volume (фикс KeyError: 'volume'),
приоритет real_volume, пустой ответ → _empty_df(), недоступный MT5 →
_empty_df(). Сигнатура fetch_mt5 не меняется.
"""

import sys

import pytest

from app_pkg.data import mt5 as mt5mod


@pytest.fixture
def fresh_state():
    """Сохраняем/восстанавливаем глобальное состояние модуля mt5."""
    saved_state = dict(mt5mod.mt5_state)
    saved_mod = mt5mod._mt5
    yield
    mt5mod.mt5_state.clear()
    mt5mod.mt5_state.update(saved_state)
    mt5mod._mt5 = saved_mod


def _ready(monkeypatch, fake):
    """Считаем MT5 уже инициализированным — fetch дойдёт до copy_rates."""
    mt5mod.mt5_state.update(init=True, ok=True, detail="connected",
                            symbols=["EURUSD"])
    monkeypatch.setattr(mt5mod, "_mt5", fake)


class _FakeMT5:
    """Псевдо-MetaTrader5: TIMEFRAME_* + copy_rates_* отдают заготовку."""

    TIMEFRAME_M15 = 15

    def __init__(self, rates):
        self._rates = rates

    def copy_rates_from_pos(self, symbol, timeframe, start_pos, count):
        return self._rates

    def copy_rates_range(self, symbol, timeframe, start, end):
        return self._rates


def _row(ts, tick_volume=10.0, real_volume=None, spread=3):
    row = {"time": ts, "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5,
           "tick_volume": tick_volume, "spread": spread}
    if real_volume is not None:
        row["real_volume"] = real_volume
    return row


def test_tick_volume_becomes_volume(monkeypatch, fresh_state):
    """Структура copy_rates без usable real_volume: tick_volume → volume."""
    rates = [_row(1_700_000_000 + i * 900) for i in range(3)]
    _ready(monkeypatch, _FakeMT5(rates))

    df = mt5mod.fetch_mt5("EURUSD", "15m", limit=3)

    assert list(df.columns) == mt5mod._EMPTY_COLUMNS  # spread отброшен
    assert len(df) == 3
    assert (df["volume"] == 10.0).all()
    assert str(df["timestamp"].dt.tz) == "UTC"
    assert df["timestamp"].is_monotonic_increasing


def test_real_volume_preferred_when_non_zero(monkeypatch, fresh_state):
    rates = [_row(1_700_000_000 + i * 900, tick_volume=10.0, real_volume=500.0)
             for i in range(2)]
    _ready(monkeypatch, _FakeMT5(rates))

    df = mt5mod.fetch_mt5("EURUSD", "15m", limit=2)

    assert (df["volume"] == 500.0).all()


def test_zero_real_volume_falls_back_to_tick(monkeypatch, fresh_state):
    """Форекс-кейс: real_volume весь нулевой → берём tick_volume."""
    rates = [_row(1_700_000_000 + i * 900, tick_volume=7.0, real_volume=0.0)
             for i in range(2)]
    _ready(monkeypatch, _FakeMT5(rates))

    df = mt5mod.fetch_mt5("EURUSD", "15m", limit=2)

    assert (df["volume"] == 7.0).all()


def test_no_volume_columns_defaults_to_zero(monkeypatch, fresh_state):
    rates = [{"time": ts, "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5}
             for ts in (1_700_000_000, 1_700_000_900)]
    _ready(monkeypatch, _FakeMT5(rates))

    df = mt5mod.fetch_mt5("EURUSD", "15m", limit=2)

    assert (df["volume"] == 0.0).all()
    assert list(df.columns) == mt5mod._EMPTY_COLUMNS


def test_empty_rates_returns_empty_df(monkeypatch, fresh_state, caplog):
    _ready(monkeypatch, _FakeMT5(None))

    with caplog.at_level("WARNING"):
        df = mt5mod.fetch_mt5("EURUSD", "15m", limit=100)

    assert df.empty
    assert list(df.columns) == mt5mod._EMPTY_COLUMNS
    assert any("пустой ответ" in rec.message for rec in caplog.records)


def test_mt5_not_initialized_returns_empty_df(monkeypatch, fresh_state):
    """MT5 недоступен (SDK не ставится / init=False) → _empty_df, без KeyError."""
    monkeypatch.setitem(sys.modules, "MetaTrader5", None)  # import упадёт
    mt5mod.mt5_state.update(init=False, ok=False, detail="", symbols=[])

    df = mt5mod.fetch_mt5("EURUSD", "15m", limit=100)

    assert df.empty
    assert list(df.columns) == mt5mod._EMPTY_COLUMNS
    assert mt5mod.mt5_state["ok"] is False


def test_full_period_uses_range_and_ignores_limit(monkeypatch, fresh_state):
    """start_sec+end_sec → copy_rates_range на весь период; limit не режет."""
    from datetime import timezone

    calls = []

    class _RangeRecorder(_FakeMT5):
        def copy_rates_range(self, symbol, timeframe, start, end):
            calls.append((symbol, timeframe, start, end))
            return self._rates

        def copy_rates_from_pos(self, *args):  # не должен вызываться
            raise AssertionError("copy_rates_from_pos не должен вызываться")

    rates = [_row(1_700_000_000 + i * 900) for i in range(3)]
    _ready(monkeypatch, _RangeRecorder(rates))

    df = mt5mod.fetch_mt5("EURUSD", "15m", limit=5,
                          start_sec=1_700_000_000.0, end_sec=1_700_044_400.0)

    assert len(df) == 3
    symbol, timeframe, start, end = calls[0]
    assert symbol == "EURUSD"
    assert timeframe == _FakeMT5.TIMEFRAME_M15
    # границы tz-aware UTC, окно = ровно запрошенный период (без локального сдвига)
    assert start.tzinfo is not None and start.utcoffset() == timezone.utc.utcoffset(None)
    assert end.tzinfo is not None and end.utcoffset() == timezone.utc.utcoffset(None)
    assert start.timestamp() == 1_700_000_000.0
    assert end.timestamp() == 1_700_044_400.0


def test_no_bounds_uses_range_anchored_to_now(monkeypatch, fresh_state):
    """Без start/end: НЕ copy_rates_from_pos (время в серверном смещении),
    а copy_rates_range, привязанный к now → истинный UTC."""
    calls = []

    class _RangeRecorder(_FakeMT5):
        def copy_rates_range(self, symbol, timeframe, start, end):
            calls.append((start, end))
            return self._rates

        def copy_rates_from_pos(self, *args):  # не должен вызываться
            raise AssertionError("copy_rates_from_pos не должен вызываться")

    import time as _time

    before = _time.time()
    # Источник в окне [now - limit*step*padding; now] вернул БОЛЬШЕ, чем limit
    # (padding окна) — fetch должен отрезать хвост ровно до limit.
    rates = [_row(1_700_000_000 + i * 900) for i in range(6)]
    _ready(monkeypatch, _RangeRecorder(rates))

    df = mt5mod.fetch_mt5("EURUSD", "15m", limit=4)
    after = _time.time()

    assert len(calls) == 1
    start, end = calls[0]
    # окно = (now - limit*step*padding, now), tz-aware UTC
    assert end.timestamp() - after < 2.0
    assert start.timestamp() < before - 4 * 900  # c учётом calendar padding
    assert start.tzinfo is not None and end.tzinfo is not None
    # хвост режется ровно до limit, отсортирован по возрастанию времени
    assert len(df) == 4
    assert df["timestamp"].is_monotonic_increasing


def test_fetch_mt5_signature_unchanged(monkeypatch, fresh_state):
    """Сигнатура: fetch_mt5(symbol, tf, limit=1000, start_sec=None, end_sec=None)."""
    import inspect

    params = inspect.signature(mt5mod.fetch_mt5).parameters
    assert list(params) == ["symbol", "tf", "limit", "start_sec", "end_sec"]
    assert params["limit"].default == 1000
    assert params["start_sec"].default is None
    assert params["end_sec"].default is None
