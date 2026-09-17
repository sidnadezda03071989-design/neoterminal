"""Тесты history_limit в get_series_df (без сети, fetch_ohlcv мокается).

Проверяем: tail из живого кеша без похода в источник, докачку СТАРЫХ баров
с PREPEND'ом (старые свечи не теряются), лимит history_limit как потолок
докачки, и фикс пустого кеша (пустой df не кэшируется).
"""

import pandas as pd
import pytest

from app_pkg.data import fetch


def _mk(n, end_ts, step=3600):
    """n свечей, заканчивающихся end_ts (включительно), шаг step секунд."""
    ts = [end_ts - i * step for i in range(n)][::-1]
    return pd.DataFrame({
        "timestamp": pd.to_datetime(ts, unit="s", utc=True),
        "open": [1.0] * n,
        "high": [2.0] * n,
        "low": [0.5] * n,
        "close": [1.5 + i for i in range(n)],
        "volume": [10.0] * n,
    })


@pytest.fixture(autouse=True)
def _clear_cache():
    """Изолируем общий data_cache между тестами."""
    fetch.data_cache.clear()
    yield
    fetch.data_cache.clear()


def test_tail_from_cache_no_refetch(monkeypatch):
    """len(cached) >= limit: отдаётся tail(limit), источник не дёргается."""
    calls = []

    def fake_ohlcv(symbol, tf, limit=1000, start_sec=None, end_sec=None):
        calls.append((limit, end_sec))
        return _mk(1000, 1_700_000_000)

    monkeypatch.setattr(fetch, "fetch_ohlcv", fake_ohlcv)

    df1 = fetch.get_series_df("BTCUSDT", "1H", limit=500, history_limit=1000)
    assert len(df1) == 500
    assert len(calls) == 1

    df2 = fetch.get_series_df("BTCUSDT", "1H", limit=1000, history_limit=1000)
    assert len(df2) == 1000
    assert len(calls) == 1  # из кеша, источник не дёргался


def test_topup_prepends_older_bars(monkeypatch):
    """В кеше 500, просят 1000: докачка 500 СТАРЫХ баров назад + PREPEND."""
    calls = []
    T = 1_700_000_000

    def fake_ohlcv(symbol, tf, limit=1000, start_sec=None, end_sec=None):
        calls.append((limit, end_sec))
        if end_sec is None:
            return _mk(500, T)  # самые свежие
        return _mk(500, int(end_sec))  # более старые (end_sec уже -1с)

    monkeypatch.setattr(fetch, "fetch_ohlcv", fake_ohlcv)

    df1 = fetch.get_series_df("BTCUSDT", "1H", limit=500, history_limit=1000)
    assert len(df1) == 500

    df2 = fetch.get_series_df("BTCUSDT", "1H", limit=1000, history_limit=1000)
    assert len(df2) == 1000
    assert len(calls) == 2
    # докачка: только недостающее, назад во времени
    assert calls[1][0] == 500
    assert calls[1][1] == pytest.approx(T - 499 * 3600 - 1.0)

    # PREPEND: старые бары сохранились, порядок по времени возрастающий
    assert df2["timestamp"].is_monotonic_increasing
    assert df2["timestamp"].is_unique
    merged = fetch.data_cache[("BTCUSDT", "1H")]["df"]
    assert len(merged) == 1000
    # стык: докачанные старые заканчиваются на (первый_кеша - 1с)
    assert int(merged["timestamp"].iloc[499].timestamp()) == T - 499 * 3600 - 1.0
    assert int(merged["timestamp"].iloc[500].timestamp()) == T - 499 * 3600

    # Третий вызов — уже без похода в источник.
    df3 = fetch.get_series_df("BTCUSDT", "1H", limit=1000, history_limit=1000)
    assert len(df3) == 1000
    assert len(calls) == 2


def test_topup_respects_history_limit_ceiling(monkeypatch):
    """history_limit — потолок: если кеш на нём, докачки нет."""
    calls = []

    def fake_ohlcv(symbol, tf, limit=1000, start_sec=None, end_sec=None):
        calls.append((limit, end_sec))
        return _mk(300, 1_700_000_000)

    monkeypatch.setattr(fetch, "fetch_ohlcv", fake_ohlcv)

    df1 = fetch.get_series_df("BTCUSDT", "1H", limit=300, history_limit=300)
    assert len(df1) == 300
    assert len(calls) == 1

    # limit=1000 > кеша, но history_limit=300 <= кеша: отдаём как есть.
    df2 = fetch.get_series_df("BTCUSDT", "1H", limit=1000, history_limit=300)
    assert len(df2) == 300
    assert len(calls) == 1


def test_topup_trims_merged_to_history_limit(monkeypatch):
    """После PREPEND кеш не раздувается сверх history_limit."""
    T = 1_700_000_000

    def fake_ohlcv(symbol, tf, limit=1000, start_sec=None, end_sec=None):
        if end_sec is None:
            return _mk(500, T)
        return _mk(1000, int(end_sec))  # источник щедрее, чем нужно

    monkeypatch.setattr(fetch, "fetch_ohlcv", fake_ohlcv)

    df = fetch.get_series_df("BTCUSDT", "1H", limit=1000, history_limit=1000)
    assert len(df) == 1000
    assert len(fetch.data_cache[("BTCUSDT", "1H")]["df"]) == 1000


def test_empty_df_not_cached(monkeypatch):
    """Пустой ответ источника: пустой df НЕ пишется в кеш, фикс бага."""
    calls = []

    def fake_ohlcv(symbol, tf, limit=1000, start_sec=None, end_sec=None):
        calls.append(1)
        return pd.DataFrame(columns=fetch._EMPTY_COLUMNS)

    monkeypatch.setattr(fetch, "fetch_ohlcv", fake_ohlcv)

    df = fetch.get_series_df("BTCUSDT", "1H", limit=500, history_limit=1000)
    assert df.empty
    assert ("BTCUSDT", "1H") not in fetch.data_cache

    # Повторный вызов снова идёт в источник (кеш «мёртвым» не остался).
    fetch.get_series_df("BTCUSDT", "1H", limit=500, history_limit=1000)
    assert len(calls) == 2


def test_default_history_limit_from_config(monkeypatch):
    """Без history_limit используется HISTORY_LIMITS[tf]."""
    seen = {}

    def fake_ohlcv(symbol, tf, limit=1000, start_sec=None, end_sec=None):
        seen["limit"] = limit
        return _mk(10, 1_700_000_000)

    monkeypatch.setattr(fetch, "fetch_ohlcv", fake_ohlcv)
    fetch.get_series_df("BTCUSDT", "5m", limit=5)
    assert seen["limit"] == 20000
