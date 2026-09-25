import pandas as pd

from app_pkg import config
from app_pkg.data import watchlist as watchlist_provider
from app_pkg.routes import data as data_routes
from app_pkg.routes.data import _daily_change


def test_daily_change_uses_previous_bar_before_day_start():
    df = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2026-09-25T23:00:00Z",
            "2026-09-26T00:00:00Z",
            "2026-09-26T01:00:00Z",
        ]),
        "open": [99.0, 101.0, 106.0],
        "close": [100.0, 105.0, 110.0],
    })

    assert _daily_change(df) == 10.0


def test_daily_change_falls_back_to_current_day_open():
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-09-26T00:00:00Z"]),
        "open": [100.0],
        "close": [110.0],
    })

    assert _daily_change(df) == 10.0
    assert set(config.SYMBOLS) == set(config.ASSET_NAMES)


def test_watchlist_item_uses_fast_live_snapshot_without_history(monkeypatch):
    def live_bar(symbol, tf):
        if (symbol, tf) == ("BTCUSDT", "1D"):
            return {"open": 100.0, "close": 105.0, "volume": 42.0}
        return None

    def no_cache(*_args, **_kwargs):
        raise AssertionError("live snapshot must not touch OHLCV cache")

    monkeypatch.setattr(data_routes, "get_live_bar", live_bar)
    monkeypatch.setattr(data_routes, "get_cached_series_df", no_cache)

    item = data_routes._watchlist_item("BTCUSDT")

    assert item["price"] == 105.0
    assert item["change"] == 5.0
    assert item["source"] == "live"


def test_watchlist_item_falls_back_to_cached_bars(monkeypatch):
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-09-26T00:00:00Z"]),
        "open": [200.0],
        "close": [190.0],
        "volume": [7.0],
    })
    monkeypatch.setattr(data_routes, "get_live_bar", lambda *_args: None)
    monkeypatch.setattr(data_routes, "get_cached_series_df", lambda *_args: df)

    item = data_routes._watchlist_item("EURUSD")

    assert item["price"] == 190.0
    assert item["change"] == -5.0
    assert item["source"] == "cache"


def test_watchlist_item_uses_bounded_forex_fallback(monkeypatch):
    monkeypatch.setattr(data_routes, "get_live_bar", lambda *_args: None)
    monkeypatch.setattr(
        data_routes,
        "get_cached_series_df",
        lambda *_args: pd.DataFrame(),
    )
    quotes = {"EURUSD": {"price": 1.08425, "change": 0.42}}

    item = data_routes._watchlist_item("EURUSD", quotes)

    assert item["price"] == 1.08425
    assert item["change"] == 0.42
    assert item["source"] == "fallback"


def test_watchlist_provider_batches_and_caches_quotes(monkeypatch):
    calls = []

    def fake_fetch_market(market, symbols):
        calls.append((market, tuple(symbols)))
        if market == "cfd":
            return [{"s": "OANDA:XAUUSD", "d": ["XAUUSD", 2400.5, 0.7]}]
        return [
            {"s": "OANDA:EURUSD", "d": ["EURUSD", 1.08, 0.3]},
            {"s": "OANDA:GBPUSD", "d": ["GBPUSD", 1.27, -0.2]},
        ]

    monkeypatch.setattr(watchlist_provider, "_fetch_market", fake_fetch_market)
    with watchlist_provider._cache_lock:
        watchlist_provider._cache.update({"ts": 0.0, "values": {}})

    first = watchlist_provider.get_forex_quotes()
    second = watchlist_provider.get_forex_quotes()

    assert first["EURUSD"]["price"] == 1.08
    assert first["XAUUSD"]["change"] == 0.7
    assert second == first
    assert len(calls) == 2

