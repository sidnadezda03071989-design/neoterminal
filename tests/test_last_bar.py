"""Тесты /api/last-bar (без сети, get_series_df мокается).

Проверяем: формат бара {time, open, high, low, close, volume} с int-временем
в секундах, tail(limit) из полного кеша без его инвалидации, непустые свежие
indicators последнего бара (совпадают с compute_indicators(df).iloc[-1]),
"candle": null / "indicators": null при пустых данных и 400 на невалидный
symbol/timeframe.
"""

import pandas as pd
import pytest

from app_pkg import config, create_app
from app_pkg.data import fetch
from app_pkg.indicators import compute_indicators

_ROWS = [
    (1700000000, 1.0, 2.0, 0.5, 1.5, 10.0),
    (1700003600, 1.5, 3.0, 1.0, 2.5, 20.0),
    (1700007200, 2.5, 4.0, 2.0, 3.5, 30.0),
]


def _df(rows):
    """Мини-DF со схемой timestamp/open/high/low/close/volume."""
    return pd.DataFrame({
        "timestamp": pd.to_datetime([r[0] for r in rows], unit="s", utc=True),
        "open": [float(r[1]) for r in rows],
        "high": [float(r[2]) for r in rows],
        "low": [float(r[3]) for r in rows],
        "close": [float(r[4]) for r in rows],
        "volume": [float(r[5]) for r in rows],
    })


@pytest.fixture(autouse=True)
def _clear_cache():
    """Изолируем общий data_cache между тестами."""
    fetch.data_cache.clear()
    yield
    fetch.data_cache.clear()


# ----------------------------------------------------------- get_series_df tail
def test_get_series_df_limit_tail_from_cache(monkeypatch):
    """limit=2 обрезает полный DF на выходе: кеш остаётся полным (tail, без reset)."""
    monkeypatch.setattr(fetch, "fetch_ohlcv",
                        lambda *a, **k: _df(_ROWS))  # как будто кеш полный

    df = fetch.get_series_df("BTCUSDT", "1H", limit=2)
    assert len(df) == 2
    assert int(df["timestamp"].iloc[-1].timestamp()) == 1700007200

    # Второй вызов — из кеша (источник больше не дёргается), снова tail(2).
    df2 = fetch.get_series_df("BTCUSDT", "1H", limit=2)
    assert len(df2) == 2
    assert list(df2["close"]) == list(df["close"])


# ------------------------------------------------------------------ роуты
@pytest.fixture()
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_route_last_bar_ok(client, monkeypatch):
    """Свежий бар + indicators последнего бара по tail(LAST_BAR_HISTORY)."""
    calls = []

    def fake_get_series_df(symbol, tf, limit=1000, force=False):
        calls.append((symbol, tf, limit))
        return _df(_ROWS)

    monkeypatch.setattr("app_pkg.routes.data.get_series_df", fake_get_series_df)
    resp = client.get("/api/last-bar?symbol=BTCUSDT&timeframe=1H")
    assert resp.status_code == 200
    data = resp.get_json()
    assert calls == [("BTCUSDT", "1H", config.LAST_BAR_HISTORY)]
    assert data["candle"]["time"] == 1700007200
    assert data["candle"]["close"] == pytest.approx(3.5)
    assert isinstance(data["ts"], float)  # utils.now_sec() — float, как и в /api/data

    # indicators: непустой словарь, значения совпадают с compute_indicators(df)
    assert isinstance(data["indicators"], dict)
    assert data["indicators"], "indicators должен быть непустым при наличии данных"
    expected = compute_indicators(_df(_ROWS)).iloc[-1]
    for key, value in expected.items():
        got = data["indicators"][key]
        if pd.isna(value):
            assert got is None
        else:
            assert got == pytest.approx(float(value))


def test_route_last_bar_indicators_match_compute(client, monkeypatch):
    """indicators последнего бара == скаляры compute_indicators(df).iloc[-1]."""
    monkeypatch.setattr(
        "app_pkg.routes.data.get_series_df",
        lambda *a, **k: _df(_ROWS),
    )
    resp = client.get("/api/last-bar?symbol=BTCUSDT&timeframe=1H")
    data = resp.get_json()
    ind = data["indicators"]
    expected = compute_indicators(_df(_ROWS)).iloc[-1]
    # sma20/sma50 на 3 барах — NaN -> None в ответе; rsi — число.
    assert ind["sma50"] is None
    assert ind["sma20"] is None
    assert ind["rsi"] == pytest.approx(float(expected["rsi"]))
    # Все значения — скаляры (float или None), никакой вложенности {time, value}.
    for v in ind.values():
        assert v is None or isinstance(v, (int, float))


def test_route_last_bar_empty_candle_null(client, monkeypatch):
    """Пустой df -> candle: null, indicators: null."""
    monkeypatch.setattr(
        "app_pkg.routes.data.get_series_df",
        lambda *a, **k: pd.DataFrame(),
    )
    resp = client.get("/api/last-bar?symbol=BTCUSDT&timeframe=1H")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["candle"] is None
    assert data["indicators"] is None


def test_route_last_bar_bad_params(client, monkeypatch):
    monkeypatch.setattr(
        "app_pkg.routes.data.get_series_df", lambda *a, **k: pd.DataFrame())
    resp = client.get("/api/last-bar?symbol=FOO&timeframe=1H")
    assert resp.status_code == 400
    assert "error" in resp.get_json()
    resp = client.get("/api/last-bar?symbol=BTCUSDT&timeframe=9m")
    assert resp.status_code == 400
    assert "error" in resp.get_json()
