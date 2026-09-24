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


# ------------------------------------------- apply_live_merge: идемпотентность
def _live_bar(ts, o, h, lo, c, v):
    """Кладёт бар в live_bars и возвращает «снятие» (cleanup)."""
    from app_pkg.data import live
    with live._live_bars_lock:
        live.live_bars[("BTCUSDT", "1m")] = {
            "ts": int(ts), "open": o, "high": h, "low": lo,
            "close": c, "volume": v, "closed": False,
        }

    def _cleanup():
        with live._live_bars_lock:
            live.live_bars.pop(("BTCUSDT", "1m"), None)
    return _cleanup


def test_apply_live_merge_is_idempotent():
    """Повторный merge с тем же live-баром даёт тот же результат.

    Регрессия «10.07 / 10.12»: раньше high/low накапливались через max/min,
    а close перезаписывался. Параллельные вызовы (/api/data и SSE-push)
    собирали бар из двух разных тиков, и последняя свеча «дрожала».
    """
    base = _df([(1700000000, 1.0, 2.0, 0.5, 1.5, 10.0)])
    cleanup = _live_bar(1700000000, 1.2, 2.4, 0.4, 2.1, 12.0)
    try:
        once = fetch.apply_live_merge("BTCUSDT", "1m", base)
        twice = fetch.apply_live_merge("BTCUSDT", "1m", once)
        assert float(twice["high"].iloc[-1]) == 2.4   # не 2.4 + накопление
        assert float(twice["low"].iloc[-1]) == 0.4
        assert float(twice["close"].iloc[-1]) == 2.1
        # Исходный df не мутирован (copy=True по умолчанию).
        assert float(base["close"].iloc[-1]) == 1.5
        assert float(base["high"].iloc[-1]) == 2.0
    finally:
        cleanup()


def test_apply_live_merge_copy_false_mutates_in_place():
    """copy=False — in-place (горячий путь), без лишней аллокации."""
    base = _df([(1700000000, 1.0, 2.0, 0.5, 1.5, 10.0)])
    cleanup = _live_bar(1700000000, 1.2, 2.7, 0.4, 2.1, 12.0)
    try:
        out = fetch.apply_live_merge("BTCUSDT", "1m", base, copy=False)
        assert out is base
        assert float(base["close"].iloc[-1]) == 2.1
    finally:
        cleanup()


def test_apply_live_merge_new_bar_appends_once():
    """Более свежий live-бар дописывается ровно один раз (идемпотентно)."""
    base = _df([(1700000000, 1.0, 2.0, 0.5, 1.5, 10.0)])
    cleanup = _live_bar(1700000060, 1.5, 2.5, 1.4, 2.2, 5.0)
    try:
        once = fetch.apply_live_merge("BTCUSDT", "1m", base)
        twice = fetch.apply_live_merge("BTCUSDT", "1m", once)
        assert len(once) == 2 and len(twice) == 2   # не 3
    finally:
        cleanup()


def test_live_bar_not_baked_into_cache(monkeypatch):
    """Live-бар НЕ попадает в data_cache.

    Регрессия: раньше _topup_older/холодный путь писали в кеш УЖЕ смерженный
    df. Для крипты TTL кеша 3600с, поэтому живой бар «запекался» на час, и
    последняя свеча застывала или расходилась с /api/last-bar.
    """
    monkeypatch.setattr(fetch, "fetch_ohlcv", lambda *a, **k: _df(_ROWS))
    cleanup = _live_bar(1700007200, 2.5, 9.9, 2.0, 9.9, 99.0)
    try:
        out = fetch.get_series_df("BTCUSDT", "1m", limit=3)
        assert float(out["close"].iloc[-1]) == 9.9       # мерж на выходе есть
        cached = fetch.data_cache[("BTCUSDT", "1m")]["df"]
        assert float(cached["close"].iloc[-1]) == 3.5, \
            "live-бар запечён в кеш — свеча застынет на TTL"
    finally:
        cleanup()


def test_data_api_and_last_bar_agree(monkeypatch, client):
    """/api/data и /api/last-bar отдают ОДИН последний бар.

    Это ровно тот инвариант, нарушение которого давало «предпоследняя 10.07,
    последняя 10.12»: два эндпоинта строили бар из разных снимков live_bars.
    """
    monkeypatch.setattr(fetch, "fetch_ohlcv", lambda *a, **k: _df(_ROWS))
    cleanup = _live_bar(1700007200, 2.5, 4.4, 2.2, 4.1, 33.0)
    try:
        d = client.get("/api/data?symbol=BTCUSDT&timeframe=1m&limit=3").get_json()
        lb = client.get("/api/last-bar?symbol=BTCUSDT&timeframe=1m").get_json()
        a, b = d["candles"][-1], lb["candle"]
        assert a["time"] == b["time"]
        for field in ("open", "high", "low", "close", "volume"):
            assert a[field] == pytest.approx(b[field]), \
                f"расхождение /api/data и /api/last-bar по {field}"
    finally:
        cleanup()


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


def test_cache_hit_applies_live_merge(monkeypatch):
    """Кеш-хит обязан применять apply_live_merge, а не отдавать снапшот кеша.

    Регрессия: для крипты TTL кеша 3600с, и ветка попадания в кеш возвращала
    cached-DF без live-мержа. Из-за этого /api/last-bar?timeframe=1m отдавал
    один и тот же бар до истечения часа — новая 1m-свеча не появлялась.
    """
    fetch_calls = {"n": 0}

    def fake_fetch_ohlcv(symbol, tf, limit=1000, **kw):
        fetch_calls["n"] += 1
        return _df(_ROWS)

    merged_flags = {"called": 0}

    def fake_merge(symbol, tf, df, copy=True):
        merged_flags["called"] += 1
        return df.copy() if copy else df

    monkeypatch.setattr(fetch, "fetch_ohlcv", fake_fetch_ohlcv)
    monkeypatch.setattr(fetch, "apply_live_merge", fake_merge)

    fetch.get_series_df("BTCUSDT", "1m", limit=2)   # холодный кеш
    assert fetch_calls["n"] == 1

    fetch.get_series_df("BTCUSDT", "1m", limit=2)   # попадание в кеш
    assert fetch_calls["n"] == 1, "второй вызов должен идти из кеша"
    assert merged_flags["called"] >= 2, \
        "кеш-хит не применил apply_live_merge (новая свеча не появится)"


def test_route_last_bar_bad_params(client, monkeypatch):
    monkeypatch.setattr(
        "app_pkg.routes.data.get_series_df", lambda *a, **k: pd.DataFrame())
    resp = client.get("/api/last-bar?symbol=FOO&timeframe=1H")
    assert resp.status_code == 400
    assert "error" in resp.get_json()
    resp = client.get("/api/last-bar?symbol=BTCUSDT&timeframe=9m")
    assert resp.status_code == 400
    assert "error" in resp.get_json()
