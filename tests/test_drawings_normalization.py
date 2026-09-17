"""Тесты нормализации рисунков ИИ в видимый диапазон графика.

Проверяем _normalize_model_drawings(candles, visible_range):
  - завышенные price/time от модели (999999, now+1год) клампятся
    в [min(low), max(high)] окна и в visible_range;
  - trendline с двумя точками на одной свече — точки разносятся
    (или рисунок отбрасывается, если свечей больше нет);
  - fib с совпавшими временем — разносится по границам окна;
  - корректные координаты модели не изменяются;
  - /api/ai-analysis прокидывает body.visible_range в нормализацию.
"""

import pytest

from app_pkg import create_app
from app_pkg.ai.agents import _normalize_model_drawings

BASE_TS = 1700000000
STEP = 3600
N = 60


def _candles(n=N):
    """Свечи: close растёт, high=close+1, low=close-1 (границы известны)."""
    out = []
    for i in range(n):
        c = 100.0 + i
        out.append({
            "time": BASE_TS + i * STEP,
            "open": c, "high": c + 1.0, "low": c - 1.0,
            "close": c, "volume": 10.0,
        })
    return out


_CANDLES = _candles()
LOW_MIN = 99.0      # min(low) = 100 - 1
HIGH_MAX = 160.0    # max(high) = 159 + 1
VR = {"from": BASE_TS, "to": BASE_TS + (N - 1) * STEP}


def test_crazy_coords_clamped():
    """price=999999, time=now+1год -> в границах рынка и visible_range."""
    far_time = BASE_TS + 365 * 24 * 3600  # +1 год
    raw = [{
        "type": "trendline",
        "points": [
            {"time": far_time, "price": 999999.0},
            {"time": BASE_TS + 10 * STEP, "price": 999999.0},
        ],
    }]
    out = _normalize_model_drawings(raw, _CANDLES, VR)

    assert len(out) == 1
    for p in out[0]["points"]:
        assert VR["from"] <= p["time"] <= VR["to"]
        assert p["time"] in [c["time"] for c in _CANDLES]  # снап к свече
        assert LOW_MIN <= p["price"] <= HIGH_MAX
    # завышенная цена упала к максимуму рынка
    assert out[0]["points"][0]["price"] == pytest.approx(HIGH_MAX)


def test_trendline_same_time_shifted():
    """Trendline: обе точки на одной свече -> вторая сдвигается на +1 свечу."""
    t = BASE_TS + 30 * STEP
    raw = [{
        "type": "trendline",
        "points": [{"time": t, "price": 130.0}, {"time": t, "price": 131.0}],
    }]
    out = _normalize_model_drawings(raw, _CANDLES, VR)

    assert len(out) == 1
    times = [p["time"] for p in out[0]["points"]]
    assert times[0] == t
    assert times[1] == t + STEP  # следующая свеча


def test_trendline_same_time_dropped_when_no_next_candle():
    """Некуда сдвигать (одна свеча) -> рисунок отбрасывается."""
    one = [_CANDLES[0]]
    raw = [{
        "type": "trendline",
        "points": [
            {"time": one[0]["time"], "price": 100.0},
            {"time": one[0]["time"], "price": 101.0},
        ],
    }]
    assert _normalize_model_drawings(raw, one, VR) == []


def test_fib_same_time_spread_to_window():
    """Fib: время совпало -> точки разносятся по границам окна."""
    t = BASE_TS + 30 * STEP
    raw = [{
        "type": "fib",
        "points": [{"time": t, "price": 120.0}, {"time": t, "price": 140.0}],
    }]
    out = _normalize_model_drawings(raw, _CANDLES, VR)

    assert len(out) == 1
    times = [p["time"] for p in out[0]["points"]]
    assert times[0] == VR["from"]
    assert times[1] == VR["to"]


def test_valid_coords_unchanged():
    """Live: корректные координаты модели остаются без изменений."""
    raw = [{
        "type": "trendline",
        "points": [
            {"time": BASE_TS + 10 * STEP, "price": 111.0},
            {"time": BASE_TS + 40 * STEP, "price": 141.0},
        ],
    }]
    out = _normalize_model_drawings(raw, _CANDLES, VR)

    assert len(out) == 1
    pts = out[0]["points"]
    assert [p["time"] for p in pts] == [BASE_TS + 10 * STEP,
                                        BASE_TS + 40 * STEP]
    assert pts[0]["price"] == pytest.approx(111.0)
    assert pts[1]["price"] == pytest.approx(141.0)


# ------------------------------------------------------------------ роут
@pytest.fixture(autouse=True)
def _isolate():
    from app_pkg.cache import _VERDICT_CACHE
    from app_pkg.data import fetch
    fetch.data_cache.clear()
    _VERDICT_CACHE.clear()
    yield
    fetch.data_cache.clear()
    _VERDICT_CACHE.clear()


@pytest.fixture()
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_route_passes_visible_range(client, monkeypatch):
    """/api/ai-analysis: body.visible_range -> нормализация рисунков."""
    import pandas as pd

    def fake_analysis(context):
        return {
            "signal": "BUY", "confidence": 0.6, "reason": "ok",
            "indicator_signals": {}, "price_levels": {},
            "suggested_drawings": [{
                "type": "trendline",
                "points": [
                    {"time": BASE_TS + 365 * 24 * 3600, "price": 999999.0},
                    {"time": BASE_TS + 5 * STEP, "price": 105.0},
                ],
            }],
        }

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(
            [c["time"] for c in _CANDLES], unit="s", utc=True),
        "open": [c["open"] for c in _CANDLES],
        "high": [c["high"] for c in _CANDLES],
        "low": [c["low"] for c in _CANDLES],
        "close": [c["close"] for c in _CANDLES],
        "volume": [c["volume"] for c in _CANDLES],
    })
    monkeypatch.setattr("app_pkg.routes.ai.get_series_df",
                        lambda *a, **k: df)
    monkeypatch.setattr("app_pkg.routes.ai.build_multi_tf_context",
                        lambda *a, **k: "")
    monkeypatch.setattr("app_pkg.routes.ai.format_tf_section",
                        lambda *a, **k: ("", None))
    monkeypatch.setattr("app_pkg.routes.ai.db_get_all_drawings",
                        lambda *a, **k: [])
    monkeypatch.setattr("app_pkg.routes.ai.db_add_drawing", lambda *a, **k: None)
    monkeypatch.setattr("app_pkg.routes.ai._llm_vision_request",
                        lambda *a, **kw: (_ for _ in ()).throw(
                            RuntimeError("vision unavailable")))
    from app_pkg import config
    if config.GROQ_AGENT_MODE == 2:
        monkeypatch.setattr("app_pkg.routes.ai.analyze_with_agents",
                            fake_analysis)
    else:
        monkeypatch.setattr("app_pkg.routes.ai.analyze_with_ai", fake_analysis)

    resp = client.post("/api/ai-analysis", json={
        "symbol": "BTCUSDT", "timeframe": "1H",
        "visible_range": VR,
    })
    assert resp.status_code == 200
    data = resp.get_json()
    drawings = data["drawings"]
    assert len(drawings) == 1
    for p in drawings[0]["points"]:
        assert VR["from"] <= p["time"] <= VR["to"]
        assert LOW_MIN <= p["price"] <= HIGH_MAX
