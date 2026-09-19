"""Тесты replay-изоляции ИИ-контекста (без сети, все источники мокаются).

Проверяем:
  - build_multi_tf_context/upto_sec: ни одна свеча и ни один тренд
    не содержат данных с time > upto_sec (replay);
  - без upto_sec (live) контекст не режется — видна последняя свеча;
  - _analysis_flow в replay берёт get_replay_df(to_sec=upto_sec) и
    обрезает снапшот свечей; в live — свежий get_series_df без обрезки;
  - routes/chat.py прокидывает replay_time -> upto_sec только в replay;
  - routes/chart-context в replay передаёт upto_sec в тренды/контекст
    с cap по visible_to;
  - cache.invalidate_verdicts_for чистит verdict-кеш по (symbol, timeframe).
"""

import re

import pandas as pd
import pytest

from app_pkg import config
from app_pkg.ai.context import build_multi_tf_context
from app_pkg.ai import chat as ai_chat
from app_pkg.cache import (
    get_cached_verdict,
    invalidate_verdicts_for,
    set_cached_verdict,
    _VERDICT_CACHE,
)
from app_pkg.data import fetch
from app_pkg import create_app

BASE_TS = 1700000000
STEP = 3600  # 1H
N = 100


def _df(n=N, step=STEP):
    """Растущий ряд: close_i = i+1 (все OHLC равны) — будущее легко детектить."""
    times = [BASE_TS + i * step for i in range(n)]
    closes = [float(i + 1) for i in range(n)]
    return pd.DataFrame({
        "timestamp": pd.to_datetime(times, unit="s", utc=True),
        "open": closes,
        "high": closes,
        "low": closes,
        "close": closes,
        "volume": [10.0] * n,
    })


@pytest.fixture(autouse=True)
def _isolate():
    """Изолируем общие кеши данных и вердиктов между тестами."""
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


# ------------------------------------------------------- context: replay/live
def test_replay_context_has_no_candles_after_upto(monkeypatch):
    """Replay: свечи и тренды не содержат time > upto_sec / цен из будущего."""
    df = _df()
    upto = BASE_TS + 59 * STEP  # 60-я свеча, close=60.0
    monkeypatch.setattr("app_pkg.ai.context.get_series_df",
                        lambda *a, **k: df)

    ctx = build_multi_tf_context("BTCUSDT", "1H", upto)

    rows = re.findall(
        r"\[(\d+) ([\d.]*) ([\d.]*) ([\d.]*) ([\d.]*) ", ctx)
    assert rows, "контекст не содержит ни одной свечи"
    assert max(int(r[0]) for r in rows) <= upto
    # Цены последних (будущих) свечей не утекли ни в свечи, ни в Trends.
    assert all(float(r[4]) <= 60.0 for r in rows)
    assert "last_price=100.0" not in ctx
    assert "last_price=60.0" in ctx


def test_live_context_not_cut(monkeypatch):
    """Live: upto_sec=None — контекст заканчивается последней свечой."""
    df = _df()
    monkeypatch.setattr("app_pkg.ai.context.get_series_df",
                        lambda *a, **k: df)

    ctx = build_multi_tf_context("BTCUSDT", "1H")

    times = [int(m.group(1)) for m in re.finditer(r"\[(\d+) ", ctx)]
    assert times and max(times) == BASE_TS + (N - 1) * STEP
    assert "last_price=100.0" in ctx


# ------------------------------------------------------------ chat analysis flow
def test_analysis_flow_replay_uses_replay_df(monkeypatch):
    """_analysis_flow(replay): get_replay_df(to_sec=upto) + обрезка снапшота."""
    df = _df(n=120)
    upto = BASE_TS + 59 * STEP
    captured = {}

    def fake_replay(symbol, tf, from_sec=None, to_sec=None, limit=1200):
        captured["to_sec"] = to_sec
        return df

    monkeypatch.setattr(ai_chat, "get_replay_df", fake_replay)
    monkeypatch.setattr(ai_chat, "get_series_df",
                        lambda *a, **k: pytest.fail("live-источник в replay"))
    monkeypatch.setattr(
        ai_chat, "build_multi_tf_context",
        lambda s, t, upto_sec=None: captured.setdefault("mf_upto", upto_sec) or "")

    def fake_analyze_replay(text):
        captured["user_text"] = text
        return {"signal": "BUY", "confidence": 0.9, "reason": "ok",
                "suggested_drawings": []}

    monkeypatch.setattr(ai_chat, "analyze_with_ai", fake_analyze_replay)
    monkeypatch.setattr(ai_chat, "db_get_all_drawings", lambda *a, **k: [])
    monkeypatch.setattr(config, "GROQ_AGENT_MODE", 1)

    _reply, _drawings, _all, _model, _fb = ai_chat._analysis_flow(
        "проанализируй рынок", "BTCUSDT", "1H", upto_sec=upto, mode="replay")

    assert captured["to_sec"] == pytest.approx(float(upto))
    assert captured.get("mf_upto") == upto
    assert "close=60.0" in captured["user_text"]
    assert "100.0" not in captured["user_text"]


def test_analysis_flow_live_untouched(monkeypatch):
    """_analysis_flow(live): свежий get_series_df, обрезки нет."""
    df = _df()
    captured = {}

    monkeypatch.setattr(ai_chat, "get_series_df",
                        lambda *a, **k: captured.setdefault("df", df))
    monkeypatch.setattr(ai_chat, "get_replay_df",
                        lambda *a, **k: pytest.fail("replay-источник в live"))
    monkeypatch.setattr(ai_chat, "build_multi_tf_context",
                        lambda s, t, upto_sec=None: "")

    def fake_analyze_live(text):
        captured["user_text"] = text
        return {"signal": "HOLD", "confidence": 0.5, "reason": "ok",
                "suggested_drawings": []}

    monkeypatch.setattr(ai_chat, "analyze_with_ai", fake_analyze_live)
    monkeypatch.setattr(ai_chat, "db_get_all_drawings", lambda *a, **k: [])
    monkeypatch.setattr(config, "GROQ_AGENT_MODE", 1)

    ai_chat._analysis_flow("проанализируй рынок", "BTCUSDT", "1H")

    assert "close=100.0" in captured["user_text"]


# ------------------------------------------------------------------- /api/chat
@pytest.mark.parametrize("payload,expected_upto", [
    ({"mode": "replay", "replay_time": 1700013200}, 1700013200),
    ({"mode": "replay", "replay_time": "1700013200.7"}, 1700013200),
    ({"mode": "replay"}, None),                       # нет replay_time
    ({"mode": "live", "replay_time": 1700013200}, None),  # live не режем
])
def test_route_chat_replay_time_passthrough(client, monkeypatch,
                                            payload, expected_upto):
    """routes/chat: upto_sec = replay_time только при mode=replay."""
    captured = {}

    def fake_chat_with_model(message, symbol, timeframe, mode="live",
                             upto_sec=None):
        captured.update(mode=mode, upto_sec=upto_sec)
        return {"reply": "ok"}

    monkeypatch.setattr("app_pkg.routes.chat.chat_with_model",
                        fake_chat_with_model)
    resp = client.post("/api/chat", json={
        "message": "привет", "symbol": "BTCUSDT", "timeframe": "1H", **payload})

    assert resp.status_code == 200
    assert captured["upto_sec"] == expected_upto


# ------------------------------------------------------------ /api/chart-context
def test_route_chart_context_replay_upto(client, monkeypatch):
    """chart-context(replay): upto_sec = время последней видимой свечи."""
    df = _df()
    captured = {}
    last_time = BASE_TS + (N - 1) * STEP
    mid_time = BASE_TS + 50 * STEP

    monkeypatch.setattr("app_pkg.routes.chart_ctx.get_replay_df",
                        lambda *a, **k: df)
    monkeypatch.setattr(
        "app_pkg.routes.chart_ctx._compute_multi_tf_trends",
        lambda s, tf, upto_sec=None: captured.setdefault("trends", upto_sec) or {})
    monkeypatch.setattr(
        "app_pkg.routes.chart_ctx.build_multi_tf_context",
        lambda s, tf, upto_sec=None: captured.setdefault("ctx", upto_sec) or "")
    monkeypatch.setattr("app_pkg.routes.chart_ctx.db_get_all_drawings",
                        lambda *a, **k: [])

    client.get("/api/chart-context?symbol=BTCUSDT&timeframe=1H"
               "&mode=replay&replay_index=99")
    assert captured["trends"] == last_time
    assert captured["ctx"] == last_time

    # Пользователь сдвинул окно: cap по visible_to.
    captured.clear()
    client.get("/api/chart-context?symbol=BTCUSDT&timeframe=1H"
               "&mode=replay&replay_index=99"
               f"&visible_from={BASE_TS}&visible_to={mid_time}")
    assert captured["trends"] == mid_time
    assert captured["ctx"] == mid_time

    # Live: обрезки нет — trends с upto_sec=None, ai_context не строится.
    captured.clear()
    monkeypatch.setattr("app_pkg.routes.chart_ctx.get_series_df",
                        lambda *a, **k: df)
    client.get("/api/chart-context?symbol=BTCUSDT&timeframe=1H&mode=live")
    assert captured["trends"] is None
    assert captured.get("ctx") is None


# ----------------------------------------------------------------------- cache
def test_invalidate_verdicts_for_clears_all_modes():
    """Ключ содержит mode: invalidate_verdicts_for чистит live и replay."""
    set_cached_verdict(("BTCUSDT", "1H", "live", None, None), {"r": 1})
    set_cached_verdict(("BTCUSDT", "1H", "replay", 500, 1700013200), {"r": 2})
    set_cached_verdict(("ETHUSDT", "1H", "live", None, None), {"r": 3})

    removed = invalidate_verdicts_for("BTCUSDT", "1H")

    assert removed == 2
    assert get_cached_verdict(("BTCUSDT", "1H", "live", None, None)) is None
    assert get_cached_verdict(("BTCUSDT", "1H", "replay", 500, 1700013200)) is None
    # Чужой символ не тронут.
    assert get_cached_verdict(("ETHUSDT", "1H", "live", None, None)) == {"r": 3}
