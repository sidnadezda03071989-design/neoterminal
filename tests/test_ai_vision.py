"""Тесты Vision-анализа в /api/ai-analysis (без сети, всё мокается).

Проверяем:
  - с screenshot (base64) вызывается _llm_vision_request и ответ vision_used: true;
  - RuntimeError("vision unavailable") -> fallback на текстовый анализ;
  - без screenshot -> текстовый путь, vision_used: false;
  - кеш: has_screenshot входит в ключ (vision и текст не пересекаются).
"""

import pytest

from app_pkg import config, create_app
from app_pkg.cache import _VERDICT_CACHE, get_cached_verdict
from app_pkg.data import fetch

_PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUg" + "A" * 200  # короткий фейковый PNG

_FAKE_VISION_RESULT = {
    "signal": "BUY",
    "confidence": 0.7,
    "reason": "vision: пробой сопротивления",
    "indicator_signals": {},
    "price_levels": {},
    "suggested_drawings": [],
}

_FAKE_TEXT_RESULT = {
    "signal": "HOLD",
    "confidence": 0.5,
    "reason": "text analysis",
    "indicator_signals": {},
    "price_levels": {},
    "suggested_drawings": [],
}


@pytest.fixture(autouse=True)
def _isolate():
    """Изолируем кеши данных и вердиктов между тестами."""
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


def _patch_common(monkeypatch):
    """Мокаем источники данных и рисовалки: только сеть QWEN остаётся живой."""
    monkeypatch.setattr("app_pkg.routes.ai.get_series_df",
                        lambda *a, **k: None)
    monkeypatch.setattr("app_pkg.routes.ai.build_multi_tf_context",
                        lambda *a, **k: "")
    monkeypatch.setattr("app_pkg.routes.ai.format_tf_section",
                        lambda *a, **k: ("", None))
    monkeypatch.setattr("app_pkg.routes.ai.db_get_all_drawings",
                        lambda *a, **k: [])
    monkeypatch.setattr("app_pkg.routes.ai.db_add_drawing", lambda *a, **k: None)


def test_vision_request_called_with_screenshot(client, monkeypatch):
    """screenshot + VISION_ENABLED -> _llm_vision_request, vision_used: true."""
    _patch_common(monkeypatch)
    captured = {}

    def fake_vision(system, user_text, image_base64, **kw):
        captured["image_len"] = len(image_base64)
        captured["text"] = user_text
        import json
        return json.dumps(_FAKE_VISION_RESULT)

    monkeypatch.setattr("app_pkg.routes.ai._llm_vision_request", fake_vision)
    monkeypatch.setattr("app_pkg.routes.ai.analyze_with_ai",
                        lambda ctx: (_ for _ in ()).throw(
                            AssertionError("text path must not run")))
    monkeypatch.setattr("app_pkg.routes.ai.analyze_with_agents",
                        lambda ctx: (_ for _ in ()).throw(
                            AssertionError("text path must not run")))

    resp = client.post("/api/ai-analysis", json={
        "symbol": "BTCUSDT", "timeframe": "1H",
        "screenshot": _PNG_BASE64,
    })
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["vision_used"] is True
    assert data["signal"] == "BUY"
    assert data["model"] == config.DEEPSEEK_VISION_MODEL
    assert captured["image_len"] == len(_PNG_BASE64)
    assert "BTCUSDT" in captured["text"]


def test_vision_unavailable_fallback_to_text(client, monkeypatch):
    """RuntimeError('vision unavailable') -> текстовый анализ, vision_used: false."""
    _patch_common(monkeypatch)

    def fake_vision(*a, **kw):
        raise RuntimeError("vision unavailable")

    monkeypatch.setattr("app_pkg.routes.ai._llm_vision_request", fake_vision)

    called = {}

    def fake_text(context):
        called["ctx"] = context
        return dict(_FAKE_TEXT_RESULT)

    if config.GROQ_AGENT_MODE == 2:
        monkeypatch.setattr("app_pkg.routes.ai.analyze_with_agents", fake_text)
    else:
        monkeypatch.setattr("app_pkg.routes.ai.analyze_with_ai", fake_text)

    resp = client.post("/api/ai-analysis", json={
        "symbol": "BTCUSDT", "timeframe": "1H",
        "screenshot": _PNG_BASE64,
    })
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["vision_used"] is False
    assert data["signal"] == "HOLD"
    assert "ctx" in called  # текстовый путь реально выполнен


def test_no_screenshot_text_path_vision_false(client, monkeypatch):
    """Без screenshot -> обычный текстовый анализ, vision_used: false."""
    _patch_common(monkeypatch)

    called = {}

    def fake_text(context):
        called["ctx"] = context
        return dict(_FAKE_TEXT_RESULT)

    def fail_vision(*a, **kw):
        raise AssertionError("vision must not be called without screenshot")

    monkeypatch.setattr("app_pkg.routes.ai._llm_vision_request", fail_vision)
    if config.GROQ_AGENT_MODE == 2:
        monkeypatch.setattr("app_pkg.routes.ai.analyze_with_agents", fake_text)
    else:
        monkeypatch.setattr("app_pkg.routes.ai.analyze_with_ai", fake_text)

    resp = client.post("/api/ai-analysis", json={
        "symbol": "BTCUSDT", "timeframe": "1H",
    })
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["vision_used"] is False
    assert data["signal"] == "HOLD"
    assert "ctx" in called


def test_vision_disabled_ignores_screenshot(client, monkeypatch):
    """VISION_ENABLED=0 -> скриншот игнорируется, текстовый путь."""
    _patch_common(monkeypatch)
    monkeypatch.setattr(config, "VISION_ENABLED", False)

    monkeypatch.setattr("app_pkg.routes.ai._llm_vision_request",
                        lambda *a, **kw: (_ for _ in ()).throw(
                            AssertionError("vision must not run")))

    fake_text = lambda ctx: dict(_FAKE_TEXT_RESULT)
    if config.GROQ_AGENT_MODE == 2:
        monkeypatch.setattr("app_pkg.routes.ai.analyze_with_agents", fake_text)
    else:
        monkeypatch.setattr("app_pkg.routes.ai.analyze_with_ai", fake_text)

    resp = client.post("/api/ai-analysis", json={
        "symbol": "BTCUSDT", "timeframe": "1H",
        "screenshot": _PNG_BASE64,
    })
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["vision_used"] is False


def test_cache_key_includes_has_screenshot(client, monkeypatch):
    """has_screenshot в ключе: vision- и текстовый вердикты не смешиваются."""
    _patch_common(monkeypatch)
    import json

    monkeypatch.setattr("app_pkg.routes.ai._llm_vision_request",
                        lambda *a, **kw: json.dumps(_FAKE_VISION_RESULT))
    fake_text = lambda ctx: dict(_FAKE_TEXT_RESULT)
    if config.GROQ_AGENT_MODE == 2:
        monkeypatch.setattr("app_pkg.routes.ai.analyze_with_agents", fake_text)
    else:
        monkeypatch.setattr("app_pkg.routes.ai.analyze_with_ai", fake_text)

    base = {"symbol": "BTCUSDT", "timeframe": "1H"}
    r1 = client.post("/api/ai-analysis",
                     json={**base, "screenshot": _PNG_BASE64}).get_json()
    r2 = client.post("/api/ai-analysis", json={**base}).get_json()

    assert r1["vision_used"] is True
    assert r2["vision_used"] is False
    # Кеши разнесены по has_screenshot.
    assert get_cached_verdict(("BTCUSDT", "1H", "live", None, None, True))
    assert get_cached_verdict(("BTCUSDT", "1H", "live", None, None, False))
