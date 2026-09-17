"""Тесты определения интента анализа чата (регрессия ValueError в re.search).

Проверяем:
  - импорт app_pkg.ai.prompts не падает (валидный _ANALYZE_INTENT_RE);
  - _detect_analysis_intent: анализ-запросы -> True, болтовня -> False;
  - _ANALYZE_INTENT_RE скомпилирован с re.IGNORECASE (регистр не важен);
  - /api/chat на любое сообщение отвечает 200 (раньше — 500).
"""

import pytest

from app_pkg import create_app
from app_pkg.ai import prompts  # noqa: F401 — импорт не должен падать
from app_pkg.ai.chat import _detect_analysis_intent
from app_pkg.cache import _VERDICT_CACHE
from app_pkg.data import fetch


def test_intent_analysis_true():
    assert _detect_analysis_intent("проанализируй BTCUSDT") is True


def test_intent_smalltalk_false():
    assert _detect_analysis_intent("привет") is False


def test_intent_what_do_you_think_true():
    assert _detect_analysis_intent("что думаешь по битку?") is True


def test_intent_case_insensitive():
    """IGNORECASE зашит в компиляцию — верхний регистр тоже матчится."""
    assert _detect_analysis_intent("ПРОАНАЛИЗИРУЙ график") is True


def test_intent_empty_false():
    assert _detect_analysis_intent("") is False
    assert _detect_analysis_intent(None) is False


@pytest.mark.parametrize("message", [
    "а что по RSI на нём?",
    "какой MACD?",
    "где уровень поддержки?",
    "что по supertrend?",
    "где сопротивление?",
])
def test_intent_indicators_true(message):
    """Индикаторы/уровни роутят в analysis даже без глагола-триггера."""
    assert _detect_analysis_intent(message) is True


@pytest.mark.parametrize("message", ["привет", "как дела?"])
def test_intent_smalltalk_still_false(message):
    """Расширение регэкспа не утащило болтовню в analysis-путь."""
    assert _detect_analysis_intent(message) is False


def test_intent_latin_substring_not_matched():
    """'email' не должен ловиться на 'ema' — латиница только по \\b."""
    assert _detect_analysis_intent("напиши мне email") is False


def test_prompts_import_and_pattern_compiled():
    """Импорт prompts не падает, паттерн — компилированный, с IGNORECASE."""
    import re
    assert isinstance(prompts._ANALYZE_INTENT_RE, re.Pattern)
    assert prompts._ANALYZE_INTENT_RE.flags & re.IGNORECASE


# ------------------------------------------------------------- /api/chat
@pytest.fixture(autouse=True)
def _isolate():
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


def test_route_chat_no_500_on_any_message(client, monkeypatch):
    """/api/chat отвечает 200 и на «привет», и на запрос анализа."""
    from app_pkg.ai import chat as ai_chat

    monkeypatch.setattr(ai_chat, "_llm_request", lambda *a, **k: None)
    # Analysis-путь ходит в _llm_request внутри app_pkg.ai.agents — мокаем и его,
    # иначе тест делает живой запрос к провайдеру (медленно и зависит от сети).
    monkeypatch.setattr(ai_chat, "analyze_with_ai", lambda ctx: None)
    monkeypatch.setattr(ai_chat, "analyze_with_agents", lambda ctx: None)
    monkeypatch.setattr(ai_chat, "db_append_chat", lambda *a, **k: 0)
    monkeypatch.setattr(ai_chat, "db_get_chat_history", lambda limit=20: [])
    monkeypatch.setattr(ai_chat, "db_get_all_drawings", lambda *a, **k: [])
    monkeypatch.setattr(ai_chat, "db_add_drawing", lambda *a, **k: None)
    monkeypatch.setattr(ai_chat, "get_series_df", lambda *a, **k: None)
    monkeypatch.setattr(ai_chat, "build_multi_tf_context", lambda *a, **k: "")

    for message in ("привет", "проанализируй BTCUSDT"):
        resp = client.post("/api/chat", json={
            "message": message, "symbol": "BTCUSDT", "timeframe": "1H",
        })
        assert resp.status_code == 200, f"500 на сообщении: {message!r}"
        assert "reply" in resp.get_json()
