"""Тесты _chat_flow: распарсивание json {"reply": ...} из ответа модели.

Проверяем:
  - '{"reply": "привет"}' -> в чат уходит "привет" без JSON-обёртки;
  - plain text (не json) -> возвращается как есть;
  - json без ключа reply -> возвращается как есть (не выдумывать);
  - None / исключение от _llm_request -> fallback "Извините...".
"""

import pytest

from app_pkg import config
from app_pkg.ai import chat as ai_chat

_FALLBACK = ("Извините, LLM-провайдер недоступен. "
             "Попробуйте позже или попросите анализ рынка.")


@pytest.fixture(autouse=True)
def _mock_db(monkeypatch):
    monkeypatch.setattr(ai_chat, "db_get_chat_history", lambda limit=20: [])
    monkeypatch.setattr(ai_chat, "db_append_chat", lambda *a, **k: 0)


def test_reply_json_parsed(monkeypatch):
    """'{"reply": "привет"}' -> 'привет' (без кавычек и JSON)."""
    monkeypatch.setattr(
        ai_chat, "_llm_request",
        lambda *a, **k: '{"reply": "привет"}')
    reply, model, fallback = ai_chat._chat_flow("привет", "BTCUSDT", "1H")
    assert reply == "привет"
    assert model == config.DEEPSEEK_MODEL
    assert fallback is False


def test_plain_text_passthrough(monkeypatch):
    """Модель вернула просто текст -> возвращается как есть."""
    monkeypatch.setattr(
        ai_chat, "_llm_request", lambda *a, **k: "просто текст")
    reply, model, fallback = ai_chat._chat_flow("привет", "BTCUSDT", "1H")
    assert reply == "просто текст"
    assert model == config.DEEPSEEK_MODEL
    assert fallback is False


def test_json_without_reply_key_passthrough(monkeypatch):
    """json без ключа reply -> возвращается как есть."""
    raw = '{"other": "x"}'
    monkeypatch.setattr(ai_chat, "_llm_request", lambda *a, **k: raw)
    reply, model, fallback = ai_chat._chat_flow("привет", "BTCUSDT", "1H")
    assert reply == raw
    assert model == config.DEEPSEEK_MODEL
    assert fallback is False


def test_none_response_fallback(monkeypatch):
    """_llm_request вернул None -> fallback 'Извините...'."""
    monkeypatch.setattr(ai_chat, "_llm_request", lambda *a, **k: None)
    reply, model, fallback = ai_chat._chat_flow("привет", "BTCUSDT", "1H")
    assert reply == _FALLBACK
    assert model == "heuristic"
    assert fallback is True


def test_request_exception_fallback(monkeypatch):
    """Исключение от _llm_request -> fallback, не 500."""
    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(ai_chat, "_llm_request", boom)
    reply, model, fallback = ai_chat._chat_flow("привет", "BTCUSDT", "1H")
    assert reply == _FALLBACK
    assert model == "heuristic"
    assert fallback is True


def test_reply_value_converted_to_str(monkeypatch):
    """reply не-строка (число) -> приводится к str, не падает."""
    monkeypatch.setattr(
        ai_chat, "_llm_request", lambda *a, **k: '{"reply": 42}')
    reply, _, _ = ai_chat._chat_flow("привет", "BTCUSDT", "1H")
    assert reply == "42"