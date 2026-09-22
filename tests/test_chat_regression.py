"""Регрессия #17 (память диалога): чат не должен отдавать сырой JSON.

Проверяем:
  - мок _llm_request вернул '{"reply":"привет"}' -> reply "привет";
  - мок вернул plain text -> отдаётся как есть;
  - мок вернул None -> fallback "Извините...";
  - в messages, ушедших в _llm_request, нет двух system-сообщений
    (и вообще нет system: его добавляет сам _llm_request);
  - текущий вопрос пользователя не дублируется.
"""

import pytest

from app_pkg.ai import chat as ai_chat

_FALLBACK = ("Извините, LLM-провайдер недоступен. "
             "Попробуйте позже или попросите анализ рынка.")


@pytest.fixture(autouse=True)
def _mock_db(monkeypatch):
    """БД чата не трогаем: история пустая, append — заглушка."""
    monkeypatch.setattr(ai_chat, "db_append_chat", lambda *a, **k: 0)
    monkeypatch.setattr(ai_chat, "db_get_chat_history", lambda limit=20: [])


def _capture_request(monkeypatch, response):
    """Мокает _llm_request, возвращает список перехваченных вызовов."""
    calls = []

    def fake(system, messages, **kw):
        calls.append({"system": system, "messages": messages})
        return response

    monkeypatch.setattr(ai_chat, "_llm_request", fake)
    return calls


def _roles(msgs):
    return [m["role"] for m in msgs]


def test_reply_json_unwrapped(monkeypatch):
    """'{"reply":"привет"}' -> в чат уходит "привет", а не JSON-обёртка."""
    _capture_request(monkeypatch, '{"reply": "привет"}')
    reply, model, fallback = ai_chat._chat_flow("привет", "BTCUSDT", "1H")
    assert reply == "привет"
    assert "{" not in reply and "reply" not in reply
    assert model != "heuristic"
    assert fallback is False


def test_plain_text_passthrough(monkeypatch):
    """Мок вернул 'plain text' -> возвращается как есть."""
    _capture_request(monkeypatch, "plain text")
    reply, _, fallback = ai_chat._chat_flow("привет", "BTCUSDT", "1H")
    assert reply == "plain text"
    assert fallback is False


def test_none_response_uses_fallback(monkeypatch):
    """Мок вернул None -> fallback 'Извините...', без исключения."""
    _capture_request(monkeypatch, None)
    reply, model, fallback = ai_chat._chat_flow("привет", "BTCUSDT", "1H")
    assert reply == _FALLBACK
    assert model == "heuristic"
    assert fallback is True


def test_no_double_system_in_msgs(monkeypatch):
    """history с system-сообщением -> в msgs system нет вообще."""
    history = [
        {"role": "system", "content": "старый system (регрессия #17)"},
        {"role": "user", "content": "привет"},
        {"role": "assistant", "content": "здравствуйте"},
    ]
    monkeypatch.setattr(ai_chat, "db_get_chat_history",
                        lambda limit=20: history)
    calls = _capture_request(monkeypatch, '{"reply": "привет"}')
    ai_chat._chat_flow("как дела?", "BTCUSDT", "1H")

    msgs = calls[0]["messages"]
    assert _roles(msgs).count("system") == 0
    assert all(r in ("user", "assistant") for r in _roles(msgs))
    # Текущий вопрос последний и ровно один.
    assert msgs[-1] == {"role": "user", "content": "как дела?"}
    assert _roles(msgs) == ["user", "assistant", "user"]


def test_current_message_not_duplicated(monkeypatch):
    """История уже кончается тем же user-сообщением -> не дублируем его."""
    monkeypatch.setattr(ai_chat, "db_get_chat_history", lambda limit=20: [
        {"role": "user", "content": "привет"},
        {"role": "user", "content": "привет"},
    ])
    calls = _capture_request(monkeypatch, "ok")
    ai_chat._chat_flow("привет", "BTCUSDT", "1H")

    msgs = calls[0]["messages"]
    assert len([m for m in msgs if m["content"] == "привет"]) == 1
    assert msgs[-1] == {"role": "user", "content": "привет"}


def test_system_prompt_is_smalltalk(monkeypatch):
    """В _llm_request уходит именно chat-промпт с lowercase 'json'."""
    from app_pkg.ai.prompts import CHAT_SYSTEM_PROMPT_SMALLTALK
    calls = _capture_request(monkeypatch, '{"reply": "ок"}')
    ai_chat._chat_flow("привет", "BTCUSDT", "1H")
    assert calls[0]["system"] == CHAT_SYSTEM_PROMPT_SMALLTALK
    assert "json" in calls[0]["system"]


def test_analysis_flow_uses_llm_request(monkeypatch):
    """Analysis-путь тоже ходит через _llm_request -> model = DEEPSEEK_MODEL.

    Регрессия: в agents.analyze_with_ai оставался вызов старого имени
    _qwen_request («name '_qwen_request' is not defined» -> heuristic).
    """
    import json

    from app_pkg import config
    from app_pkg.ai import agents
    from app_pkg.ai.prompts import QWEN_SYSTEM_PROMPT

    monkeypatch.setattr(ai_chat, "get_series_df", lambda *a, **k: None)
    monkeypatch.setattr(ai_chat, "build_multi_tf_context", lambda *a, **k: "")
    monkeypatch.setattr(ai_chat, "db_get_all_drawings", lambda *a, **k: [])
    monkeypatch.setattr(ai_chat, "db_add_drawing", lambda *a, **k: None)
    monkeypatch.setattr(ai_chat, "append_chat_message", lambda *a, **k: 0)
    monkeypatch.setattr(config, "GROQ_AGENT_MODE", 0)

    captured = {}

    def fake_llm(system, messages, **kw):
        captured["system"] = system
        captured["roles"] = [m["role"] for m in messages]
        return json.dumps({"signal": "BUY", "confidence": 0.8, "reason": "ok",
                           "indicator_signals": {}, "price_levels": {},
                           "suggested_drawings": []})

    monkeypatch.setattr(agents, "_llm_request", fake_llm)

    reply, _drawings, _all, model, fallback = ai_chat._analysis_flow(
        "проанализируй BTCUSDT", "BTCUSDT", "1H")
    assert model == config.DEEPSEEK_MODEL
    assert model != "heuristic"
    assert fallback is False
    assert "BUY" in reply
    assert captured["system"] == QWEN_SYSTEM_PROMPT
    assert captured["roles"] == ["user"]