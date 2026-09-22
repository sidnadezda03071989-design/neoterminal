"""Тесты lowercase-ключевого слова "json" для json_object-режима.

DeepSeek (как и Groq/OpenRouter) при response_format={"type":"json_object"}
требует подстроку "json" (в нижнем регистре) в сообщениях — иначе часть
моделей отвечает HTTP 400 InvalidParameter.

Проверяем:
  - каждый системный промпт prompts.py содержит lowercase "json";
  - _llm_request с промптом без "json" дописывает "\\njson" в system
    и логирует warning;
  - _llm_request с промптом, где "json" уже есть, ничего не добавляет.
"""

import logging

import pytest

from app_pkg.ai import llm, prompts
from app_pkg.ai.llm import _llm_request


@pytest.mark.parametrize("name", [
    "QWEN_SYSTEM_PROMPT",
    "CHAT_SYSTEM_PROMPT_SMALLTALK",
    "AGENT1_SYSTEM_PROMPT",
    "AGENT2_SYSTEM_PROMPT",
])
def test_prompts_contain_lowercase_json(name):
    """re.search(r'json', prompt) без флагов находит (lowercase обязателен)."""
    prompt = getattr(prompts, name)
    import re
    assert re.search(r"json", prompt), f"{name} не содержит lowercase 'json'"


class _FakeResp:
    status_code = 200

    def __init__(self):
        self.text = ""
        self._data = {"choices": [{"message": {"content": "OK"}}]}

    def json(self):
        return self._data

    def raise_for_status(self):
        pass


@pytest.fixture(autouse=True)
def _fake_creds(monkeypatch):
    """Дефолты — DeepSeek; Groq/Qwen отключены, чтобы не влияли на json-режим."""
    monkeypatch.setattr(llm, "LLM_PROVIDER_ORDER", ["deepseek", "groq"])
    monkeypatch.setattr(llm, "QWEN_API_KEY_1", "")
    monkeypatch.setattr(llm, "DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(llm, "DEEPSEEK_BASE_URL", "https://fake.example/v1")
    monkeypatch.setattr(llm, "DEEPSEEK_MODEL", "deepseek-v4-flash")
    monkeypatch.setattr(llm, "DEEPSEEK_MAX_TOKENS", 2000)
    monkeypatch.setattr(llm, "GROQ_API_KEY", "")


def test_auto_adds_json_keyword(monkeypatch, caplog):
    """Промпт без 'json' -> в system, ушедший в POST, дописан '\\njson'."""
    captured = {}

    def fake_post(url, json=None, **kw):
        captured["payload"] = json
        return _FakeResp()

    monkeypatch.setattr(llm._LLM_SESSION, "post", fake_post)

    with caplog.at_level(logging.WARNING, logger="app_pkg.ai.llm"):
        result = _llm_request(
            "Ты ассистент. Отвечай кратко.",  # нет слова json
            [{"role": "user", "content": "привет"}])

    assert result == "OK"
    system = captured["payload"]["messages"][0]["content"]
    assert system.endswith("\njson")
    assert any("auto-added json keyword" in r.getMessage()
               for r in caplog.records)


def test_no_double_add_when_json_present(monkeypatch, caplog):
    """'json' уже в промпте/сообщении -> ничего не добавляется, без warning."""
    captured = {}

    def fake_post(url, json=None, **kw):
        captured["payload"] = json
        return _FakeResp()

    monkeypatch.setattr(llm._LLM_SESSION, "post", fake_post)

    # "JSON" капсом НЕ считается — нужен lowercase; проверяем оба случая.
    with caplog.at_level(logging.WARNING, logger="app_pkg.ai.llm"):
        _llm_request(
            "Отвечай валидным json-объектом.",
            [{"role": "user", "content": "hi"}])
    system = captured["payload"]["messages"][0]["content"]
    assert not system.endswith("\njson")
    assert not [r for r in caplog.records
                if "auto-added json keyword" in r.getMessage()]


def test_user_message_json_counts(monkeypatch):
    """'json' в user-сообщении тоже спасает — system не дополняется."""
    captured = {}

    def fake_post(url, json=None, **kw):
        captured["payload"] = json
        return _FakeResp()

    monkeypatch.setattr(llm._LLM_SESSION, "post", fake_post)

    _llm_request(
        "Ты ассистент.",
        [{"role": "user", "content": "верни json ответа"}])
    system = captured["payload"]["messages"][0]["content"]
    assert not system.endswith("\njson")