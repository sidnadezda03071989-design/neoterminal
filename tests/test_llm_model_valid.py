"""Тесты _llm_request (DeepSeek/aitunnel + fallback Groq): HTTP-коды, json-режим.

Мокаем requests.Session.post (сеть не трогаем, реальные провайдеры не дёргаем):
  - HTTP 200 -> content модели, warning'ов нет, response_format в payload,
    запрос уходит на DEEPSEEK_BASE_URL с DEEPSEEK_MODEL;
  - HTTP 400 (json-режим не поддержан) -> попытка 2 без response_format
    и с «Отвечай строго валидным JSON» в system;
  - HTTP 400 на обеих попытках -> warning с телом ответа + HTTPError;
  - HTTP 401/403 у DeepSeek и Groq -> warning'и + None (см. отдельный файл
    test_deepseek_fallback.py на сам fallback);
  - HTTP 429 -> fallback к следующему провайдеру; 429 у ВСЕХ ->
    RuntimeError("rate limit").
"""

import logging

import pytest
from requests import HTTPError

from app_pkg.ai import llm


class _FakeResp:
    def __init__(self, status_code, text="", data=None):
        self.status_code = status_code
        self.text = text
        self._data = data if data is not None else {
            "choices": [{"message": {"content": "OK"}}],
        }

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise HTTPError(f"HTTP {self.status_code}")


@pytest.fixture(autouse=True)
def _fake_creds(monkeypatch):
    """Креды DeepSeek+Groq; цепочка сужена до них (Qwen заглушён)."""
    monkeypatch.setattr(llm, "LLM_PROVIDER_ORDER", ["deepseek", "groq"])
    monkeypatch.setattr(llm, "QWEN_API_KEY_1", "")
    monkeypatch.setattr(llm, "DEEPSEEK_API_KEY", "test-key-ds")
    monkeypatch.setattr(llm, "DEEPSEEK_BASE_URL", "https://fake.example/v1")
    monkeypatch.setattr(llm, "DEEPSEEK_MODEL", "deepseek-v4-flash")
    monkeypatch.setattr(llm, "DEEPSEEK_MAX_TOKENS", 2000)
    monkeypatch.setattr(llm, "GROQ_API_KEY", "test-key-groq")
    monkeypatch.setattr(llm, "GROQ_BASE_URL", "https://groq.example/v1")
    monkeypatch.setattr(llm, "GROQ_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setattr(llm, "GROQ_MAX_TOKENS", 2000)


def test_http_200_no_warning(monkeypatch, caplog):
    """HTTP 200 -> content модели, HTTP-warning'ов нет, response_format есть."""
    captured = {}

    def fake_post(url, json=None, **kw):
        captured["url"] = url
        captured["payload"] = json
        captured["headers"] = kw.get("headers")
        return _FakeResp(200, "", {"choices": [
            {"message": {"role": "assistant", "content": "OK"}}]})

    monkeypatch.setattr(llm._LLM_SESSION, "post", fake_post)

    with caplog.at_level(logging.WARNING, logger="app_pkg.ai.llm"):
        result = llm._llm_request(
            "Отвечай валидным json-объектом.",
            [{"role": "user", "content": "say OK"}])

    assert result == "OK"
    # Дефолты — DeepSeek (основной провайдер).
    assert captured["url"] == "https://fake.example/v1/chat/completions"
    assert captured["payload"]["response_format"] == {"type": "json_object"}
    assert captured["payload"]["model"] == "deepseek-v4-flash"
    headers = captured["headers"]
    assert headers["Authorization"] == "Bearer test-key-ds"
    # Ни Groq, ни HTTP-Referer/X-Title в заголовках нет.
    assert "HTTP-Referer" not in headers
    assert "X-Title" not in headers
    # Никаких warning'ов вообще (json в промпте есть, авто-добавления нет).
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


def test_fallback_on_400_drops_response_format(monkeypatch):
    """Попытка 1 (400 с response_format) -> попытка 2 без него + JSON-инструкция."""
    posts = []

    def fake_post(url, json=None, **kw):
        posts.append(json)
        if len(posts) == 1:
            return _FakeResp(400, "response_format is not supported")
        return _FakeResp(200, "", {"choices": [
            {"message": {"role": "assistant", "content": "OK"}}]})

    monkeypatch.setattr(llm._LLM_SESSION, "post", fake_post)

    result = llm._llm_request(
        "Ты ассистент. Верни json.",
        [{"role": "user", "content": "say OK"}])

    assert result == "OK"
    assert len(posts) == 2
    assert posts[0].get("response_format") == {"type": "json_object"}
    assert "response_format" not in posts[1]
    system = posts[1]["messages"][0]["content"]
    assert "строго валидным JSON" in system


def test_http_400_both_attempts_logs_warning(monkeypatch, caplog):
    """400 на обеих попытках -> warning с телом + HTTPError (fallback нет)."""
    monkeypatch.setattr(
        llm._LLM_SESSION, "post",
        lambda *a, **k: _FakeResp(400, '{"error": "invalid model"}'))

    with caplog.at_level(logging.WARNING, logger="app_pkg.ai.llm"):
        with pytest.raises(HTTPError):
            llm._llm_request("system", [{"role": "user", "content": "hi"}])

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any(
        "HTTP 400" in r.getMessage() and "invalid model" in r.getMessage()
        for r in warnings
    ), f"ожидаемый warning не найден: {[r.getMessage() for r in warnings]}"


def test_both_providers_forbidden_returns_none(monkeypatch, caplog):
    """403 у DeepSeek и у Groq -> None + body обоих ответов в логе."""
    monkeypatch.setattr(
        llm._LLM_SESSION, "post",
        lambda *a, **k: _FakeResp(403, "forbidden"))

    with caplog.at_level(logging.WARNING, logger="app_pkg.ai.llm"):
        result = llm._llm_request("system", [{"role": "user", "content": "hi"}])

    assert result is None
    assert any("HTTP 403" in r.getMessage() and "forbidden" in r.getMessage()
               for r in caplog.records)


def test_rate_limit_429_all_providers(monkeypatch):
    """HTTP 429 у ВСЕХ провайдеров цепочки -> RuntimeError('rate limit')."""
    calls = []

    def fake_post(url, json=None, **kw):
        calls.append(url)
        return _FakeResp(429, "too many requests")

    monkeypatch.setattr(llm._LLM_SESSION, "post", fake_post)

    with pytest.raises(RuntimeError, match="rate limit"):
        llm._llm_request("system", [{"role": "user", "content": "hi"}])
    # Цепочка deepseek -> groq: каждый опрошен ровно один раз (429 не
    # ретраится с json-режимом повторно — это не HTTP 400).
    assert len(calls) == 2


def test_system_duplicates_removed(monkeypatch, caplog):
    """System-дубликаты в messages удаляются (двойной system ломает API)."""
    captured = {}

    def fake_post(url, json=None, **kw):
        captured["payload"] = json
        return _FakeResp(200)

    monkeypatch.setattr(llm._LLM_SESSION, "post", fake_post)

    with caplog.at_level(logging.WARNING, logger="app_pkg.ai.llm"):
        llm._llm_request(
            "Отвечай валидным json-объектом.",
            [{"role": "system", "content": "чужой system"},
             {"role": "user", "content": "hi"}])

    msgs = captured["payload"]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == "Отвечай валидным json-объектом."
    assert any("removed 1 system message" in r.getMessage()
               for r in caplog.records)


def test_no_key_returns_none(monkeypatch):
    """Без ключей всех провайдеров цепочки запрос не выполняется -> None."""
    monkeypatch.setattr(llm, "DEEPSEEK_API_KEY", "")
    monkeypatch.setattr(llm, "GROQ_API_KEY", "")
    monkeypatch.setattr(llm, "QWEN_API_KEY_1", "")
    assert llm._llm_request(
        "system", [{"role": "user", "content": "hi"}]) is None