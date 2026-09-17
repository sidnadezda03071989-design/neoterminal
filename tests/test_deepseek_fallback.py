"""Fallback DeepSeek -> Groq в app_pkg.ai.llm (без сети, мокаем Session.post).

Цепочка провайдеров — LLM_PROVIDER_ORDER; здесь порядок сужен до
['deepseek','groq'] и Qwen-ключ заглушён, чтобы тесты проверяли именно
связку DeepSeek -> Groq.
Проверяем:
  - DeepSeek отвечает 403 -> retry на Groq (другой base_url и model),
    результат возвращается, в логе 'provider deepseek failed (HTTP 403),
    fallback to groq';
  - HTTP 401 и 404 от DeepSeek тоже триггерят fallback;
  - DeepSeek 403 + Groq 403 -> None (оба провайдера недоступны);
  - ключа DeepSeek нет, но Groq настроен -> запрос сразу уходит в Groq;
  - vision: DeepSeek 403 -> retry vision-запроса на Groq;
  - сетевые сбои (ConnectionError/Timeout/ConnectionResetError) у DeepSeek
    -> тот же retry на Groq ('provider deepseek failed (network: ...)');
    сеть легла у обоих -> None (чат) / RuntimeError (vision).
"""

import logging

import pytest
import requests
from requests import HTTPError

from app_pkg.ai import llm

_DS_URL = "https://ds.example/v1"
_GROQ_URL = "https://groq.example/v1"


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
def _creds(monkeypatch):
    """Провайдеры DeepSeek+Groq настроены; Qwen заглушён (не в этой связке)."""
    monkeypatch.setattr(llm, "LLM_PROVIDER_ORDER", ["deepseek", "groq"])
    monkeypatch.setattr(llm, "QWEN_API_KEY_1", "")
    monkeypatch.setattr(llm, "DEEPSEEK_API_KEY", "key-ds")
    monkeypatch.setattr(llm, "DEEPSEEK_BASE_URL", _DS_URL)
    monkeypatch.setattr(llm, "DEEPSEEK_MODEL", "deepseek-v4-flash")
    monkeypatch.setattr(llm, "DEEPSEEK_MAX_TOKENS", 2000)
    monkeypatch.setattr(llm, "DEEPSEEK_VISION_MODEL",
                        "deepseek-v4-flash-vision-exp")
    monkeypatch.setattr(llm, "GROQ_API_KEY", "key-groq")
    monkeypatch.setattr(llm, "GROQ_BASE_URL", _GROQ_URL)
    monkeypatch.setattr(llm, "GROQ_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setattr(llm, "GROQ_MAX_TOKENS", 2000)
    monkeypatch.setattr(llm, "GROQ_VISION_MODEL",
                        "llama-3.2-90b-vision-preview")
    monkeypatch.setattr(llm, "_vision_supported", lambda *a, **k: True)


def _router(status_by_host, calls, text="forbidden"):
    """fake_post: статус по провайдеру (ds/groq) + запись вызовов."""
    def fake_post(url, json=None, **kw):  # noqa: A002
        host = "groq" if url.startswith(_GROQ_URL) else "ds"
        status = status_by_host[host]
        calls.append({"host": host, "url": url, "payload": json})
        return _FakeResp(status, text if status >= 400 else "")
    return fake_post


def test_deepseek_403_falls_back_to_groq(monkeypatch, caplog):
    """403 от DeepSeek -> retry на Groq, ответ возвращается."""
    calls = []
    monkeypatch.setattr(llm._LLM_SESSION, "post",
                        _router({"ds": 403, "groq": 200}, calls))

    with caplog.at_level(logging.WARNING, logger="app_pkg.ai.llm"):
        result = llm._llm_request("Отвечай валидным json.",
                                  [{"role": "user", "content": "say OK"}])

    assert result == "OK"
    assert [c["host"] for c in calls] == ["ds", "groq"]
    assert calls[0]["payload"]["model"] == "deepseek-v4-flash"
    assert calls[1]["payload"]["model"] == "llama-3.3-70b-versatile"
    assert calls[1]["url"].startswith(_GROQ_URL)
    assert any("provider deepseek failed (HTTP 403), fallback to groq"
               in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("status", [401, 404])
def test_401_and_404_trigger_fallback(monkeypatch, status):
    """HTTP 401/404 от DeepSeek тоже уводят запрос на Groq."""
    calls = []
    monkeypatch.setattr(llm._LLM_SESSION, "post",
                        _router({"ds": status, "groq": 200}, calls))

    result = llm._llm_request("Отвечай json.",
                              [{"role": "user", "content": "hi"}])
    assert result == "OK"
    assert [c["host"] for c in calls] == ["ds", "groq"]


def test_both_providers_unavailable_returns_none(monkeypatch):
    """403 у DeepSeek и у Groq -> None (не исключение)."""
    calls = []
    monkeypatch.setattr(llm._LLM_SESSION, "post",
                        _router({"ds": 403, "groq": 403}, calls))

    assert llm._llm_request("Отвечай json.",
                            [{"role": "user", "content": "hi"}]) is None
    assert [c["host"] for c in calls] == ["ds", "groq"]


def test_no_deepseek_key_uses_groq(monkeypatch):
    """Ключа DeepSeek нет -> запрос сразу уходит в Groq."""
    monkeypatch.setattr(llm, "DEEPSEEK_API_KEY", "")
    calls = []
    monkeypatch.setattr(llm._LLM_SESSION, "post",
                        _router({"ds": 200, "groq": 200}, calls))

    assert llm._llm_request("Отвечай json.",
                            [{"role": "user", "content": "hi"}]) == "OK"
    assert [c["host"] for c in calls] == ["groq"]
    assert calls[0]["payload"]["model"] == "llama-3.3-70b-versatile"


def test_vision_falls_back_to_groq(monkeypatch):
    """Vision: 403 от DeepSeek -> retry на Groq vision-модель."""
    calls = []
    monkeypatch.setattr(llm._LLM_SESSION, "post",
                        _router({"ds": 403, "groq": 200}, calls))

    png = "iVBORw0KGgoAAAANSUhEUg" + "A" * 200
    result = llm._llm_vision_request("Верни json.", "BTCUSDT 1H", png)

    assert result == "OK"
    assert [c["host"] for c in calls] == ["ds", "groq"]
    assert calls[0]["payload"]["model"] == "deepseek-v4-flash-vision-exp"
    assert calls[1]["payload"]["model"] == "llama-3.2-90b-vision-preview"


def test_vision_both_unavailable_raises(monkeypatch):
    """Vision: 404 у обоих провайдеров -> RuntimeError('vision unavailable')."""
    calls = []
    monkeypatch.setattr(llm._LLM_SESSION, "post",
                        _router({"ds": 404, "groq": 404}, calls))

    png = "iVBORw0KGgoAAAANSUhEUg" + "A" * 200
    with pytest.raises(RuntimeError, match="vision unavailable"):
        llm._llm_vision_request("Верни json.", "BTCUSDT 1H", png)
    assert [c["host"] for c in calls] == ["ds", "groq"]


# ------------------------------------------------- сетевые сбои (fallback)
def _network_router(error, calls, groq_status=200):
    """fake_post: DeepSeek падает сетью, Groq отвечает groq_status."""
    def fake_post(url, json=None, **kw):  # noqa: A002
        host = "groq" if url.startswith(_GROQ_URL) else "ds"
        calls.append({"host": host, "url": url, "payload": json})
        if host == "ds":
            raise error
        return _FakeResp(groq_status, "" if groq_status < 400 else "boom")
    return fake_post


@pytest.mark.parametrize("error", [
    requests.ConnectionError("DNS fail"),
    requests.Timeout("read timeout"),
    ConnectionResetError("connection reset"),
])
def test_network_error_falls_back_to_groq(monkeypatch, caplog, error):
    """Сетевая ошибка DeepSeek -> retry на Groq, ответ возвращается."""
    calls = []
    monkeypatch.setattr(llm._LLM_SESSION, "post",
                        _network_router(error, calls))

    with caplog.at_level(logging.WARNING, logger="app_pkg.ai.llm"):
        result = llm._llm_request("Отвечай валидным json.",
                                  [{"role": "user", "content": "say OK"}])

    assert result == "OK"
    assert [c["host"] for c in calls] == ["ds", "groq"]
    assert calls[1]["payload"]["model"] == "llama-3.3-70b-versatile"
    assert calls[1]["url"].startswith(_GROQ_URL)
    assert any("provider deepseek failed (network"
               in r.getMessage() for r in caplog.records)


def test_network_error_both_providers_returns_none(monkeypatch):
    """Сеть легла у DeepSeek и у Groq -> None (не исключение)."""
    calls = []

    def fake_post(url, json=None, **kw):  # noqa: A002
        calls.append(url)
        raise requests.ConnectionError("network down")

    monkeypatch.setattr(llm._LLM_SESSION, "post", fake_post)

    assert llm._llm_request("Отвечай json.",
                            [{"role": "user", "content": "hi"}]) is None
    assert len(calls) == 2


def test_vision_network_error_falls_back_to_groq(monkeypatch):
    """Vision: сетевая ошибка DeepSeek -> retry на Groq vision-модель."""
    calls = []
    monkeypatch.setattr(
        llm._LLM_SESSION, "post",
        _network_router(requests.ConnectionError("boom"), calls))

    png = "iVBORw0KGgoAAAANSUhEUg" + "A" * 200
    assert llm._llm_vision_request("Верни json.", "BTCUSDT 1H", png) == "OK"
    assert [c["host"] for c in calls] == ["ds", "groq"]
    assert calls[1]["payload"]["model"] == "llama-3.2-90b-vision-preview"


def test_vision_network_error_both_unavailable_raises(monkeypatch):
    """Vision: сеть легла у обоих -> RuntimeError('vision unavailable')."""
    def fake_post(url, json=None, **kw):  # noqa: A002
        raise requests.ReadTimeout("timeout")

    monkeypatch.setattr(llm._LLM_SESSION, "post", fake_post)

    png = "iVBORw0KGgoAAAANSUhEUg" + "A" * 200
    with pytest.raises(RuntimeError, match="vision unavailable"):
        llm._llm_vision_request("Верни json.", "BTCUSDT 1H", png)


def test_http_fallback_then_network_error_returns_none(monkeypatch):
    """DeepSeek 403 -> retry на Groq, у Groq сеть упала -> None."""
    calls = []

    def fake_post(url, json=None, **kw):  # noqa: A002
        calls.append(url)
        if url.startswith(_GROQ_URL):
            raise requests.ConnectionError("groq down")
        return _FakeResp(403, "forbidden")

    monkeypatch.setattr(llm._LLM_SESSION, "post", fake_post)

    assert llm._llm_request("Отвечай json.",
                            [{"role": "user", "content": "hi"}]) is None
    assert len(calls) == 2


def test_vision_http_fallback_then_network_error_raises(monkeypatch):
    """Vision: 403 DeepSeek -> Groq, сеть упала -> RuntimeError."""
    def fake_post(url, json=None, **kw):  # noqa: A002
        if url.startswith(_GROQ_URL):
            raise requests.ConnectionError("groq down")
        return _FakeResp(403, "forbidden")

    monkeypatch.setattr(llm._LLM_SESSION, "post", fake_post)

    png = "iVBORw0KGgoAAAANSUhEUg" + "A" * 200
    with pytest.raises(RuntimeError, match="vision unavailable"):
        llm._llm_vision_request("Верни json.", "BTCUSDT 1H", png)