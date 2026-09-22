"""Fallback-цепочка провайдеров Qwen -> Groq -> DeepSeek (app_pkg/ai/llm.py).

Сеть не трогаем: мокаем requests.Session.post и _vision_supported.
Проверяем:
  - Qwen 200 -> используется Qwen, Groq/DeepSeek не вызываются;
  - Qwen 403 -> Groq 200 -> используется Groq (+ warning с fallback);
  - Qwen 403 + Groq 403 -> DeepSeek 200 -> используется DeepSeek;
  - все 403 -> None;
  - Qwen без ключа -> сразу Groq;
  - LLM_PROVIDER_ORDER=deepseek,qwen -> порядок соблюдён;
  - HTTP 429 -> fallback к следующему (залимиченный Groq не блокирует
    рабочий DeepSeek); 429 у ВСЕХ -> RuntimeError("rate limit"),
    в т.ч. для vision (а не «vision unavailable»);
  - Vision: Qwen-vl 200 -> используется Qwen (qwen-vl-max);
  - Vision: Qwen-vl первым даже при порядке groq,deepseek; Qwen-vl 404 ->
    Groq-vision 200 -> используется Groq.
"""

import logging

import pytest
from requests import HTTPError

from app_pkg.ai import llm

_QWEN_URL = "https://qwen.example/v1"
_GROQ_URL = "https://groq.example/v1"
_DS_URL = "https://ds.example/v1"

_PNG = "iVBORw0KGgoAAAANSUhEUg" + "A" * 200


class _FakeResp:
    """Минимальный ответ провайдера (200 по умолчанию, content='OK')."""

    def __init__(self, status_code=200, text="", content="OK"):
        self.status_code = status_code
        self.text = text
        self._data = {"choices": [{"message": {"role": "assistant",
                                               "content": content}}]}

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise HTTPError(f"HTTP {self.status_code}")


@pytest.fixture(autouse=True)
def _creds(monkeypatch):
    """Все три провайдера настроены; vision-capability не ходит в сеть."""
    monkeypatch.setattr(llm, "QWEN_API_KEY_1", "key-qwen")
    monkeypatch.setattr(llm, "QWEN_BASE_URL_1", _QWEN_URL)
    monkeypatch.setattr(llm, "QWEN_MODEL_1", "qwen-plus")
    monkeypatch.setattr(llm, "QWEN_VL_MODEL", "qwen-vl-max")
    monkeypatch.setattr(llm, "QWEN_TIMEOUT", 60.0)
    monkeypatch.setattr(llm, "QWEN_MAX_TOKENS", 2000)
    monkeypatch.setattr(llm, "GROQ_API_KEY", "key-groq")
    monkeypatch.setattr(llm, "GROQ_BASE_URL", _GROQ_URL)
    monkeypatch.setattr(llm, "GROQ_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setattr(llm, "GROQ_VISION_MODEL",
                        "llama-3.2-90b-vision-preview")
    monkeypatch.setattr(llm, "GROQ_TIMEOUT", 60.0)
    monkeypatch.setattr(llm, "GROQ_MAX_TOKENS", 2000)
    monkeypatch.setattr(llm, "DEEPSEEK_API_KEY", "key-ds")
    monkeypatch.setattr(llm, "DEEPSEEK_BASE_URL", _DS_URL)
    monkeypatch.setattr(llm, "DEEPSEEK_MODEL", "deepseek-v4-flash")
    monkeypatch.setattr(llm, "DEEPSEEK_VISION_MODEL",
                        "deepseek-v4-flash-vision-exp")
    monkeypatch.setattr(llm, "DEEPSEEK_TIMEOUT", 60.0)
    monkeypatch.setattr(llm, "DEEPSEEK_MAX_TOKENS", 2000)
    monkeypatch.setattr(llm, "LLM_PROVIDER_ORDER",
                        ["qwen", "groq", "deepseek"])
    monkeypatch.setattr(llm, "_vision_supported", lambda *a, **k: True)


def _host(url):
    """Имя провайдера по url — как в calls[]."""
    if url.startswith(_QWEN_URL):
        return "qwen"
    if url.startswith(_GROQ_URL):
        return "groq"
    return "ds"


def _router(status_by_host, calls):
    """fake_post: статус по провайдеру + запись вызовов (host/url/payload)."""
    def fake_post(url, json=None, **kw):
        host = _host(url)
        status = status_by_host[host]
        calls.append({"host": host, "url": url, "payload": json})
        return _FakeResp(status, "" if status < 400 else "error")
    return fake_post
def test_qwen_200_uses_qwen_only(monkeypatch):
    """Qwen отвечает 200 -> Groq/DeepSeek не вызываются."""
    calls = []
    monkeypatch.setattr(llm._LLM_SESSION, "post",
                        _router({"qwen": 200, "groq": 200, "ds": 200}, calls))

    result = llm._llm_request("Отвечай валидным json.",
                              [{"role": "user", "content": "say OK"}])

    assert result == "OK"
    assert [c["host"] for c in calls] == ["qwen"]
    assert calls[0]["payload"]["model"] == "qwen-plus"
    assert calls[0]["url"] == f"{_QWEN_URL}/chat/completions"


def test_qwen_403_falls_back_to_groq(monkeypatch, caplog):
    """Qwen 403 (квота) -> Groq 200 -> ответ Groq, warning с fallback."""
    calls = []
    monkeypatch.setattr(llm._LLM_SESSION, "post",
                        _router({"qwen": 403, "groq": 200, "ds": 200}, calls))

    with caplog.at_level(logging.WARNING, logger="app_pkg.ai.llm"):
        result = llm._llm_request("Отвечай json.",
                                  [{"role": "user", "content": "hi"}])

    assert result == "OK"
    assert [c["host"] for c in calls] == ["qwen", "groq"]
    assert calls[1]["payload"]["model"] == "llama-3.3-70b-versatile"
    assert any("provider qwen failed (HTTP 403), fallback to groq"
               in r.getMessage() for r in caplog.records)


def test_qwen_and_groq_403_falls_back_to_deepseek(monkeypatch, caplog):
    """Qwen 403 + Groq 403 -> DeepSeek 200 -> ответ DeepSeek."""
    calls = []
    monkeypatch.setattr(llm._LLM_SESSION, "post",
                        _router({"qwen": 403, "groq": 403, "ds": 200}, calls))

    with caplog.at_level(logging.WARNING, logger="app_pkg.ai.llm"):
        result = llm._llm_request("Отвечай json.",
                                  [{"role": "user", "content": "hi"}])

    assert result == "OK"
    assert [c["host"] for c in calls] == ["qwen", "groq", "ds"]
    assert calls[2]["payload"]["model"] == "deepseek-v4-flash"
    assert any("provider groq failed (HTTP 403), fallback to deepseek"
               in r.getMessage() for r in caplog.records)


def test_all_providers_403_returns_none(monkeypatch):
    """Все три провайдера 403 -> None (не исключение)."""
    calls = []
    monkeypatch.setattr(llm._LLM_SESSION, "post",
                        _router({"qwen": 403, "groq": 403, "ds": 403}, calls))

    assert llm._llm_request("Отвечай json.",
                            [{"role": "user", "content": "hi"}]) is None
    assert [c["host"] for c in calls] == ["qwen", "groq", "ds"]


def test_qwen_429_falls_back_to_groq(monkeypatch, caplog):
    """Qwen 429 (rate limit) -> Groq 200 -> ответ Groq, 429 не терминален."""
    calls = []
    monkeypatch.setattr(llm._LLM_SESSION, "post",
                        _router({"qwen": 429, "groq": 200, "ds": 200}, calls))

    with caplog.at_level(logging.WARNING, logger="app_pkg.ai.llm"):
        result = llm._llm_request("Отвечай json.",
                                  [{"role": "user", "content": "hi"}])

    assert result == "OK"
    assert [c["host"] for c in calls] == ["qwen", "groq"]
    assert any("provider qwen rate limited (HTTP 429), fallback to groq"
               in r.getMessage() for r in caplog.records)


def test_groq_429_falls_back_to_deepseek(monkeypatch):
    """Groq 429 не блокирует рабочий DeepSeek (раньше цепочка обрывалась)."""
    calls = []
    monkeypatch.setattr(llm._LLM_SESSION, "post",
                        _router({"qwen": 403, "groq": 429, "ds": 200}, calls))

    result = llm._llm_request("Отвечай json.",
                              [{"role": "user", "content": "hi"}])

    assert result == "OK"
    assert [c["host"] for c in calls] == ["qwen", "groq", "ds"]
    assert calls[2]["payload"]["model"] == "deepseek-v4-flash"


def test_all_providers_429_raises_rate_limit(monkeypatch):
    """429 у ВСЕХ -> RuntimeError('rate limit'), а не None."""
    calls = []
    monkeypatch.setattr(llm._LLM_SESSION, "post",
                        _router({"qwen": 429, "groq": 429, "ds": 429}, calls))

    with pytest.raises(RuntimeError, match="rate limit"):
        llm._llm_request("Отвечай json.",
                         [{"role": "user", "content": "hi"}])
    assert [c["host"] for c in calls] == ["qwen", "groq", "ds"]


def test_vision_all_429_raises_rate_limit(monkeypatch):
    """Vision: 429 у всех -> RuntimeError('rate limit'), не 'vision unavail'."""
    calls = []
    monkeypatch.setattr(llm._LLM_SESSION, "post",
                        _router({"qwen": 429, "groq": 429, "ds": 429}, calls))

    with pytest.raises(RuntimeError, match="rate limit"):
        llm._llm_vision_request("Верни json.", "BTCUSDT 1H", _PNG)
    assert [c["host"] for c in calls] == ["qwen", "groq", "ds"]


def test_qwen_without_key_goes_straight_to_groq(monkeypatch, caplog):
    """Ключа Qwen нет -> провайдер пропущен, запрос сразу уходит в Groq."""
    monkeypatch.setattr(llm, "QWEN_API_KEY_1", "")
    calls = []
    monkeypatch.setattr(llm._LLM_SESSION, "post",
                        _router({"qwen": 200, "groq": 200, "ds": 200}, calls))

    with caplog.at_level(logging.WARNING, logger="app_pkg.ai.llm"):
        result = llm._llm_request("Отвечай json.",
                                  [{"role": "user", "content": "hi"}])

    assert result == "OK"
    assert [c["host"] for c in calls] == ["groq"]
    assert any("provider qwen skipped (no key)" in r.getMessage()
               for r in caplog.records)


def test_provider_order_from_config_is_respected(monkeypatch):
    """LLM_PROVIDER_ORDER=deepseek,qwen -> первым опрашивается DeepSeek."""
    monkeypatch.setattr(llm, "LLM_PROVIDER_ORDER", ["deepseek", "qwen"])
    calls = []
    monkeypatch.setattr(llm._LLM_SESSION, "post",
                        _router({"qwen": 200, "groq": 200, "ds": 200}, calls))

    assert llm._llm_request("Отвечай json.",
                            [{"role": "user", "content": "hi"}]) == "OK"
    assert [c["host"] for c in calls] == ["ds"]
    assert calls[0]["payload"]["model"] == "deepseek-v4-flash"


def test_vision_qwen_vl_200_used(monkeypatch):
    """Vision: Qwen-vl 200 -> qwen-vl-max и image_url в payload."""
    calls = []
    monkeypatch.setattr(llm._LLM_SESSION, "post",
                        _router({"qwen": 200, "groq": 200, "ds": 200}, calls))

    result = llm._llm_vision_request("Верни json.", "BTCUSDT 1H", _PNG)

    assert result == "OK"
    assert [c["host"] for c in calls] == ["qwen"]
    assert calls[0]["payload"]["model"] == "qwen-vl-max"
    content = calls[0]["payload"]["messages"][1]["content"]
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_vision_qwen_vl_first_and_404_falls_back_to_groq(monkeypatch, caplog):
    """Vision: Qwen-vl первым (даже при порядке groq,deepseek); 404 -> Groq."""
    monkeypatch.setattr(llm, "LLM_PROVIDER_ORDER", ["groq", "deepseek"])
    calls = []
    monkeypatch.setattr(llm._LLM_SESSION, "post",
                        _router({"qwen": 404, "groq": 200, "ds": 200}, calls))

    with caplog.at_level(logging.WARNING, logger="app_pkg.ai.llm"):
        result = llm._llm_vision_request("Верни json.", "BTCUSDT 1H", _PNG)

    assert result == "OK"
    # qwen вызывался первым несмотря на LLM_PROVIDER_ORDER, затем groq.
    assert [c["host"] for c in calls] == ["qwen", "groq"]
    assert calls[1]["payload"]["model"] == "llama-3.2-90b-vision-preview"
    assert any("provider qwen failed (HTTP 404), fallback to groq"
               in r.getMessage() for r in caplog.records)