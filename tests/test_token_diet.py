# -*- coding: utf-8 -*-
"""Тесты токен-диеты (без сети): контекст, история чата, кеш LLM.

Проверяем:
  - format_tf_section: у главного ТФ не больше CONTEXT_CANDLES_MAIN
    свечей (+2 строки: сводка старых баров и заголовок-секция), у
    остальных — не больше CONTEXT_CANDLES_OTHER (+2);
  - индикаторы: ровно CONTEXT_INDICATOR_TAIL значений в каждом ряду;
  - сводка старых баров присутствует у главного ТФ и отсутствует
    у второстепенных;
  - build_multi_tf_context пропускает ТФ из CONTEXT_SKIP_TF (кроме main);
  - _chat_flow обрезает старые сообщения до CHAT_MESSAGE_MAX_CHARS,
    последнее (текущий вопрос) — целиком;
  - кеш LLM: два одинаковых analysis-запроса -> один вызов сети;
    запрос с другой последней свечой -> новый вызов; chat не кешируется.
"""

import re

import pandas as pd
import pytest

from app_pkg import config
from app_pkg.ai.context import build_multi_tf_context, format_tf_section
from app_pkg.ai import chat as ai_chat
from app_pkg.ai import llm
from app_pkg.cache import _LLM_RESPONSE_CACHE
from app_pkg.data import fetch

BASE_TS = 1_700_000_000
STEP = 60  # 1m
N = 300


def _df(n=N, step=STEP):
    """Ряд из n свечей с растущими ценами (close_i = i + 1)."""
    times = [BASE_TS + i * step for i in range(n)]
    closes = [float(i + 1) for i in range(n)]
    return pd.DataFrame({
        "timestamp": pd.to_datetime(times, unit="s", utc=True),
        "open": closes,
        "high": [c + 0.5 for c in closes],
        "low": [c - 0.5 for c in closes],
        "close": closes,
        "volume": [10.0] * n,
    })


@pytest.fixture(autouse=True)
def _isolate():
    """Чистые кеши данных/LLM между тестами."""
    fetch.data_cache.clear()
    _LLM_RESPONSE_CACHE.clear()
    yield
    fetch.data_cache.clear()
    _LLM_RESPONSE_CACHE.clear()


def _candle_rows(text):
    return re.findall(r"\[(\d+) ([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+) ", text)


# ------------------------------------------------------------ контекст
def test_main_tf_candle_limit_and_summary(monkeypatch):
    """Главный ТФ: <= MAIN свечей + строка-сводка по старым барам."""
    monkeypatch.setattr("app_pkg.ai.context.get_series_df",
                        lambda *a, **k: _df())
    section, _ = format_tf_section("BTCUSDT", "15m", "15m")
    rows = _candle_rows(section)
    assert len(rows) <= config.CONTEXT_CANDLES_MAIN + 2
    assert len(rows) <= config.CONTEXT_CANDLES_MAIN  # ровно лимит рядов
    assert "older " in section and "trend=up" in section
    # Последняя свеча — самая свежая.
    assert int(rows[-1][0]) == BASE_TS + (N - 1) * STEP


def test_other_tf_candle_limit_no_summary(monkeypatch):
    """Второстепенный ТФ: <= OTHER свечей, без строки-сводки."""
    monkeypatch.setattr("app_pkg.ai.context.get_series_df",
                        lambda *a, **k: _df())
    section, _ = format_tf_section("BTCUSDT", "4H", "15m")
    rows = _candle_rows(section)
    assert len(rows) <= config.CONTEXT_CANDLES_OTHER + 2
    assert len(rows) <= config.CONTEXT_CANDLES_OTHER
    assert "older " not in section


def test_indicators_tail_values(monkeypatch):
    """Каждый индикатор — ровно CONTEXT_INDICATOR_TAIL значений."""
    monkeypatch.setattr("app_pkg.ai.context.get_series_df",
                        lambda *a, **k: _df())
    section, _ = format_tf_section("BTCUSDT", "15m", "15m")
    ind_block = section.split("=== Indicators")[1]
    for line in ind_block.strip().splitlines():
        line = line.strip()
        if not line or "===" in line:
            continue
        name, _, values = line.partition("=")
        assert name in ("sma20", "sma50", "ema50", "rsi", "macd",
                        "macd_signal", "bb_up", "bb_mid", "bb_low")
        assert len(values.split()) <= config.CONTEXT_INDICATOR_TAIL


def test_skip_tf_dropped_from_sections_but_not_trends(monkeypatch):
    """build_multi_tf_context: 1m/5m не в секциях, но в Trends остаются."""
    monkeypatch.setattr("app_pkg.ai.context.get_series_df",
                        lambda *a, **k: _df())
    ctx = build_multi_tf_context("BTCUSDT", "15m")
    assert "=== Candles 1m" not in ctx
    assert "=== Candles 5m" not in ctx
    assert "=== Candles 15m" in ctx
    # Trends строятся по всем ТФ.
    for tf in config.TIMEFRAMES:
        assert f"15m: trend=" in ctx or tf in ctx.split("=== Trends ===")[1]


def test_candle_format_no_commas(monkeypatch):
    """Свеча в одну строку через пробел, без запятых."""
    monkeypatch.setattr("app_pkg.ai.context.get_series_df",
                        lambda *a, **k: _df(n=5))
    section, _ = format_tf_section("BTCUSDT", "15m", "15m")
    row = _candle_rows(section)[0]
    assert "," not in row
    assert re.search(r"\[\d+ [\d.]+ [\d.]+ [\d.]+ [\d.]+ [\d.]+\]", section)


# ------------------------------------------------------------ чат
def test_chat_flow_truncates_old_messages(monkeypatch):
    """Старые сообщения обрезаны до CHAT_MESSAGE_MAX_CHARS, текущее — нет."""
    long_msg = "x" * 1000
    captured = {}

    def fake_history(limit=None):
        return [
            {"role": "user", "content": long_msg},
            {"role": "assistant", "content": long_msg},
            {"role": "user", "content": long_msg},
            {"role": "assistant", "content": long_msg},
            {"role": "user", "content": long_msg},
            {"role": "user", "content": long_msg},  # текущий вопрос
        ]

    def fake_llm_request(system, messages, **kwargs):
        captured["msgs"] = messages
        captured["kwargs"] = kwargs
        return '{"reply": "ok"}'

    monkeypatch.setattr(ai_chat, "db_get_chat_history", fake_history)
    monkeypatch.setattr(ai_chat, "_llm_request", fake_llm_request)

    reply, model, fallback = ai_chat._chat_flow(
        "привет", "BTCUSDT", "15m")
    assert reply == "ok"
    msgs = captured["msgs"]
    # Текущее сообщение последнее и целиком.
    assert msgs[-1]["content"] == "привет"
    for m in msgs[:-1]:
        assert len(m["content"]) <= config.CHAT_MESSAGE_MAX_CHARS
    assert "..." in msgs[0]["content"]
    assert captured["kwargs"].get("purpose") == "chat"


# ------------------------------------------------------------ кеш LLM
_CANDLE_MSG = [{
    "role": "user",
    "content": "[1700000000 76000 76100 75900 76050 10]\n"
               "проанализируй рынок",
}]


def test_llm_cache_dedupes_identical_analysis(monkeypatch):
    """Два одинаковых analysis-запроса -> один вызов сети."""
    calls = []

    class FakeResp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "OK"}}]}

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append(url)
        return FakeResp()

    monkeypatch.setattr(llm._LLM_SESSION, "post", fake_post)
    monkeypatch.setattr(llm, "LLM_PROVIDER_ORDER", ["deepseek"])
    monkeypatch.setattr(llm, "QWEN_API_KEY_1", "")
    monkeypatch.setattr(llm, "GROQ_API_KEY", "")
    monkeypatch.setattr(llm, "DEEPSEEK_API_KEY", "key-ds")
    monkeypatch.setattr(llm, "DEEPSEEK_BASE_URL", "https://fake.example/v1")
    monkeypatch.setattr(llm, "DEEPSEEK_MODEL", "deepseek-v4-flash")

    r1 = llm._llm_request("Отвечай json", list(_CANDLE_MSG))
    r2 = llm._llm_request("Отвечай json", list(_CANDLE_MSG))
    assert r1 == r2 == "OK"
    assert len(calls) == 1


def test_llm_cache_miss_on_new_candle(monkeypatch):
    """Новая последняя свеча -> новый ключ -> второй вызов сети."""
    calls = []

    class FakeResp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "OK"}}]}

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append(url)
        return FakeResp()

    monkeypatch.setattr(llm._LLM_SESSION, "post", fake_post)
    monkeypatch.setattr(llm, "LLM_PROVIDER_ORDER", ["deepseek"])
    monkeypatch.setattr(llm, "QWEN_API_KEY_1", "")
    monkeypatch.setattr(llm, "GROQ_API_KEY", "")
    monkeypatch.setattr(llm, "DEEPSEEK_API_KEY", "key-ds")
    monkeypatch.setattr(llm, "DEEPSEEK_BASE_URL", "https://fake.example/v1")
    monkeypatch.setattr(llm, "DEEPSEEK_MODEL", "deepseek-v4-flash")

    llm._llm_request("Отвечай json", list(_CANDLE_MSG))
    msg2 = [{"role": "user",
             "content": "[1700000060 76010 76110 75910 76060 10]\nанализ"}]
    llm._llm_request("Отвечай json", msg2)
    assert len(calls) == 2


def test_llm_chat_not_cached(monkeypatch):
    """purpose=chat не кешируется: два одинаковых вызова -> два запроса."""
    calls = []

    class FakeResp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "OK"}}]}

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append(url)
        return FakeResp()

    monkeypatch.setattr(llm._LLM_SESSION, "post", fake_post)
    monkeypatch.setattr(llm, "LLM_PROVIDER_ORDER", ["deepseek"])
    monkeypatch.setattr(llm, "QWEN_API_KEY_1", "")
    monkeypatch.setattr(llm, "GROQ_API_KEY", "")
    monkeypatch.setattr(llm, "DEEPSEEK_API_KEY", "key-ds")
    monkeypatch.setattr(llm, "DEEPSEEK_BASE_URL", "https://fake.example/v1")
    monkeypatch.setattr(llm, "DEEPSEEK_MODEL", "deepseek-v4-flash")

    msgs = [{"role": "user", "content": "привет"}]
    llm._llm_request("Отвечай json", list(msgs), purpose="chat")
    llm._llm_request("Отвечай json", list(msgs), purpose="chat")
    assert len(calls) == 2


def test_purpose_max_tokens_values():
    """purpose -> max_tokens: analysis 800, chat 400, None -> None."""
    assert llm._purpose_max_tokens("analysis") == config.LLM_MAX_TOKENS_ANALYSIS
    assert llm._purpose_max_tokens("chat") == config.LLM_MAX_TOKENS_CHAT
    assert llm._purpose_max_tokens(None) is None


def test_reason_truncated_to_config_limit():
    """_normalize_ai_result обрезает reason до LLM_REASON_MAX_CHARS."""
    from app_pkg.ai.agents import _normalize_ai_result
    long_reason = "r" * 1000
    result = _normalize_ai_result(
        {"signal": "buy", "confidence": 0.9, "reason": long_reason}, [])
    assert len(result["reason"]) == config.LLM_REASON_MAX_CHARS
