# -*- coding: utf-8 -*-
"""Тесты токен-диеты v3 (без сети): компактный снапшот и адаптивный AI Backtest.

Проверяем:
  - compact_snapshot: короткие ключи схемы v3, округление по полям,
    отсутствие "ok" и блоков без данных (missing = no data);
  - mtf-агрегаты вместо сырых свечей: {tf: {tr, trend, rsi, adx}};
  - llm: passthrough max_tokens/temperature и учёт usage-токенов;
  - триггеры адаптивного шага (a-e) на синтетике;
  - кэш вердиктов (cache_hits) и carry-forward (confidence *= 0.9);
  - пакетный режим batch=5/2 и fallback разбора по одному;
  - метрики run (tokens_prompt/tokens_completion/llm_calls/cache_hits);
  - критерии 50 баров: llm_calls <= 40% шагов, токены минимум в 3 раза
    меньше baseline, совпадение сигналов adaptive/baseline >= 80%.
"""

import json

import pandas as pd
import pytest

from app_pkg.ai import ai_backtest as aibt
from app_pkg.ai import llm
from app_pkg.data import fetch
from app_pkg.data import market_snapshot as ms

BASE_TS = 1_700_000_000


@pytest.fixture(autouse=True)
def _isolate():
    fetch.data_cache.clear()
    aibt._RUNS.clear()
    yield
    fetch.data_cache.clear()
    aibt._RUNS.clear()


def _bars_df(n, step=3600, base_ts=BASE_TS):
    """Ровный ряд из n баров — для _bar_timestamps движка."""
    return pd.DataFrame({
        "timestamp": pd.to_datetime([base_ts + i * step for i in range(n)],
                                    unit="s", utc=True),
        "open": [100.0] * n, "high": [101.0] * n, "low": [99.0] * n,
        "close": [100.0] * n, "volume": [1.0] * n,
    })


def _raw(**over):
    """Полный валидный raw-снимок get_raw_market_data (ok:true во всех блоках)."""
    raw = {
        "technicals": {
            "rsi": 62.3456, "atr": 1.2345, "bb_pct_b": 0.876,
            "sma20_diff_pct": 1.234, "close": 100.123, "atr_percentile": 0.512,
            "dist_to_high_pct": -2.345, "dist_to_low_pct": 3.456, "ok": True,
        },
        "scanner_edge": {
            "winrate": 0.606, "sharpe": 4.567, "max_dd": 0.0076,
            "profit_factor": 2.234, "trades": 80.0, "test_sharpe": 3.999,
            "params": {"oversold": 30, "overbought": 70}, "ok": True,
        },
        "sentiment": {"ls_ratio": 1.234, "long_pct": 0.5567,
                      "fear_greed": 0.424, "ok": True},
        "clock": {"hour_utc": 14.0, "dow": 0.0, "session": "london_ny",
                  "london_ny_overlap": 1.0, "market_open": 1.0, "ok": True},
        "macro": {"us10y": {"value": 4.123}, "fed_rate": {"value": 5.5},
                  "dxy": {"value": 104.567}, "ok": True},
        "calendar": {"events": [], "high_impact_in_2h": 1.0,
                     "next_event_in_min": 42.0, "ok": True},
        "derivatives": {"open_interest": 12345.678, "oi_change_1h_pct": 2.345,
                        "funding_rate": 0.0001234, "basis_pct": 0.045,
                        "top_ls_ratio": 1.234, "ok": True},
        "news_sentiment": {"avg_sentiment": -0.234, "bullish_count": 3.0,
                           "bearish_count": 5.0, "ok": True},
    }
    raw.update(over)
    return raw


def _blank_raw():
    """Raw-снимок, где все блоки без данных (ok:false / все-null)."""
    return {
        "technicals": {"rsi": None, "atr": None, "bb_pct_b": None,
                       "sma20_diff_pct": None, "close": None,
                       "atr_percentile": None, "dist_to_high_pct": None,
                       "dist_to_low_pct": None, "ok": False},
        "scanner_edge": {"winrate": None, "sharpe": None, "max_dd": None,
                         "profit_factor": None, "trades": None,
                         "test_sharpe": None, "params": {}, "ok": False},
        "sentiment": {"ls_ratio": None, "long_pct": None, "fear_greed": None,
                      "ok": False},
        "clock": {"hour_utc": None, "dow": None, "session": None,
                  "london_ny_overlap": None, "market_open": None, "ok": False},
        "macro": {"us10y": {"value": None}, "fed_rate": {"value": None},
                  "dxy": {"value": None}, "ok": False},
        "calendar": {"events": [], "high_impact_in_2h": None,
                     "next_event_in_min": None, "ok": False},
        "derivatives": {"open_interest": None, "oi_change_1h_pct": None,
                        "funding_rate": None, "basis_pct": None,
                        "top_ls_ratio": None, "ok": False},
        "news_sentiment": {"avg_sentiment": None, "bullish_count": None,
                           "bearish_count": None, "ok": False},
    }


# ------------------------------------------------------------- компактность
def test_compact_keys_and_rounding(monkeypatch):
    """Короткие ключи v3 + округление по полям; ok:true не отдаётся."""
    monkeypatch.setattr(ms, "get_raw_market_data", lambda *a, **k: _raw())
    monkeypatch.setattr(ms, "_mtf_compact",
                        lambda *a, **k: {"1h": {"tr": 1, "rsi": 62}})

    snap = ms.compact_snapshot("BTCUSDT", "1H")

    assert set(snap) <= {"t", "se", "s", "c", "m", "cal", "d", "ns", "mtf"}
    assert "ok" not in json.dumps(snap)
    assert set(snap["t"]) == {"rsi", "atr", "atr_pct", "bb", "sma", "ap", "dh",
                              "dl", "close"}
    t = snap["t"]
    assert t["rsi"] == 62.35          # 2dp
    assert t["bb"] == 0.88
    assert t["sma"] == 1.23
    assert t["ap"] == 0.51
    assert t["dh"] == -2.35
    assert t["dl"] == 3.46
    assert t["atr"] == 1.2            # 1dp
    assert t["atr_pct"] == 0.01       # atr/close, 2dp
    assert t["close"] == 100.1        # 1dp
    se = snap["se"]
    assert se["wr"] == 0.61 and se["sharpe"] == 4.57 and se["pf"] == 2.23
    assert se["dd"] == 0.01 and se["tsh"] == 4.0
    assert se["n"] == 80 and isinstance(se["n"], int)
    assert se["p"]["oversold"] == 30
    assert snap["s"]["ls"] == 1.23 and snap["s"]["long_pct"] == 0.6
    assert snap["s"]["fng"] == 0.42
    assert snap["c"]["ses"] == "london_ny" and snap["c"]["h"] == 14
    assert snap["m"] == {"us10y": 4.12, "fed": 5.5, "dxy": 104.57}
    assert snap["cal"] == {"hi2h": 1, "next_min": 42}
    assert snap["d"]["fund"] == 0.0001
    assert snap["ns"]["avg"] == -0.23 and snap["ns"]["bull"] == 3


def test_compact_omits_missing_blocks(monkeypatch):
    """ok:false / {} / все-null блоки НЕ попадают в JSON (missing = no data)."""
    monkeypatch.setattr(ms, "get_raw_market_data", lambda *a, **k: _blank_raw())
    monkeypatch.setattr(ms, "_mtf_compact", lambda *a, **k: {})
    assert ms.compact_snapshot("BTCUSDT", "1H") == {}

    # Пустые dict и все-null при ok:true — тоже пропуск.
    raw = _raw()
    raw["sentiment"] = {}
    raw["macro"] = {"us10y": {"value": None}, "fed_rate": {"value": None},
                    "dxy": {"value": None}, "ok": True}
    raw["derivatives"] = {"open_interest": None, "oi_change_1h_pct": None,
                          "funding_rate": None, "basis_pct": None,
                          "top_ls_ratio": None, "ok": True}
    monkeypatch.setattr(ms, "get_raw_market_data", lambda *a, **k: raw)
    snap = ms.compact_snapshot("BTCUSDT", "1H")
    assert "s" not in snap and "m" not in snap and "d" not in snap
    assert "t" in snap and "ns" in snap


def test_mtf_aggregates_no_raw_candles(monkeypatch):
    """Сырые свечи заменены агрегатами {tf:{tr,trend,rsi,adx}}."""
    monkeypatch.setattr(ms, "get_raw_market_data",
                        lambda *a, **k: _blank_raw())
    up = _bars_df(80)
    up["close"] = [100.0 + i for i in range(80)]     # close > sma50
    monkeypatch.setattr(ms, "get_series_df", lambda *a, **k: up)

    snap = ms.compact_snapshot("BTCUSDT", "1H")

    assert set(snap) == {"mtf"}
    assert set(snap["mtf"]) == {"15m", "1h", "4h", "1d"}
    assert snap["mtf"]["1h"]["tr"] == 1
    assert snap["mtf"]["1h"]["trend"] == "up"
    assert isinstance(snap["mtf"]["1h"]["rsi"], int)
    assert snap["mtf"]["1h"]["adx"] is None or isinstance(
        snap["mtf"]["1h"]["adx"], int)

    down = _bars_df(80)
    down["close"] = [100.0 - i for i in range(80)]   # close < sma50
    monkeypatch.setattr(ms, "get_series_df", lambda *a, **k: down)
    assert ms.compact_snapshot("BTCUSDT", "1H")["mtf"]["4h"]["tr"] == -1
    assert ms.compact_snapshot("BTCUSDT", "1H")["mtf"]["4h"]["trend"] == "down"
    # Никаких OHLC-массивов в снимке — только tr/trend/rsi/adx.
    assert "high" not in json.dumps(snap) and "low" not in json.dumps(snap)


# --------------------------------------------------------------- llm-учёт
def test_llm_build_payload_temperature_and_max_tokens():
    """max_tokens/temperature прокидываются в payload; дефолт T=0.3."""
    msg = [{"role": "user", "content": "x"}]
    p = llm._build_payload("m", "sys", msg, 120, 0.0)
    assert p["max_tokens"] == 120 and p["temperature"] == 0.0
    assert llm._build_payload("m", "sys", msg, 800)["temperature"] == 0.3


def test_collect_usage_accumulates():
    """usage.{prompt,completion,total}_tokens суммируются в аккумулятор."""
    class _Resp:
        def json(self):
            return {"usage": {"prompt_tokens": 7, "completion_tokens": 3,
                              "total_tokens": 10}}

    out = {}
    llm._collect_usage(_Resp(), out)
    llm._collect_usage(_Resp(), out)
    assert out == {"prompt_tokens": 14, "completion_tokens": 6,
                   "total_tokens": 20}


def test_build_payload_reasoning_effort_optional():
    """reasoning_effort идёт в payload только когда задан (None -> нет ключа)."""
    msg = [{"role": "user", "content": "x"}]
    assert "reasoning_effort" not in llm._build_payload("m", "sys", msg, 120)
    p = llm._build_payload("m", "sys", msg, 120, 0.0, "none")
    assert p["reasoning_effort"] == "none"


def test_attempt_chat_drops_reasoning_effort_on_400(monkeypatch):
    """HTTP 400 -> повтор без response_format И без reasoning_effort.

    Часть провайдеров не знает reasoning_effort: 400 не должен ломать
    цепочку, поэтому необязательный параметр снимается на повторе.
    """
    class _R:
        def __init__(self, sc):
            self.status_code = sc

    calls = []

    def fake_post(url, headers, payload, timeout, model):
        calls.append(dict(payload))
        return _R(400 if len(calls) == 1 else 200)

    monkeypatch.setattr(llm, "_post_chat", fake_post)
    resp = llm._attempt_chat(
        "https://x/v1", "k", "m", "sys",
        [{"role": "user", "content": "hi"}], 120, 10, 0.0, "none")

    assert calls[0]["reasoning_effort"] == "none"
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert "reasoning_effort" not in calls[1]
    assert "response_format" not in calls[1]
    assert resp.status_code == 200


def test_completion_content_ignores_truncated_reasoning():
    """finish_reason=length + пустой content -> None (обрезанный reasoning).

    Иначе панель получала chain-of-thought («We need to process the input
    snapshot...») вместо JSON-уровней. Пустой content при finish_reason=stop
    по-прежнему читается из reasoning (gpt-oss-120b кладёт ответ туда).
    """
    class _Resp:
        def __init__(self, data):
            self._d = data

        def json(self):
            return self._d

    def _resp(finish, content="", reasoning=""):
        return _Resp({"choices": [{
            "finish_reason": finish,
            "message": {"content": content, "reasoning": reasoning},
        }]})

    assert llm._completion_content(
        _resp("length", "", "We need to process")) is None
    assert llm._completion_content(_resp("stop", "", '{"a":1}')) == '{"a":1}'
    assert llm._completion_content(_resp("length", '{"a":1}')) == '{"a":1}'


def test_verdict_calls_request_reasoning_effort_none(monkeypatch):
    """Вердикт-вызовы (live-панель и движок) глушат reasoning (effort=none)."""
    seen = []

    def fake(system, messages, **kwargs):
        seen.append(kwargs.get("reasoning_effort"))
        return json.dumps({"pu": 0.6, "pd": 0.4, "pf": 1.5, "sig": "L",
                           "tg": [[101.0, 0.6], [99.0, 0.4]]})

    monkeypatch.setattr(aibt, "_llm_request", fake)
    monkeypatch.setattr(aibt, "compact_snapshot", lambda *a, **k: {})

    aibt._llm_verdict("sys", "payload", None, {})
    levels, price, err = aibt.levels_for_slice("BTCUSDT", "1H",
                                               current_price=100.0)

    assert seen == ["none", "none"]
    assert err is None and levels


def test_slice_price_replay_uses_replay_window(monkeypatch):
    """В реплее цена берётся из get_replay_df (окно до барьера), не из live."""
    df = _bars_df(5, step=3600, base_ts=BASE_TS)
    called = {}

    def fake_replay(symbol, tf, to_sec=None, limit=None):
        called["to_sec"] = to_sec
        return df

    monkeypatch.setattr(aibt, "get_replay_df", fake_replay)
    monkeypatch.setattr(aibt, "get_series_df",
                        lambda *a, **k: pytest.fail("live get_series_df в реплее"))

    price = aibt._slice_price("BTCUSDT", "1H", upto_sec=BASE_TS + 3600 * 4)

    assert price == pytest.approx(100.0)
    assert called["to_sec"] == float(BASE_TS + 3600 * 4)


def test_slice_df_replay_uses_replay_window(monkeypatch):
    """Снимок в реплее строится из исторического окна до барьера."""
    df = _bars_df(10, step=3600, base_ts=BASE_TS)

    monkeypatch.setattr(ms, "get_replay_df",
                        lambda *a, **k: df)
    monkeypatch.setattr(ms, "get_series_df",
                        lambda *a, **k: pytest.fail("live get_series_df в реплее"))

    out = ms._slice_df("BTCUSDT", "1H", upto_sec=BASE_TS + 3600 * 5)
    assert len(out) == 6


# -------------------------------------------------------------- триггеры
def test_adaptive_triggers(monkeypatch):
    """(a-e) триггеры: зоны rsi, скачок rsi/close, смена ses/hi2h, первый шаг."""
    p = {"oversold": 30.0, "overbought": 70.0}
    base = {"t": {"rsi": 50.0, "close": 100.0, "atr": 2.0},
            "c": {"ses": "asia", "open": 1}, "cal": {"hi2h": 0}}
    last = {"rsi": 50.0, "close": 100.0}

    assert aibt._should_call(base, None, None, p) is True          # (e)
    assert aibt._should_call(base, base, last, p) is False         # тишина
    near = {**base, "t": {"rsi": 28.0, "close": 100.0, "atr": 2.0}}
    assert aibt._should_call(near, base, last, p) is True          # (a)
    jump = {**base, "t": {"rsi": 58.5, "close": 100.0, "atr": 2.0}}
    assert aibt._should_call(jump, base, last, p) is True          # (b)
    move = {**base, "t": {"rsi": 50.0, "close": 101.5, "atr": 2.0}}
    assert aibt._should_call(move, base, last, p) is True          # (c)
    ses = {**base, "c": {"ses": "london", "open": 1}}
    assert aibt._should_call(ses, base, last, p) is True           # (d)
    hi = {**base, "cal": {"hi2h": 1}}
    assert aibt._should_call(hi, base, last, p) is True            # (d)
    small = {**base, "t": {"rsi": 55.0, "close": 101.0, "atr": 2.0}}
    assert aibt._should_call(small, base, last, p) is False


def _quiet_snap():
    return {"t": {"rsi": 50.0, "close": 100.0, "atr": 2.0},
            "c": {"ses": "asia", "open": 1}, "cal": {"hi2h": 0},
            "se": {"p": {"oversold": 30, "overbought": 70}}}


def _fake_llm(calls, sig="L"):
    def fake(system, messages, **kwargs):
        calls.append(messages[0]["content"])
        usage = kwargs.get("usage_out")
        if isinstance(usage, dict):
            usage["prompt_tokens"] = 10
            usage["completion_tokens"] = 5
        return json.dumps({"sig": sig, "pu": 0.6, "pd": 0.3, "pf": 0.1,
                           "tg": [[101, 0.6]]})
    return fake


# ------------------------------------------------------------ кэш вердиктов
def test_verdict_cache_hit_and_carry_forward(monkeypatch):
    """Смена hi2h триггерит вызов с тем же ключом -> cache_hit, без сети."""
    snaps = [_quiet_snap(), {**_quiet_snap(), "cal": {"hi2h": 1}}]
    seq = iter(snaps + [snaps[-1]] * 20)
    monkeypatch.setattr(aibt, "compact_snapshot", lambda *a, **k: next(seq))
    monkeypatch.setattr(aibt, "charon_prompt_text", lambda: "SYS")
    monkeypatch.setattr(aibt, "get_series_df",
                        lambda *a, **k: _bars_df(2))
    calls = []
    monkeypatch.setattr(aibt, "_llm_request", _fake_llm(calls))

    res = aibt.run_verdict_backtest("BTCUSDT", "1H", bars=2, adaptive=True)

    assert res["llm_calls"] == 1
    assert res["cache_hits"] == 1
    assert res["tokens_prompt"] == 10 and res["tokens_completion"] == 5
    assert res["signals"] == ["L", "L"]


def test_carry_forward_decays_confidence(monkeypatch):
    """Пропущенный шаг: сигнал прежний, confidence *= 0.9."""
    monkeypatch.setattr(aibt, "compact_snapshot",
                        lambda *a, **k: _quiet_snap())
    monkeypatch.setattr(aibt, "charon_prompt_text", lambda: "SYS")
    monkeypatch.setattr(aibt, "get_series_df", lambda *a, **k: _bars_df(4))
    calls = []
    monkeypatch.setattr(aibt, "_llm_request", _fake_llm(calls))

    res = aibt.run_verdict_backtest("BTCUSDT", "1H", bars=4, adaptive=True)

    confs = [s["conf"] for s in res["steps"]]
    assert res["llm_calls"] == 1
    assert confs[0] == 0.6
    assert confs[1] == pytest.approx(0.54)
    assert confs[2] == pytest.approx(0.486)
    assert res["steps"][1]["carried"] is True


# ---------------------------------------------------------------- batch
def test_parse_verdict_batch_and_invalid():
    """JSON-массив -> n вердиктов; несовпадение длины/не-JSON -> None."""
    arr = json.dumps([
        {"sig": "L", "pu": 0.6, "pd": 0.3, "pf": 0.1, "tg": [[101, 0.6]]},
        {"sig": "F", "pu": 0.3, "pd": 0.3, "pf": 0.4},
    ])
    out = aibt.parse_verdict_batch(arr, 2)
    assert [v["sig"] for v in out] == ["L", "F"]
    assert out[0]["tg"] == [[101, 0.6]]

    assert aibt.parse_verdict_batch("не json", 2) is None
    assert aibt.parse_verdict_batch(json.dumps([{"sig": "L"}]), 2) is None
    # Обёртка {"v": [...]} тоже поддерживается.
    assert aibt.parse_verdict_batch(
        json.dumps({"v": [{"sig": "S", "pu": 0.2, "pd": 0.7, "pf": 0.1}]}),
        1)[0]["sig"] == "S"


def test_batch_packing_sends_arrays(monkeypatch):
    """batch=2: по 2 снимка на вызов (adaptive=False -> все шаги в сеть)."""
    monkeypatch.setattr(aibt, "compact_snapshot",
                        lambda *a, **k: _quiet_snap())
    monkeypatch.setattr(aibt, "charon_prompt_text", lambda: "SYS")
    monkeypatch.setattr(aibt, "get_series_df", lambda *a, **k: _bars_df(4))
    payloads = []

    def fake(system, messages, **kwargs):
        content = messages[0]["content"]
        payloads.append(content)
        usage = kwargs.get("usage_out")
        if isinstance(usage, dict):
            usage["prompt_tokens"] = 10
            usage["completion_tokens"] = 5
        if content.strip().startswith("["):
            count = content.count('"t"')
            return json.dumps([{"sig": "L", "pu": 0.5, "pd": 0.3, "pf": 0.2}
                               for _ in range(count)])
        return json.dumps({"sig": "L", "pu": 0.5, "pd": 0.3, "pf": 0.2})

    monkeypatch.setattr(aibt, "_llm_request", fake)

    res = aibt.run_verdict_backtest("BTCUSDT", "1H", bars=4, batch=2,
                                    adaptive=False)
    assert res["llm_calls"] == 2
    assert res["signals"] == ["L", "L", "L", "L"]
    assert sum(1 for p in payloads if p.strip().startswith("[")) == 2


def test_batch_fallback_to_singles(monkeypatch):
    """Невалидный массив -> пакет разбирается по одному (fallback)."""
    monkeypatch.setattr(aibt, "compact_snapshot",
                        lambda *a, **k: _quiet_snap())
    monkeypatch.setattr(aibt, "charon_prompt_text", lambda: "SYS")
    monkeypatch.setattr(aibt, "get_series_df", lambda *a, **k: _bars_df(2))

    def fake(system, messages, **kwargs):
        usage = kwargs.get("usage_out")
        if isinstance(usage, dict):
            usage["prompt_tokens"] = 10
            usage["completion_tokens"] = 5
        if messages[0]["content"].strip().startswith("["):
            return "мусор"  # пакет невалиден
        return json.dumps({"sig": "L", "pu": 0.5, "pd": 0.3, "pf": 0.2})

    monkeypatch.setattr(aibt, "_llm_request", fake)

    res = aibt.run_verdict_backtest("BTCUSDT", "1H", bars=2, batch=2,
                                    adaptive=False)
    assert res["llm_calls"] == 3  # 1 пакетный + 2 одиночных
    assert res["signals"] == ["L", "L"]


# ---------------------------------------------------------------- метрики
def test_run_metrics_contain_token_diet(monkeypatch):
    """Метрики run: tokens_prompt/completion, llm_calls, cache_hits, steps."""
    monkeypatch.setattr(aibt, "compact_snapshot",
                        lambda *a, **k: _quiet_snap())
    monkeypatch.setattr(aibt, "charon_prompt_text", lambda: "SYS")
    monkeypatch.setattr(aibt, "get_series_df", lambda *a, **k: _bars_df(5))
    calls = []
    monkeypatch.setattr(aibt, "_llm_request", _fake_llm(calls))

    res = aibt.run_verdict_backtest("BTCUSDT", "1H", bars=5, adaptive=True)
    for key in ("tokens_prompt", "tokens_completion", "llm_calls",
                "cache_hits", "steps", "llm_calls_ratio"):
        assert key in res["metrics"]
    assert {"tokens_prompt", "tokens_completion", "llm_calls",
            "cache_hits"} <= set(res)
    assert res["metrics"]["steps"] == 5


# ------------------------------------------- критерии диеты (50 баров, офлайн)
def test_adaptive_50_bars_diet_criteria(monkeypatch):
    """50 шагов: llm_calls <= 40%, токены >=3x меньше baseline, сигналы >=80%."""
    steps = 50
    monkeypatch.setattr(aibt, "compact_snapshot",
                        lambda *a, **k: _quiet_snap())
    monkeypatch.setattr(aibt, "charon_prompt_text", lambda: "SYS")
    monkeypatch.setattr(aibt, "get_series_df",
                        lambda *a, **k: _bars_df(steps))
    calls = []
    monkeypatch.setattr(aibt, "_llm_request", _fake_llm(calls))

    base = aibt.run_verdict_backtest("BTCUSDT", "1H", bars=steps,
                                     adaptive=False)
    adapt = aibt.run_verdict_backtest("BTCUSDT", "1H", bars=steps,
                                      adaptive=True)

    assert adapt["metrics"]["llm_calls"] <= 0.4 * steps
    base_tokens = base["tokens_prompt"] + base["tokens_completion"]
    adapt_tokens = adapt["tokens_prompt"] + adapt["tokens_completion"]
    assert adapt_tokens * 3 <= base_tokens

    match = sum(1 for a, b in zip(adapt["signals"], base["signals"]) if a == b)
    assert match >= 0.8 * steps


# --------------------------------------- живой AI Backtest: компактный снимок
def test_levels_for_slice_uses_compact_snapshot_and_tg(monkeypatch):
    """Живая панель: в LLM уходит компактный снимок, ответ tg -> уровни.

    Регрессия: промпт v3 описывает ключи t/se/s/... и схему
    {"pu","pd","pf","sig","tg":[[price,prob]]}; старый verbose-снимок
    (technicals/scanner_edge/sentiment) и старый парсер давали 0 уровней.
    """
    calls = {}

    def fake_llm(system, messages, **kwargs):
        calls["system"] = system
        calls["messages"] = messages
        return json.dumps({"pu": 0.6, "pd": 0.0, "pf": 1.5, "sig": "L",
                           "tg": [[102.5, 0.6], [98.0, 0.4]]})

    monkeypatch.setattr(aibt, "_llm_request", fake_llm)
    monkeypatch.setattr(aibt, "_slice_price", lambda *a, **k: 100.0)
    monkeypatch.setattr(aibt, "charon_prompt_text", lambda: "RULES")
    monkeypatch.setattr(aibt, "compact_snapshot",
                        lambda *a, **k: {"t": {"rsi": 25.0, "close": 100.0}})
    monkeypatch.setattr(aibt, "_structure_levels", lambda *a, **k: [])

    levels, price, err = aibt.levels_for_slice("BTCUSDT", "1H")

    assert err is None and price == pytest.approx(100.0)
    assert calls["system"] == "RULES"
    assert calls["messages"] == [{"role": "user", "content": (
        '{"current_price":100.0,"t":{"rsi":25.0,"close":100.0}}')}]
    by_side = {lv["side"]: lv for lv in levels}
    assert by_side["UP"]["price"] == pytest.approx(102.5)
    assert by_side["DOWN"]["price"] == pytest.approx(98.0)
    assert by_side["UP"]["probability"] == pytest.approx(0.6)


def test_parse_tg_probabilities_normalized_to_100():
    """v3 tg: вероятности уровней нормализуются к сумме 1.00 (100%)."""
    raw = json.dumps({"sig": "L", "pu": 0.6, "pd": 0.3, "pf": 0.1,
                      "tg": [[102.0, 0.5], [98.0, 0.5], [96.0, 0.25]]})
    levels = aibt.parse_probability_levels(raw, current_price=100.0)

    assert sum(lv["probability"] for lv in levels) == pytest.approx(1.0,
                                                                    abs=1e-3)
    by_price = {lv["price"]: lv["probability"] for lv in levels}
    assert by_price[102.0] == pytest.approx(0.4)
    assert by_price[98.0] == pytest.approx(0.4)
    assert by_price[96.0] == pytest.approx(0.2)


def test_parse_tg_dedupes_same_price_keeping_max_prob():
    """Одинаковая цена в одной стороне -> одна строка (max вероятность), сумма = 1."""
    raw = json.dumps({"sig": "S", "tg": [[98.0, 0.1], [98.0, 0.3],
                                         [102.0, 0.3], [96.0, 0.3]]})
    levels = aibt.parse_probability_levels(raw, current_price=100.0)

    prices = [lv["price"] for lv in levels]
    assert prices.count(98.0) == 1
    assert sum(lv["probability"] for lv in levels) == pytest.approx(1.0,
                                                                    abs=1e-3)
