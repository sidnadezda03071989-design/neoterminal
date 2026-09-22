# -*- coding: utf-8 -*-
"""Тесты детерминированного фильтра сигналов (app_pkg/ai/signal_filter.py).

Проверяем правила спеки:
  1. MTF обязательный фильтр: 4h против сигнала режет сторону (×0.7);
  2. MTF consensus: +0.05 за каждый согласный ТФ; consensus/alignment в мета;
  3. горизонт сигнала (horizon_bars, generated_at) — обязательные метки;
  4. confidence-агрегат: ниже 0.5 сигнал не торгуем даже при pu>0.6;
  5. ADX<20 (флэт): pu, pd ×0.6;
  6. VWAP + OBV подтверждение: +0.05;
  а также filter_levels по стороне уровня и интеграцию в run_verdict_backtest
  / levels_for_slice (без сети).
"""

import json

import pandas as pd
import pytest

from app_pkg import config
from app_pkg.ai import ai_backtest as aibt
from app_pkg.ai import signal_filter as sf
from app_pkg.data import fetch

BASE_TS = 1_700_000_000


@pytest.fixture(autouse=True)
def _isolate():
    fetch.data_cache.clear()
    aibt._RUNS.clear()
    yield
    fetch.data_cache.clear()
    aibt._RUNS.clear()


def _v(sig="L", pu=0.6, pd=0.3, pf=0.1):
    return {"sig": sig, "pu": pu, "pd": pd, "pf": pf,
            "tg": [[101.0, 0.6]], "conf": pu if sig == "L" else pd}


def _mtf(blocks):
    return {"mtf": {name: {"trend": trend, "tr": {"up": 1, "down": -1,
                                                 "flat": 0}[trend]}
                    for name, trend in blocks.items()}}


def _bars_df(n, step=3600, base_ts=BASE_TS):
    return pd.DataFrame({
        "timestamp": pd.to_datetime([base_ts + i * step for i in range(n)],
                                    unit="s", utc=True),
        "open": [100.0] * n, "high": [101.0] * n, "low": [99.0] * n,
        "close": [100.0] * n, "volume": [1.0] * n,
    })


# --------------------------------------------------- правила 1-2: MTF
def test_mtf_4h_down_against_long_cuts_pu_and_blocks():
    """LONG против 4h 'down': pu ×0.7; confidence 0.35 < 0.5 -> sig=F."""
    out = sf.filter_verdict(_v("L"), _mtf({"4h": "down"}))
    assert out["sig"] == "F"                      # п.4 блокирует сделку
    assert out["pu"] == pytest.approx(0.42)       # 0.6 × 0.7
    assert out["conf"] == pytest.approx(0.1)      # после F conf = pf
    f = out["filter"]
    assert f["confidence"] == pytest.approx(0.35)
    assert f["alignment"] is None                 # 1h нет — расхождение не счесть
    assert "mtf_4h_against" in f["flags"]
    assert "confidence_blocked" in f["flags"]


def test_mtf_4h_up_against_short_cuts_pd():
    """SHORT против 4h 'up': pd ×0.7 и блокировка по confidence."""
    out = sf.filter_verdict(_v("S", pu=0.3, pd=0.6), _mtf({"4h": "up"}))
    assert out["sig"] == "F"
    assert out["pd"] == pytest.approx(0.42)


def test_mtf_consensus_adds_bonus_and_keeps_signal():
    """Все 4 ТФ up + LONG: pu +4×0.05, alignment=+1, confidence>=0.5."""
    snap = _mtf({"15m": "up", "1h": "up", "4h": "up", "1d": "up"})
    out = sf.filter_verdict(_v("L"), snap)
    assert out["sig"] == "L"
    assert out["pu"] == pytest.approx(0.8)        # 0.6 + 0.20
    f = out["filter"]
    assert f["consensus"] == 4
    assert f["alignment"] == 1
    assert f["confidence"] == pytest.approx(0.7)  # 0.5 + 4×0.05
    assert "mtf_consensus" in f["flags"]


def test_mtf_against_consensus_and_vwap_keep_signal():
    """4h против, но 2 согласных ТФ + VWAP/OBV: confidence 0.5, sig остаётся L."""
    snap = {
        "mtf": _mtf({"15m": "up", "1h": "down", "4h": "down", "1d": "up"})["mtf"],
        "v": {"vwd": 1.2, "obv": 0.3},
    }
    out = sf.filter_verdict(_v("L"), snap)
    assert out["sig"] == "L"
    assert out["pu"] == pytest.approx(0.57)       # 0.6×0.7 + 2×0.05 + 0.05(vwap)
    f = out["filter"]
    assert f["confidence"] == pytest.approx(0.5)  # 0.5 -0.15 +0.10 +0.05
    assert f["consensus"] == 2
    assert f["alignment"] == 1                    # 1h down = 4h down
    assert {"mtf_4h_against", "mtf_consensus", "vwap_obv_confirm"} \
        <= set(f["flags"])


def test_alignment_divergence_penalizes_confidence():
    """1h up + 4h down: alignment -1, confidence 0.3 -> сделку не берём."""
    out = sf.filter_verdict(_v("L"), _mtf({"1h": "up", "4h": "down"}))
    assert out["filter"]["alignment"] == -1
    # 0.5 -0.15 (4h против) +0.05 (1h согласен) -0.10 (расхождение)
    assert out["filter"]["confidence"] == pytest.approx(0.3)
    assert out["sig"] == "F"


# ----------------------------------------------------------- правило 5: ADX
def test_adx_flat_cuts_both_sides():
    """tr.adx < 20 (флэт): pu и pd ×0.6, confidence 0.4 -> sig=F."""
    out = sf.filter_verdict(_v("L"), {"tr": {"adx": 15.0}})
    assert out["pu"] == pytest.approx(0.36)
    assert out["pd"] == pytest.approx(0.18)
    assert out["filter"]["confidence"] == pytest.approx(0.4)
    assert out["sig"] == "F"
    assert "adx_flat" in out["filter"]["flags"]


def test_adx_trend_boosts_confidence():
    """tr.adx >= 25 по направлению: confidence 0.55, сигнал не режем."""
    out = sf.filter_verdict(_v("L"), {"tr": {"adx": 30.0}})
    assert out["sig"] == "L"
    assert out["pu"] == pytest.approx(0.6)
    assert out["filter"]["confidence"] == pytest.approx(0.55)
    assert "adx_trend" in out["filter"]["flags"]


# ----------------------------------------------------------- правило 6: VWAP+OBV
def test_vwap_obv_confirm_long():
    """LONG + v.vwd>0 + v.obv>0: pu +0.05 (0.6 -> 0.65), confidence 0.55."""
    out = sf.filter_verdict(_v("L"), {"v": {"vwd": 1.5, "obv": 0.4}})
    assert out["sig"] == "L"
    assert out["pu"] == pytest.approx(0.65)
    assert out["filter"]["confidence"] == pytest.approx(0.55)


def test_vwap_obv_confirm_short():
    """SHORT + v.vwd<0 + v.obv<0 (зеркало LONG-ветки): pd +0.05."""
    out = sf.filter_verdict(_v("S", pu=0.3, pd=0.6),
                            {"v": {"vwd": -1.5, "obv": -0.4}})
    assert out["sig"] == "S"
    assert out["pd"] == pytest.approx(0.65)


def test_vwap_obv_partial_not_confirmed():
    """Только VWAP или только OBV — подтверждения нет (ветка не срабатывает)."""
    out = sf.filter_verdict(_v("L"), {"v": {"vwd": 1.5, "obv": -0.4}})
    assert out["pu"] == pytest.approx(0.6)
    assert "vwap_obv_confirm" not in out["filter"]["flags"]


# ------------------------------------------- границы и отсутствие данных
def test_missing_blocks_are_noop():
    """Нет mtf/tr/v — правила молчат (missing = no data), сигнал без изменений."""
    out = sf.filter_verdict(_v("L"), {"t": {"rsi": 50.0, "close": 100.0}})
    assert out["sig"] == "L"
    assert out["pu"] == pytest.approx(0.6)
    assert out["pd"] == pytest.approx(0.3)
    assert out["conf"] == pytest.approx(0.6)
    assert out["filter"]["confidence"] == pytest.approx(0.5)
    assert out["filter"]["flags"] == []


def test_filter_verdict_none_and_implausible_sig():
    """None -> None; сигнал без стороны дополняется метами без правок."""
    assert sf.filter_verdict(None) is None
    out = sf.filter_verdict({"sig": "X", "pu": 0.6})
    assert out["sig"] == "X"
    assert out["pu"] == 0.6
    assert out["filter"]["horizon_bars"] == config.SIGNAL_FILTER_HORIZON_BARS


def test_horizon_and_generated_at_metadata():
    """Горизонт и generated_at (п.3) кладутся в filter; дефолты подставляются."""
    out = sf.filter_verdict(_v("L"), {}, generated_at=1712345678,
                            horizon_bars=42)
    assert out["filter"]["horizon_bars"] == 42
    assert out["filter"]["generated_at"] == 1712345678
    defaulted = sf.filter_verdict(_v("L"), {})
    assert defaulted["filter"]["horizon_bars"] == config.SIGNAL_FILTER_HORIZON_BARS
    assert isinstance(defaulted["filter"]["generated_at"], int)


# ---------------------------------------------------------- filter_levels
def test_filter_levels_mtf_against_side():
    """4h down: UP-уровни ×0.7; DOWN получает +0.05 consensus (4h вниз)."""
    levels = [{"side": "UP", "price": 110.0, "probability": 0.6},
              {"side": "DOWN", "price": 90.0, "probability": 0.7}]
    out = sf.filter_levels(levels, _mtf({"4h": "down"}))
    by_side = {lv["side"]: lv for lv in out}
    assert by_side["UP"]["probability"] == pytest.approx(0.42)   # 0.6×0.7
    assert by_side["DOWN"]["probability"] == pytest.approx(0.75)  # 0.7+0.05
    assert by_side["UP"]["price"] == 110.0


def test_filter_levels_adx_flat_and_consensus_bonus():
    """ADX<20 режет обе стороны; consensus добавляет бонус по стороне."""
    snap = {
        "mtf": _mtf({"15m": "up", "1h": "up", "4h": "down", "1d": "up"})["mtf"],
        "tr": {"adx": 15.0},
    }
    levels = [{"side": "UP", "price": 110.0, "probability": 0.6},
              {"side": "DOWN", "price": 90.0, "probability": 0.7}]
    out = sf.filter_levels(levels, snap)
    by_side = {lv["side"]: lv for lv in out}
    # UP: 0.6×0.6 (adx) ×0.7 (4h down) + 3×0.05 -> 0.252 + 0.15 = 0.402
    assert by_side["UP"]["probability"] == pytest.approx(0.402)
    # DOWN: 0.7×0.6 + 1×0.05 = 0.47
    assert by_side["DOWN"]["probability"] == pytest.approx(0.47)


def test_filter_levels_noop_without_data():
    levels = [{"side": "UP", "price": 110.0, "probability": 0.6}]
    assert sf.filter_levels(levels, {"t": {"close": 100.0}})[0] \
        == levels[0]
    assert sf.filter_levels(None) == []


# ------------------------------------------------------------- интеграция
def test_run_verdict_backtest_applies_filter(monkeypatch):
    """Вердикты в движке проходят через фильтр: блокировка 4h видна в steps."""
    snaps = [_mtf({"4h": "down"}), {"t": {"rsi": 50.0, "close": 100.0}}]
    seq = iter(snaps)
    monkeypatch.setattr(aibt, "compact_snapshot",
                        lambda *a, **k: next(seq))
    monkeypatch.setattr(aibt, "charon_prompt_text", lambda: "SYS")
    monkeypatch.setattr(aibt, "get_series_df",
                        lambda *a, **k: _bars_df(2))

    def fake_llm(system, messages, **kwargs):
        usage = kwargs.get("usage_out")
        if isinstance(usage, dict):
            usage["prompt_tokens"] = 10
            usage["completion_tokens"] = 5
        return json.dumps({"sig": "L", "pu": 0.6, "pd": 0.3, "pf": 0.1})

    monkeypatch.setattr(aibt, "_llm_request", fake_llm)

    res = aibt.run_verdict_backtest("BTCUSDT", "1H", bars=2, adaptive=False)
    assert res["signals"] == ["F", "L"]
    assert res["steps"][0]["filter"]["confidence"] == pytest.approx(0.35)
    assert res["steps"][0]["filter"]["applied"]
    assert res["steps"][1]["filter"]["flags"] == []
    assert res["steps"][0]["conf"] == pytest.approx(0.1)


def test_levels_for_slice_applies_filter(monkeypatch):
    """levels_for_slice: уровни проходят filter_levels на том же снимке."""
    calls = []

    def fake_llm(system, messages, **kwargs):
        calls.append(messages)
        return json.dumps({"targets": [
            {"side": "UP", "price": 110, "probability": 0.6},
            {"side": "DOWN", "price": 90, "probability": 0.7},
        ]})

    monkeypatch.setattr(aibt, "_llm_request", fake_llm)
    monkeypatch.setattr(aibt, "_slice_price", lambda *a, **k: 100.0)
    monkeypatch.setattr(aibt, "charon_prompt_text", lambda: "SYS")
    monkeypatch.setattr(aibt, "compact_snapshot",
                        lambda *a, **k: _mtf({"4h": "down"}))

    levels, price, err = aibt.levels_for_slice("BTCUSDT", "1H")
    assert err is None
    by_side = {lv["side"]: lv for lv in levels}
    assert by_side["UP"]["probability"] == pytest.approx(0.42)
    # DOWN: 4h down не против DOWN; consensus +0.05
    assert by_side["DOWN"]["probability"] == pytest.approx(0.75)