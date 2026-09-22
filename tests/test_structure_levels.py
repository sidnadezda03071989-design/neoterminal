# -*- coding: utf-8 -*-
"""Детерминированные структурные уровни (app_pkg/ai/structure_levels.py).

Главная регрессия: уровни AI Backtest рисуются «по логике» из реальных
экстремумов свечей, а не формульной сеткой LLM (равный шаг, линейный спад
вероятности), одинаковой для каждого прогона.
"""

import math

import pandas as pd
import pytest

from app_pkg.ai.structure_levels import (
    atr_last,
    structure_levels_from_df,
)


def _df(highs, lows, closes, volume=None):
    n = len(closes)
    volume = volume if volume is not None else [1000.0] * n
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="1h",
                                   tz="UTC"),
        "open": closes, "high": highs, "low": lows,
        "close": closes, "volume": volume,
    })


def _swing_series(base=100.0, amp=8.0, n=60):
    """Цена с выраженными локальными экстремумами (фракталы ±2 бара)."""
    closes, highs, lows = [], [], []
    v = base
    for i in range(n):
        v = base + amp * math.sin(i * 0.35)
        closes.append(round(v, 2))
        highs.append(round(v + 1.2, 2))
        lows.append(round(v - 1.2, 2))
    return highs, lows, closes


def test_atr_last_constant_range():
    h = [100.0] * 30
    l = [97.0] * 30
    c = [99.0] * 30
    assert atr_last(h, l, c, period=14) == pytest.approx(3.0)


def test_atr_last_insufficient_bars():
    assert atr_last([1.0] * 5, [1.0] * 5, [1.0] * 5, period=14) is None


def test_structure_deterministic_for_same_df():
    hs, ls, cs = _swing_series()
    df = _df(hs, ls, cs)
    cp = cs[-1]
    a = structure_levels_from_df(df, cp)
    b = structure_levels_from_df(df, cp)
    assert a == b


def test_structure_levels_shape_and_sides():
    hs, ls, cs = _swing_series(base=100.0, amp=8.0, n=80)
    df = _df(hs, ls, cs)
    cp = cs[-1]
    levels = structure_levels_from_df(df, cp, max_per_side=5)
    assert len(levels) >= 4  # минимум 2 на сторону

    ups = [lv for lv in levels if lv["side"] == "UP"]
    downs = [lv for lv in levels if lv["side"] == "DOWN"]
    assert 2 <= len(ups) <= 5
    assert 2 <= len(downs) <= 5
    for lv in levels:
        assert set(lv) >= {"side", "price", "probability", "diff", "diff_pct"}
        assert MIN_PROB_LE(lv["probability"])
        assert lv["diff"] == pytest.approx(lv["price"] - cp, abs=1e-6)
    assert [lv["price"] for lv in ups] == sorted(lv["price"] for lv in ups)
    assert [lv["price"] for lv in downs] == sorted(
        (lv["price"] for lv in downs), reverse=True)
    # Ближний уровень вероятнее дальнего (монотонность по дистанции).
    assert ups[0]["probability"] >= ups[-1]["probability"]
    assert downs[0]["probability"] >= downs[-1]["probability"]


def MIN_PROB_LE(v):
    return 0.10 <= v <= 0.50


def test_structure_uses_real_swing_extremes():
    """Свинг-хай реально ложится в UP-уровни, свинг-лоу — в DOWN."""
    closes = [100.0 + i * 0.2 for i in range(40)]
    highs = [c + 2.0 for c in closes]
    lows = [c - 2.0 for c in closes]
    # Свинг-хай (бар 25) и свинг-лоу (бар 35) — реальные экстремумы,
    # окружение заметно ниже/выше; оба в пределах 2.5·ATR от цены среза.
    highs[25] = 115.0
    lows[25] = 100.0
    highs[35] = 104.0
    lows[35] = 98.0
    df = _df(highs, lows, closes)
    cp = closes[-1]  # ~107.8
    levels = structure_levels_from_df(df, cp)
    up_prices = [lv["price"] for lv in levels if lv["side"] == "UP"]
    down_prices = [lv["price"] for lv in levels if lv["side"] == "DOWN"]
    assert any(abs(p - 115.0) <= 0.1 for p in up_prices)
    assert any(abs(p - 98.0) <= 0.1 for p in down_prices)
    assert any(abs(p - 100.0) <= 0.1 for p in down_prices)


def test_structure_differs_between_datasets():
    """Два разных рынка -> разные уровни (контрпример «одинаковых для теста»)."""
    hs1, ls1, cs1 = _swing_series(base=100.0, amp=8.0, n=80)
    hs2, ls2, cs2 = _swing_series(base=200.0, amp=15.0, n=80)
    l1 = structure_levels_from_df(_df(hs1, ls1, cs1), cs1[-1])
    l2 = structure_levels_from_df(_df(hs2, ls2, cs2), cs2[-1])
    assert l1 != l2
    p1 = [lv["price"] for lv in l1]
    p2 = [lv["price"] for lv in l2]
    assert abs(sum(p1) - sum(p2)) > 1.0


def test_no_swings_filled_by_atr_anchors():
    """Монотонный тренд (нет фракталов) -> сторона всё равно не пустеет."""
    closes = [50.0 + i * 0.5 for i in range(60)]
    df = _df(closes, closes, closes)
    cp = closes[-1]
    levels = structure_levels_from_df(df, cp)
    ups = [lv for lv in levels if lv["side"] == "UP"]
    downs = [lv for lv in levels if lv["side"] == "DOWN"]
    assert len(ups) >= 2 and len(downs) >= 2
    # Якоря не в 0/негатив: у DOWN все цены положительны.
    assert all(lv["price"] > 0 for lv in downs)


def test_empty_on_short_data():
    closes = [100.0, 101.0, 102.0]
    assert structure_levels_from_df(_df(closes, closes, closes), 101.0) == []


def test_empty_on_bad_price():
    hs, ls, cs = _swing_series()
    df = _df(hs, ls, cs)
    assert structure_levels_from_df(df, None) == []
    assert structure_levels_from_df(df, -5.0) == []


def test_probabilities_scale_with_volatility():
    """Более волатильный рынок -> более разнесённые уровни (не шаблон ATR)."""
    hs1, ls1, cs1 = _swing_series(base=100.0, amp=3.0, n=80)
    hs2, ls2, cs2 = _swing_series(base=100.0, amp=20.0, n=80)
    l1 = structure_levels_from_df(_df(hs1, ls1, cs1), cs1[-1])
    l2 = structure_levels_from_df(_df(hs2, ls2, cs2), cs2[-1])
    span1 = max(lv["price"] for lv in l1) - min(lv["price"] for lv in l1)
    span2 = max(lv["price"] for lv in l2) - min(lv["price"] for lv in l2)
    assert span2 > span1 * 2