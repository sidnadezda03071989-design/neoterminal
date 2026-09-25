"""Тесты правил 21-24 (app_pkg/ai/apply_rules.py).

Проверяет:
  - Каждое правило корректно меняет pu/pd при соблюдении условий.
  - Правила пропускаются, когда блок d отсутствует.
  - Правила пропускаются, когда поле в d равно None.
  - _d_val корректно достаёт значения из snapshot["d"].
"""
from __future__ import annotations

import pytest

from app_pkg.ai.apply_rules import (
    _d_val,
    _rule_21_funding_extreme,
    _rule_22_oi_divergence,
    _rule_23_cvd_trend,
    _rule_24_taker_imbalance,
)


def _snap(d_block: dict | None) -> dict:
    return {"d": d_block} if d_block is not None else {}


# ---------------------------------------------------------------------------
# _d_val
# ---------------------------------------------------------------------------


def test_d_val_returns_value_from_d_block():
    assert _d_val({"d": {"funding_zscore": 2.5}}, "funding_zscore") == 2.5


def test_d_val_returns_none_when_d_missing():
    assert _d_val({}, "funding_zscore") is None


def test_d_val_returns_none_when_d_is_none():
    assert _d_val({"d": None}, "funding_zscore") is None


def test_d_val_returns_none_when_field_missing():
    assert _d_val({"d": {"oi": 100}}, "funding_zscore") is None


def test_d_val_returns_none_when_field_none():
    assert _d_val({"d": {"funding_zscore": None}}, "funding_zscore") is None


# ---------------------------------------------------------------------------
# Rule 21: funding_extreme
# ---------------------------------------------------------------------------


def test_rule_21_funding_extreme_positive_zscore():
    """funding_zscore > 2.0 -> pu-=.10, pd+=.10."""
    pu, pd = 0.50, 0.50
    snap = {"d": {"funding_zscore": 2.5}}
    pu, pd = _rule_21_funding_extreme(pu, pd, snap)
    assert pu == pytest.approx(0.40)
    assert pd == pytest.approx(0.60)


def test_rule_21_funding_extreme_negative_zscore():
    """funding_zscore < -2.0 -> pu+=.10, pd-=.10."""
    pu, pd = 0.50, 0.50
    snap = {"d": {"funding_zscore": -3.0}}
    pu, pd = _rule_21_funding_extreme(pu, pd, snap)
    assert pu == pytest.approx(0.60)
    assert pd == pytest.approx(0.40)


def test_rule_21_funding_extreme_normal_zscore():
    """|funding_zscore| <= 2.0 -> no change."""
    pu, pd = 0.50, 0.50
    snap = {"d": {"funding_zscore": 1.0}}
    pu_orig, pd_orig = pu, pd
    pu, pd = _rule_21_funding_extreme(pu, pd, snap)
    assert pu == pu_orig
    assert pd == pd_orig


def test_rule_21_funding_extreme_d_block_missing():
    pu, pd = 0.50, 0.50
    snap = {"d": None}
    pu, pd = _rule_21_funding_extreme(pu, pd, snap)
    assert pu == 0.50
    assert pd == 0.50


# ---------------------------------------------------------------------------
# Rule 22: oi_divergence
# ---------------------------------------------------------------------------


def test_rule_22_oi_divergence_growth():
    """oi_change_4 > 0.05 -> pu+=.05, pd+=.05."""
    pu, pd = 0.50, 0.50
    snap = {"d": {"oi_change_4": 0.10}}
    pu, pd = _rule_22_oi_divergence(pu, pd, snap)
    assert pu == pytest.approx(0.55)
    assert pd == pytest.approx(0.55)


def test_rule_22_oi_divergence_decline():
    """oi_change_4 < -0.05 -> pu*=0.9, pd*=0.9."""
    pu, pd = 0.50, 0.50
    snap = {"d": {"oi_change_4": -0.08}}
    pu, pd = _rule_22_oi_divergence(pu, pd, snap)
    assert pu == pytest.approx(0.45)
    assert pd == pytest.approx(0.45)


def test_rule_22_oi_divergence_no_change():
    """|oi_change_4| <= 0.05 -> no change."""
    pu, pd = 0.50, 0.50
    snap = {"d": {"oi_change_4": 0.03}}
    pu, pd = _rule_22_oi_divergence(pu, pd, snap)
    assert pu == 0.50
    assert pd == 0.50


def test_rule_22_oi_divergence_field_none():
    pu, pd = 0.50, 0.50
    snap = {"d": {"oi_change_4": None}}
    pu, pd = _rule_22_oi_divergence(pu, pd, snap)
    assert pu == 0.50
    assert pd == 0.50


# ---------------------------------------------------------------------------
# Rule 23: cvd_trend
# ---------------------------------------------------------------------------


def test_rule_23_cvd_trend_positive_slope():
    """cvd_slope > 0.2 -> pu+=.05."""
    pu, pd = 0.50, 0.50
    snap = {"d": {"cvd_slope": 0.5}}
    pu, pd = _rule_23_cvd_trend(pu, pd, snap)
    assert pu == pytest.approx(0.55)
    assert pd == 0.50


def test_rule_23_cvd_trend_negative_slope():
    """cvd_slope < -0.2 -> pd+=.05."""
    pu, pd = 0.50, 0.50
    snap = {"d": {"cvd_slope": -0.5}}
    pu, pd = _rule_23_cvd_trend(pu, pd, snap)
    assert pu == 0.50
    assert pd == pytest.approx(0.55)


def test_rule_23_cvd_trend_no_change():
    """|cvd_slope| <= 0.2 -> no change."""
    pu, pd = 0.50, 0.50
    snap = {"d": {"cvd_slope": 0.1}}
    pu, pd = _rule_23_cvd_trend(pu, pd, snap)
    assert pu == 0.50
    assert pd == 0.50


def test_rule_23_cvd_trend_field_none():
    pu, pd = 0.50, 0.50
    snap = {"d": {"cvd_slope": None}}
    pu, pd = _rule_23_cvd_trend(pu, pd, snap)
    assert pu == 0.50
    assert pd == 0.50


# ---------------------------------------------------------------------------
# Rule 24: taker_imbalance
# ---------------------------------------------------------------------------


def test_rule_24_taker_imbalance_high_ratio():
    """buy_sell_ratio > 1.2 -> pu+=.03."""
    pu, pd = 0.50, 0.50
    snap = {"d": {"buy_sell_ratio": 1.5}}
    pu, pd = _rule_24_taker_imbalance(pu, pd, snap)
    assert pu == pytest.approx(0.53)
    assert pd == 0.50


def test_rule_24_taker_imbalance_low_ratio():
    """buy_sell_ratio < 0.8 -> pd+=.03."""
    pu, pd = 0.50, 0.50
    snap = {"d": {"buy_sell_ratio": 0.6}}
    pu, pd = _rule_24_taker_imbalance(pu, pd, snap)
    assert pu == 0.50
    assert pd == pytest.approx(0.53)


def test_rule_24_taker_imbalance_no_change():
    """0.8 <= buy_sell_ratio <= 1.2 -> no change."""
    pu, pd = 0.50, 0.50
    snap = {"d": {"buy_sell_ratio": 1.0}}
    pu, pd = _rule_24_taker_imbalance(pu, pd, snap)
    assert pu == 0.50
    assert pd == 0.50


def test_rule_24_taker_imbalance_field_none():
    pu, pd = 0.50, 0.50
    snap = {"d": {"buy_sell_ratio": None}}
    pu, pd = _rule_24_taker_imbalance(pu, pd, snap)
    assert pu == 0.50
    assert pd == 0.50


# ---------------------------------------------------------------------------
# Combined: rules skip when d block missing or field None
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rule_fn", [
    _rule_21_funding_extreme,
    _rule_22_oi_divergence,
    _rule_23_cvd_trend,
    _rule_24_taker_imbalance,
])
def test_rules_skip_when_d_block_missing(rule_fn):
    """Любое правило не меняет pu/pd, если d блок отсутствует."""
    pu, pd = 0.50, 0.50
    pu_out, pd_out = rule_fn(pu, pd, {})
    assert pu_out == 0.50
    assert pd_out == 0.50


@pytest.mark.parametrize("rule_fn, field, high_val, low_val", [
    (_rule_21_funding_extreme, "funding_zscore", 3.0, -3.0),
    (_rule_22_oi_divergence, "oi_change_4", 0.10, -0.10),
    (_rule_23_cvd_trend, "cvd_slope", 0.5, -0.5),
    (_rule_24_taker_imbalance, "buy_sell_ratio", 1.5, 0.6),
])
def test_rules_skip_when_field_none(rule_fn, field, high_val, low_val):
    """Правило не меняет pu/pd, если поле в d блоке равно None."""
    pu, pd_base = 0.50, 0.50
    snap = {"d": {field: None}}
    pu_out, pd_out = rule_fn(pu, pd_base, snap)
    assert pu_out == pu
    assert pd_out == pd_base