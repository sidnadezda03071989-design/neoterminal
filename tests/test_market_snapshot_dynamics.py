"""Тесты динамики метрик Market Snapshot (Charon, Фаза 1/3).

Проверяем разделение ответственности «Python считает — LLM читает»:
  - helper-функции _slope/_delta/_percentile_rank (краевые случаи);
  - новые поля динамики в raw-блоках (technicals/trend/momentum/divergence):
    *_delta, *_slope, *_pct50, price_slope, rsi_price_div;
  - все новые поля доходят до compact_snapshot (иначе LLM их не видит);
  - кросс-метрика rsi_price_div (RSI падает при растущей цене -> +1 bearish);
  - правило 17 (trend mortality) на готовых числах снимка режет pu;
  - текст правил 17-20 и шкал присутствует в config/charon_prompt.txt.

Герметично: ни сети, ни БД (get_series_df/get_replay_df и внешние блоки
подменяются; _mtf_compact -> {}).
"""

import numpy as np
import pandas as pd
import pytest

from app_pkg import config
from app_pkg.data import market_snapshot as ms

_END = 1700000000


def _df(close, step=3600):
    """Синтетический OHLCV-DF с таймстампами до _END включительно."""
    close = np.asarray(close, dtype=float)
    n = len(close)
    ts = pd.to_datetime([_END - i * step for i in range(n)][::-1],
                        unit="s", utc=True)
    return pd.DataFrame({
        "timestamp": ts,
        "open": close - 0.05,
        "high": close + 0.6,
        "low": close - 0.6,
        "close": close,
        "volume": np.full(n, 10.0),
    })


def _dying_trend_df():
    """Сильный аптренд, затем пила: ADX падает (наклон < 0)."""
    up = 100.0 + 1.0 * np.arange(250)
    alt = up[-1] + 0.8 * ((np.arange(8) % 2) * 2 - 1)
    return _df(np.concatenate([up, alt]))


def _bearish_div_df():
    """Цена растёт (затухающими шагами), RSI падает -> bearish divergence."""
    i = np.arange(120)
    return _df(100.0 + 4.0 * np.sqrt(i) + 0.4 * np.sin(i / 1.5))


@pytest.fixture()
def hermetic(monkeypatch):
    """Снимок герметичен: ни БД, ни сети."""
    monkeypatch.setattr(ms, "get_series_df", lambda *a, **k: _dying_trend_df())
    monkeypatch.setattr(ms, "get_replay_df", lambda *a, **k: _dying_trend_df())
    monkeypatch.setattr(ms.db, "db_get_best_scan_stats", lambda s, tf: None)
    monkeypatch.setattr(ms, "get_crowd_snapshot", lambda s: {})
    monkeypatch.setattr(ms, "get_macro_snapshot", lambda s: None)
    monkeypatch.setattr(ms, "get_econ_calendar", lambda c, *a, **k: None)
    monkeypatch.setattr(ms, "get_derivatives_snapshot", lambda s: {})
    monkeypatch.setattr(ms, "get_micro_snapshot", lambda s: {})
    monkeypatch.setattr(ms, "get_news_sentiment", lambda s, **k: None)
    monkeypatch.setattr(ms, "_mtf_compact", lambda *a, **k: {})


# ----------------------------------------------------------- helper: _slope
def test_slope_linear_series():
    assert ms._slope(list(range(1, 11)), 10) == pytest.approx(1.0)


def test_slope_flat_series():
    assert ms._slope([5.0, 5.0, 5.0, 5.0, 5.0], 5) == 0.0


def test_slope_short_series():
    assert ms._slope([1.0, 2.0, 3.0, 4.0, 5.0], 10) is None


# ----------------------------------------------------------- helper: _delta
def test_delta_basic():
    assert ms._delta([1.0, 2.0, 3.0]) == 1.0
    assert ms._delta([3.0]) is None


# -------------------------------------------------- helper: _percentile_rank
def test_percentile_rank_median():
    series = [float(x) for x in range(1, 51)]      # 1..50
    assert ms._percentile_rank(25.5, series) == pytest.approx(0.5)


def test_percentile_rank_min():
    series = [float(x) for x in range(1, 51)]
    assert ms._percentile_rank(1.0, series) == 0.0


# ---------------------------------------------------- динамика в raw-снимке
def test_adx_slope_in_snapshot(hermetic):
    """adx_slope присутствует в trend и отрицателен на падающем ADX."""
    raw = ms.get_raw_market_data("BTCUSDT", "1h")
    trend = raw["trend"]
    assert "adx_slope" in trend
    assert trend["adx_slope"] is not None
    assert trend["adx_slope"] < -0.5          # тренд умирает
    assert trend["adx_pct50"] is not None
    for key in ("adx_delta", "adx_pct50", "adx_max_50"):
        assert key in trend, key
    # ADX был сильным: пик за 50 баров > 30
    assert trend["adx_max_50"] is not None and trend["adx_max_50"] > 30


def test_compact_preserves_dynamics(hermetic):
    """Все новые поля динамики доходят до compact_snapshot (иначе LLM слеп)."""
    snap = ms.compact_snapshot("BTCUSDT", "1h")
    for key in ("rsi_delta", "rsi_slope", "rsi_pct50", "atr_delta", "atr_slope",
                "bb_pct_delta", "bb_pct_slope"):
        assert key in snap["t"], key
    for key in ("adx_delta", "adx_slope", "adx_pct50", "adx_max_50"):
        assert key in snap["tr"], key
    for key in ("rsi_slope_short", "hist_delta", "macd_cross"):
        assert key in snap["mo"], key
    for key in ("price_slope", "rsi_price_div"):
        assert key in snap["div"], key
    # macd_cross всегда валиден: -1/0/+1
    assert snap["mo"]["macd_cross"] in (-1, 0, 1)
    assert snap["div"]["rsi_price_div"] in (-1, 0, 1)


def test_rsi_price_div_bearish(monkeypatch):
    """RSI падает при растущей цене -> rsi_price_div == +1 (bearish)."""
    monkeypatch.setattr(ms, "get_series_df", lambda *a, **k: _bearish_div_df())
    d = ms._divergence_block(_bearish_div_df())
    assert d["price_slope"] is not None and d["price_slope"] > 0
    assert d["rsi_price_div"] == 1


# ------------------------------------------------- правило 17 на числах снимка
def _rule17(tr):
    """Правило 17 (trend mortality, 3 условия) -> (d_pu, d_pd).

    LLM НЕ считает: adx_max_50/adx/adx_slope уже посчитаны Python, правило
    лишь применяет пороги к готовым числам снимка.
    """
    if (tr["adx_max_50"] > 30 and tr["adx_slope"] < -0.5
            and tr["adx"] > 20):
        return -0.10, +0.10
    return 0.0, 0.0


def test_rule_17_trend_mortality_reduces_pu():
    """Был сильный тренд (пик >30), ADX падает, ещё жив (>20) -> режем pu."""
    tr = {"adx_max_50": 45.0, "adx": 26.0, "adx_slope": -1.7}
    assert _rule17(tr) == (pytest.approx(-0.10), pytest.approx(0.10))


def test_rule_17_no_fire_without_strong_peak():
    """Слабый тренд (пик <=30) -> правило 17 молчит."""
    tr = {"adx_max_50": 28.0, "adx": 26.0, "adx_slope": -1.7}
    assert _rule17(tr) == (0.0, 0.0)


def test_rule_17_no_fire_when_adx_already_dead():
    """Тренд уже сдох (adx <= 20) -> правило 17 молчит."""
    tr = {"adx_max_50": 45.0, "adx": 18.0, "adx_slope": -1.7}
    assert _rule17(tr) == (0.0, 0.0)


def test_prompt_contains_dynamics_rules():
    """Правила 17-20 и новые шкалы есть в config/charon_prompt.txt."""
    text = config.CHARON_PROMPT_FILE.read_text(encoding="utf-8")
    assert "17" in text and "trend mortality" in text
    assert "18" in text and "momentum acceleration" in text
    assert "19" in text and "RSI-price divergence" in text
    assert "20" in text and "volatility contraction" in text
    for field in ("adx_slope", "adx_pct50", "adx_max_50", "rsi_delta",
                  "rsi_price_div", "bb_pct_slope", "atr_slope", "macd_cross",
                  "pct50"):
        assert field in text, field
