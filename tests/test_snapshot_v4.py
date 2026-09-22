# -*- coding: utf-8 -*-
"""Тесты v4 Market Snapshot: mtf-тренд по EMA, дивергенции, свечные паттерны.

Проверяем (Уровень 4 — локальные расчёты по OHLCV, без сети/БД):
  - _mtf_compact: trend up/down/flat по ЧЁТКОМУ определению ema20/50
    (мёртвая зона 0.1% + подтверждение close>ema20) — едино для всех ТФ;
  - _divergence_signal/_divergence_combined: bear/bull/none по двум последним
    экстремумам (запись вызовов и краевые случаи);
  - _divergence_block: поля в {-1,0,1}, ok:true на синтетике, null-схема;
  - _candle_block: body/фитили/engulfing/pinbar по последней свече;
  - компакт: div/cnd/mcr/ebc появляются в снимке, missing = no data.
"""

import numpy as np
import pandas as pd
import pytest

from app_pkg.data import market_snapshot as ms

_END = int(pd.Timestamp("2026-02-20 22:00:00", tz="UTC").timestamp())


def _df(close, open_=None, high=None, low=None, volume=None, n=90, step=900):
    """Синтетический OHLCV-DF (n баров, таймстампы до _END)."""
    close = np.asarray(close, dtype=float)
    if high is None:
        high = close + 0.5
    if low is None:
        low = close - 0.5
    if open_ is None:
        open_ = close - 0.1
    if volume is None:
        volume = np.full(len(close), 10.0)
    ts = pd.to_datetime([_END - i * step for i in range(len(close))][::-1],
                        unit="s", utc=True)
    return pd.DataFrame({
        "timestamp": ts, "open": np.asarray(open_, dtype=float),
        "high": np.asarray(high, dtype=float), "low": np.asarray(low, dtype=float),
        "close": close, "volume": np.asarray(volume, dtype=float),
    })


# ------------------------------------------------------------- mtf тренд (EMA)
def _mtf_series(monkeypatch, close):
    df = _df(close, n=len(close))
    monkeypatch.setattr(ms, "get_series_df", lambda *a, **k: df)
    return ms._mtf_compact("BTCUSDT", None)["1h"]


def test_mtf_trend_up_uses_ema_definition(monkeypatch):
    close = 100.0 + np.arange(80) * 0.25
    tf = _mtf_series(monkeypatch, close)
    assert tf["tr"] == 1 and tf["trend"] == "up"


def test_mtf_trend_down(monkeypatch):
    close = 100.0 - np.arange(80) * 0.25
    tf = _mtf_series(monkeypatch, close)
    assert tf["tr"] == -1 and tf["trend"] == "down"


def test_mtf_trend_flat_constant(monkeypatch):
    tf = _mtf_series(monkeypatch, np.full(80, 100.0))
    assert tf["tr"] == 0 and tf["trend"] == "flat"


def test_mtf_trend_flat_dead_zone(monkeypatch):
    """Мёртвая зона ±0.1%: пологий рост, где close>ema20, но ema20<ema50*1.001."""
    close = 100.0 + 0.0005 * np.arange(80)
    tf = _mtf_series(monkeypatch, close)
    assert tf["trend"] == "flat"


def test_mtf_trend_flat_no_confirmation(monkeypatch):
    """ema20>ema50*1.001, но close не выше ema20 — это не up."""
    n = 80
    close = 100.0 + np.arange(n) * 0.3
    close[-1] = close[-1] - 30.0  # откат в конце ниже ema20
    tf = _mtf_series(monkeypatch, close)
    assert tf["trend"] == "flat"


# --------------------------------------------------------- дивергенции (чистые)
def test_divergence_signal_bullish():
    """Цена бьёт минимум, индикатор нет -> +1 (на дне дива, на вершинах нет)."""
    c = np.array([100, 99, 98, 97, 96, 95, 94, 95, 96, 97, 96, 93, 94, 95,
                  96, 97, 96, 95, 94, 93])
    v = np.array([50, 49, 48, 47, 46, 45, 44, 45, 46, 47, 46, 44.2, 44.8,
                  45.2, 45.8, 46.4, 46.8, 46.5, 46.1, 45.6])
    # Экстремумы-минимумы индикатора: инд.6 (44) и инд.11 (44.2); цена при
    # этом 94 -> 93 (новый минимум) = бычья дивергенция.
    assert ms._divergence_signal(c, v, "low") == 1
    # Вершинная структура без дивы: цена 97 -> 96 (не новый high) -> high=0.
    assert ms._divergence_signal(c, v, "high") == 0


def test_divergence_signal_bearish():
    """Цена бьёт максимум, индикатор нет -> -1."""
    c = np.array([100, 101, 102, 103, 104, 105, 106, 105, 104, 103, 102, 107,
                  106, 105, 104, 103, 102, 101, 100, 99])
    v = np.array([20, 21, 22, 23, 24, 25, 26, 25, 24, 23, 22, 25.6, 25.2,
                  24.8, 24.4, 24, 23.6, 23.2, 22.8, 22.4])
    assert ms._divergence_signal(c, v, "high") == -1
    assert ms._divergence_signal(c, v, "low") == 0


def test_divergence_signal_none_when_indicator_confirms():
    """Индикатор тоже бьёт экстремум -> нет дивергенции (0)."""
    c = np.array([100, 99, 98, 97, 96, 95, 94, 95, 96, 97, 96, 93, 94, 95,
                  96, 97, 96, 95, 94, 93])
    v = np.array([50, 49, 48, 47, 46, 45, 44, 45, 46, 47, 46, 43.9, 44.1,
                  44.3, 44.5, 44.7, 44.9, 44.8, 44.7, 44.6])
    # Минимумы: 44 (6) -> 43.9 (11); цена 94 -> 93 (новый low) — индикатор
    # тоже... нет, ниже: dip подтверждён, бычья дива отсутствует.
    assert ms._divergence_signal(c, v, "low") == 0
    assert ms._divergence_signal(c, v, "high") == 0  # цена вершины не растёт


def test_divergence_signal_too_short():
    assert ms._divergence_signal([100, 99], [50, 49], "low") == 0
    assert ms._divergence_combined([100, 99, 98], [50, 49, 48]) == 0


def test_divergence_combined_prefers_bull():
    c = np.array([100, 99, 98, 97, 96, 95, 94, 95, 96, 97, 98, 93, 94, 95,
                  96, 97, 98, 97, 96, 95])
    v = np.array([50, 49, 48, 47, 46, 45, 44, 45, 46, 47, 46, 44.2, 44.8,
                  45.2, 45.8, 46.4, 46.8, 46.5, 46.1, 45.6])
    assert ms._divergence_combined(c, v) == 1


# ------------------------------------------------------------ дивергенции block
def test_divergence_block_fields_in_scope(monkeypatch):
    df = _df(100.0 + 0.02 * np.arange(90) + 0.4 * np.sin(np.arange(90) / 6.0))
    d = ms._divergence_block(df)
    assert d["ok"] is True
    for key in ("rsi_price", "macd_price", "obv_price"):
        assert d[key] in (-1, 0, 1), key


def test_divergence_block_plumbing(monkeypatch):
    """Значения пропускаются в блок как есть (обратная совместимость контракта)."""
    seq = iter([1, -1, 0])

    def fake_combined(closes, ind):
        return next(seq)

    monkeypatch.setattr(ms, "_divergence_combined", fake_combined)
    d = ms._divergence_block(_df(100.0 + np.arange(90)))
    assert d == {"rsi_price": 1, "macd_price": -1, "obv_price": 0, "ok": True}


def test_divergence_block_few_bars():
    assert ms._divergence_block(_df(np.full(20, 100.0)))["ok"] is False
    assert ms._divergence_block(pd.DataFrame())["ok"] is False


# ----------------------------------------------------------------- свечные блок
def test_candle_block_bull_engulfing():
    o = [100.0] * 40 + [102.0, 99.0]
    c = [100.0] * 40 + [101.0, 103.0]
    h = [101.5] * 40 + [103.5, 104.0]
    l = [98.5] * 40 + [98.0, 98.5]
    df = _df(c, open_=o, high=h, low=l)
    out = ms._candle_block(df)
    assert out["ok"] is True
    rng = h[-1] - l[-1]
    assert out["body_ratio"] == pytest.approx((c[-1] - o[-1]) / rng, abs=1e-3)
    assert out["upper_wick_ratio"] == pytest.approx(
        (h[-1] - max(o[-1], c[-1])) / rng, abs=1e-3)
    assert out["lower_wick_ratio"] == pytest.approx(
        (min(o[-1], c[-1]) - l[-1]) / rng, abs=1e-3)
    assert out["engulfing"] == 1  # последний быкует и поглощает прошлый плюс
    assert out["pinbar"] in (-1, 0, 1)


def test_candle_block_bear_engulfing():
    o = [100.0] * 40 + [99.0, 104.0]
    c = [100.0] * 40 + [103.0, 98.0]
    h = [101.0] * 40 + [102.0, 105.0]
    l = [99.0] * 40 + [98.0, 97.0]
    out = ms._candle_block(_df(c, open_=o, high=h, low=l))
    assert out["engulfing"] == -1


def test_candle_block_pinbar_bullish():
    """Длинный нижний фитиль >= 2× тело и >= верхнего фитиля -> +1."""
    o = [100.0] * 40 + [100.0]
    c = [100.0] * 40 + [100.2]
    h = [101.0] * 40 + [100.5]
    l = [99.0] * 40 + [97.0]   # фитиль 3.0 ниже тела
    out = ms._candle_block(_df(c, open_=o, high=h, low=l))
    assert out["pinbar"] == 1


def test_candle_block_doji_ratios_none():
    """Нулевой диапазон свечи -> доли None, паттерны 0 (без выдумок)."""
    o = [100.0] * 45
    c = [100.0] * 45
    h = [100.0] * 45
    l = [100.0] * 45
    out = ms._candle_block(_df(c, open_=o, high=h, low=l))
    assert out["ok"] is True
    assert out["body_ratio"] is None
    assert out["upper_wick_ratio"] is None
    assert out["lower_wick_ratio"] is None
    assert out["engulfing"] == 0 and out["pinbar"] == 0


def test_candle_block_few_bars():
    assert ms._candle_block(_df([100.0]))["ok"] is False
    assert ms._candle_block(pd.DataFrame())["ok"] is False


# ------------------------------------------------------------------ компакт
def test_compact_emits_div_cnd_mcr_ebc(monkeypatch):
    raw = {
        "divergence": {"rsi_price": 1, "macd_price": -1, "obv_price": 0,
                       "ok": True},
        "candle": {"body_ratio": 0.722, "upper_wick_ratio": 0.151,
                   "lower_wick_ratio": 0.127, "engulfing": 1, "pinbar": 0,
                   "ok": True},
        "micro": {"spread_norm": 0.0004, "ob_imb_10": 0.18, "bid_sum_10": 124000,
                  "ask_sum_10": 98000, "large_trades_ratio": 0.22, "ok": True},
        "context": {"corr_btc_30": None, "eth_btc_corr_30": 0.86, "ok": True},
    }
    monkeypatch.setattr(ms, "get_raw_market_data", lambda *a, **k: raw)
    monkeypatch.setattr(ms, "_mtf_compact", lambda *a, **k: {})
    snap = ms.compact_snapshot("BTCUSDT", "1H")
    assert snap["div"] == {"rsi": 1, "macd": -1, "obv": 0}
    assert snap["cnd"] == {"br": 0.722, "uw": 0.151, "lw": 0.127, "eng": 1,
                           "pin": 0}
    assert snap["mcr"] == {"sp": 0.0004, "obi": 0.18, "bs": 124000,
                           "as": 98000, "ltr": 0.22}
    assert snap["cg"] == {"ebc": 0.86}          # null-поля контекста пропали
    assert "ok" not in str(snap)


def test_compact_omits_empty_div_cnd_mcr(monkeypatch):
    raw = {
        "divergence": {"rsi_price": None, "macd_price": None, "obv_price": None,
                       "ok": True},
        "candle": {"body_ratio": None, "upper_wick_ratio": None,
                   "lower_wick_ratio": None, "engulfing": None, "pinbar": None,
                   "ok": False},
        "micro": {"spread_norm": None, "ob_imb_10": None, "bid_sum_10": None,
                  "ask_sum_10": None, "large_trades_ratio": None, "ok": False},
    }
    monkeypatch.setattr(ms, "get_raw_market_data", lambda *a, **k: raw)
    monkeypatch.setattr(ms, "_mtf_compact", lambda *a, **k: {})
    snap = ms.compact_snapshot("BTCUSDT", "1H")
    assert "div" not in snap and "cnd" not in snap and "mcr" not in snap