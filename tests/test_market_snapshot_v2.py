"""Тесты структурных блоков v2 Market Snapshot (app_pkg/data/market_snapshot.py).

Проверяем (Уровень 4 — локальные расчёты по OHLCV, без сети/БД):
  - _volume_block: rel_vol/obv_slope/vol_zscore/vol_percentile/vwap_dev;
  - _trend_block:  adx/plus_di/minus_di/di_spread/ema20_50_ratio/reg_slope_20;
  - _momentum_block: macd/macd_signal/macd_hist/macd_hist_slope/rsi_slope;
  - _volatility_block: bb_width/bb_width_pct/hv20/atr_change_pct/atr_ratio;
  - _regime_block: hurst/autocorr_lag1/efficiency_ratio;
  - _context_block: corr_btc_30 (крипта) и null-заглушки индексов;
  - мало баров / пустой df -> null-схема с ok:false, без exception;
  - унификация масштабов: fear_greed 0-100 -> 0-1, *_pct в долях, остальные
    доли (long_pct/winrate/max_dd) НЕ трогаются; кеш crowd не мутируется;
  - compact_snapshot: новые блоки v/tr/mo/vl/rg (cg исчезает без данных),
    s.fng — дробь 0-1; весь снимок — валидный JSON.
"""

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from app_pkg.data import market_snapshot as ms

_END = int(datetime(2026, 2, 20, 22, 0, tzinfo=timezone.utc).timestamp())


def _df(close, volume=None, end_ts=_END, step=900, high=None, low=None):
    """Синтетический OHLCV-DF (таймстампы до end_ts включительно)."""
    close = np.asarray(close, dtype=float)
    n = len(close)
    ts = pd.to_datetime([end_ts - i * step for i in range(n)][::-1],
                        unit="s", utc=True)
    if volume is None:
        volume = np.full(n, 10.0)
    volume = np.asarray(volume, dtype=float)
    if high is None:
        high = close + 0.5
    if low is None:
        low = close - 0.5
    return pd.DataFrame({
        "timestamp": ts,
        "open": close - 0.1,
        "high": np.asarray(high, dtype=float),
        "low": np.asarray(low, dtype=float),
        "close": close,
        "volume": volume,
    })


def _synthetic_df():
    """300 баров с мягким трендом и растущим объёмом — все окна покрыты."""
    n = 300
    rng = np.arange(n)
    close = 100.0 + 0.02 * rng + 0.4 * np.sin(rng / 6.0)
    return _df(close, volume=10.0 + 0.06 * rng)


def _structural_raw():
    """Raw-блоки v/tr/mo/vl/rg с данными + context с null-заглушками."""
    def b(**kw):
        return {**kw, "ok": True}

    return {
        "volume": b(rel_vol=1.2, vwap_dev=-0.3),
        "trend": b(adx=28.4, plus_di=30.0, minus_di=10.0, di_spread=20.0,
                   ema20_50_ratio=1.125, reg_slope_20=0.25),
        "momentum": b(macd=1.2, macd_signal=0.9, macd_hist=0.3,
                      macd_hist_slope=0.05, rsi_slope=2.0),
        "volatility": b(bb_width=0.125, bb_width_pct=0.9, hv20=41.2,
                        atr_change_pct=3.4, atr_ratio_short_long=1.1),
        "regime": b(hurst=0.62, autocorr_lag1=-0.1, efficiency_ratio=0.7),
        "context": b(corr_btc_30=None, btc_dominance=None, corr_dxy_30=None,
                     beta_index=None, index_return_1d=None, index_rsi=None),
    }


# --------------------------------------------------------------- volume
def test_volume_block_trending_volume():
    close = np.linspace(100.0, 100.0 + 149 * 0.01, 150)
    volume = np.linspace(1.0, 150.0, 150)
    v = ms._volume_block(_df(close, volume))
    assert v["ok"] is True
    assert v["rel_vol"] == pytest.approx(150.0 / 140.5, abs=0.02)
    assert v["vol_zscore"] > 1.0
    assert 0.8 <= v["vol_percentile"] <= 1.0
    assert v["obv_slope"] is not None and v["obv_slope"] > 0
    assert v["vwap_dev"] is not None and abs(v["vwap_dev"]) < 2.0


def test_volume_block_few_bars_null():
    v = ms._volume_block(_df(np.full(10, 100.0)))
    assert v["ok"] is False
    for key in ("rel_vol", "obv_slope", "vol_zscore", "vol_percentile",
                "vwap_dev"):
        assert v[key] is None


def test_volume_block_empty_df():
    assert ms._volume_block(pd.DataFrame())["ok"] is False


# ---------------------------------------------------------------- trend
def test_trend_block_uptrend():
    close = np.linspace(100.0, 100.0 + 119 * 0.5, 120)
    t = ms._trend_block(_df(close))
    assert t["ok"] is True
    assert t["plus_di"] is not None and t["minus_di"] is not None
    assert t["minus_di"] < t["plus_di"]
    assert t["di_spread"] > 0
    assert t["adx"] > 20
    assert t["ema20_50_ratio"] > 1.0
    assert t["reg_slope_20"] > 0


def test_trend_block_flat_market():
    t = ms._trend_block(_df(np.full(120, 100.0)))
    assert t["ok"] is True
    assert t["ema20_50_ratio"] == 1.0
    assert t["reg_slope_20"] == 0.0
    assert t["adx"] is None  # ATR=0 -> ADX не определён, честный null


def test_trend_block_few_bars():
    assert ms._trend_block(_df(np.linspace(100.0, 150.0, 30)))["ok"] is False


# ------------------------------------------------------------- momentum
def test_momentum_block_present():
    m = ms._momentum_block(_df(np.linspace(100.0, 130.0, 60)))
    assert m["ok"] is True
    for key in ("macd", "macd_signal", "macd_hist", "macd_hist_slope",
                "rsi_slope"):
        assert m[key] is not None


def test_momentum_block_few_bars():
    assert ms._momentum_block(_df(np.linspace(100.0, 110.0, 20)))["ok"] is False


# ----------------------------------------------------------- volatility
def test_volatility_block_growing_vol():
    n = 150
    i = np.arange(n)
    close = 100.0 + 0.002 * i ** 2            # ускорение -> растущий std
    amp = 0.5 + 0.5 * i                        # растущий диапазон -> растущий ATR
    v = ms._volatility_block(_df(close, high=close + amp, low=close - amp))
    assert v["ok"] is True
    assert v["bb_width"] is not None and v["bb_width"] > 0
    assert v["bb_width_pct"] is not None and v["bb_width_pct"] > 0.7
    assert v["hv20"] is not None and v["hv20"] > 0
    assert v["atr_change_pct"] is not None
    ratio = v["atr_ratio_short_long"]
    assert ratio is not None and ratio > 1.0


def test_volatility_block_flat():
    v = ms._volatility_block(_df(np.full(120, 100.0)))
    assert v["ok"] is True
    assert v["bb_width"] == 0.0
    assert v["bb_width_pct"] == 0.5
    assert v["hv20"] == 0.0
    assert v["atr_change_pct"] == 0.0
    assert v["atr_ratio_short_long"] == 1.0


# ---------------------------------------------------------------- regime
def test_regime_block_trending():
    close = 100.0 + np.arange(150) * 0.3
    r = ms._regime_block(_df(close))
    assert r["ok"] is True
    assert r["hurst"] is not None and r["hurst"] > 0.5
    assert r["efficiency_ratio"] is not None and r["efficiency_ratio"] > 0.5


def test_regime_block_mean_reverting():
    n = 150
    base = 100.0 + 0.5 * np.arange(n)
    wave = 2.0 * ((np.arange(n) % 2) * 2 - 1)
    close = base + wave
    r = ms._regime_block(_df(close))
    assert r["ok"] is True
    assert r["efficiency_ratio"] is not None and r["efficiency_ratio"] < 0.3
    assert r["autocorr_lag1"] is not None and r["autocorr_lag1"] < 0


def test_regime_block_few_bars():
    assert ms._regime_block(_df(np.linspace(100.0, 120.0, 30)))["ok"] is False


# --------------------------------------------------------------- context
def test_context_block_crypto_correlation(monkeypatch):
    n = 60
    base = 100.0 + np.arange(n) * 0.05 + 0.5 * np.sin(np.arange(n) / 3.0)
    btc_df = _df(base)
    eth_df = _df(2.0 * base + 5.0)
    def fake_slice(sym, tf, *a, **k):
        if sym not in ("BTCUSDT", "ETHUSDT"):
            pytest.fail(f"не ждали _slice_df({sym})")
        return {"BTCUSDT": btc_df, "ETHUSDT": eth_df}[sym]

    monkeypatch.setattr(ms, "_slice_df", fake_slice)
    c = ms._context_block(eth_df, "ETHUSDT", "1H")
    assert c["ok"] is True
    assert c["corr_btc_30"] == pytest.approx(1.0, abs=1e-3)
    assert c["eth_btc_corr_30"] == pytest.approx(1.0, abs=1e-3)
    assert c["btc_dominance"] is None
    assert c["corr_dxy_30"] is None and c["beta_index"] is None


def test_context_block_btc_no_self_correlation(monkeypatch):
    n = 60
    rng = np.arange(n)
    btc_df = _df(100.0 + 0.5 * rng)
    eth_df = _df(200.0 + 1.0 * rng)
    monkeypatch.setattr(
        ms, "_slice_df",
        lambda sym, tf, *a, **k: btc_df if sym == "BTCUSDT" else eth_df)
    c = ms._context_block(_df(100.0 + 0.5 * rng), "BTCUSDT", "1H")
    assert c["ok"] is True
    assert c["corr_btc_30"] is None  # к себе пары корреляцию не считает
    assert c["eth_btc_corr_30"] == pytest.approx(1.0, abs=1e-3)
    assert c["btc_dominance"] is None


def test_context_block_eth_btc_corr_skipped_in_replay(monkeypatch):
    """Эфир-биток-корреляция считается только в live (реплей бережёт сеть)."""
    n = 60
    rng = np.arange(n)
    btc_df = _df(100.0 + 0.5 * rng)
    eth_df = _df(200.0 + 1.0 * rng)
    called = []
    monkeypatch.setattr(
        ms, "_slice_df",
        lambda sym, tf, *a, **k: (called.append(sym) or
                                  (btc_df if sym == "BTCUSDT" else eth_df)))
    c = ms._context_block(eth_df, "ETHUSDT", "1H", upto_sec=_END)
    assert c["ok"] is True
    assert c["eth_btc_corr_30"] is None
    assert called == ["BTCUSDT"]  # только corr_btc_30, без ETH-досрезки


def test_context_block_forex_stubs():
    c = ms._context_block(_df(np.linspace(1.0, 1.2, 60)), "EURUSD", "1H")
    assert c["ok"] is True
    for key in ("corr_dxy_30", "beta_index", "index_return_1d", "index_rsi",
                "eth_btc_corr_30"):
        assert c[key] is None


def test_context_block_fetch_error_is_null(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no btc feed")

    monkeypatch.setattr(ms, "_slice_df", boom)
    c = ms._context_block(_df(np.linspace(100.0, 150.0, 60)), "ETHUSDT", "1H")
    assert c["ok"] is True
    assert c["corr_btc_30"] is None


def test_context_block_empty_df():
    c = ms._context_block(pd.DataFrame(), "BTCUSDT", "1H")
    assert c["ok"] is False
    for key, val in c.items():
        if key != "ok":
            assert val is None


# ------------------------------------------------ шкала 0-1 (унификация)
def test_new_pct_fields_in_fraction_scale():
    close = np.linspace(100.0, 100.0 + 149 * 0.01, 150)
    v = ms._volume_block(_df(close, np.linspace(1.0, 150.0, 150)))
    assert 0.0 <= v["vol_percentile"] <= 1.0

    n = 150
    i = np.arange(n)
    close2 = 100.0 + 0.002 * i ** 2
    amp = 0.5 + 0.5 * i
    vl = ms._volatility_block(_df(close2, high=close2 + amp, low=close2 - amp))
    assert 0.0 <= vl["bb_width_pct"] <= 1.0


def test_scanner_edge_fraction_fields_untouched(monkeypatch):
    fake = {"train": {}, "test": {"winrate": 0.606, "max_dd": 0.0076},
            "params": {}}
    monkeypatch.setattr(ms.db, "db_get_best_scan_stats", lambda s, tf: fake)
    se = ms._scanner_edge("BTCUSDT", "15m")
    assert se["winrate"] == 0.606
    assert se["max_dd"] == 0.0076


# ------------------------------------------------------- get_raw_market_data
@pytest.fixture()
def hermetic(monkeypatch):
    """Снимок герметичен: ни БД, ни сети."""
    monkeypatch.setattr(ms, "get_series_df", lambda *a, **k: _synthetic_df())
    monkeypatch.setattr(ms, "get_replay_df", lambda *a, **k: _synthetic_df())
    monkeypatch.setattr(ms.db, "db_get_best_scan_stats", lambda s, tf: None)
    monkeypatch.setattr(ms, "get_crowd_snapshot", lambda s: {})
    monkeypatch.setattr(ms, "get_macro_snapshot", lambda s: None)
    monkeypatch.setattr(ms, "get_econ_calendar", lambda c, *a, **k: None)
    monkeypatch.setattr(ms, "get_derivatives_snapshot", lambda s: {})
    monkeypatch.setattr(ms, "get_micro_snapshot", lambda s: {})
    monkeypatch.setattr(ms, "get_news_sentiment", lambda s, **k: None)


def test_raw_contains_all_structural_blocks(hermetic):
    raw = ms.get_raw_market_data("BTCUSDT", "15m")
    for name in ("volume", "trend", "momentum", "volatility", "regime",
                 "context"):
        assert name in raw
        assert isinstance(raw[name], dict)
        assert "ok" in raw[name]
    assert raw["technicals"]["ok"] is True
    dumped = json.dumps(raw)
    assert json.loads(dumped) == raw  # NaN/Inf из снимка исключены


def test_raw_new_blocks_null_on_empty_df(monkeypatch):
    monkeypatch.setattr(ms, "get_series_df", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(ms, "get_replay_df", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(ms.db, "db_get_best_scan_stats", lambda s, tf: None)
    monkeypatch.setattr(ms, "get_crowd_snapshot", lambda s: {})
    monkeypatch.setattr(ms, "get_macro_snapshot", lambda s: None)
    monkeypatch.setattr(ms, "get_econ_calendar", lambda c, *a, **k: None)
    monkeypatch.setattr(ms, "get_derivatives_snapshot", lambda s: {})
    monkeypatch.setattr(ms, "get_micro_snapshot", lambda s: {})
    monkeypatch.setattr(ms, "get_news_sentiment", lambda s, **k: None)

    raw = ms.get_raw_market_data("EURUSD", "15m")
    for name in ("volume", "trend", "momentum", "volatility", "regime",
                 "context"):
        block = raw[name]
        assert block["ok"] is False, name
        for key, val in block.items():
            if key != "ok":
                assert val is None, (name, key)


def test_normalize_crowd_scale_taker_clip():
    """buySellRatio-экстремумы не сдвигают распределение: клип после r/(1+r)."""
    base = {"ls_ratio": 1.0, "long_pct": 0.5, "fear_greed": 55,
            "ok": True, "ts": 1}
    out = ms._normalize_crowd_scale({**base, "taker_buy_sell": 100.0})
    assert out["taker_buy_sell"] == 0.95          # 100 -> 0.9901 -> clip
    out = ms._normalize_crowd_scale({**base, "taker_buy_sell": 0.01})
    assert out["taker_buy_sell"] == 0.05          # 0.01 -> 0.0099 -> clip
    out = ms._normalize_crowd_scale({**base, "taker_buy_sell": 1.05})
    assert out["taker_buy_sell"] == pytest.approx(0.5122, abs=1e-4)
    sent = {**base, "taker_buy_sell": 100.0}
    ms._normalize_crowd_scale(sent)
    assert sent["taker_buy_sell"] == 100.0        # без мутации входа


def test_fear_greed_normalized_and_no_cache_mutation(hermetic, monkeypatch):
    crowd = {"ls_ratio": 2.2, "long_pct": 0.689, "short_pct": 0.311,
             "fear_greed": 52, "ok": True, "ts": 1}
    monkeypatch.setattr(ms, "get_crowd_snapshot", lambda s: crowd)
    r1 = ms.get_raw_market_data("BTCUSDT", "15m")
    r2 = ms.get_raw_market_data("BTCUSDT", "15m")
    assert r1["sentiment"]["fear_greed"] == 0.52
    assert r2["sentiment"]["fear_greed"] == 0.52
    assert crowd["fear_greed"] == 52  # кеш crowd не мутирован
    assert r1["sentiment"]["long_pct"] == 0.689  # доли не тронуты


# ------------------------------------------------------------- компакт
def test_compact_sentiment_fng_fraction(monkeypatch):
    raw = {"sentiment": {"ls_ratio": 2.2, "long_pct": 0.689,
                         "fear_greed": 0.52, "ok": True}}
    monkeypatch.setattr(ms, "get_raw_market_data", lambda *a, **k: raw)
    monkeypatch.setattr(ms, "_mtf_compact", lambda *a, **k: {})
    snap = ms.compact_snapshot("BTCUSDT", "1H")
    assert snap["s"]["fng"] == 0.52
    assert snap["s"]["long_pct"] == 0.7


def test_compact_includes_structural_blocks(monkeypatch):
    monkeypatch.setattr(ms, "get_raw_market_data",
                        lambda *a, **k: _structural_raw())
    monkeypatch.setattr(ms, "_mtf_compact", lambda *a, **k: {})
    snap = ms.compact_snapshot("BTCUSDT", "1H")
    assert snap["v"] == {"rv": 1.2, "vwd": -0.3}
    assert snap["tr"] == {"adx": 28.4, "pdi": 30.0, "mdi": 10.0, "ds": 20.0,
                          "er": 1.125, "rs": 0.25}
    assert snap["mo"] == {"macd": 1.2, "ms": 0.9, "mh": 0.3, "mhs": 0.05,
                          "rsi_s": 2.0}
    assert snap["vl"] == {"bw": 0.125, "bwp": 0.9, "hv": 41.2, "atc": 3.4,
                          "atr_r": 1.1}
    assert snap["rg"] == {"h": 0.62, "ac1": -0.1, "er": 0.7}
    assert "cg" not in snap  # контекст без данных исчезает (missing = no data)


def test_compact_structural_json_clean(monkeypatch):
    monkeypatch.setattr(ms, "get_raw_market_data",
                        lambda *a, **k: _structural_raw())
    monkeypatch.setattr(ms, "_mtf_compact", lambda *a, **k: {})
    snap = ms.compact_snapshot("BTCUSDT", "1H")
    assert json.loads(json.dumps(snap)) == snap
    assert "ok" not in json.dumps(snap)