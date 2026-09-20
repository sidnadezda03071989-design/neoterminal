# -*- coding: utf-8 -*-
"""Фиксы движка бэктеста/сканера (запрос пользователя):

  1. Нет заглядывания в будущее: выход по TP/SL проверяется со СЛЕДУЮЩЕГО
     бара после входа (вход по close — фитиль бара входа уже в прошлом);
  2. Минимальная величина сделки: |TP - SL| >= BACKTEST_MIN_TP_SL_CANDLES
     средних свечей графика (mean(high-low)) — иначе сделка не открывается;
  3. Сделки не открываются одна внутри другой: строго последовательно.

Сканер использует те же прогоны run_backtest (train/test окна df=...), поэтому
фиксы покрывают и скан.
"""
import numpy as np
import pandas as pd

from app_pkg import config
from app_pkg.ai import backtest as bt
from app_pkg.ai.backtest import run_backtest


def _make_df(n=1500, seed=7):
    """OHLCV случайного блуждания, как в test_backtest_metrics."""
    rng = np.random.default_rng(seed)
    ret = rng.normal(0, 0.002, n)
    close = 100 * np.exp(np.cumsum(ret))
    spread = np.abs(rng.normal(0, 0.003, n))
    high = close * (1 + spread)
    low = close * (1 - spread)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum.reduce([high, open_, close])
    low = np.minimum.reduce([low, open_, close])
    ts = pd.date_range("2025-01-01", periods=n, freq="15min")
    return pd.DataFrame({
        "timestamp": ts, "open": open_, "high": high, "low": low,
        "close": close, "volume": rng.uniform(100, 1000, n),
    })


def _run(df, **kw):
    return run_backtest("BTCUSDT", "15m", 0, 0, "sma_cross",
                        {"fast": 20, "slow": 50}, 10000, df=df, **kw)


# --------------------------------------------------- 1) заглядывание в будущее
def test_no_trade_exits_on_its_entry_bar(monkeypatch):
    """Выход по TP/SL не проверяется на баре ВХОДА: вход по close, фитиль
    этого бара уже в прошлом. Заставляем _exit_on_bar всегда возвращать
    касание — при корректном движке первое закрытие возможно ТОЛЬКО на
    следующем баре (exit_time == entry_time + STEP).
    """
    def fake_exit(position, high, low, open_=None):
        return position["entry"], "tp"

    monkeypatch.setattr(bt, "_exit_on_bar", fake_exit)
    r = _run(_make_df())
    assert "error" not in r
    assert r["total_trades"] > 0
    step = 900  # 15m в секундах
    for t in r["trades"]:
        # если бы выход проверялся на баре входа — эти времена были бы равны
        # (сделка «открылась и закрылась» в один бар из данных будущего)
        assert t["exit_time"] == t["entry_time"] + step


def test_no_trade_exits_before_or_on_its_entry():
    """Нет нулевых/обратных сделок: exit строго ПОСЛЕ entry во всех прогонах."""
    for dataset in ("full", "train", "test"):
        r = _run(_make_df(), dataset=dataset)
        assert "error" not in r, dataset
        for t in r["trades"]:
            assert t["exit_time"] > t["entry_time"], (dataset, t)


# ------------------------------------------------ 2) минимум TP↔SL (3 свечи)
def test_micro_trades_are_filtered_out(monkeypatch):
    """sl_atr=0.02 → TP↔SL короче трёх средних свечей → сделки НЕ открываются.
    Тот же df с выключенным фильтром даёт сделки — виноват именно фильтр."""
    df = _make_df()
    monkeypatch.setattr(config, "BACKTEST_SL_ATR", 0.02)
    monkeypatch.setattr(config, "BACKTEST_MIN_TP_SL_CANDLES", 3.0)
    r = _run(df)
    assert "error" not in r
    assert r["total_trades"] == 0

    monkeypatch.setattr(config, "BACKTEST_MIN_TP_SL_CANDLES", 0.0)
    r2 = _run(df)
    assert "error" not in r2
    assert r2["total_trades"] > 0


def test_every_trade_respects_min_tp_sl_distance(monkeypatch):
    """Все записанные сделки имеют |TP-SL| >= 3 × средняя свеча окна."""
    df = _make_df()
    avg_candle = float(np.nanmean(df["high"].to_numpy() - df["low"].to_numpy()))
    threshold = config.BACKTEST_MIN_TP_SL_CANDLES * avg_candle
    r = _run(df)
    assert "error" not in r
    assert r["total_trades"] > 0
    for t in r["trades"]:
        assert abs(t["tp_price"] - t["sl_price"]) >= threshold


def test_filter_applies_to_scan_windows(monkeypatch):
    """Сканер гонит train/test окна через тот же run_backtest (df=..., dataset):
    фильтр микро-сделок работает и в окнах скана."""
    df = _make_df()
    monkeypatch.setattr(config, "BACKTEST_SL_ATR", 0.02)
    for dataset in ("train", "test"):
        r = _run(df, dataset=dataset)
        assert "error" not in r, dataset
        assert r["total_trades"] == 0, dataset


# ------------------------------------------------- 3) сделки не пересекаются
def test_no_trades_on_same_bar_or_overlapping():
    """Строгая последовательность: entry_i+1 > exit_i (одна сделка не может
    открыться внутри другой ни в одном dataset)."""
    for dataset in ("full", "train", "test"):
        r = _run(_make_df(), dataset=dataset)
        assert "error" not in r, dataset
        trades = sorted(r["trades"], key=lambda t: t["entry_time"])
        for a, b in zip(trades, trades[1:]):  # noqa: RUF007 — как в metrics
            assert a["exit_time"] < b["entry_time"], (dataset, a, b)