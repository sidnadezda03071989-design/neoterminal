# -*- coding: utf-8 -*-
"""Тесты параметризации стратегий и передачи готового df в run_backtest.

Проверяем (фиксы сканера):
  - SMACross/RSIReversal/MACDCross читают params и дают РАЗНЫЕ сделки
    для разных комбинаций (раньше все 90 комбинаций были одинаковыми);
  - серии индикаторов предрасчитаны в _precompute(df) один раз, next()
    только читает готовые numpy-массивы (O(1) на бар);
  - run_backtest(df=...) НЕ зовёт get_replay_df;
  - run_scan тянет данные ОДИН раз на символ, а не на комбинацию.
"""

import numpy as np
import pandas as pd
import pytest

from app_pkg import config
from app_pkg.ai import backtest as bt
from app_pkg.ai import scanner
from app_pkg.ai.backtest import SMACross, run_backtest

T = 1_700_000_000
STEP = 900  # 15m
N = 600     # баров в синтетике


def _mk_df(n=N, end_ts=T, step=STEP):
    """Синтетика: восходящий тренд + синус — пересечения SMA/RSI/MACD есть."""
    ts = [end_ts - i * step for i in range(n)][::-1]
    closes = [100.0 + 20.0 * np.sin(i / 15.0) + 0.05 * i for i in range(n)]
    return pd.DataFrame({
        "timestamp": pd.to_datetime(ts, unit="s", utc=True),
        "open": closes,
        "high": [c + 1.0 for c in closes],
        "low": [c - 1.0 for c in closes],
        "close": closes,
        "volume": [10.0] * n,
    })


_DF = _mk_df()


def _trade_sig(result):
    """Подпись сделок: список (entry_time, exit_time, entry_price)."""
    return [(t["entry_time"], t["exit_time"], t["entry_price"])
            for t in result.get("trades") or []]


# ------------------------------------------------------- параметры стратегий
def test_sma_cross_params_read():
    """SMACross читает fast/slow из params, дефолт 20/50."""
    s1 = SMACross({"fast": 5, "slow": 10})
    s2 = SMACross({"fast": 20, "slow": 50})
    s3 = SMACross()
    assert (s1.fast, s1.slow) == (5, 10)
    assert (s2.fast, s2.slow) == (20, 50)
    assert (s3.fast, s3.slow) == (20, 50)  # дефолты без params


def test_sma_cross_different_params_different_trades():
    """fast=5/slow=10 vs fast=20/slow=50 — разные сделки."""
    r1 = run_backtest("BTCUSDT", "15m", None, None, "sma_cross",
                      {"fast": 5, "slow": 10}, df=_DF)
    r2 = run_backtest("BTCUSDT", "15m", None, None, "sma_cross",
                      {"fast": 20, "slow": 50}, df=_DF)
    assert "error" not in r1 and "error" not in r2
    assert r1["total_trades"] > 0 and r2["total_trades"] > 0
    assert _trade_sig(r1) != _trade_sig(r2)


def test_rsi_reversal_different_period_different_trades():
    """period=7 vs period=21 — разные сделки."""
    r1 = run_backtest("BTCUSDT", "15m", None, None, "rsi_reversal",
                      {"period": 7}, df=_DF)
    r2 = run_backtest("BTCUSDT", "15m", None, None, "rsi_reversal",
                      {"period": 21}, df=_DF)
    assert "error" not in r1 and "error" not in r2
    assert _trade_sig(r1) != _trade_sig(r2)


def test_macd_cross_different_params_different_trades():
    """fast=8/slow=21 vs 12/26 — разные сделки."""
    r1 = run_backtest("BTCUSDT", "15m", None, None, "macd_cross",
                      {"fast": 8, "slow": 21}, df=_DF)
    r2 = run_backtest("BTCUSDT", "15m", None, None, "macd_cross",
                      {"fast": 12, "slow": 26}, df=_DF)
    assert "error" not in r1 and "error" not in r2
    assert _trade_sig(r1) != _trade_sig(r2)


def test_series_precomputed_once():
    """Серии предрасчитаны в _precompute(df); next() только читает их.

    (Бывший test_series_cache_reused: ленивый кеш _series заменён на
    предрасчёт всех серий в конструкторе — быстрее и без ветвлений в next().)
    """
    s = SMACross({"fast": 5, "slow": 10}, df=_DF)
    expected = _DF["close"].rolling(5).mean().to_numpy()
    np.testing.assert_allclose(s.sma_fast, expected, equal_nan=True)

    # next() не пересчитывает серии: это те же numpy-массивы
    arr_fast = s.sma_fast
    arr_slow = s.sma_slow
    for _ in range(3):
        s.next(_DF, None, 100, None, 0, [])
    assert s.sma_fast is arr_fast and s.sma_slow is arr_slow


# ------------------------------------------------------------ df в run_backtest
def test_run_backtest_with_df_skips_get_replay_df(monkeypatch):
    """run_backtest(df=...) не зовёт get_replay_df; без df — зовёт."""
    calls = []

    def fake_replay(*a, **k):
        calls.append(1)
        return _DF

    monkeypatch.setattr(bt, "get_replay_df", fake_replay)

    res = run_backtest("BTCUSDT", "15m", None, None, "sma_cross",
                       {"fast": 5, "slow": 10}, df=_DF)
    assert "error" not in res
    assert calls == []  # df передан — источник данных не дёргается

    res2 = run_backtest("BTCUSDT", "15m", None, None, "sma_cross",
                        {"fast": 5, "slow": 10})
    assert "error" not in res2
    assert calls == [1]  # без df — ровно один запрос


# --------------------------------------------------------------- scanner flow
def test_run_scan_fetches_data_once_per_symbol(monkeypatch):
    """run_scan: get_replay_df 1 раз на символ (2 символа → 2, не 90)."""
    replay_symbols = []

    def fake_replay(symbol, tf, from_sec, to_sec, limit=None):
        replay_symbols.append(symbol)
        return _mk_df()

    monkeypatch.setattr(scanner, "get_replay_df", fake_replay)
    monkeypatch.setattr(config, "SCAN_MIN_TRADES", 0)  # не отсеивать синтетику
    monkeypatch.setitem(config.SCAN_GRIDS, "sma_cross",
                        {"fast": [5, 10, 15, 20, 25]})
    events = []
    monkeypatch.setattr(scanner, "_ws_push",
                        lambda event, data: events.append((event, data)))

    run_id = scanner.run_scan(["BTCUSDT", "ETHUSDT"], "15m", ["sma_cross"])
    from app_pkg import db
    try:
        assert replay_symbols == ["BTCUSDT", "ETHUSDT"]  # 1 запрос на символ
        rows = db.db_get_scan_results(run_id, limit=1000)
        assert len(rows) == 10  # 2 символа × 5 комбинаций
        # Комбинации с разными fast дают РАЗНЫЙ combined_sharpe
        sharpes = {r["combined_sharpe"] for r in rows}
        assert len(sharpes) > 1
        # Прогресс дошёл до конца
        assert events[-1][1] == {"run_id": run_id, "done": 10, "total": 10,
                                 "current": None}
    finally:
        db.db_clear_scan_run(run_id)
