"""Тесты 9 новых стратегий backtest.py (EMA, BB, Supertrend, Stoch, CCI,
VWAP, ADX).

Для каждой стратегии проверяем:
  - __init__ с df читает параметры корректно;
  - _precompute создаёт numpy-массивы правильной длины (len(df));
  - next() возвращает None до прогрева (i < warmup);
  - next() возвращает dict {"action", "price"} при сигнале на синтетике;
  - разные params -> разные результаты (не идентичные);
  - не падает на df с NaN и на пустом df;
  - run_backtest с новой стратегией возвращает результат без ошибки.

Плюс: STRATEGY_MAP содержит все 12 стратегий (3 старых + 9 новых).
"""

import numpy as np
import pandas as pd
import pytest

from app_pkg.ai.backtest import (
    STRATEGY_MAP,
    ADXTrend,
    BollingerBreakout,
    BollingerReversal,
    CCIReversal,
    EMACross,
    StochasticCross,
    StochasticReversal,
    SupertrendFollow,
    VWAPReversal,
    run_backtest,
)

T = 1_700_000_000
STEP = 900  # 15m
N = 800


def _mk_df(n=N, nan_holes=False):
    """Синтетика: тренд + синус — сигналы есть у всех стратегий."""
    ts = [T - i * STEP for i in range(n)][::-1]
    closes = [100.0 + 20.0 * np.sin(i / 18.0) + 0.08 * i for i in range(n)]
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(ts, unit="s", utc=True),
        "open": closes,
        "high": [c + 1.5 for c in closes],
        "low": [c - 1.5 for c in closes],
        "close": closes,
        "volume": [100.0 + (i % 7) * 10.0 for i in range(n)],
    })
    if nan_holes:
        for j in (100, 300, 500):
            df.loc[j, "close"] = np.nan
            df.loc[j + 1, "volume"] = np.nan
    return df


_DF = _mk_df()

# (класс, ключ STRATEGY_MAP, params A, params B, attrs, warmup баров)
NEW_STRATEGIES = [
    (EMACross, "ema_cross", {"fast": 5, "slow": 10},
     {"fast": 20, "slow": 50}, {"fast": 5, "slow": 10}, 50),
    (BollingerReversal, "bb_reversal", {"period": 10, "std": 1.5},
     {"period": 20, "std": 2.5}, {"period": 10, "std_mult": 1.5}, 20),
    (BollingerBreakout, "bb_breakout", {"period": 10, "std": 1.5},
     {"period": 20, "std": 2.5}, {"period": 10, "std_mult": 1.5}, 20),
    (SupertrendFollow, "supertrend", {"period": 5, "multiplier": 1.5},
     {"period": 14, "multiplier": 3.0},
     {"period": 5, "multiplier": 1.5}, 14),
    (StochasticReversal, "stoch_reversal",
     {"k_period": 7, "d_period": 2, "oversold": 25, "overbought": 75},
     {"k_period": 14, "d_period": 3, "oversold": 35, "overbought": 65},
     {"k_period": 7, "oversold": 25}, 21),
    (StochasticCross, "stoch_cross", {"k_period": 5, "d_period": 2},
     {"k_period": 14, "d_period": 5}, {"k_period": 5, "d_period": 2}, 21),
    (CCIReversal, "cci_reversal",
     {"period": 10, "oversold": -80, "overbought": 80},
     {"period": 20, "oversold": -120, "overbought": 120},
     {"period": 10, "oversold": -80}, 21),
    (VWAPReversal, "vwap_reversal", {"threshold": 0.001},
     {"threshold": 0.02}, {"threshold": 0.001}, 1),
    (ADXTrend, "adx_trend", {"period": 7, "threshold": 15},
     {"period": 14, "threshold": 35}, {"period": 7, "threshold": 15}, 28),
]

_IDS = [s[1] for s in NEW_STRATEGIES]
_PARAMS = dict(zip(
    ["cls", "key", "p1", "p2", "attrs", "warmup"],
    zip(*NEW_STRATEGIES)))


def test_strategy_map_has_12_strategies():
    """STRATEGY_MAP: 3 старых + 9 новых = 12 ключей."""
    assert set(STRATEGY_MAP) == {
        "sma_cross", "rsi_reversal", "macd_cross",
        "ema_cross", "bb_reversal", "bb_breakout", "supertrend",
        "stoch_reversal", "stoch_cross", "cci_reversal", "vwap_reversal",
        "adx_trend",
    }
    assert len(STRATEGY_MAP) == 12


@pytest.mark.parametrize(("cls", "key", "p1", "p2", "attrs", "warmup"),
                         NEW_STRATEGIES, ids=_IDS)
def test_params_read_and_precompute_arrays(cls, key, p1, p2, attrs, warmup):
    """__init__ с df читает params; _precompute даёт numpy-массивы len(df)."""
    s = cls(p1, df=_DF)
    for name, val in attrs.items():
        assert getattr(s, name) == val, f"{key}.{name} != {val}"
    arrays = [v for v in vars(s).values() if isinstance(v, np.ndarray)]
    assert arrays, f"{key}: нет предрасчитанных numpy-массивов"
    for arr in arrays:
        assert len(arr) == len(_DF)


@pytest.mark.parametrize(("cls", "key", "p1", "p2", "attrs", "warmup"),
                         NEW_STRATEGIES, ids=_IDS)
def test_next_none_before_warmup(cls, key, p1, p2, attrs, warmup):
    """next() возвращает None до прогрева (i < warmup)."""
    s = cls(p1, df=_DF)
    for i in range(min(warmup, 5)):
        assert s.next(_DF, None, i, None, 10000, []) is None, \
            f"{key}: сигнал на баре {i} до прогрева {warmup}"


@pytest.mark.parametrize(("cls", "key", "p1", "p2", "attrs", "warmup"),
                         NEW_STRATEGIES, ids=_IDS)
def test_next_signal_is_dict(cls, key, p1, p2, attrs, warmup):
    """next() при сигнале возвращает dict с "action" и "price"."""
    s = cls(p1, df=_DF)
    sig = None
    for i in range(max(warmup, 2), len(_DF)):
        sig = s.next(_DF, None, i, None, 10000, [])
        if sig is not None:
            break
    assert sig is not None, f"{key}: нет ни одного сигнала на синтетике"
    assert sig["action"] in ("BUY", "SELL")
    assert isinstance(sig["price"], float) and sig["price"] > 0


@pytest.mark.parametrize(("cls", "key", "p1", "p2", "attrs", "warmup"),
                         NEW_STRATEGIES, ids=_IDS)
def test_different_params_different_results(cls, key, p1, p2, attrs, warmup):
    """Разные params -> разные результаты (сделки или эквити не идентичны)."""
    r1 = run_backtest("X", "15m", None, None, key, p1, df=_DF)
    r2 = run_backtest("X", "15m", None, None, key, p2, df=_DF)
    assert "error" not in r1, f"{key}: {r1.get('error')}"
    assert "error" not in r2, f"{key}: {r2.get('error')}"
    assert r1["total_trades"] > 0, f"{key}: нет сделок на синтетике"
    key_a = (r1["total_trades"], round(r1["total_return"], 4))
    key_b = (r2["total_trades"], round(r2["total_return"], 4))
    assert key_a != key_b, f"{key}: разные params дали идентичный результат"


@pytest.mark.parametrize(("cls", "key", "p1", "p2", "attrs", "warmup"),
                         NEW_STRATEGIES, ids=_IDS)
def test_nan_and_empty_df_no_crash(cls, key, p1, p2, attrs, warmup):
    """next() не падает на df с NaN; конструктор не падает на пустом df."""
    df_nan = _mk_df(nan_holes=True)
    s = cls(p1, df=df_nan)
    for i in range(len(df_nan)):
        sig = s.next(df_nan, None, i, None, 10000, [])
        assert sig is None or set(sig) == {"action", "price"}
    empty = pd.DataFrame(columns=["timestamp", "open", "high", "low",
                                  "close", "volume"])
    s2 = cls(p1, df=empty)  # _precompute на пустом df не должен падать
    assert s2.next(empty, None, 0, None, 10000, []) is None


@pytest.mark.parametrize(("cls", "key", "p1", "p2", "attrs", "warmup"),
                         NEW_STRATEGIES, ids=_IDS)
def test_run_backtest_nan_df_no_crash(cls, key, p1, p2, attrs, warmup):
    """run_backtest не падает на df с NaN-дырками (движок пропускает бар)."""
    r = run_backtest("X", "15m", None, None, key, p1, df=_mk_df(nan_holes=True))
    assert "error" not in r, f"{key}: {r.get('error')}"
    assert len(r["equity_curve"]) <= len(_DF)


@pytest.mark.parametrize(("cls", "key", "p1", "p2", "attrs", "warmup"),
                         NEW_STRATEGIES, ids=_IDS)
@pytest.mark.debt
def test_run_backtest_works(cls, key, p1, p2, attrs, warmup):
    """run_backtest с новой стратегией возвращает полную статистику."""
    r = run_backtest("X", "15m", None, None, key, p1, df=_DF)
    assert "error" not in r, f"{key}: {r.get('error')}"
    assert len(r["equity_curve"]) == len(_DF)
    assert "sharpe_ratio" in r and "max_drawdown" in r
    assert "win_rate" in r and "trades" in r
