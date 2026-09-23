"""BLOCK-40: метрики бэктеста — profit factor, winrate, expectancy.

Синтетический df без сетевого доступа: run_backtest(df=...) данные не ходит.
"""
import itertools

import numpy as np
import pandas as pd
import pytest

from app_pkg.ai.backtest import run_backtest


def _make_df(n=3000, seed=7, drift=0.0):
    """OHLCV случайного блуждания: high/low симметричны — без сдвига."""
    rng = np.random.default_rng(seed)
    ret = rng.normal(drift, 0.002, n)
    close = 100 * np.exp(np.cumsum(ret))
    spread = np.abs(rng.normal(0, 0.003, n))
    high = close * (1 + spread)
    low = close * (1 - spread)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum.reduce([high, open_, close])
    low = np.minimum.reduce([low, open_, close])
    # timestamp — datetime64, как в реальном пайплайне: utils.epoch_secs
    # парсит именно даты (int-секунды трактуются как наносекунды → 1970).
    ts = pd.date_range("2025-01-01", periods=n, freq="15min")
    return pd.DataFrame({
        "timestamp": ts, "open": open_, "high": high, "low": low,
        "close": close, "volume": rng.uniform(100, 1000, n),
    })


def _result(**kw):
    df = _make_df(**{k: v for k, v in kw.items() if k in ("seed", "drift")})
    return run_backtest("BTCUSDT", "15m", 0, 0, "sma_cross",
                        {"fast": 20, "slow": 50}, 10000,
                        df=df, tp_atr=2.0, sl_atr=1.0, dataset="full")


def test_profit_factor_is_gross_profit_over_gross_loss():
    r = _result()
    trades = r["trades"]
    gross = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    loss = -sum(t["pnl"] for t in trades if t["pnl"] < 0)
    assert loss > 0  # на случайном блуждании убыточные сделки есть
    assert r["profit_factor"] == pytest.approx(round(gross / loss, 2))


def test_winrate_excludes_zero_pnl_from_denominator():
    """Сделки с pnl == 0 не считаются убытками: winrate = wins / (wins+losses)."""
    r = _result()
    trades = r["trades"]
    wins = sum(1 for t in trades if t["pnl"] > 0)
    losses = sum(1 for t in trades if t["pnl"] < 0)
    expected = wins / (wins + losses) if (wins + losses) else 0
    assert r["win_rate"] == pytest.approx(expected, abs=1e-4)


def test_expectancy_matches_definition():
    r = _result()
    trades = r["trades"]
    wins = [t["pnl_pct"] for t in trades if t["pnl"] > 0]
    losses = [-t["pnl_pct"] for t in trades if t["pnl"] < 0]
    if not wins or not losses:
        pytest.skip("нужны и прибыльные, и убыточные сделки")
    wr = len(wins) / (len(wins) + len(losses))
    exp = wr * (sum(wins) / len(wins)) - (1 - wr) * (sum(losses) / len(losses))
    assert r["expectancy"] == pytest.approx(exp, abs=1e-4)


def test_rr_ratio_is_avg_win_over_avg_loss():
    r = _result()
    trades = r["trades"]
    wins = [t["pnl_pct"] for t in trades if t["pnl"] > 0]
    losses = [-t["pnl_pct"] for t in trades if t["pnl"] < 0]
    assert r["rr_ratio"] == pytest.approx(
        (sum(wins) / len(wins)) / (sum(losses) / len(losses)), abs=0.02)


def test_every_trade_has_tp_sl_levels_for_drawing():
    """Зоны на графике (BLOCK-40) требуют tp_price/sl_price у 100% сделок."""
    r = _result()
    assert r["trades"]
    for t in r["trades"]:
        assert t["tp_price"] is not None
        assert t["sl_price"] is not None


def test_tp_is_strictly_twice_as_far_as_sl():
    """BLOCK-41: тейк СТРОГО в два раза дальше от входа, чем стоп."""
    r = _result()
    assert r["trades"]
    for t in r["trades"]:
        tp_dist = abs(t["tp_price"] - t["entry_price"])
        sl_dist = abs(t["entry_price"] - t["sl_price"])
        assert sl_dist > 0
        assert tp_dist == pytest.approx(2.0 * sl_dist, rel=1e-6)


def test_trades_do_not_overlap():
    """BLOCK-41: сделка не может быть внутри другой — новая позиция
    открывается только после закрытия предыдущей (exit < следующий entry)."""
    r = _result()
    trades = sorted(r["trades"], key=lambda t: t["entry_time"])
    for a, b in itertools.pairwise(trades):
        assert a["exit_time"] is not None
        assert a["exit_time"] < b["entry_time"], "перекрытие сделок"
