# -*- coding: utf-8 -*-
"""Бэктест торговых стратегий на исторических данных."""

import logging
from abc import ABC, abstractmethod

import numpy as np

from app_pkg import config, utils
from app_pkg.data.fetch import get_replay_df
from app_pkg.indicators import compute_indicators

log = logging.getLogger(__name__)


class BacktestStrategy(ABC):
    """База стратегии: next() возвращает {"action": "BUY"|"SELL", "price"}."""

    def __init__(self, params=None):
        self.params = params or {}

    @abstractmethod
    def next(self, candles, ind, i, position, cash, equity_log):
        pass


class SMACross(BacktestStrategy):
    """SMA20 x SMA50 cross."""

    def next(self, candles, ind, i, position, cash, equity_log):
        if i < 50:
            return None
        sma20 = ind["sma20"].iloc[i]
        sma50 = ind["sma50"].iloc[i]
        sma20_1 = ind["sma20"].iloc[i - 1]
        sma50_1 = ind["sma50"].iloc[i - 1]
        if _isna(sma20) or _isna(sma50):
            return None
        price = utils._clean(candles["close"].iloc[i])
        if price is None:
            return None
        if not position and sma20 > sma50 and sma20_1 <= sma50_1:
            return {"action": "BUY", "price": price}
        if position and sma20 < sma50 and sma20_1 >= sma50_1:
            return {"action": "SELL", "price": price}
        return None


class RSIReversal(BacktestStrategy):
    """RSI < 30 — BUY, RSI > 70 — SELL."""

    def next(self, candles, ind, i, position, cash, equity_log):
        if i < 14:
            return None
        rsi = utils._clean(ind["rsi"].iloc[i])
        price = utils._clean(candles["close"].iloc[i])
        if price is None or rsi is None:
            return None
        if not position and rsi < 30:
            return {"action": "BUY", "price": price}
        if position and rsi > 70:
            return {"action": "SELL", "price": price}
        return None


class MACDCross(BacktestStrategy):
    """Пересечение MACD и signal-линии."""

    def next(self, candles, ind, i, position, cash, equity_log):
        if i < 26:
            return None
        macd = utils._clean(ind["macd"].iloc[i])
        sig = utils._clean(ind["macd_signal"].iloc[i])
        macd_1 = utils._clean(ind["macd"].iloc[i - 1])
        sig_1 = utils._clean(ind["macd_signal"].iloc[i - 1])
        if macd is None or sig is None:
            return None
        price = utils._clean(candles["close"].iloc[i])
        if price is None:
            return None
        if not position and macd > sig and macd_1 <= sig_1:
            return {"action": "BUY", "price": price}
        if position and macd < sig and macd_1 >= sig_1:
            return {"action": "SELL", "price": price}
        return None


STRATEGY_MAP = {
    "sma_cross": SMACross,
    "rsi_reversal": RSIReversal,
    "macd_cross": MACDCross,
}


def _isna(v):
    return v is None or (isinstance(v, float) and np.isnan(v))


def run_backtest(symbol, tf, from_sec, to_sec, strategy_name, params,
                 initial_cash=10000, replay_limit=None):
    """Запуск бэктеста; возвращает dict со статистикой и кривой эквити."""
    strategy_cls = STRATEGY_MAP.get(strategy_name)
    if not strategy_cls:
        return {"error": f"Unknown strategy: {strategy_name}"}

    limit = replay_limit or config.BACKTEST_LIMIT
    df = get_replay_df(symbol, tf, from_sec, to_sec, limit=limit)
    if df is None or len(df) < 50:
        return {"error": "Недостаточно данных"}
    ind = compute_indicators(df)
    strategy = strategy_cls(params)

    cash = float(initial_cash)
    position = None
    equity_log = []
    trades = []
    entry_bar = 0

    for i in range(len(df)):
        price = utils._clean(df["close"].iloc[i])
        tstamp = int(utils.epoch_secs(df["timestamp"])[i])
        sig = strategy.next(df, ind, i, position, cash, equity_log)
        if sig and sig["action"] == "BUY" and not position:
            shares = cash / sig["price"] if sig["price"] else 0
            position = {"shares": shares, "entry": sig["price"]}
            entry_bar = i
            cash = 0
        elif sig and sig["action"] == "SELL" and position:
            value = position["shares"] * sig["price"]
            pnl = value - (position["shares"] * position["entry"])
            trades.append({
                "entry_time": int(utils.epoch_secs(df["timestamp"])[entry_bar]),
                "exit_time": tstamp,
                "entry_price": position["entry"],
                "exit_price": sig["price"],
                "pnl": round(pnl, 2),
                "r_ratio": 0,
            })
            cash = value
            position = None
        equity = cash + (position["shares"] * price if position else 0)
        equity_log.append({"time": tstamp, "equity": round(equity, 2)})

    if not equity_log:
        return {"error": "Пустая кривая эквити"}

    total_return = (equity_log[-1]["equity"] - initial_cash) / initial_cash
    daily_returns = []
    for i in range(1, len(equity_log)):
        prev = equity_log[i - 1]["equity"]
        if prev:
            daily_returns.append(
                (equity_log[i]["equity"] - prev) / prev)
    sharpe = 0
    if daily_returns and np.std(daily_returns) > 0:
        sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(365)

    max_dd = 0
    peak = equity_log[0]["equity"]
    for e in equity_log:
        if e["equity"] > peak:
            peak = e["equity"]
        dd = (peak - e["equity"]) / peak
        if dd > max_dd:
            max_dd = dd

    total_trades = len(trades)
    wins = sum(1 for t in trades if t["pnl"] > 0)
    win_rate = wins / total_trades if total_trades else 0

    return {
        "total_return": round(total_return, 4),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown": round(max_dd, 4),
        "total_trades": total_trades,
        "win_rate": round(win_rate, 2),
        "trades": trades[-20:],
        "equity_curve": equity_log,
    }