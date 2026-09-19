# -*- coding: utf-8 -*-
"""Бэктест торговых стратегий на исторических данных."""

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from app_pkg import config, utils
from app_pkg.data.fetch import get_replay_df
from app_pkg.indicators import (
    _adx, _atr, _supertrend, compute_indicators, rsi_wilder,
)

log = logging.getLogger(__name__)


class BacktestStrategy(ABC):
    """База стратегии: next() возвращает {"action": "BUY"|"SELL", "price"}.

    params приходят из grid-search сканера (SCAN_GRIDS) или /api/backtest.
    df — опциональный DataFrame: если передан, все производные серии
    (rolling/ewm/RSI) предрасчитываются в _precompute() ОДИН раз ДО цикла
    по барам, и next() только читает готовые numpy-массивы (O(1) на бар).
    """

    def __init__(self, params=None, df=None):
        self.params = params or {}
        if df is not None:
            self._precompute(df)

    def _precompute(self, df):
        """Предрасчёт производных серий под свои params; переопределяется."""
        pass

    @abstractmethod
    def next(self, candles, ind, i, position, cash, equity_log):
        pass


class SMACross(BacktestStrategy):
    """Пересечение SMA fast x SMA slow (дефолт 20/50)."""

    def __init__(self, params=None, df=None):
        self.params = params or {}
        self.fast = self.params.get("fast", 20)
        self.slow = self.params.get("slow", 50)
        self.sma_fast = None
        self.sma_slow = None
        super().__init__(params, df)

    def _precompute(self, df):
        """SMA fast/slow — по одному rolling на весь df.

        Раньше rolling считался при первом next() и читался через
        pandas .iloc на каждый бар — O(n) обращений к Series; теперь
        серия один раз конвертируется в numpy-массив.
        """
        closes = df["close"]
        self.sma_fast = closes.rolling(self.fast).mean().to_numpy()
        self.sma_slow = closes.rolling(self.slow).mean().to_numpy()

    def next(self, candles, ind, i, position, cash, equity_log):
        if i < max(self.fast, self.slow):
            return None
        sma_fast_i = self.sma_fast[i]
        sma_slow_i = self.sma_slow[i]
        sma_fast_1 = self.sma_fast[i - 1]
        sma_slow_1 = self.sma_slow[i - 1]
        if _isna(sma_fast_i) or _isna(sma_slow_i) \
                or _isna(sma_fast_1) or _isna(sma_slow_1):
            return None
        price = utils._clean(candles["close"].iloc[i])
        if price is None:
            return None
        if not position and sma_fast_i > sma_slow_i and sma_fast_1 <= sma_slow_1:
            return {"action": "BUY", "price": price}
        if position and sma_fast_i < sma_slow_i and sma_fast_1 >= sma_slow_1:
            return {"action": "SELL", "price": price}
        return None


class RSIReversal(BacktestStrategy):
    """RSI < oversold — BUY, RSI > overbought — SELL (дефолт 14/30/70)."""

    def __init__(self, params=None, df=None):
        self.params = params or {}
        self.period = self.params.get("period", 14)
        self.oversold = self.params.get("oversold", 30)
        self.overbought = self.params.get("overbought", 70)
        self.rsi = None
        super().__init__(params, df)

    def _precompute(self, df):
        """RSI Wilder с периодом из params — один раз на весь df."""
        self.rsi = rsi_wilder(df["close"], self.period).to_numpy()

    def next(self, candles, ind, i, position, cash, equity_log):
        if i < max(self.period, 2):
            return None
        rsi_i = utils._clean(self.rsi[i])
        price = utils._clean(candles["close"].iloc[i])
        if price is None or rsi_i is None:
            return None
        if not position and rsi_i < self.oversold:
            return {"action": "BUY", "price": price}
        if position and rsi_i > self.overbought:
            return {"action": "SELL", "price": price}
        return None


class MACDCross(BacktestStrategy):
    """Пересечение MACD и signal-линии (дефолт 12/26/9)."""

    def __init__(self, params=None, df=None):
        self.params = params or {}
        self.fast = self.params.get("fast", 12)
        self.slow = self.params.get("slow", 26)
        self.signal = self.params.get("signal", 9)
        self.macd = None
        self.sig = None
        super().__init__(params, df)

    def _precompute(self, df):
        """MACD-линия и signal — по одному ewm-расчёту на весь df."""
        closes = df["close"]
        ema_fast = closes.ewm(span=self.fast, adjust=False).mean()
        ema_slow = closes.ewm(span=self.slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        self.macd = macd.to_numpy()
        self.sig = macd.ewm(span=self.signal, adjust=False).mean().to_numpy()

    def next(self, candles, ind, i, position, cash, equity_log):
        if i < self.slow:
            return None
        macd_i = utils._clean(self.macd[i])
        sig_i = utils._clean(self.sig[i])
        macd_1 = utils._clean(self.macd[i - 1])
        sig_1 = utils._clean(self.sig[i - 1])
        if macd_i is None or sig_i is None:
            return None
        price = utils._clean(candles["close"].iloc[i])
        if price is None:
            return None
        if not position and macd_i > sig_i and macd_1 <= sig_1:
            return {"action": "BUY", "price": price}
        if position and macd_i < sig_i and macd_1 >= sig_1:
            return {"action": "SELL", "price": price}
        return None


# ------------------------------------------------- общие хелперы предрасчёта
def _bb_arrays(df, period, std_mult):
    """Bollinger Bands (up, low) + closes как numpy-массивы длины df."""
    closes = pd.Series(df["close"], dtype="float64")
    sma = closes.rolling(period).mean()
    sd = closes.rolling(period).std()
    up = (sma + std_mult * sd).to_numpy()
    low = (sma - std_mult * sd).to_numpy()
    return up, low, closes.to_numpy()


def _stoch_arrays(df, k_period, d_period):
    """Stochastic %K/%D как numpy-массивы (np.nan при нулевом диапазоне)."""
    closes = pd.Series(df["close"], dtype="float64")
    lows = pd.Series(df["low"], dtype="float64")
    highs = pd.Series(df["high"], dtype="float64")
    low_k = lows.rolling(k_period).min()
    high_k = highs.rolling(k_period).max()
    rng = (high_k - low_k).replace(0, np.nan)
    k = 100.0 * (closes - low_k) / rng
    d = k.rolling(d_period).mean()
    return k.to_numpy(), d.to_numpy()


class EMACross(BacktestStrategy):
    """Пересечение EMA fast x EMA slow (дефолт 12/26)."""

    def __init__(self, params=None, df=None):
        self.params = params or {}
        self.fast = int(self.params.get("fast", 12))
        self.slow = int(self.params.get("slow", 26))
        self.ema_fast = None
        self.ema_slow = None
        super().__init__(params, df)

    def _precompute(self, df):
        closes = pd.Series(df["close"], dtype="float64")
        self.ema_fast = closes.ewm(
            span=self.fast, adjust=False).mean().to_numpy()
        self.ema_slow = closes.ewm(
            span=self.slow, adjust=False).mean().to_numpy()

    def next(self, candles, ind, i, position, cash, equity_log):
        if i < max(self.fast, self.slow):
            return None
        f_i, s_i = self.ema_fast[i], self.ema_slow[i]
        f_1, s_1 = self.ema_fast[i - 1], self.ema_slow[i - 1]
        if _isna(f_i) or _isna(s_i) or _isna(f_1) or _isna(s_1):
            return None
        price = utils._clean(candles["close"].iloc[i])
        if price is None:
            return None
        if not position and f_i > s_i and f_1 <= s_1:
            return {"action": "BUY", "price": price}
        if position and f_i < s_i and f_1 >= s_1:
            return {"action": "SELL", "price": price}
        return None


class BollingerReversal(BacktestStrategy):
    """Mean reversion: BUY у нижней границы BB, SELL у верхней."""

    def __init__(self, params=None, df=None):
        self.params = params or {}
        self.period = int(self.params.get("period", 20))
        self.std_mult = float(self.params.get("std", 2.0))
        self.bb_up = None
        self.bb_low = None
        super().__init__(params, df)

    def _precompute(self, df):
        self.bb_up, self.bb_low, _ = _bb_arrays(df, self.period, self.std_mult)

    def next(self, candles, ind, i, position, cash, equity_log):
        if i < self.period:
            return None
        up_i, low_i = self.bb_up[i], self.bb_low[i]
        if _isna(up_i) or _isna(low_i):
            return None
        price = utils._clean(candles["close"].iloc[i])
        if price is None:
            return None
        if not position and price <= low_i:
            return {"action": "BUY", "price": price}
        if position and price >= up_i:
            return {"action": "SELL", "price": price}
        return None


class BollingerBreakout(BacktestStrategy):
    """Trend following: BUY на пробое верхней границы BB, SELL — нижней."""

    def __init__(self, params=None, df=None):
        self.params = params or {}
        self.period = int(self.params.get("period", 20))
        self.std_mult = float(self.params.get("std", 2.0))
        self.bb_up = None
        self.bb_low = None
        self.closes = None
        super().__init__(params, df)

    def _precompute(self, df):
        self.bb_up, self.bb_low, self.closes = _bb_arrays(
            df, self.period, self.std_mult)

    def next(self, candles, ind, i, position, cash, equity_log):
        if i < self.period:
            return None
        up_i, up_1 = self.bb_up[i], self.bb_up[i - 1]
        low_i, low_1 = self.bb_low[i], self.bb_low[i - 1]
        close_i, close_1 = self.closes[i], self.closes[i - 1]
        if (_isna(up_i) or _isna(up_1) or _isna(low_i) or _isna(low_1)
                or _isna(close_i) or _isna(close_1)):
            return None
        price = utils._clean(candles["close"].iloc[i])
        if price is None:
            return None
        if not position and close_i > up_i and close_1 <= up_1:
            return {"action": "BUY", "price": price}
        if position and close_i < low_i and close_1 >= low_1:
            return {"action": "SELL", "price": price}
        return None


class SupertrendFollow(BacktestStrategy):
    """Следование за Supertrend: BUY при пробое линии вверх, SELL — вниз."""

    def __init__(self, params=None, df=None):
        self.params = params or {}
        self.period = int(self.params.get("period", 10))
        self.multiplier = float(self.params.get("multiplier", 3.0))
        self.st = None
        self.closes = None
        super().__init__(params, df)

    def _precompute(self, df):
        # _supertrend приватная, но это единственный способ учесть кастомные
        # period/multiplier: compute_indicators хардкодит 10/3.0.
        self.st = _supertrend(
            df["high"].astype(float), df["low"].astype(float),
            df["close"].astype(float), self.period, self.multiplier).to_numpy()
        self.closes = pd.Series(df["close"], dtype="float64").to_numpy()

    def next(self, candles, ind, i, position, cash, equity_log):
        if i < max(self.period, 1):
            return None
        st_i, st_1 = self.st[i], self.st[i - 1]
        close_i, close_1 = self.closes[i], self.closes[i - 1]
        if _isna(st_i) or _isna(st_1) or _isna(close_i) or _isna(close_1):
            return None
        price = utils._clean(candles["close"].iloc[i])
        if price is None:
            return None
        # Смена направления = цена пересекла линию st. При переходе тренда
        # вверх линия ПАДАЕТ с upper на lower band (st_i < st_1), поэтому
        # BUY — пересечение close снизу вверх при скачке линии вниз.
        if not position and st_i < st_1 and close_1 < st_1 and close_i > st_i:
            return {"action": "BUY", "price": price}
        if position and st_i > st_1 and close_1 > st_1 and close_i < st_i:
            return {"action": "SELL", "price": price}
        return None


class StochasticReversal(BacktestStrategy):
    """Отбой от зон: BUY из перепроданности с разворотом %K вверх,
    SELL из перекупленности с разворотом вниз."""

    def __init__(self, params=None, df=None):
        self.params = params or {}
        self.k_period = int(self.params.get("k_period", 14))
        self.d_period = int(self.params.get("d_period", 3))
        self.oversold = self.params.get("oversold", 20)
        self.overbought = self.params.get("overbought", 80)
        self.k = None
        self.d = None
        super().__init__(params, df)

    def _precompute(self, df):
        self.k, self.d = _stoch_arrays(df, self.k_period, self.d_period)

    def next(self, candles, ind, i, position, cash, equity_log):
        if i < self.k_period + 1:
            return None
        k_i, k_1 = self.k[i], self.k[i - 1]
        if _isna(k_i) or _isna(k_1):
            return None
        price = utils._clean(candles["close"].iloc[i])
        if price is None:
            return None
        if not position and k_i < self.oversold and k_i > k_1:
            return {"action": "BUY", "price": price}
        if position and k_i > self.overbought and k_i < k_1:
            return {"action": "SELL", "price": price}
        return None


class StochasticCross(BacktestStrategy):
    """Пересечение %K и %D: BUY при пересечении вверх, SELL — вниз."""

    def __init__(self, params=None, df=None):
        self.params = params or {}
        self.k_period = int(self.params.get("k_period", 14))
        self.d_period = int(self.params.get("d_period", 3))
        self.k = None
        self.d = None
        super().__init__(params, df)

    def _precompute(self, df):
        self.k, self.d = _stoch_arrays(df, self.k_period, self.d_period)

    def next(self, candles, ind, i, position, cash, equity_log):
        if i < self.k_period + self.d_period + 1:
            return None
        k_i, d_i = self.k[i], self.d[i]
        k_1, d_1 = self.k[i - 1], self.d[i - 1]
        if _isna(k_i) or _isna(d_i) or _isna(k_1) or _isna(d_1):
            return None
        price = utils._clean(candles["close"].iloc[i])
        if price is None:
            return None
        if not position and k_i > d_i and k_1 <= d_1:
            return {"action": "BUY", "price": price}
        if position and k_i < d_i and k_1 >= d_1:
            return {"action": "SELL", "price": price}
        return None


class CCIReversal(BacktestStrategy):
    """Отбой от уровней CCI: BUY из зоны ниже oversold с разворотом вверх,
    SELL из зоны выше overbought с разворотом вниз."""

    def __init__(self, params=None, df=None):
        self.params = params or {}
        self.period = int(self.params.get("period", 20))
        self.oversold = self.params.get("oversold", -100)
        self.overbought = self.params.get("overbought", 100)
        self.cci = None
        super().__init__(params, df)

    def _precompute(self, df):
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        tp = (high + low + close) / 3.0
        tp_sma = tp.rolling(self.period).mean()
        md = tp.rolling(self.period).apply(
            lambda x: float(np.mean(np.abs(x - np.mean(x)))), raw=True)
        md = md.replace(0, np.nan)  # md == 0 -> NaN, как в compute_indicators
        self.cci = ((tp - tp_sma) / (0.015 * md)).to_numpy()

    def next(self, candles, ind, i, position, cash, equity_log):
        if i < self.period + 1:
            return None
        cci_i, cci_1 = self.cci[i], self.cci[i - 1]
        if _isna(cci_i) or _isna(cci_1):
            return None
        price = utils._clean(candles["close"].iloc[i])
        if price is None:
            return None
        if not position and cci_i < self.oversold and cci_i > cci_1:
            return {"action": "BUY", "price": price}
        if position and cci_i > self.overbought and cci_i < cci_1:
            return {"action": "SELL", "price": price}
        return None


class VWAPReversal(BacktestStrategy):
    """Mean reversion от VWAP: BUY при отклонении ниже -threshold,
    SELL при отклонении выше +threshold."""

    def __init__(self, params=None, df=None):
        self.params = params or {}
        # threshold — legacy-имя (панель бэктеста), vwap_threshold — ключ
        # грида сканера (config.SCAN_GRIDS); поддержаны оба.
        self.threshold = float(self.params.get(
            "vwap_threshold", self.params.get("threshold", 0.005)))
        self.vwap = None
        super().__init__(params, df)

    def _precompute(self, df):
        tp = (df["high"].astype(float) + df["low"].astype(float)
              + df["close"].astype(float)) / 3.0
        vol = df["volume"].astype(float)
        vwap = (tp * vol).cumsum() / vol.cumsum().replace(0, np.nan)
        self.vwap = vwap.to_numpy()

    def next(self, candles, ind, i, position, cash, equity_log):
        if i < 1:
            return None
        vwap_i = self.vwap[i]
        if _isna(vwap_i):
            return None
        price = utils._clean(candles["close"].iloc[i])
        if price is None:
            return None
        if not position and price < vwap_i * (1.0 - self.threshold):
            return {"action": "BUY", "price": price}
        if position and price > vwap_i * (1.0 + self.threshold):
            return {"action": "SELL", "price": price}
        return None


class ADXTrend(BacktestStrategy):
    """Трендовая по ADX + DI: BUY при сильном тренде вверх (+DI > -DI),
    SELL при тренде вниз. ADX берётся из compute_indicators (период 14),
    +DI/-DI считаются здесь по Wilder с кастомным периодом."""

    def __init__(self, params=None, df=None):
        self.params = params or {}
        self.period = int(self.params.get("period", 14))
        # threshold — legacy-имя (панель бэктеста), adx_threshold — ключ
        # грида сканера (config.SCAN_GRIDS); поддержаны оба.
        self.threshold = float(self.params.get(
            "adx_threshold", self.params.get("threshold", 25)))
        self.plus_di = None
        self.minus_di = None
        self.adx = None
        super().__init__(params, df)

    def _precompute(self, df):
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        up = high.diff()
        down = -low.diff()
        plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0),
                            index=high.index)
        minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0),
                             index=high.index)
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1.0 / self.period, adjust=False).mean()

        def _wilder(s):
            return s.ewm(alpha=1.0 / self.period, adjust=False).mean()

        self.plus_di = (100.0 * _wilder(plus_dm) / atr).to_numpy()
        self.minus_di = (100.0 * _wilder(minus_dm) / atr).to_numpy()
        # ADX с фиксированным периодом 14 — ровно как в compute_indicators,
        # но без пересчёта всех 18 колонок ради одной (см. BLOCK3).
        self.adx = _adx(high, low, close, 14).to_numpy()

    def next(self, candles, ind, i, position, cash, equity_log):
        if i < max(self.period * 2, 2):
            return None
        adx_i = self.adx[i]
        p_i, m_i = self.plus_di[i], self.minus_di[i]
        p_1, m_1 = self.plus_di[i - 1], self.minus_di[i - 1]
        if (_isna(adx_i) or _isna(p_i) or _isna(m_i)
                or _isna(p_1) or _isna(m_1)):
            return None
        price = utils._clean(candles["close"].iloc[i])
        if price is None:
            return None
        if not position and adx_i > self.threshold and p_i > m_i and p_1 <= m_1:
            return {"action": "BUY", "price": price}
        if position and adx_i > self.threshold and m_i > p_i and m_1 <= p_1:
            return {"action": "SELL", "price": price}
        return None


STRATEGY_MAP = {
    "sma_cross": SMACross,
    "rsi_reversal": RSIReversal,
    "macd_cross": MACDCross,
    "ema_cross": EMACross,
    "bb_reversal": BollingerReversal,
    "bb_breakout": BollingerBreakout,
    "supertrend": SupertrendFollow,
    "stoch_reversal": StochasticReversal,
    "stoch_cross": StochasticCross,
    "cci_reversal": CCIReversal,
    "vwap_reversal": VWAPReversal,
    "adx_trend": ADXTrend,
}


def _isna(v):
    return v is None or (isinstance(v, float) and np.isnan(v))


def run_backtest(symbol, tf, from_sec, to_sec, strategy_name, params,
                 initial_cash=10000, replay_limit=None, df=None, ind=None,
                 tp_atr=None, sl_atr=None):
    """Запуск бэктеста; возвращает dict со статистикой и кривой эквити.

    df — опциональный готовый DataFrame (свечи): если передан, get_replay_df
    НЕ вызывается. Grid-search сканер так передаёт заранее нарезанные
    train/test окна — 90 комбинаций × 10 символов дают 10 запросов данных
    вместо 900.

    ind — опциональные готовые индикаторы (compute_indicators), выровненные
    по df. Сканер считает их ОДИН раз на пару (symbol, tf) и переиспользует
    во всех комбинациях — иначе compute_indicators пересчитывался бы на
    каждый бэктест. Если ind не передан или не совпадает по длине — считается
    здесь.

    tp_atr / sl_atr — take-profit / stop-loss в множителях ATR(14) от цены
    входа (например 2.0 = +2×ATR / -2×ATR). Если заданы, LONG-позиция
    закрывается по уровню в тот же бар (по high/low свечи), не дожидаясь
    сигнала SELL. В trades добавляется поле "exit_reason": tp/sl/signal.
    """
    strategy_cls = STRATEGY_MAP.get(strategy_name)
    if not strategy_cls:
        return {"error": f"Unknown strategy: {strategy_name}"}

    if df is None:
        if config.BACKTEST_USE_FULL_HISTORY:
            # Полная история: максимум доступных свечей (20k крипта / 13k форекс),
            # если вызывающий не передал свой replay_limit.
            limit = replay_limit or config.BACKTEST_MAX_CANDLES
        else:
            limit = replay_limit or config.BACKTEST_LIMIT
        df = get_replay_df(symbol, tf, from_sec, to_sec, limit=limit)
    if df is None or len(df) < 50:
        return {"error": "Недостаточно данных"}
    # Жёсткий потолок свечей: MT5 может отдать куда больше запрошенного
    # лимита (напр. 23115 баров XAUUSD при limit=5000) — обрезаем до
    # BACKTEST_MAX_CANDLES (= MAX_DATA_LIMIT), но НЕ до SCAN_REPLAY_LIMIT:
    # полноисторийный бэктест работает на 20k свечей.
    if len(df) > config.BACKTEST_MAX_CANDLES:
        df = df.iloc[-config.BACKTEST_MAX_CANDLES:].reset_index(drop=True)
    if ind is None or len(ind) != len(df):
        ind = compute_indicators(df)
    # df передаётся в конструктор: все rolling/ewm/RSI серии предрасчитываются
    # один раз ДО цикла по барам (см. BacktestStrategy._precompute).
    strategy = strategy_cls(params, df=df)

    cash = float(initial_cash)
    position = None
    equity_log = []
    trades = []
    entry_bar = 0

    ts = utils.epoch_secs(df["timestamp"])  # один раз, не в цикле (O(n^2) иначе)
    # Предрасчёт массивов high/low/ATR: проверка TP/SL по уровням свечи
    # (аналог сигнала стратегии) не должна растекаться в O(n^2).
    high_arr = df["high"].to_numpy(dtype=float) if "high" in df else None
    low_arr = df["low"].to_numpy(dtype=float) if "low" in df else None
    atr_arr = (_atr(df["high"], df["low"], df["close"], 14).to_numpy(dtype=float)
               if (tp_atr or sl_atr) and "high" in df and "low" in df else None)
    for i in range(len(df)):
        price = utils._clean(df["close"].iloc[i])
        if price is None:
            # NaN-бар (дырка в данных): нет валидной цены. Пропускаем бар
            # целиком — иначе position["shares"] * None валит цикл, а
            # оценка эквити по мусорной цене искажает sharpe/max_dd.
            continue
        tstamp = int(ts[i])
        sig = strategy.next(df, ind, i, position, cash, equity_log)
        # TP/SL LONG: уровни из ATR-множителей; проверяем ДО сигнала стратегии.
        # Если на одном баре пробиты оба уровня, консервативно считаем, что
        # первым сработал SL (риск важнее прибыли).
        if position and atr_arr is not None:
            exit_price = None
            exit_reason = None
            atr_i = utils._clean(atr_arr[i])
            if atr_i is not None and atr_i > 0:
                sl_price = None
                tp_price = None
                if sl_atr:
                    sl_price = position["entry"] - sl_atr * atr_i
                if tp_atr:
                    tp_price = position["entry"] + tp_atr * atr_i
                lo = utils._clean(low_arr[i])
                hi = utils._clean(high_arr[i])
                if sl_price is not None and lo is not None and lo <= sl_price:
                    exit_price, exit_reason = sl_price, "sl"
                elif (tp_price is not None and hi is not None
                      and hi >= tp_price):
                    exit_price, exit_reason = tp_price, "tp"
            if exit_price is not None:
                value = position["shares"] * exit_price
                pnl = value - (position["shares"] * position["entry"])
                entry_shares = position["shares"]
                entry_price = position["entry"]
                trades.append({
                    "entry_time": int(ts[entry_bar]),
                    "exit_time": tstamp,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "direction": "BUY",
                    "tp_price": round(tp_price, 8) if tp_price is not None else None,
                    "sl_price": round(sl_price, 8) if sl_price is not None else None,
                    "tp_time": tstamp if exit_reason == "tp" else None,
                    "sl_time": tstamp if exit_reason == "sl" else None,
                    "pnl": round(pnl, 2),
                    "pnl_pct": round((pnl / (entry_shares * entry_price)) * 100, 4)
                               if entry_shares and entry_price else 0,
                    "r_ratio": 0,
                    "exit_reason": exit_reason,
                })
                cash = value
                position = None
                equity = cash
                equity_log.append({"time": tstamp, "equity": round(equity, 2)})
                continue  # бар закрыт TP/SL — сигнал стратегии не обрабатываем
        if sig and sig["action"] == "BUY" and not position:
            shares = cash / sig["price"] if sig["price"] else 0
            position = {"shares": shares, "entry": sig["price"]}
            entry_bar = i
            cash = 0
        elif sig and sig["action"] == "SELL" and position:
            value = position["shares"] * sig["price"]
            pnl = value - (position["shares"] * position["entry"])
            trades.append({
                "entry_time": int(ts[entry_bar]),
                "exit_time": tstamp,
                "entry_price": position["entry"],
                "exit_price": sig["price"],
                "direction": "BUY",
                "tp_price": None,
                "sl_price": None,
                "tp_time": None,
                "sl_time": None,
                "pnl": round(pnl, 2),
                "pnl_pct": round((pnl / (position["shares"] * position["entry"])) * 100, 4)
                           if position["shares"] and position["entry"] else 0,
                "r_ratio": 0,
                "exit_reason": "signal",
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

    # trades/trades_full — ПОЛНЫЙ список (без среза [-20:]): счётчик сделок в
    # статистике должен совпадать с числом блоков на графике (BLOCK-30).
    trades_out = trades
    if config.BACKTEST_MAX_TRADES_RETURNED:
        trades_out = trades[-config.BACKTEST_MAX_TRADES_RETURNED:]

    return {
        "total_return": round(total_return, 4),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown": round(max_dd, 4),
        "total_trades": total_trades,
        "win_rate": round(win_rate, 2),
        "trades": trades_out,
        "trades_full": trades_out,
        "equity_curve": equity_log,
        "candles_used": len(df),
        "bars_from": int(ts[0]),
        "bars_to": int(ts[-1]),
    }