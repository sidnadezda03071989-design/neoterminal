"""Бэктест торговых стратегий на исторических данных."""

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from app_pkg import config, utils
from app_pkg.data.fetch import get_replay_df
from app_pkg.indicators import (
    _adx,
    _atr,
    _supertrend,
    compute_indicators,
    rsi_wilder,
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


def rr_levels(entry_price, risk_atr, atr_entry, rr=None, direction="BUY"):
    """Уровни сделки СТРОГО с R/R = rr (по умолчанию config.BACKTEST_RR).

    Стоп — первичен: риск = |risk_atr|×ATR от entry.
    Long:  SL = entry − risk, TP = entry + rr×risk.
    Short: SL = entry + risk, TP = entry − rr×risk.
    Расстояние |TP−entry| ровно в rr раз больше |entry−SL| — всегда.
    Возвращает (tp_price, sl_price, risk); (None, None, None) если уровни
    построить нельзя — сделка не открывается.
    """
    r = None if rr is None else abs(float(rr))
    risk = None
    entry = utils._clean(entry_price)
    if entry and risk_atr and atr_entry and atr_entry > 0:
        risk = abs(float(risk_atr)) * float(atr_entry)
    if not entry or not r or r <= 0 or not risk or risk <= 0:
        return None, None, None
    is_long = str(direction or "BUY").upper() != "SELL"
    if is_long:
        sl_price = round(entry - risk, 8)
        tp_price = round(entry + r * risk, 8)
    else:
        sl_price = round(entry + risk, 8)
        tp_price = round(entry - r * risk, 8)
    return tp_price, sl_price, risk


def _hit_sl(is_long, sl_price, hi, lo, op):
    if sl_price is None:
        return False
    if is_long:
        if op is not None and op <= sl_price:
            return True
        return lo is not None and lo <= sl_price
    if op is not None and op >= sl_price:
        return True
    return hi is not None and hi >= sl_price


def _hit_tp(is_long, tp_price, hi, lo, op):
    if tp_price is None:
        return False
    if is_long:
        if op is not None and op >= tp_price:
            return True
        return hi is not None and hi >= tp_price
    if op is not None and op <= tp_price:
        return True
    return lo is not None and lo <= tp_price


def _exit_on_bar(position, high, low, open_=None):
    """Первая линия, которой коснулась цена на баре: (цена, "tp"|"sl").

    Гэп через уровень: open уже за SL/TP — этот уровень. Оба уровня на
    одном баре (фитиль задел и стоп, и тейк) — всегда SL: по OHLC нельзя
    узнать порядок касаний, тейк после стопа считать нельзя.
    """
    hi = utils._clean(high)
    lo = utils._clean(low)
    op = utils._clean(open_)
    sl_price = position.get("sl_price")
    tp_price = position.get("tp_price")
    is_long = str(position.get("direction") or "BUY").upper() != "SELL"
    sl_hit = _hit_sl(is_long, sl_price, hi, lo, op)
    tp_hit = _hit_tp(is_long, tp_price, hi, lo, op)
    if sl_hit:
        return sl_price, "sl"
    if tp_hit:
        return tp_price, "tp"
    return None, None


def run_backtest(symbol, tf, from_sec, to_sec, strategy_name, params,
                 initial_cash=10000, replay_limit=None, df=None, ind=None,
                 tp_atr=None, sl_atr=None, dataset="full", rr=None):
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

    sl_atr — риск сделки (стоп) в множителях ATR(14) от цены входа:
    SL = entry − sl_atr×ATR, дефолт config.BACKTEST_SL_ATR.

    rr — соотношение риск/прибыль (по умолчанию config.BACKTEST_RR = 2).
    Тейк строится СТРОГО от стопа: TP = entry + rr×(entry − SL), то есть
    расстояние TP↔entry ровно в rr раз больше расстояния entry↔SL — всегда,
    для каждой сделки, независимо от переданного tp_atr (он оставлен только
    для обратной совместимости вызовов).

    Сделка живёт ТОЛЬКО между линиями: выход — касание TP или SL на баре
    (_exit_on_bar), что первым коснулось. Сигнальные выходы и закрытие
    «по последней свече» убраны (BLOCK-42): статистика (winrate, expectancy,
    pf, R/R) считается строго по результатам касаний, поэтому в trades
    exit_reason всегда "tp" или "sl". Соответственно risk/reward, а не
    направление сигнала, определяют результат каждой сделки.

    Сделки, у которых уровни построить нельзя (нет ATR/rr/риска) или
    расстояние TP↔SL меньше BACKTEST_MIN_TP_SL_CANDLES средних свечей окна
    (mean(high-low)) — НЕ открываются: у 100% записанных сделок есть
    tp_price/sl_price/tp_time/sl_time, а позиция, не дожившая до уровня,
    просто не попадает в trades — это открытая позиция, а не результат
    (BLOCK-42).

    Сделки не открываются одна внутри другой: выход (касание TP/SL) закрывает
    позицию, после чего бар пропускается (continue) — новая сделка может
    войти только со следующего бара, поэтому entry_time > exit_time прошлой.

    dataset — "full" (вся история), "train" (первые SCAN_TRAIN_SPLIT=70%) или
    "test" (последние 30%, out-of-sample). При train/test история делится на
    train/test окна, бэктест гонится только на выбранной части; в ответ
    добавляются train_range/test_range — временные диапазоны обоих окон
    (BLOCK-33). Сканер по-прежнему зовёт df=... с dataset по умолчанию
    "full" — его поведение не меняется.
    """
    strategy_cls = STRATEGY_MAP.get(strategy_name)
    if not strategy_cls:
        return {"error": f"Unknown strategy: {strategy_name}"}
    # BLOCK-42: R/R и риск — из config, если вызывающий не задал своё.
    rr = config.BACKTEST_RR if rr is None else rr
    risk_atr = config.BACKTEST_SL_ATR if sl_atr is None else sl_atr
    if dataset not in ("full", "train", "test"):
        return {"error": f"Unknown dataset: {dataset}"}

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

    # Сплит train/test (BLOCK-33): dataset != "full" — прогнать стратегию только
    # на выбранной части истории. Сплит до расчёта индикаторов: если индикаторы
    # были переданы под ПОЛНУЮ длину, проверка len(ind) != len(df) ниже
    # пересчитает их на нарезанном df.
    train_range = None
    test_range = None
    if dataset in ("train", "test"):
        full_n = len(df)
        split_idx = max(1, int(full_n * config.SCAN_TRAIN_SPLIT))
        ts_full = utils.epoch_secs(df["timestamp"])
        train_range = {
            "from": int(ts_full[0]),
            "to": int(ts_full[split_idx - 1]),
        }
        test_range = {
            "from": int(ts_full[min(split_idx, full_n - 1)]),
            "to": int(ts_full[full_n - 1]),
        }
        if dataset == "train":
            df = df.iloc[:split_idx].reset_index(drop=True)
        else:
            df = df.iloc[split_idx:].reset_index(drop=True)
        if len(df) < 50:
            return {"error": "Недостаточно данных в выбранном dataset"}

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
    # BLOCK-42: статистика касаний линий — сколько сделок закрылось по тейку
    # и сколько по стопу. Считается по тем же сделкам, что уходят в trades,
    # поэтому tp_trades + sl_trades == total_trades всегда.
    stats = {"tp": 0, "sl": 0}

    ts = utils.epoch_secs(df["timestamp"])  # один раз, не в цикле (O(n^2) иначе)
    # Предрасчёт массивов high/low/ATR: проверка TP/SL по уровням свечи
    # (аналог сигнала стратегии) не должна растекаться в O(n^2).
    high_arr = df["high"].to_numpy(dtype=float) if "high" in df else None
    low_arr = df["low"].to_numpy(dtype=float) if "low" in df else None
    open_arr = df["open"].to_numpy(dtype=float) if "open" in df else None
    # ATR нужен ВСЕГДА (BLOCK-42): уровни TP/SL строятся для каждой сделки,
    # без них сделка не открывается вообще. Период — config.BACKTEST_ATR_PERIOD
    # (дефолт 14): atr_at_entry в trades считается ровно этим ATR.
    atr_arr = (_atr(df["high"], df["low"], df["close"],
                    config.BACKTEST_ATR_PERIOD).to_numpy(dtype=float)
               if "high" in df and "low" in df else None)
    # Фильтр микро-сделок: расстояние TP↔SL должно быть не меньше
    # BACKTEST_MIN_TP_SL_CANDLES×средняя_свеча окна (mean(high-low)). Если
    # окно пустое/плоское — фильтр выключен (min_tp_sl=None).
    min_tp_sl = None
    if high_arr is not None and low_arr is not None \
            and config.BACKTEST_MIN_TP_SL_CANDLES > 0:
        avg_candle = utils._clean(float(np.nanmean(high_arr - low_arr)))
        if avg_candle and avg_candle > 0:
            min_tp_sl = config.BACKTEST_MIN_TP_SL_CANDLES * avg_candle
    for i in range(len(df)):
        price = utils._clean(df["close"].iloc[i])
        if price is None:
            # NaN-бар (дырка в данных): нет валидной цены. Пропускаем бар
            # целиком — иначе position["shares"] * None валит цикл, а
            # оценка эквити по мусорной цене искажает sharpe/max_dd.
            continue
        tstamp = int(ts[i])
        sig = strategy.next(df, ind, i, position, cash, equity_log)
        # Выход ТОЛЬКО по касанию TP/SL, со следующего бара после входа
        # (вход по close — фитиль бара входа уже в прошлом, это не выход).
        # Оба уровня на одном баре → SL: тейк после стопа не засчитывается.
        if position is not None and i > entry_bar:
            exit_price, exit_reason = _exit_on_bar(
                position,
                high_arr[i] if high_arr is not None else None,
                low_arr[i] if low_arr is not None else None,
                open_arr[i] if open_arr is not None else None)
            if exit_price is not None:
                value = position["shares"] * exit_price
                entry_shares = position["shares"]
                entry_price = position["entry"]
                pnl = value - (entry_shares * entry_price)
                if exit_reason == "tp":
                    position["tp_time"] = tstamp
                else:
                    position["sl_time"] = tstamp
                stats[exit_reason] += 1
                trades.append({
                    "entry_time": int(ts[entry_bar]),
                    "exit_time": tstamp,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "direction": position.get("direction") or "BUY",
                    "tp_price": position.get("tp_price"),
                    "sl_price": position.get("sl_price"),
                    # ATR на баре входа (config.BACKTEST_ATR_PERIOD): фронтенд
                    # строит зоны TP/SL строго по этим уровням и atr_at_entry.
                    "atr_at_entry": position.get("atr_at_entry"),
                    "tp_time": position.get("tp_time"),
                    "sl_time": position.get("sl_time"),
                    "pnl": round(pnl, 2),
                    "pnl_pct": round((pnl / (entry_shares * entry_price)) * 100, 4)
                               if entry_shares and entry_price else 0,
                    "r_ratio": rr if exit_reason == "tp" else -1.0,
                    "exit_reason": exit_reason,
                })
                cash = value
                position = None
                equity = cash
                equity_log.append({"time": tstamp, "equity": round(equity, 2)})
                continue  # бар закрыт TP/SL — сигнал стратегии не обрабатываем
        if sig and sig["action"] == "BUY" and not position:
            atr_entry = utils._clean(atr_arr[i]) if atr_arr is not None else None
            tp_price, sl_price, _risk = rr_levels(
                sig["price"], risk_atr, atr_entry, rr, direction="BUY")
            if tp_price is not None and \
                    (min_tp_sl is None or abs(tp_price - sl_price) >= min_tp_sl):
                shares = cash / sig["price"] if sig["price"] else 0
                position = {
                    "shares": shares,
                    "entry": sig["price"],
                    "direction": "BUY",
                    "tp_price": tp_price,
                    "sl_price": sl_price,
                    "atr_at_entry": atr_entry,
                    "tp_time": None,
                    "sl_time": None,
                }
                entry_bar = i
                cash = 0
            # else: нет ATR/риска (или TP↔SL короче трёх средних свечей) —
            # уровни не построить/сделка микроразмера — не открываем

    if not equity_log:
        # Ни одной закрытой сделки: сигналов не было либо все отфильтрованы
        # (TP↔SL короче трёх средних свечей). Валидный прогон с нулевыми
        # метриками, а НЕ ошибка — иначе скан пометит комбинацию ошибкой,
        # хотя это просто «0 сделок» (отсев SCAN_MIN_TRADES и так её выкинет).
        return {
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "total_trades": 0,
            "win_rate": 0.0,
            "avg_win_pct": 0.0,
            "avg_loss_pct": 0.0,
            "expectancy": 0.0,
            "rr_ratio": None,
            "profit_factor": 0.0,
            "tp_touches": 0,
            "sl_touches": 0,
            "trades": [],
            "trades_full": [],
            "equity_curve": [],
            "candles_used": len(df),
            "bars_from": int(ts[0]),
            "bars_to": int(ts[-1]),
            "dataset": dataset,
            "train_range": train_range,
            "test_range": test_range,
        }

    # BLOCK-42: позиция, открытая, но не коснувшаяся ни TP, ни SL к концу
    # данных, НЕ попадает в сделки: результат — это линия, которую цена
    # коснулась первой. Открытая позиция = не результат. Чтобы эквити-кривая
    # не «запрыгивала» последней позицией по цене закрытия, вычитаем её
    # нереализованную прибыль/убыток из final equity, а в equity_log не
              # не добавляем ничего (она уже сформирована на момент открытия позиции).
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
        peak = max(peak, e["equity"])
        dd = (peak - e["equity"]) / peak
        max_dd = max(max_dd, dd)

    total_trades = len(trades)
    # BLOCK-40: winrate — доля прибыльных среди закрытых С НЕНУЛЕВЫМ
    # результатом. Раньше сделки с pnl == 0 (брокерские нули, round до 0.00)
    # падали в знаменатель как убытки и искажали винрейт.
    wins = sum(1 for t in trades if t["pnl"] > 0)
    losses = sum(1 for t in trades if t["pnl"] < 0)
    win_rate = wins / (wins + losses) if (wins + losses) else 0

    # BLOCK-38: expectancy — ГЛАВНАЯ метрика (средний профит на сделку в % от
    # цены входа): winrate без неё обманывает — 36% выигрышей при R/R 2:1 дают
    # плюс, а 70% при R/R 1:3 — минус. Считаем по pnl_pct сделок:
    #   avg_win_pct  — средний плюс среди прибыльных,
    #   avg_loss_pct — средний минус по модулю среди убыточных,
    #   expectancy   — winrate×avg_win − (1−winrate)×avg_loss (% за сделку),
    #   rr_ratio     — avg_win / avg_loss (во сколько раз профит больше убытка);
    #                  None, если убыточных сделок нет — R/R не определён.
    win_pcts = [t["pnl_pct"] for t in trades if t["pnl"] > 0]
    loss_pcts = [-t["pnl_pct"] for t in trades if t["pnl"] < 0]
    avg_win_pct = sum(win_pcts) / len(win_pcts) if win_pcts else 0.0
    avg_loss_pct = sum(loss_pcts) / len(loss_pcts) if loss_pcts else 0.0
    expectancy = win_rate * avg_win_pct - (1 - win_rate) * avg_loss_pct
    rr_ratio = (avg_win_pct / avg_loss_pct) if avg_loss_pct > 0 else None

    # BLOCK-40: profit factor — ЧЁТКО валовый профит / валовый убыток:
    # сумма pnl прибыльных сделок / модуль суммы pnl убыточных. Убыточных нет —
    # 999.0 (кап: бесконечность невалидна в JSON/SQLite), сделок нет — 0.0.
    gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gross_loss = -sum(t["pnl"] for t in trades if t["pnl"] < 0)
    if gross_loss > 0:
        profit_factor = round(gross_profit / gross_loss, 2)
    else:
        profit_factor = 999.0 if gross_profit > 0 else 0.0

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
        # BLOCK-35: 4 знака вместо 2 — иначе train/test winrate с разным числом
        # сделок схлопывались в одно число (0.67/183 и 0.67/76 в панели).
        "win_rate": round(win_rate, 4),
        # BLOCK-38: expectancy и R/R — их панель сканера показывает рядом с
        # winrate (через scanner._metrics -> train_json/test_json).
        "avg_win_pct": round(avg_win_pct, 4),
        "avg_loss_pct": round(avg_loss_pct, 4),
        "expectancy": round(expectancy, 4),
        "rr_ratio": round(rr_ratio, 2) if rr_ratio is not None else None,
        # BLOCK-40: валовый профит / валовый убыток — панель бэктеста и сканер.
        "profit_factor": profit_factor,
        # BLOCK-42: статистика касаний линий — tp/sl считается по сделкам,
        # поэтому stats["tp"] + stats["sl"] == total_trades всегда. Их сумма
        # строго равна числу сделок в trades: никаких "signal"/"end".
        "tp_touches": stats["tp"],
        "sl_touches": stats["sl"],
        "trades": trades_out,
        "trades_full": trades_out,
        "equity_curve": equity_log,
        "candles_used": len(df),
        "bars_from": int(ts[0]),
        "bars_to": int(ts[-1]),
        # BLOCK-33: выбранный dataset и временные диапазоны train/test окон
        # (None при dataset="full" — сплита не было).
        "dataset": dataset,
        "train_range": train_range,
        "test_range": test_range,
    }