"""Индикаторы и сериализация данных для графика и ИИ.

Контракт с фронтендом:
    df_to_ohlcv      -> list[{time, open, high, low, close, volume}]
    df_to_indicators -> dict{name: [{time, value}, ...]} — СЛОВАРЬ строго,
                        каждый массив выровнен с df по длине.
    df_to_payload    -> (candles, indicators, last_price)
"""

import logging

import numpy as np
import pandas as pd

from app_pkg import utils

logger = logging.getLogger(__name__)


def rsi_wilder(series, period: int = 14) -> pd.Series:
    """RSI (Wilder) ТОЛЬКО через ewm(alpha=1/period, adjust=False).

    avg_loss == 0  -> 100
    avg_gain == 0  -> 0
    NaN            -> 50
    """
    s = pd.Series(series, dtype="float64")
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()

    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

    rsi = rsi.where(avg_loss != 0, 100.0)                       # avg_loss==0 -> 100
    rsi = rsi.where(~((avg_gain == 0) & (avg_loss != 0)), 0.0)  # avg_gain==0 -> 0
    rsi = rsi.fillna(50.0)                                      # NaN -> 50
    return rsi


def _atr(high, low, close, period: int = 14) -> pd.Series:
    """Average True Range (Wilder smoothing)."""
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def _supertrend(high, low, close, period: int = 10, multiplier: float = 3.0) -> pd.Series:
    """Supertrend (ATR-10, множитель 3.0)."""
    hl2 = (high + low) / 2.0
    atr = _atr(high, low, close, period)
    basic_upper = (hl2 + multiplier * atr).to_numpy(dtype=float)
    basic_lower = (hl2 - multiplier * atr).to_numpy(dtype=float)

    n = len(close)
    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)
    st = np.full(n, np.nan)
    if n == 0:
        return pd.Series(st, index=close.index)

    cl = close.to_numpy(dtype=float)
    final_upper[0] = basic_upper[0]
    final_lower[0] = basic_lower[0]
    st[0] = final_lower[0]
    for i in range(1, n):
        prev_upper = final_upper[i - 1]
        prev_lower = final_lower[i - 1]
        prev_close = cl[i - 1]
        final_upper[i] = basic_upper[i] if (
            np.isnan(prev_upper) or basic_upper[i] < prev_upper or prev_close > prev_upper
        ) else prev_upper
        final_lower[i] = basic_lower[i] if (
            np.isnan(prev_lower) or basic_lower[i] > prev_lower or prev_close < prev_lower
        ) else prev_lower
        st[i] = final_upper[i] if cl[i] <= final_upper[i] else final_lower[i]
    return pd.Series(st, index=close.index)


def _adx(high, low, close, period: int = 14) -> pd.Series:
    """ADX (Wilder, 14)."""
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)

    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()

    w = lambda s: s.ewm(alpha=1.0 / period, adjust=False).mean()
    plus_di = 100.0 * w(plus_dm) / atr
    minus_di = 100.0 * w(minus_dm) / atr
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    dx = dx.replace([np.inf, -np.inf], np.nan)
    return dx.ewm(alpha=1.0 / period, adjust=False).mean()

# ------------------------------------------------------------ индикаторы
_IND_COLUMNS = (
    "sma20", "sma50", "ema50", "bb_up", "bb_mid", "bb_low", "rsi",
    "macd", "macd_signal", "macd_hist", "vwap", "supertrend",
    "stoch_k", "stoch_d", "adx", "cci", "obv", "pivot",
)


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Полный набор индикаторов, выровненный по df (index совпадает)."""
    out = pd.DataFrame(index=df.index)
    for col in _IND_COLUMNS:
        out[col] = np.nan
    if df is None or df.empty:
        return out

    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)

    out["sma20"] = close.rolling(20).mean()
    out["sma50"] = close.rolling(50).mean()
    out["ema50"] = close.ewm(span=50, adjust=False).mean()

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    out["bb_mid"] = bb_mid
    out["bb_up"] = bb_mid + 2.0 * bb_std
    out["bb_low"] = bb_mid - 2.0 * bb_std

    out["rsi"] = rsi_wilder(close, 14)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    tp = (high + low + close) / 3.0
    cum_vol = volume.cumsum().replace(0, np.nan)
    out["vwap"] = (tp * volume).cumsum() / cum_vol

    out["supertrend"] = _supertrend(high, low, close, 10, 3.0)

    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    rng = (high14 - low14).replace(0, np.nan)
    out["stoch_k"] = 100.0 * (close - low14) / rng
    out["stoch_d"] = out["stoch_k"].rolling(3).mean()

    out["adx"] = _adx(high, low, close, 14)

    tp_sma20 = tp.rolling(20).mean()
    md = tp.rolling(20).apply(
        lambda x: float(np.mean(np.abs(x - np.mean(x)))), raw=True)
    denom = (0.015 * md).replace(0, np.nan)
    out["cci"] = (tp - tp_sma20) / denom

    direction = np.sign(close.diff().fillna(0))
    out["obv"] = (direction * volume).cumsum()

    out["pivot"] = tp
    return out


def df_to_ohlcv(df: pd.DataFrame) -> list:
    """Candles для фронтенда: list[{time, open, high, low, close, volume}]."""
    if df is None or df.empty:
        return []
    ts = utils.epoch_secs(df["timestamp"])
    rows = []
    for i in range(len(df)):
        rows.append({
            "time": int(ts[i]),
            "open": utils._clean(df["open"].iloc[i]),
            "high": utils._clean(df["high"].iloc[i]),
            "low": utils._clean(df["low"].iloc[i]),
            "close": utils._clean(df["close"].iloc[i]),
            "volume": utils._clean(df["volume"].iloc[i]),
        })
    return rows


def df_last_candle(df: pd.DataFrame):
    """Последняя свеча в формате фронта {time, open, high, low, close, volume}.

    time — int Unix-секунды. Пустой df/None -> None (без исключений).
    Используется в /api/last-bar и в live-пуше (app_pkg.data.live).
    """
    if df is None or df.empty:
        return None
    ts = utils.epoch_secs(df["timestamp"].iloc[-1:])
    last = df.iloc[-1]
    return {
        "time": int(ts[0]),
        "open": utils._clean(last["open"]),
        "high": utils._clean(last["high"]),
        "low": utils._clean(last["low"]),
        "close": utils._clean(last["close"]),
        "volume": utils._clean(last["volume"]),
    }


def df_to_indicators(df: pd.DataFrame, ind: pd.DataFrame) -> dict:
    """СЛОВАРЬ индикаторов: {name: [{time, value}, ...]}.

    Каждый массив выровнен с df по длине. ФРОНТ ЖДЁТ ИМЕННО ЭТО —
    если вернётся список — все индикаторы на фронте сломаются.
    """
    result = {}
    if df is None or df.empty:
        return result
    ts = utils.epoch_secs(df["timestamp"])
    for col in ind.columns:
        series = ind[col]
        arr = []
        for i in range(len(df)):
            if i < len(series):
                arr.append({"time": int(ts[i]), "value": utils._clean(series.iloc[i])})
            else:
                arr.append({"time": int(ts[i]), "value": None})
        result[col] = arr
    return result


def df_to_payload(df: pd.DataFrame):
    """(candles, indicators, last_price) одним вызовом."""
    if df is None or df.empty:
        return [], {}, None
    candles = df_to_ohlcv(df)
    ind = compute_indicators(df)
    indicators = df_to_indicators(df, ind)
    last_price = utils._clean(float(df["close"].iloc[-1]))
    return candles, indicators, last_price


def slice_payload(df: pd.DataFrame, upto_index=None):
    """Payload по df, обрезанному до upto_index (для replay).

    Возвращает (candles, indicators, last_price, total). Не мутирует df.
    """
    if df is None or df.empty:
        return [], {}, None, 0
    df = df.copy()
    if upto_index is not None:
        upto_index = int(upto_index)
        upto_index = max(0, min(upto_index, len(df) - 1))
        df = df.iloc[:upto_index + 1]
    candles, indicators, last_price = df_to_payload(df)
    return candles, indicators, last_price, len(candles)