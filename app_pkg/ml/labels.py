# -*- coding: utf-8 -*-
"""Прототип ML-разметки: triple-barrier labeling (Лопес де Прадо).

Подготавливает обучающий датасет из OHLCV (офлайн, без сети): для каждого
бара строятся «тройные барьеры» от цены ЗАКРЫТИЯ (бар входа):

    upper  = close * (1 + k * atr_pct)
    lower  = close * (1 - k * atr_pct)
    horizon = 20 баров  (вертикальный барьер)

Метка бара (числа, пригодные для классификации):
    +1 — верхний барьер достигнут (и начат) ПЕРВЫМ в окне t+1..t+horizon;
    -1 — нижний барьер достигнут первым, либо НА ОДНОМ баре задеты оба
         (порядок касаний внутри бара по OHLC неизвестен — приоритет
         нижнего, та же конвенция, что в бэктестере `_exit_on_bar`);
     0 — ни один барьер не достигнут за horizon (таймаут);
   NaN — будущего меньше horizon (метка невычислима, строка выпадает).

atr_pct — ATR(14)/close на баре входа: барьеры масштабируются к текущей
волатильности, поэтому высоковолатильные активы не получают тривиальных
меток. Если atr_pct <= 0 (плоский ряд) — барьер вырожден, метка = 0.

build_labeled_dataset(...) собирает признаки (по умолчанию — компактный
набор из OHLCV, default_features) + метки и пишет CSV на диск для будущего
обучения модели. Соглашение по «обеим линиям на одном баре» совпадает
с торговым движком — разметка и бэктест смотрят на рынок одинаково.
"""

import os

import numpy as np
import pandas as pd

from app_pkg import utils
from app_pkg.data import market_snapshot
from app_pkg.indicators import _atr, rsi_wilder

DEFAULT_ATR_PERIOD = 14
DEFAULT_K = 2.0
DEFAULT_HORIZON = 20


def _atr_pct(high, low, close, atr_period=DEFAULT_ATR_PERIOD):
    """ATR(period)/close как доля цены (0 там, где close не валиден)."""
    high = pd.Series(np.asarray(high, dtype=float), dtype="float64")
    low = pd.Series(np.asarray(low, dtype=float), dtype="float64")
    close_s = pd.Series(np.asarray(close, dtype=float), dtype="float64")
    atr = _atr(high, low, close_s, atr_period).to_numpy(dtype=float)
    c = close_s.to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.where(c > 0, atr / np.where(c > 0, c, np.nan), 0.0)
    return np.nan_to_num(pct, nan=0.0, posinf=0.0, neginf=0.0)


def triple_barrier_labels(df, k=DEFAULT_K, horizon=DEFAULT_HORIZON,
                          atr_period=DEFAULT_ATR_PERIOD, price_col="close",
                          high_col="high", low_col="low"):
    """Метки {-1, 0, 1, NaN} на каждый бар df (см. докстринг модуля).

    Возвращает (labels, meta):
      labels — np.ndarray (object): события барьеров для каждого бара,
      meta — dict: k/horizon/atr_period, размер и баланс классов.
    """
    meta = {"k": float(k), "horizon": int(horizon),
            "atr_period": int(atr_period)}
    if df is None or df.empty:
        meta.update(n_rows=0, n_computable=0, n_dropped=0,
                    class_balance={-1: 0, 0: 0, 1: 0})
        return np.array([], dtype=object), meta
    close = df[price_col].astype(float).to_numpy()
    high = df[high_col].astype(float).to_numpy()
    low = df[low_col].astype(float).to_numpy()
    n = len(close)
    k = float(k)
    horizon = int(horizon)
    out = np.full(n, np.nan, dtype=object)
    if horizon <= 0 or k <= 0:
        return out, meta

    upper = close * (1.0 + k * _atr_pct(high, low, close, atr_period))
    lower = close * (1.0 - k * _atr_pct(high, low, close, atr_period))
    last = n - horizon - 1  # последний бар, для которого есть 20 барьера-баров

    for t in range(last + 1):
        if lower[t] >= upper[t]:  # вырожденный барьер (atr_pct <= 0)
            out[t] = 0
            continue
        label = None
        for j in range(t + 1, t + horizon + 1):
            low_hit = low[j] <= lower[t]
            up_hit = high[j] >= upper[t]
            if low_hit:       # приоритет нижнего (обе линии на баре -> SL)
                label = -1
                break
            if up_hit:
                label = 1
                break
        out[t] = 0 if label is None else label  # таймаут -> 0

    meta["n_rows"] = n
    computable = max(0, int(last + 1))
    meta["n_computable"] = computable
    meta["n_dropped"] = n - computable
    vals = [int(v) for v in out[:computable] if not np.isnan(v)]
    meta["class_balance"] = {
        -1: vals.count(-1), 0: vals.count(0), 1: vals.count(1)}
    return out, meta


def default_features(df, atr_period=DEFAULT_ATR_PERIOD):
    """Компактный набор признаков из OHLCV (офлайн, без сети).

    Колонки: rsi, atr_pct, bb_pct_b, ema_diff_pct, vol_zscore, roc_5,
    dist_hi_20, dist_lo_20. NaN на прогревных барах (rolling/ewm) остаются
    как NaN — build_labeled_dataset их отбрасывает.

    Это ТОЛЬКО baseline («пол», а не цель): OHLCV сам по себе не
    предсказывает направление — accuracy ~50%. Реальные фичи архитектуры —
    charon_features (снимки market_snapshot).
    """
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    vol = df["volume"].astype(float)
    feat = pd.DataFrame(index=df.index)
    feat["rsi"] = rsi_wilder(close, 14)
    feat["atr_pct"] = _atr(high, low, close, atr_period) / close
    sma20 = close.rolling(20).mean()
    sd20 = close.rolling(20).std().replace(0, np.nan)
    feat["bb_pct_b"] = (close - sma20) / (2.0 * sd20)
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    feat["ema_diff_pct"] = (ema20 - ema50) / ema50
    vol_sma = vol.rolling(20).mean()
    vol_sd = vol.rolling(20).std()
    vol_z = (vol - vol_sma) / vol_sd
    feat["vol_zscore"] = vol_z.where(vol_sd > 0, 0.0)
    feat["roc_5"] = close.pct_change(5)
    feat["dist_hi_20"] = close / high.rolling(20).max() - 1.0
    feat["dist_lo_20"] = close / low.rolling(20).min() - 1.0
    return feat


# ----------------------------------------------- Charon-фичи (реальные)
# X датасета собирается из СНИМКОВ market_snapshot — тех же чисел, что
# видит нейросеть в live (rsi/adx/hurst/di/bb/candle/clock...), а не из
# 8 OHLCV-индикаторов. Все окна блоков <= 100 баров, поэтому причинные
# индикаторы можно посчитать ОДИН раз на полном ряду (значение в баре t
# зависит только от t-100..t и совпадает со «снимком по окну» market_snapshot)
# и досчитать по-барово только дорогие хвосты (hurst, ранг-перцентили,
# автокорреляция, дивергенции). Блоки, требующие сети/БД (context/
# scanner_edge/sentiment/macro/calendar/derivatives/micro/news_sentiment),
# в офлайн-датасет не включаются: missing = no data (семантика v3 схемы).

_SESSION_CODE = {"asia": 0, "london": 1, "london_ny": 2, "ny": 3,
                 "off_hours": 4, "weekend": 5}

_CLOCK_NUM_FIELDS = ("hour_utc", "dow", "london_ny_overlap", "market_open")

# Минимальное число свечей блока (срез короче -> ok:false => NaN-строки).
_CHARON_BLOCK_GATE = {
    "technicals": 25, "volume": 20, "trend": 50, "momentum": 40,
    "volatility": 30, "regime": 55, "divergence": 40, "candle": 2,
}

# Разрядность округления полей (совпадает с _round(...) в market_snapshot).
_CHARON_DIGITS = {
    "technicals": {"rsi": 4, "atr": 8, "bb_pct_b": 4, "sma20_diff_pct": 4,
                   "atr_percentile": 4, "dist_to_high_pct": 4,
                   "dist_to_low_pct": 4},
    "volume": {"rel_vol": 2, "obv_slope": 2, "vol_zscore": 2,
               "vol_percentile": 2, "vwap_dev": 2},
    "trend": {"adx": 1, "plus_di": 1, "minus_di": 1, "di_spread": 1,
              "ema20_50_ratio": 3, "reg_slope_20": 2},
    "momentum": {"macd": 2, "macd_signal": 2, "macd_hist": 2,
                 "macd_hist_slope": 2, "rsi_slope": 2},
    "volatility": {"bb_width": 3, "bb_width_pct": 2, "hv20": 1,
                   "atr_change_pct": 2, "atr_ratio_short_long": 2},
    "regime": {"hurst": 2, "autocorr_lag1": 2, "efficiency_ratio": 2},
    "divergence": {"rsi_price": 0, "macd_price": 0, "obv_price": 0},
    "candle": {"body_ratio": 3, "upper_wick_ratio": 3, "lower_wick_ratio": 3,
               "engulfing": 0, "pinbar": 0},
}


def _rolling_rank_pct(arr, window):
    """Ранговая перцентиль каждого значения в trailing-окне: (доля < + 0.5*==)/w."""
    a = np.asarray(arr, dtype="float64")
    n = len(a)
    out = np.full(n, np.nan)
    if n - window + 1 <= 0:
        return out
    for i in range(window - 1, n):
        hist = a[i - window + 1:i + 1]
        cur = a[i]
        less = float((hist < cur).sum())
        equal = float((hist == cur).sum())
        out[i] = (less + 0.5 * equal) / float(window)
    return out


def _rolling_ols_slope(arr, window=20):
    """Наклон OLS последних window значений (как _slope, без inf/nan-строк)."""
    a = np.asarray(arr, dtype="float64")
    n = len(a)
    out = np.full(n, np.nan)
    if n < window or window < 2:
        return out
    x = np.arange(window, dtype="float64") - (window - 1) / 2.0
    denom = float(np.dot(x, x))
    if not denom:
        return out
    for i in range(window - 1, n):
        y = a[i - window + 1:i + 1]
        out[i] = float(np.dot(x, y - y.mean())) / denom
    return out


def _round_col(arr, digits):
    """Округление как _round(...) в market_snapshot (Python round, None -> NaN).

    np.round и round расходятся на бинарных тью-точках (напр. 0.445): снимок
    округляет именно Python-ом — воспроизводим его по-элементно, чтобы X
    датасета совпадал с live-числами Харона бит в бит.
    """
    fn = np.frompyfunc(lambda v: (
        None if np.isnan(v) or v in (np.inf, -np.inf)
        else round(float(v), digits)), 1, 1)
    return fn(arr).astype("float64")


def charon_features(df, symbol=None, lookback=None):
    """Снимки Харона по ВСЕМ барам df (офлайн), плоский числовой вектор.

    Для каждого бара t воспроизводятся локальные блоки market_snapshot
    ровно как для «последней свечи» в live: technicals, volume, trend,
    momentum, volatility, regime, divergence, candle + clock (сессии от
    времени свечи). Все поля — колонки 'block.field' (числа; None -> NaN).
    Взгляд ТОЛЬКО в прошлое: у бара t нет данных из будущего — метки
    (triple-barrier) не пересекаются с признаками.

    Блоки, требующие сети/БД (context, scanner_edge, sentiment, macro,
    calendar, derivatives, micro, news_sentiment), в офлайн-датасете
    отсутствуют. Прогревные бары (короче минимума блока) дают NaN и
    отбрасываются build_labeled_dataset'ом.

    Передаётся в build_labeled_dataset как features_fn (по умолчанию).
    """
    if df is None or df.empty:
        return pd.DataFrame(dtype="float64")
    df2 = df.reset_index(drop=True)
    n = len(df2)
    has_ts = "timestamp" in df2.columns
    open_ = df2["open"].astype(float).to_numpy(dtype="float64")
    high = df2["high"].astype(float).to_numpy(dtype="float64")
    low = df2["low"].astype(float).to_numpy(dtype="float64")
    close = df2["close"].astype(float).to_numpy(dtype="float64")
    vol = df2["volume"].astype(float).to_numpy(dtype="float64")
    c = pd.Series(close, dtype="float64")

    cols = {name + "." + key: np.full(n, np.nan)
            for name, fields in _CHARON_DIGITS.items() for key in fields}

    def put(name, key, arr):
        cols[name + "." + key] = np.asarray(arr, dtype="float64")

    # ---- причинные индикаторы (полный ряд; окна <= 100 <= lookback)
    idx = np.arange(n, dtype="float64")
    rsi_v = rsi_wilder(c, 14).to_numpy(dtype="float64")
    atr14 = _atr(pd.Series(high, dtype="float64"),
                 pd.Series(low, dtype="float64"), c, 14).to_numpy()
    atr5 = _atr(pd.Series(high, dtype="float64"),
                pd.Series(low, dtype="float64"), c, 5).to_numpy()
    atr20 = _atr(pd.Series(high, dtype="float64"),
                 pd.Series(low, dtype="float64"), c, 20).to_numpy()
    sma20 = c.rolling(20).mean().to_numpy(dtype="float64")
    bb_std = c.rolling(20).std().to_numpy(dtype="float64")
    bb_up = sma20 + 2.0 * bb_std
    bb_low = sma20 - 2.0 * bb_std
    bb_rng = bb_up - bb_low
    width = np.where(np.isfinite(sma20) & (sma20 != 0), bb_rng / sma20, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log(close / np.roll(close, 1))
    lr[0] = np.nan
    ema12 = c.ewm(span=12, adjust=False).mean().to_numpy(dtype="float64")
    ema26 = c.ewm(span=26, adjust=False).mean().to_numpy(dtype="float64")
    macd = ema12 - ema26
    signal = (pd.Series(macd).ewm(span=9, adjust=False).mean()
              .to_numpy(dtype="float64"))
    hist = macd - signal
    direction = np.sign(np.concatenate(([0.0], np.diff(close))))
    obv = np.cumsum(direction * vol)
    ema20 = c.ewm(span=20, adjust=False).mean().to_numpy(dtype="float64")
    ema50 = c.ewm(span=50, adjust=False).mean().to_numpy(dtype="float64")
    pdi, mdi, adx = market_snapshot._di(
        pd.Series(high, dtype="float64"), pd.Series(low, dtype="float64"),
        c, 14)

    # ---- technicals
    put("technicals", "rsi", rsi_v)
    put("technicals", "atr", atr14)
    bb_pct = np.where(bb_rng != 0, (close - bb_low) / bb_rng, np.nan)
    put("technicals", "bb_pct_b", bb_pct)
    put("technicals", "sma20_diff_pct",
        np.where(np.isfinite(sma20) & (sma20 != 0),
                 (close - sma20) / sma20 * 100.0, np.nan))
    put("technicals", "atr_percentile", _rolling_rank_pct(atr14, 100))
    # technicals.close намеренно НЕ включён: абсолютная цена — переоценённый
    # по gain признак без переносимости между датасетами (см. feature-аудит).
    h50 = (pd.Series(high).rolling(50).max().to_numpy(dtype="float64"))
    l50 = (pd.Series(low).rolling(50).min().to_numpy(dtype="float64"))
    put("technicals", "dist_to_high_pct",
        np.where(np.isfinite(h50) & (close != 0),
                 (close - h50) / close * 100.0, np.nan))
    put("technicals", "dist_to_low_pct",
        np.where(np.isfinite(l50) & (close != 0),
                 (close - l50) / close * 100.0, np.nan))

    # ---- volume
    vol_sma20 = (pd.Series(vol).rolling(20).mean().to_numpy(dtype="float64"))
    put("volume", "rel_vol",
        np.where(np.isfinite(vol_sma20) & (vol_sma20 != 0), vol / vol_sma20,
                 np.nan))
    put("volume", "obv_slope", _rolling_ols_slope(obv, 20))
    vol_hist_std = (pd.Series(vol).rolling(100).std(ddof=1)
                    .to_numpy(dtype="float64"))
    vol_hist_mean = (pd.Series(vol).rolling(100).mean().to_numpy(dtype="float64"))
    put("volume", "vol_zscore",
        np.where(vol_hist_std > 0, (vol - vol_hist_mean) / vol_hist_std, np.nan))
    put("volume", "vol_percentile", _rolling_rank_pct(vol, 100))
    tp = (high + low + close) / 3.0
    typ_vol = tp * vol
    cum_tp = (pd.Series(typ_vol).rolling(20).sum().to_numpy(dtype="float64"))
    cum_v = (pd.Series(vol).rolling(20).sum().to_numpy(dtype="float64"))
    vwap20 = np.where(cum_v > 0, cum_tp / np.where(cum_v > 0, cum_v, np.nan),
                      np.nan)
    put("volume", "vwap_dev",
        np.where(np.isfinite(vwap20) & (vwap20 != 0),
                 (close - vwap20) / vwap20 * 100.0, np.nan))

    # ---- trend
    put("trend", "adx", adx)
    put("trend", "plus_di", pdi)
    put("trend", "minus_di", mdi)
    put("trend", "di_spread", pdi - mdi)
    put("trend", "ema20_50_ratio",
        np.where(np.isfinite(ema50) & (ema50 != 0), ema20 / ema50, np.nan))
    slope_c = _rolling_ols_slope(close, 20)
    put("trend", "reg_slope_20",
        np.where(atr14 > 0, slope_c / atr14, slope_c))

    # ---- momentum
    put("momentum", "macd", macd)
    put("momentum", "macd_signal", signal)
    put("momentum", "macd_hist", hist)
    put("momentum", "macd_hist_slope",
        np.concatenate(([np.nan], np.diff(hist))))
    put("momentum", "rsi_slope", np.concatenate(([np.nan], np.diff(rsi_v))))

    # ---- volatility
    put("volatility", "bb_width", width)
    put("volatility", "bb_width_pct", _rolling_rank_pct(width, 100))
    lr_ser = pd.Series(lr)
    hv = lr_ser.rolling(20).std(ddof=1).to_numpy(dtype="float64")
    put("volatility", "hv20", hv * np.sqrt(252.0) * 100.0)
    prev_atr = np.roll(atr14, 1)
    prev_atr[0] = np.nan
    put("volatility", "atr_change_pct",
        np.where(prev_atr != 0, (atr14 - prev_atr) / prev_atr * 100.0, np.nan))
    put("volatility", "atr_ratio_short_long",
        np.where(atr20 > 0, atr5 / atr20, np.nan))

    # ---- regime (lagger per-row хвосты)
    hurst_a = np.full(n, np.nan)
    auto_a = np.full(n, np.nan)
    er_a = np.full(n, np.nan)
    for t in range(max(0, 54), n):
        sl_close = close[max(0, t - 299):t + 1]
        h = market_snapshot._hurst_rs(pd.Series(sl_close), 100)
        if h is not None:
            hurst_a[t] = h
        vals = lr[max(0, t - market_snapshot._LOOKBACK + 1):t + 1]
        lw = vals[~np.isnan(vals)][-100:]
        if lw.size >= 10:
            x = lw[:-1]
            y = lw[1:]
            sx = x.std(ddof=1)
            sy = y.std(ddof=1)
            if sx > 0 and sy > 0:
                sxy = float(((x - x.mean()) * (y - y.mean())).sum())
                auto_a[t] = sxy / ((len(x) - 1) * sx * sy)
        if t >= 20:
            p = close[t - 20:t + 1]
            net = abs(p[-1] - p[0])
            path = float(np.abs(np.diff(p)).sum())
            if path > 0:
                er_a[t] = net / path
    put("regime", "hurst", hurst_a)
    put("regime", "autocorr_lag1", auto_a)
    put("regime", "efficiency_ratio", er_a)

    # ---- divergence (по-барово: локальные экстремумы за 20 баров)
    # Окно как у live-пробы (<= _LOOKBACK баров): иначе O(n^2) на больших df
    # и, что важнее, снимок смотрел бы в несуществующую глубину истории.
    _LB = market_snapshot._LOOKBACK
    div_r = np.full(n, np.nan)
    div_m = np.full(n, np.nan)
    div_o = np.full(n, np.nan)
    for t in range(n):
        lo = max(0, t - _LB + 1)
        if t < 39:
            continue
        div_r[t] = market_snapshot._divergence_combined(close[lo:t + 1],
                                                        rsi_v[lo:t + 1])
        div_m[t] = market_snapshot._divergence_combined(close[lo:t + 1],
                                                        hist[lo:t + 1])
        div_o[t] = market_snapshot._divergence_combined(close[lo:t + 1],
                                                        obv[lo:t + 1])
    put("divergence", "rsi_price", div_r)
    put("divergence", "macd_price", div_m)
    put("divergence", "obv_price", div_o)

    # ---- candle (последняя свеча + предыдущая), доджи -> ratios NaN
    rng = high - low
    body = np.abs(close - open_)
    body_ratio = np.where(rng > 0, body / rng, np.nan)
    put("candle", "body_ratio", body_ratio)
    put("candle", "upper_wick_ratio",
        np.where(rng > 0, (high - np.maximum(open_, close)) / rng, np.nan))
    put("candle", "lower_wick_ratio",
        np.where(rng > 0, (np.minimum(open_, close) - low) / rng, np.nan))
    po = np.roll(open_, 1)
    ph = np.roll(high, 1)
    pl = np.roll(low, 1)
    pc = np.roll(close, 1)
    po[0] = ph[0] = pl[0] = pc[0] = np.nan
    prng = ph - pl
    pbody = pc - po
    b = close - open_
    engulf = np.zeros(n)
    both_rng = np.isfinite(prng) & (prng != 0) & (rng != 0)
    up = both_rng & (b > 0) & (pbody < 0) & (close >= po) & \
        (open_ <= pc) & (np.abs(b) > np.abs(pbody))
    dn = both_rng & (b < 0) & (pbody > 0) & (open_ >= pc) & \
        (close <= po) & (np.abs(b) > np.abs(pbody))
    engulf = np.where(up, 1.0, np.where(dn, -1.0, 0.0))
    put("candle", "engulfing", engulf)
    lw_raw = np.where(rng > 0, (np.minimum(open_, close) - low), np.nan)
    uw_raw = np.where(rng > 0, (high - np.maximum(open_, close)), np.nan)
    pin = np.zeros(n)
    bull_pin = (rng > 0) & (lw_raw >= 2 * body) & (lw_raw >= uw_raw)
    bear_pin = (rng > 0) & (uw_raw >= 2 * body) & (uw_raw >= lw_raw)
    pin = np.where(bull_pin, 1.0, np.where(bear_pin, -1.0, 0.0))
    put("candle", "pinbar", pin)

    # ---- clock (сессии от времени свечи)
    clk = {name: np.full(n, np.nan) for name in _CLOCK_NUM_FIELDS}
    clk["session_code"] = np.full(n, np.nan)
    if has_ts:
        tser = pd.to_datetime(df2["timestamp"])
        if getattr(tser.dt, "tz", None) is None:
            tser = tser.dt.tz_localize("UTC")
        ts_sec = tser.astype("int64").to_numpy() // 10 ** 9
        for t in range(n):
            s = market_snapshot.session_info(int(ts_sec[t]), symbol)
            for f in _CLOCK_NUM_FIELDS:
                clk[f][t] = s[f]
            clk["session_code"][t] = _SESSION_CODE.get(s["session"], np.nan)
        for f, arr in clk.items():
            cols["clock." + f] = arr   # без timestamp столбцы clock не включаем

    # ---- сборка: гейты (прогрев блока), округление (цифры снимка)
    out = pd.DataFrame(index=df2.index)
    for prefix in _CHARON_DIGITS:
        gate = _CHARON_BLOCK_GATE[prefix]
        mask = idx >= gate - 1
        for field in _CHARON_DIGITS[prefix]:
            arr = cols[prefix + "." + field]
            arr = np.where(mask, arr, np.nan)
            arr = _round_col(arr, _CHARON_DIGITS[prefix][field])
            out[prefix + "." + field] = arr
    if has_ts:
        for f in list(_CLOCK_NUM_FIELDS) + ["session_code"]:
            out["clock." + f] = cols["clock." + f]
    return out


def build_labeled_dataset(df, k=DEFAULT_K, horizon=DEFAULT_HORIZON,
                          features_fn=None, out_dir=None,
                          csv_name="labeled_dataset.csv", dropna=True):
    """Собирает датасет (признаки, метки) из OHLCV df и пишет CSV на диск.

    features_fn — callable(df) -> DataFrame признаков; по умолчанию
    charon_features (снимки market_snapshot по каждому бару — реальные
    фичи архитектуры, а не 8 OHLCV-индикаторов; OHLCV-базу можно получить
    явным features_fn=default_features как «пол» для сравнения). Строки
    выравниваются по барам: метке бара t отвечают признаки бара t (метка
    смотрит ТОЛЬКО в будущее, признаки — в прошлое, пересечения нет).

    dropna — отбрасывать строки с NaN-меткой (нет будущего на horizon)
    и NaN-признаками (прогрев блоков/индикаторов). CSV сохраняется с
    колонками признаков + label; в out_dir (None — только вернуть, ничего
    не писать).

    Возвращает (X, y, meta):
      X — DataFrame признаков (label исключена),
      y — np.ndarray меток {-1, 0, 1},
      meta — k/horizon/atr_period, число строк, баланс классов, путь CSV.
    """
    labels, meta = triple_barrier_labels(df, k=k, horizon=horizon)
    feat = (features_fn or charon_features)(df)
    rows = feat.copy()
    rows["label"] = labels
    if dropna:
        rows = rows.dropna(subset=["label"])
        rows = rows.dropna()          # прогревные звон-строки признаков
    meta["n_labeled"] = int(len(rows))
    meta["n_skipped"] = int(meta["n_computable"] - len(rows))
    if not rows.empty:
        counts = rows["label"].value_counts().to_dict()
        meta["class_balance"] = {int(k): int(v) for k, v in counts.items()}
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, csv_name)
        rows.to_csv(path, index_label="row")
        meta["csv"] = path
    y = rows["label"].to_numpy()
    X = rows.drop(columns=["label"])
    return X, y, meta