# -*- coding: utf-8 -*-
"""Сырые рыночные данные для ИИ (Market Snapshot).

Единственный источник правды для нейросети: НИКАКОГО текста, только числа.
Нейросеть не получает «словесное описание рынка» — она получает этот JSON
и делает выводы по формулам системного промпта.

get_raw_market_data(symbol, timeframe, upto_sec=None, ts_override=None) -> dict:
    technicals   — RSI(14), ATR(14), BB %B, SMA20 diff %, ATR-перцентиль,
                   дистанции до 50-барных экстремумов, последняя цена;
    volume       — rel_vol/obv_slope/vol_zscore/vol_percentile/vwap_dev (локально по OHLCV);
    trend        — adx/plus_di/minus_di/di_spread/ema20_50_ratio/reg_slope_20;
    momentum     — macd/macd_signal/macd_hist/macd_hist_slope/rsi_slope;
    volatility   — bb_width/bb_width_pct/hv20/atr_change_pct/atr_ratio_short_long;
    regime       — hurst/autocorr_lag1/efficiency_ratio (режим рынка);
    divergence   — rsi_price/macd_price/obv_price (дивергенции цены против
                   индикатора за 20 баров: -1 bearish / +1 bullish / 0 none);
    candle       — body_ratio/upper_wick_ratio/lower_wick_ratio/engulfing/pinbar
                   по последней свече (feature, не правило);
    context      — corr_btc_30 + eth_btc_corr_30 (крипта, локально по OHLCV
                   BTCUSDT/ETHUSDT) и null-заглушки
                   btc_dominance/corr_dxy_30/beta_index/index_return_1d/index_rsi;
    scanner_edge — лучшая комбинация сканера
                   (winrate/sharpe/max_dd/profit_factor/trades/test_sharpe/params);
    sentiment    — crowd: ls_ratio/long_pct/short_pct/taker_buy_sell/fear_greed
                   (крипта) или null-схема {ls_ratio, long_pct, fear_greed: null,
                   ok: false} при ошибке/не-крипте;
    macro        — us10y/fed_rate/dxy (FRED) — форекс/все активы;
    calendar     — экономический календарь (Finnhub) для валюты пары;
    derivatives  — open_interest/oi_change_1h_pct/funding_rate/basis_pct/
                   top_ls_ratio (крипта);
    micro        — микроструктура стакана (Binance spot, крипта):
                   spread_norm/ob_imb_10/bid_sum_10/ask_sum_10/large_trades_ratio;
    news_sentiment — lexicon-скоринг заголовков ленты актива (без LLM);
    clock        — сессии по времени ПОСЛЕДНЕЙ свечи среза (реплей-безопасно,
                   не по системным часам), кроме случаев ts_override для тестов.

Единое правило блоков: при ошибке/отсутствии источника поля = null и
"ok": false — без exception и без опасных 0/50-заглушек; "не применимо"
к активу ({} от геттера) -> {"applicable": false, "ok": false}.

Единая шкала значений: ВСЕ процентные/долевые поля — в долях 0-1 (long_pct,
short_pct, winrate, max_dd и все новые *_pct); fear_greed из crowd (0-100)
и taker_buy_sell из crowd (binance buySellRatio, ratio -> r/(1+r)) приводятся
к 0-1 уже в get_raw_market_data (копия, без мутации кеша геттера).
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from app_pkg import config, db, utils
from app_pkg.data.derivatives import get_derivatives_snapshot
from app_pkg.data.fetch import get_replay_df, get_series_df
from app_pkg.data.macro import get_econ_calendar, get_macro_snapshot
from app_pkg.data.micro import get_micro_snapshot
from app_pkg.data.news_sentiment import get_news_sentiment
from app_pkg.data.sentiment import get_crowd_snapshot
from app_pkg.indicators import _atr, rsi_wilder

log = logging.getLogger(__name__)

# Периоды индикаторов (фиксированные — единый контракт для ИИ).
_RSI_PERIOD = 14
_ATR_PERIOD = 14
_BB_PERIOD = 20
_BB_STD = 2.0
_SMA_PERIOD = 20
# Свечей на расчёт: ~стабильный хвост RSI/BB/SMA без "прогрева" первых баров.
_LOOKBACK = 300
# Распределение ATR для перцентиля волатильности (Уровень 3).
_ATR_WINDOW = 100
# Окно экстремумов для дистанций close (Уровень 3).
_LEVELS_WINDOW = 50

# Доп. блоки (sentiment/macro/calendar/derivatives/news_sentiment) тянутся
# параллельно отдельным пулом с жёстким таймаутом: медленный/отвалившийся
# источник деградирует в null-схему своего блока, но НЕ держит снимок.
_EXTRA_BLOCK_WORKERS = 5
_EXTRA_BLOCK_TIMEOUT_SEC = 15.0
_block_pool = ThreadPoolExecutor(max_workers=_EXTRA_BLOCK_WORKERS,
                                 thread_name_prefix="snapshot-blocks")

# Null-схема sentiment (контракт снимка: ровно эти 4 поля).
_SENTIMENT_NULL_BLOCK = {
    "ls_ratio": None, "long_pct": None, "fear_greed": None, "ok": False,
}


def _round(v, digits=6):
    """float-значение или None (NaN/Inf/нечисло не должны попасть в JSON)."""
    v = utils._clean(v)
    return round(v, digits) if v is not None else None


def _slice_df(symbol, timeframe, upto_sec=None):
    """Срез свечей для снимка с барьером реплея upto_sec (None в live).

    В реплее берём ИСТОРИЧЕСКОЕ окно, заканчивающееся на барьере
    (get_replay_df): live-хвост get_series_df не содержит старых баров, и
    барьер старше окна давал пустой снимок.
    """
    if upto_sec is not None:
        df = get_replay_df(symbol, timeframe, to_sec=float(upto_sec),
                           limit=_LOOKBACK)
    else:
        df = get_series_df(symbol, timeframe, limit=_LOOKBACK)
    if df is None or df.empty:
        return df
    if upto_sec is not None:
        mask = df["timestamp"] <= pd.to_datetime(int(upto_sec), unit="s",
                                                 utc=True)
        df = df[mask]
    return df


def _atr_percentile(df):
    """Перцентиль текущего ATR(14) в распределении за последние 100 баров.

    Ранговая (fractional) перцентиль: (доля строго ниже + 0.5 * равных) / N,
    поэтому константная волатильность даёт ~0.5, растущая — ближе к 1.0.
    Меньше 100 баров или нечисловой ряд -> None (без exception).
    """
    if df is None or len(df) < _ATR_WINDOW:
        return None
    atr = _atr(df["high"], df["low"], df["close"], _ATR_PERIOD)
    hist = atr.tail(_ATR_WINDOW).dropna()
    if len(hist) < _ATR_WINDOW:
        return None
    cur = utils._clean(hist.iloc[-1])
    if cur is None:
        return None
    less = float((hist < cur).sum())
    equal = float((hist == cur).sum())
    return _round((less + 0.5 * equal) / float(len(hist)), 4)


def _dist_to_extremes(df):
    """Дистанции close до 50-барных экстремумов, в % от цены close.

    dist_to_high_pct = (close - max_high_50) / close * 100 (обычно <= 0);
    dist_to_low_pct  = (close - min_low_50) / close * 100 (обычно >= 0).
    Нормировка на close, а не на сам экстремум — фича = удалённость цены
    от уровня в процентах от текущей цены.
    Меньше 50 баров или нулевой экстремум -> pair (None, None).
    """
    if df is None or len(df) < _LEVELS_WINDOW:
        return None, None
    last_close = _round(df["close"].iloc[-1], 8)
    if last_close is None:
        return None, None
    high50 = utils._clean(df["high"].tail(_LEVELS_WINDOW).max())
    low50 = utils._clean(df["low"].tail(_LEVELS_WINDOW).min())
    if not high50 or not low50 or not last_close:
        return None, None
    dist_high = _round((last_close - high50) / last_close * 100.0, 4)
    dist_low = _round((last_close - low50) / last_close * 100.0, 4)
    return dist_high, dist_low


def _technicals_blank():
    """Null-схема technicals при отсутствии данных (ok: false, без 0)."""
    return {
        "rsi": None, "atr": None, "bb_pct_b": None,
        "sma20_diff_pct": None, "close": None,
        "atr_percentile": None, "dist_to_high_pct": None,
        "dist_to_low_pct": None, "ok": False,
    }


def _technicals_from_df(df):
    """Индикаторы по последним свечам: RSI, ATR, BB %B, SMA20 diff %.

    df — УЖЕ обрезанный срез (_slice_df: барьер реплея). Все значения —
    числа (None, если данных меньше необходимого минимума); при отсутствии
    данных — null-схема с "ok": false.
    """
    if df is None or df.empty:
        return _technicals_blank()
    if len(df) < max(_RSI_PERIOD, _BB_PERIOD, _SMA_PERIOD) + 5:
        return _technicals_blank()

    close = pd.Series(df["close"], dtype="float64")
    last_close = _round(close.iloc[-1], 8)

    rsi = _round(rsi_wilder(close, _RSI_PERIOD).iloc[-1], 4)
    atr = _round(_atr(df["high"], df["low"], df["close"],
                      _ATR_PERIOD).iloc[-1], 8)

    sma20 = close.rolling(_SMA_PERIOD).mean()
    bb_mid = sma20
    bb_std = close.rolling(_BB_PERIOD).std()
    bb_up = bb_mid + _BB_STD * bb_std
    bb_low = bb_mid - _BB_STD * bb_std
    bb_range = utils._clean((bb_up - bb_low).iloc[-1])
    bb_pct_b = None
    if bb_range:
        bb_pct_b = _round((last_close - bb_low.iloc[-1]) / bb_range, 4)

    sma20_last = utils._clean(sma20.iloc[-1])
    sma20_diff_pct = None
    if sma20_last and last_close:
        sma20_diff_pct = _round((last_close - sma20_last) / sma20_last * 100.0,
                                4)

    atr_percentile = _atr_percentile(df)
    dist_high, dist_low = _dist_to_extremes(df)

    return {
        "rsi": rsi, "atr": atr, "bb_pct_b": bb_pct_b,
        "sma20_diff_pct": sma20_diff_pct, "close": last_close,
        "atr_percentile": atr_percentile,
        "dist_to_high_pct": dist_high, "dist_to_low_pct": dist_low,
        "ok": True,
    }


# ----------------------------------------------- структурные блоки (Уровень 4)
# 6 блоков считаются ЛОКАЛЬНО из OHLCV-среза (без внешних API). Единые
# контракты: доля/перцентиль *_pct — в 0-1; "ok": false при пустом/малом df.

def _slope(series, window):
    """Наклон OLS (простой МНК) последних window значений, без scipy."""
    try:
        y = pd.Series(series, dtype="float64")
        y = y.replace([np.inf, -np.inf], np.nan).dropna()
        if len(y) < 2:
            return None
        y = y.tail(window)
        if len(y) < 2:
            return None
        x = np.arange(len(y), dtype="float64") - (len(y) - 1) / 2.0
        denom = float(np.dot(x, x))
        if not denom:
            return None
        return float(np.dot(x, y.to_numpy(dtype="float64")
                            - y.to_numpy(dtype="float64").mean()) / denom)
    except Exception:  # noqa: BLE001 — расчётный хелпер
        return None


def _rank_pct(hist, cur):
    """Ранговая перцентиль cur в историческом ряду: (доля < + 0.5*f ==) / N."""
    if cur is None or len(hist) < 2:
        return None
    less = float((hist < cur).sum())
    equal = float((hist == cur).sum())
    return (less + 0.5 * equal) / float(len(hist))


def _di(high, low, close, period=14):
    """plus_di / minus_di / adx (Wilder, период) как три pd.Series."""
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0),
                        index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0),
                         index=low.index)
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()

    def wmean(s):
        return s.ewm(alpha=1.0 / period, adjust=False).mean()

    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = 100.0 * wmean(plus_dm) / atr
        minus_di = 100.0 * wmean(minus_dm) / atr
        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    dx = dx.replace([np.inf, -np.inf], np.nan)
    adx = dx.ewm(alpha=1.0 / period, adjust=False).mean()
    return plus_di, minus_di, adx


def _hurst_rs(series, window=100):
    """Показатель Хёрста R/S (log-income ряда) по последним window барам."""
    try:
        y = pd.Series(series, dtype="float64")
        y = y.replace([np.inf, -np.inf], np.nan).dropna()
        if len(y) < 50:
            return None
        clipped = np.clip(y.to_numpy(dtype="float64"), a_min=1e-12, a_max=None)
        r = np.diff(np.log(clipped))
        r = r[:window]
        n = len(r)
        if n < 50:
            return None
        lags = range(2, min(n // 2, 100) + 1)
        xs, tau = [], []
        with np.errstate(divide="ignore", invalid="ignore"):
            for lag in lags:
                n_chunks = n // lag
                if n_chunks < 1:
                    continue
                chunk = r[:n_chunks * lag].reshape(n_chunks, lag)
                dev = np.cumsum(chunk - chunk.mean(axis=1, keepdims=True), axis=1)
                rng = dev.max(axis=1) - dev.min(axis=1)
                std = chunk.std(axis=1)
                mask = std > 0
                if not mask.any():  # весь чанк константен — лаг не информативен
                    continue
                rs = (rng[mask] / std[mask]).mean()
                if rs is not None and rs > 0:
                    xs.append(float(np.log(lag)))
                    tau.append(float(np.log(rs)))
        if len(xs) < 3:
            return None
        return float(np.polyfit(xs, tau, 1)[0])
    except Exception:  # noqa: BLE001 — расчётный хелпер
        return None


def _corr_of_returns(df_a, df_b, window=30):
    """Корреляция доходностей двух DF по общим таймстампам (последние window)."""
    try:
        a = df_a.set_index("timestamp")["close"].astype(float)
        b = df_b.set_index("timestamp")["close"].astype(float)
        joined = pd.concat([a.rename("a"), b.rename("b")], axis=1,
                           join="inner").dropna()
        if len(joined) < 3:
            return None
        joined = joined.tail(max(3, window))
        ra = joined["a"].pct_change()
        rb = joined["b"].pct_change()
        pairs = pd.concat([ra.rename("a"), rb.rename("b")], axis=1).dropna()
        if len(pairs) < 3:
            return None
        if pairs["a"].std() == 0.0 or pairs["b"].std() == 0.0:
            return None  # константный ряд -> корреляция не определена
        c = float(pairs["a"].corr(pairs["b"]))
        return None if np.isnan(c) else c
    except Exception:  # noqa: BLE001 — расчётный хелпер
        return None


def _volume_blank():
    return {"rel_vol": None, "obv_slope": None, "vol_zscore": None,
            "vol_percentile": None, "vwap_dev": None, "ok": False}


def _volume_block(df):
    """Объём: rel_vol, OBV-наклон, z-score, перцентиль, отклонение от VWAP20."""
    if df is None or df.empty:
        return _volume_blank()
    try:
        if len(df) < 20:
            return _volume_blank()
        vol = pd.Series(df["volume"], dtype="float64")
        close = pd.Series(df["close"], dtype="float64")
        last_vol = utils._clean(vol.iloc[-1])
        sma20_vol = utils._clean(vol.rolling(20).mean().iloc[-1])
        rel_vol = None
        if last_vol is not None and sma20_vol:
            rel_vol = _round(last_vol / sma20_vol, 2)

        direction = np.sign(close.diff().fillna(0))
        obv = (direction * vol).cumsum()
        obv_slope = _round(_slope(obv, 20), 2)

        vol_zscore = None
        vol_percentile = None
        if len(df) >= 100:
            hist = vol.tail(100).replace([np.inf, -np.inf], np.nan).dropna()
            if len(hist) >= 20 and last_vol is not None:
                sd = float(hist.std(ddof=1))
                if sd and sd > 0:
                    vol_zscore = _round(
                        (last_vol - float(hist.mean())) / sd, 2)
                vol_percentile = _round(_rank_pct(hist, last_vol), 2)

        vwap_dev = None
        seg = df.tail(20)
        tp = ((seg["high"].astype(float) + seg["low"].astype(float)
               + seg["close"].astype(float)) / 3.0).to_numpy(dtype="float64")
        v20 = seg["volume"].astype(float).to_numpy(dtype="float64")
        cum_vol = np.cumsum(v20)
        last_close = utils._clean(close.iloc[-1])
        if cum_vol[-1] > 0 and last_close is not None:
            vwap20 = float(np.cumsum(tp * v20)[-1] / cum_vol[-1])
            if vwap20 > 0:
                vwap_dev = _round((last_close - vwap20) / vwap20 * 100.0, 2)

        return {"rel_vol": rel_vol, "obv_slope": obv_slope,
                "vol_zscore": vol_zscore, "vol_percentile": vol_percentile,
                "vwap_dev": vwap_dev, "ok": True}
    except Exception as exc:  # noqa: BLE001
        log.warning("volume block failed: %s", exc)
        return _volume_blank()


def _trend_blank():
    return {"adx": None, "plus_di": None, "minus_di": None,
            "di_spread": None, "ema20_50_ratio": None, "reg_slope_20": None,
            "ok": False}


def _trend_block(df):
    """Сила тренда: ADX/DM (14), EMA20/50-спред, регресс-наклон на ATR."""
    if df is None or df.empty:
        return _trend_blank()
    try:
        if len(df) < 50:
            return _trend_blank()
        high = pd.Series(df["high"], dtype="float64")
        low = pd.Series(df["low"], dtype="float64")
        close = pd.Series(df["close"], dtype="float64")
        pdi, mdi, adx_s = _di(high, low, close, 14)
        adx = _round(utils._clean(adx_s.iloc[-1]), 1)
        plus_di = _round(utils._clean(pdi.iloc[-1]), 1)
        minus_di = _round(utils._clean(mdi.iloc[-1]), 1)
        di_spread = None
        if plus_di is not None and minus_di is not None:
            di_spread = _round(plus_di - minus_di, 1)

        ema20 = utils._clean(close.ewm(span=20, adjust=False).mean().iloc[-1])
        ema50 = utils._clean(close.ewm(span=50, adjust=False).mean().iloc[-1])
        ema20_50_ratio = None
        if ema20 is not None and ema50:
            ema20_50_ratio = _round(ema20 / ema50, 3)

        atr_last = utils._clean(_atr(high, low, close, 14).iloc[-1])
        slope = _slope(close, 20)
        reg_slope_20 = None
        if slope is not None and atr_last and atr_last > 0:
            reg_slope_20 = _round(slope / atr_last, 2)
        elif slope is not None:
            reg_slope_20 = _round(slope, 2)

        return {"adx": adx, "plus_di": plus_di, "minus_di": minus_di,
                "di_spread": di_spread, "ema20_50_ratio": ema20_50_ratio,
                "reg_slope_20": reg_slope_20, "ok": True}
    except Exception as exc:  # noqa: BLE001
        log.warning("trend block failed: %s", exc)
        return _trend_blank()


def _momentum_blank():
    return {"macd": None, "macd_signal": None, "macd_hist": None,
            "macd_hist_slope": None, "rsi_slope": None, "ok": False}


def _momentum_block(df):
    """Импульс: MACD (12,26,9), наклон гистограммы и RSI."""
    if df is None or df.empty:
        return _momentum_blank()
    try:
        if len(df) < 40:
            return _momentum_blank()
        close = pd.Series(df["close"], dtype="float64")
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal

        h_now = utils._clean(hist.iloc[-1])
        h_prev = utils._clean(hist.iloc[-2])
        macd_hist_slope = None
        if h_now is not None and h_prev is not None:
            macd_hist_slope = _round(h_now - h_prev, 2)

        rsi = rsi_wilder(close, 14)
        r_now = utils._clean(rsi.iloc[-1])
        r_prev = utils._clean(rsi.iloc[-2])
        rsi_slope = None
        if r_now is not None and r_prev is not None:
            rsi_slope = _round(r_now - r_prev, 2)

        return {"macd": _round(utils._clean(macd.iloc[-1]), 2),
                "macd_signal": _round(utils._clean(signal.iloc[-1]), 2),
                "macd_hist": _round(utils._clean(hist.iloc[-1]), 2),
                "macd_hist_slope": macd_hist_slope,
                "rsi_slope": rsi_slope,
                "ok": True}
    except Exception as exc:  # noqa: BLE001
        log.warning("momentum block failed: %s", exc)
        return _momentum_blank()


def _volatility_blank():
    return {"bb_width": None, "bb_width_pct": None, "hv20": None,
            "atr_change_pct": None, "atr_ratio_short_long": None, "ok": False}


def _volatility_block(df):
    """Динамика волатильности: BB-ширина, HV20, прирост/спред ATR."""
    if df is None or df.empty:
        return _volatility_blank()
    try:
        if len(df) < 30:
            return _volatility_blank()
        high = pd.Series(df["high"], dtype="float64")
        low = pd.Series(df["low"], dtype="float64")
        close = pd.Series(df["close"], dtype="float64")
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_up = bb_mid + 2.0 * bb_std
        bb_low = bb_mid - 2.0 * bb_std

        width = (bb_up - bb_low) / bb_mid
        width_cur = utils._clean(width.iloc[-1])
        bb_width = _round(width_cur, 3) if width_cur is not None else None
        bb_width_pct = None
        if len(df) >= 100:
            hist = width.replace([np.inf, -np.inf], np.nan).dropna().tail(100)
            if len(hist) >= 20:
                bb_width_pct = _round(_rank_pct(hist, width_cur), 2)

        lr = np.log(close / close.shift(1))
        lr = lr.replace([np.inf, -np.inf], np.nan).dropna().tail(20)
        hv20 = None
        if len(lr) >= 5:
            hv20 = _round(float(lr.std(ddof=1)) * np.sqrt(252.0) * 100.0, 1)

        atr14 = _atr(high, low, close, 14)
        a_now = utils._clean(atr14.iloc[-1])
        a_prev = utils._clean(atr14.iloc[-2])
        atr_change_pct = None
        if a_now is not None and a_prev is not None and a_prev:
            atr_change_pct = _round((a_now - a_prev) / a_prev * 100.0, 2)

        atr5 = utils._clean(_atr(high, low, close, 5).iloc[-1])
        atr20 = utils._clean(_atr(high, low, close, 20).iloc[-1])
        atr_ratio = None
        if atr5 is not None and atr20:
            atr_ratio = _round(atr5 / atr20, 2)

        return {"bb_width": bb_width, "bb_width_pct": bb_width_pct,
                "hv20": hv20, "atr_change_pct": atr_change_pct,
                "atr_ratio_short_long": atr_ratio, "ok": True}
    except Exception as exc:  # noqa: BLE001
        log.warning("volatility block failed: %s", exc)
        return _volatility_blank()


def _regime_blank():
    return {"hurst": None, "autocorr_lag1": None, "efficiency_ratio": None,
            "ok": False}


def _regime_block(df):
    """Режим рынка: Хёрст (R/S), автокорреляция lag1, эффективность (Кауфман)."""
    if df is None or df.empty:
        return _regime_blank()
    try:
        if len(df) < 55:
            return _regime_blank()
        close = pd.Series(df["close"], dtype="float64")
        hurst = _round(_hurst_rs(close, 100), 2)

        lr = np.log(close / close.shift(1))
        lr = lr.replace([np.inf, -np.inf], np.nan).dropna().tail(100)
        autocorr_lag1 = None
        if len(lr) >= 10:
            with np.errstate(divide="ignore", invalid="ignore"):
                x = lr.to_numpy(dtype="float64")[:-1]
                y = lr.to_numpy(dtype="float64")[1:]
                sx, sy = float(x.std(ddof=1)), float(y.std(ddof=1))
                if sx and sy and sx > 0 and sy > 0:
                    sxy = float(((x - x.mean()) * (y - y.mean())).sum())
                    ac = sxy / ((len(x) - 1) * sx * sy)
                    autocorr_lag1 = _round(ac, 2)

        prices = close.tail(21).replace([np.inf, -np.inf], np.nan).dropna()
        er = None
        if len(prices) >= 21:
            net = float(abs(prices.iloc[-1] - prices.iloc[0]))
            path = float(prices.diff().abs().sum())
            if path and path > 0:
                er = _round(net / path, 2)

        return {"hurst": hurst, "autocorr_lag1": autocorr_lag1,
                "efficiency_ratio": er, "ok": True}
    except Exception as exc:  # noqa: BLE001
        log.warning("regime block failed: %s", exc)
        return _regime_blank()


# ------------------------------------------- дивергенции price vs индикатор
# Сравнение двух последних экстремумов (same-side) цены и индикатора за окно:
# цена делает новый high, а индикатор нет -> bearish (-1); новый low без
# подтверждения индикатора -> bullish (+1). 0 = расхождения нет.

_DIV_WINDOW = 20
_DIV_NEIGHBOR = 2


def _extrema_indexes(vals, side, k=_DIV_NEIGHBOR):
    """Индексы локальных пиков/впадин (в пределах ±k баров)."""
    n = len(vals)
    idx = []
    for i in range(1, n - 1):
        seg = vals[max(0, i - k):min(n, i + k + 1)]
        m = np.max(seg) if side == "high" else np.min(seg)
        if vals[i] == m:
            idx.append(i)
    return idx


def _divergence_signal(closes, indicator, side, window=_DIV_WINDOW):
    """Расхождение последних двух экстремумов side: -1/0/1 (чистая функция).

    side="high": цена выше-побьёт, индикатор ниже-прогуляется -> -1 (bearish);
    side="low":  цена ниже-побьёт, индикатор выше-прогуляется -> +1 (bullish).
    """
    c = np.asarray(closes, dtype="float64")[-window:]
    v = np.asarray(indicator, dtype="float64")[-window:]
    if len(c) < 10 or len(v) < 10:
        return 0
    idx = _extrema_indexes(v, side)
    if len(idx) < 2:
        return 0
    i1, i2 = idx[-2], idx[-1]
    if i1 == i2:
        return 0
    if side == "high":
        # цена обновляет максимум, а индикатор нет -> бычий импульс слабеет
        return -1 if (c[i1] < c[i2] and v[i1] > v[i2]) else 0
    # side == "low": цена бьёт минимум, индикатор уже не бьёт -> бычья дива
    return 1 if (c[i1] > c[i2] and v[i1] < v[i2]) else 0


def _divergence_combined(closes, indicator, window=_DIV_WINDOW):
    """-1 bearish / +1 bullish / 0 none. Бычья приоритетнее при конфликте."""
    bull = _divergence_signal(closes, indicator, "low", window)
    bear = _divergence_signal(closes, indicator, "high", window)
    if bull == 1:
        return 1
    if bear == -1:
        return -1
    return 0


def _divergence_blank():
    return {"rsi_price": None, "macd_price": None, "obv_price": None,
            "ok": False}


def _divergence_block(df):
    """Дивергенции RSI/MACD-hist/OBV против цены (последние 20 баров).

    Значения: -1 bearish, +1 bullish, 0 no divergence. Меньше 40 баров
    (нестабильные RSI/MACD) -> null-схема с ok:false.
    """
    if df is None or df.empty:
        return _divergence_blank()
    try:
        if len(df) < 40:
            return _divergence_blank()
        close = pd.Series(df["close"], dtype="float64")
        rsi = rsi_wilder(close, _RSI_PERIOD)

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        hist = ema12 - ema26 - (ema12 - ema26).ewm(span=9,
                                                   adjust=False).mean()

        vol = pd.Series(df["volume"], dtype="float64")
        direction = np.sign(close.diff().fillna(0))
        obv = (direction * vol).cumsum()

        return {
            "rsi_price": _divergence_combined(close, rsi),
            "macd_price": _divergence_combined(close, hist),
            "obv_price": _divergence_combined(close, obv),
            "ok": True,
        }
    except Exception as exc:  # noqa: BLE001
        log.warning("divergence block failed: %s", exc)
        return _divergence_blank()


def _candle_blank():
    return {"body_ratio": None, "upper_wick_ratio": None,
            "lower_wick_ratio": None, "engulfing": None, "pinbar": None,
            "ok": False}


def _candle_block(df):
    """Свечные паттерны ПОСЛЕДНЕГО бара (feature, не правило).

    body_ratio = |close-open|/range; upper/lower_wick_ratio = доля фитилей;
    engulfing = +1 бычье поглощение, -1 медвежье, 0 нет; pinbar = +1 бычий
    пин, -1 медвежий, 0 нет. Нулевой диапазон -> ratios None (доджи не врём).
    """
    if df is None or df.empty:
        return _candle_blank()
    try:
        if len(df) < 2:
            return _candle_blank()
        last = df.iloc[-1]
        prev = df.iloc[-2]
        o = utils._clean(last["open"])
        h = utils._clean(last["high"])
        l = utils._clean(last["low"])
        c = utils._clean(last["close"])
        if None in (o, h, l, c):
            return _candle_blank()
        rng = h - l
        body_ratio = None
        upper_wick_ratio = None
        lower_wick_ratio = None
        if rng and rng > 0:
            body_ratio = _round(abs(c - o) / rng, 3)
            upper_wick_ratio = _round((h - max(o, c)) / rng, 3)
            lower_wick_ratio = _round((min(o, c) - l) / rng, 3)

        engulfing = 0
        po = utils._clean(prev["open"])
        ph = utils._clean(prev["high"])
        pl = utils._clean(prev["low"])
        pc = utils._clean(prev["close"])
        if None not in (po, ph, pl, pc):
            prng = ph - pl
            pbody = pc - po
            body = c - o
            if prng and rng:
                if body > 0 > pbody and c >= po and o <= pc and abs(body) > abs(pbody):
                    engulfing = 1
                elif body < 0 < pbody and o >= pc and c <= po and abs(body) > abs(pbody):
                    engulfing = -1

        pinbar = 0
        if rng and rng > 0:
            body = abs(c - o)
            if lower_wick_ratio is not None and upper_wick_ratio is not None:
                lw = lower_wick_ratio * rng
                uw = upper_wick_ratio * rng
                if lw >= 2 * body and lw >= uw:
                    pinbar = 1
                elif uw >= 2 * body and uw >= lw:
                    pinbar = -1

        return {"body_ratio": body_ratio, "upper_wick_ratio": upper_wick_ratio,
                "lower_wick_ratio": lower_wick_ratio, "engulfing": engulfing,
                "pinbar": pinbar, "ok": True}
    except Exception as exc:  # noqa: BLE001
        log.warning("candle block failed: %s", exc)
        return _candle_blank()


def _context_blank():
    return {"corr_btc_30": None, "eth_btc_corr_30": None,
            "btc_dominance": None, "corr_dxy_30": None,
            "beta_index": None, "index_return_1d": None, "index_rsi": None,
            "ok": False}


def _context_block(df, symbol, timeframe=None, upto_sec=None):
    """Контекст индекса: corr_btc_30 + eth_btc_corr_30 (крипта, локально).

    corr_btc_30     — корреляция доходностей пары ↔ BTCUSDT (30 баров);
    eth_btc_corr_30 — рыночный gauge: корреляция ETHUSDT ↔ BTCUSDT (30 баров,
                      один на всех крипто-парах);
    btc_dominance / corr_dxy_30 / beta_index / index_return_1d / index_rsi —
    null-заглушки (требуют внешнего индекса/API). Только OHLCV-источник.
    """
    if df is None or df.empty:
        return _context_blank()
    try:
        out = {"corr_btc_30": None, "eth_btc_corr_30": None,
               "btc_dominance": None, "corr_dxy_30": None,
               "beta_index": None, "index_return_1d": None,
               "index_rsi": None}
        s = str(symbol or "").upper()
        is_crypto = s.endswith(("USDT", "BUSD", "USDC"))
        if is_crypto and timeframe:
            btc_df = None
            try:
                btc_df = _slice_df("BTCUSDT", timeframe, upto_sec)
                if s != "BTCUSDT" and btc_df is not None and not btc_df.empty:
                    corr = _corr_of_returns(df, btc_df, 30)
                    out["corr_btc_30"] = (_round(corr, 2)
                                          if corr is not None else None)
            except Exception as exc:  # noqa: BLE001
                log.warning("context corr_btc_30(%s) failed: %s", s, exc)
                out["corr_btc_30"] = None
            try:
                # Только в live: в реплее каждая досрезка ETH/BTC была бы
                # отдельной сетевой пачкой на шаг бэктеста (get_replay_df без
                # кеша) — это утроило бы запросы к Binance. В реплее поле null.
                if upto_sec is None:
                    eth_df = _slice_df("ETHUSDT", timeframe, upto_sec)
                    if (btc_df is not None and not btc_df.empty
                            and eth_df is not None and not eth_df.empty):
                        corr = _corr_of_returns(eth_df, btc_df, 30)
                        out["eth_btc_corr_30"] = (_round(corr, 2)
                                                  if corr is not None else None)
            except Exception as exc:  # noqa: BLE001
                log.warning("context eth_btc_corr_30(%s) failed: %s", s, exc)
                out["eth_btc_corr_30"] = None
        elif _is_forex_symbol(s):
            out["corr_dxy_30"] = None
        out["ok"] = True
        return out
    except Exception as exc:  # noqa: BLE001
        log.warning("context block failed: %s", exc)
        return _context_blank()


def _scanner_edge(symbol, timeframe):
    """Лучшая комбинация сканера: только цифры (как в Reference Data).

    winrate/max_dd/profit_factor/trades/test_sharpe — out-of-sample (test)
    окно; пропущенные в БД поля -> null (не 0 и не заглушка). Скана не
    было — вся схема null с "ok": false.
    """
    s = db.db_get_best_scan_stats(symbol, timeframe)
    if not s:
        return {
            "winrate": None, "sharpe": None, "max_dd": None,
            "profit_factor": None, "trades": None, "test_sharpe": None,
            "params": {}, "ok": False,
        }
    test = s.get("test") or {}
    params = {}
    for k, v in (s.get("params") or {}).items():
        v = utils._clean(v)
        params[str(k)] = v if v is not None else None

    def _opt(key):
        if key not in test:
            return None
        return _round(test.get(key), 4)

    trades = None
    if "trades" in test:
        tr = utils._clean(test.get("trades"))
        if tr is not None:
            trades = int(tr)

    return {
        "winrate": _round(test.get("winrate"), 4),
        "sharpe": _round(s.get("sharpe"), 4),
        "max_dd": _round(test.get("max_dd"), 4),
        "profit_factor": _opt("profit_factor"),
        "trades": trades,
        "test_sharpe": _opt("sharpe"),
        "params": params,
        "ok": True,
    }


def _is_forex_symbol(symbol):
    """Форекс-символ: 6 букв вне списка крипты (EURUSD, USDJPY, XAUUSD...)."""
    s = str(symbol or "").upper()
    return s not in config.CRYPTO_SYMBOLS and len(s) == 6 and s.isalpha()


def session_info(unix_ts, symbol=None):
    """Сессия по Unix-времени (UTC): час, день недели, статус рынка.

    session: 00-07 asia, 08-12 london, 13-16 london_ny, 17-21 ny, иначе
    off_hours; сб/вс -> weekend. london_ny_overlap = 1 только при 13-16 UTC
    в будни. market_open = 0 только для форекса в выходные; крипта всегда 1.
    Чистая функция — без I/O, удобна для тестов и реплея.
    """
    dt = datetime.fromtimestamp(float(unix_ts), tz=timezone.utc)
    hour_utc = int(dt.hour)
    dow = int(dt.weekday())
    weekend = dow >= 5
    if weekend:
        session = "weekend"
    elif hour_utc <= 7:
        session = "asia"
    elif hour_utc <= 12:
        session = "london"
    elif hour_utc <= 16:
        session = "london_ny"
    elif hour_utc <= 21:
        session = "ny"
    else:
        session = "off_hours"
    overlap = 1 if (not weekend and 13 <= hour_utc <= 16) else 0
    market_open = 0 if (_is_forex_symbol(symbol) and weekend) else 1
    return {
        "hour_utc": hour_utc,
        "dow": dow,
        "session": session,
        "london_ny_overlap": overlap,
        "market_open": market_open,
        "ok": True,
    }


def _clock_blank():
    """Null-схема clock, если время свечи недоступно (ok: false)."""
    return {
        "hour_utc": None, "dow": None, "session": None,
        "london_ny_overlap": None, "market_open": None, "ok": False,
    }


def _df_last_ts_sec(df):
    """Unix-секунды последней свечи среза (tz-safe)."""
    last_ts = pd.to_datetime(df["timestamp"].iloc[-1])
    if getattr(last_ts, "tz", None) is None:
        last_ts = last_ts.tz_localize("UTC")
    return int(last_ts.timestamp())


# --------------------------------------------------- доп. блоки данных
def _macro_item_blank():
    return {"value": None, "prev_value": None, "change": None, "date": None}


def _macro_blank():
    return {"us10y": _macro_item_blank(), "fed_rate": _macro_item_blank(),
            "dxy": _macro_item_blank(), "ok": False, "ts": None}


def _calendar_blank():
    return {"events": [], "high_impact_in_2h": 0, "next_event_in_min": None,
            "ok": False, "ts": None}


def _derivatives_blank():
    return {"open_interest": None, "oi_change_1h_pct": None,
            "funding_rate": None, "basis_pct": None, "top_ls_ratio": None,
            "ok": False, "ts": None}


def _micro_blank():
    return {"spread_norm": None, "ob_imb_10": None, "bid_sum_10": None,
            "ask_sum_10": None, "large_trades_ratio": None, "ok": False,
            "ts": None}


def _news_sentiment_blank():
    return {"avg_sentiment": None, "bullish_count": 0, "bearish_count": 0,
            "neutral_count": 0, "sample_size": 0, "worst_headline_score": None,
            "best_headline_score": None, "ok": False, "ts": None}


def _blank_for(block):
    return {"macro": _macro_blank, "calendar": _calendar_blank,
            "derivatives": _derivatives_blank, "micro": _micro_blank,
            "news_sentiment": _news_sentiment_blank}.get(block, dict)()


def _resolve_extra_block(name, future, symbol):
    """Результат фьючерса -> блок: ошибка -> null-схема, {} -> не применимо."""
    try:
        data = future.result(timeout=_EXTRA_BLOCK_TIMEOUT_SEC)
    except Exception as exc:  # noqa: BLE001 — TimeoutError/сбой потока
        log.warning("snapshot block %s(%s) failed: %s", name, symbol, exc)
        return _blank_for(name)
    if not isinstance(data, dict) or not data:
        if name == "sentiment":
            return _SENTIMENT_NULL_BLOCK
        return {"applicable": False, "ok": False}
    return data


def _build_extra_blocks(symbol, clock_ts):
    """Внешние блоки macro/calendar/derivatives/news_sentiment (с деградацией).

    clock_ts — время среза (последняя свеча/ts_override): уходит в календарь,
    чтобы «сегодня/+1 день» и флаги считались от точки реплея, а не от
    системных часов.
    """
    jobs = [
        ("macro", get_macro_snapshot, symbol),
        ("derivatives", get_derivatives_snapshot, symbol),
        ("micro", get_micro_snapshot, symbol),
        ("news_sentiment", get_news_sentiment, symbol),
    ]
    ccy = symbol[:3] if _is_forex_symbol(symbol) else None
    if ccy:
        jobs.append(("calendar", get_econ_calendar, ccy, clock_ts))

    futures = {name: _block_pool.submit(fn, *args)
               for name, fn, *args in jobs}
    blocks = {}
    for name, future in futures.items():
        blocks[name] = _resolve_extra_block(name, future, symbol)
    if not ccy:
        blocks["calendar"] = {"applicable": False, "ok": False}
    return blocks


def _resolve_crowd(symbol):
    """sentiment = get_crowd_snapshot; {} или exception -> null-схема."""
    future = _block_pool.submit(get_crowd_snapshot, symbol)
    try:
        data = future.result(timeout=_EXTRA_BLOCK_TIMEOUT_SEC)
    except Exception as exc:  # noqa: BLE001
        log.warning("snapshot block sentiment(%s) failed: %s", symbol, exc)
        data = {}
    if not isinstance(data, dict) or not data:
        return _SENTIMENT_NULL_BLOCK
    return data


def _normalize_crowd_scale(sent):
    """Единая шкала 0-1 (КОПИЯ блока — кеш get_crowd_snapshot не мутируем).

    fear_greed 0-100 -> 0-1; taker_buy_sell (binance buySellRatio, ratio-шкала,
    типично 0.3-3) -> доля покупок r/(1+r) в 0-1, чтобы пороги ''>0.6/''<0.4
    в промпте читались как доля покупательского давления.
    long_pct/short_pct/winrate/max_dd уже в долях — не трогаем. Остальные поля
    (ls_ratio/ts/ok) проходят как есть.
    """
    if not isinstance(sent, dict):
        return sent
    out = dict(sent)
    fg = utils._clean(out.get("fear_greed"))
    if fg is not None:
        out["fear_greed"] = _round(fg / 100.0, 4)
    tb = utils._clean(out.get("taker_buy_sell"))
    if tb is not None and tb > 0:
        share = tb / (1.0 + tb)
        # Клип на [0.05, 0.95]: редкие «магниты» (buySellRatio 0.1 / 10+ на
        # низколиквидных парах) не должны сдвигать распределение входа модели.
        share = max(0.05, min(0.95, share))
        out["taker_buy_sell"] = _round(share, 4)
    return out


def get_raw_market_data(symbol, timeframe, upto_sec=None, ts_override=None):
    """Чистый словарь с цифрами для ИИ (без единого текстового пояснения).

    upto_sec — барьер реплея (срез без взгляда в будущее); None в live.
    ts_override — для тестов: принудительное время блоков clock/calendar;
    иначе они считаются по времени ПОСЛЕДНЕЙ свечи среза, чтобы время в
    AI Backtest (replay) соответствовало исторической точке, а не системным
    часам.
    При неизвестной паре technicals/clock уходят в null-схему с ok:false —
    ИИ получает структуру всегда.

    Доп. блоки (sentiment/macro/calendar/derivatives/news_sentiment) тянутся
    из своих геттеров (внутри у них кеши; снимок НЕ кэшируется). Каждый вызов
    обёрнут: ошибка -> null-схема блока с ok:false, "не применимо" ({} от
    геттера) -> {"applicable": false, "ok": false}.

    Единая шкала: все процентные/долевые поля — доли 0-1 (fear_greed из
    crowd 0-100 приводится к 0-1 через _normalize_crowd_scale).
    """
    symbol = str(symbol or "").upper()
    timeframe = str(timeframe or "").strip()
    df = _slice_df(symbol, timeframe, upto_sec)

    clock_ts = ts_override
    if clock_ts is None and df is not None and not df.empty:
        clock_ts = _df_last_ts_sec(df)
    clock = session_info(clock_ts, symbol) if clock_ts is not None \
        else _clock_blank()

    blocks = _build_extra_blocks(symbol, clock_ts)
    blocks["sentiment"] = _normalize_crowd_scale(_resolve_crowd(symbol))

    return {
        "technicals": _technicals_from_df(df),
        "volume": _volume_block(df),
        "trend": _trend_block(df),
        "momentum": _momentum_block(df),
        "volatility": _volatility_block(df),
        "regime": _regime_block(df),
        "divergence": _divergence_block(df),
        "candle": _candle_block(df),
        "context": _context_block(df, symbol, timeframe, upto_sec),
        "scanner_edge": _scanner_edge(symbol, timeframe),
        "sentiment": blocks["sentiment"],
        "clock": clock,
        "macro": blocks["macro"],
        "calendar": blocks["calendar"],
        "derivatives": blocks["derivatives"],
        "micro": blocks["micro"],
        "news_sentiment": blocks["news_sentiment"],
    }


# =================================================== токен-диета (схема v3)
# Максимально сжатый снимок для LLM: короткие ключи, округление до значимых
# разрядов, ОТСУТСТВИЕ блоков без данных (missing = no data) и НИКОГДА не
# отдаём "ok": true. Сырые OHLC-массивы по ТФ заменены агрегатами mtf:
# {"tf": {"tr", "trend", "rsi", "adx"}} (trend — ema20/50-определение, см.
# _mtf_compact: up/down/flat с мёртвой зоной 0.1% и подтверждением close>ema20).
_MTF_TFS = ("15m", "1H", "4H", "1D")


def _put(out, key, value):
    """Положить значение в компактный блок, если оно непустое (None -> нет)."""
    if value is None:
        return
    if value == {} or value == []:
        return
    out[key] = value


def _r2(v):
    return _round(v, 2)


def _r1(v):
    return _round(v, 1)


def _rint(v):
    """Целое (n/trades/счётчики) или None."""
    v = utils._clean(v)
    if v is None:
        return None
    try:
        return int(round(v))
    except (TypeError, ValueError):
        return None


def _block_has_data(block):
    """У блока есть данные: не ok:false/applicable:false и есть непустое поле."""
    if not isinstance(block, dict) or not block:
        return False
    if block.get("ok") is False or block.get("applicable") is False:
        return False
    for key, value in block.items():
        if key in ("ok", "applicable"):
            continue
        if value is None or value == {} or value == []:
            continue
        return True
    return False


def _compact_technicals(raw):
    """technicals -> t{rsi,atr,atr_pct,bb,sma,ap,dh,dl,close} (2dp; atr/close 1dp)."""
    t = raw.get("technicals") or {}
    if not _block_has_data(t):
        return None
    out = {}
    _put(out, "rsi", _r2(t.get("rsi")))
    _put(out, "atr", _r1(t.get("atr")))
    atr_val = utils._clean(t.get("atr"))
    close_val = utils._clean(t.get("close"))
    if atr_val and close_val:
        _put(out, "atr_pct", _r2(atr_val / close_val))
    _put(out, "bb", _r2(t.get("bb_pct_b")))
    _put(out, "sma", _r2(t.get("sma20_diff_pct")))
    _put(out, "ap", _r2(t.get("atr_percentile")))
    _put(out, "dh", _r2(t.get("dist_to_high_pct")))
    _put(out, "dl", _r2(t.get("dist_to_low_pct")))
    _put(out, "close", _r1(t.get("close")))
    return out or None


def _compact_scanner(raw):
    """scanner_edge -> se{wr,sharpe,pf,dd,n,tsh,p} (2dp; n int)."""
    s = raw.get("scanner_edge") or {}
    if not _block_has_data(s):
        return None
    out = {}
    _put(out, "wr", _r2(s.get("winrate")))
    _put(out, "sharpe", _r2(s.get("sharpe")))
    _put(out, "pf", _r2(s.get("profit_factor")))
    _put(out, "dd", _r2(s.get("max_dd")))
    _put(out, "n", _rint(s.get("trades")))
    _put(out, "tsh", _r2(s.get("test_sharpe")))
    params = {}
    for k, v in (s.get("params") or {}).items():
        cv = utils._clean(v)
        params[str(k)] = cv if cv is not None else None
    if params:
        out["p"] = params
    return out or None


def _compact_sentiment(raw):
    """sentiment -> s{ls,long_pct,fng,tbs} (ls 2dp; long_pct 1dp; fng/tbs 2dp в 0-1)."""
    s = raw.get("sentiment") or {}
    if not _block_has_data(s):
        return None
    out = {}
    _put(out, "ls", _r2(s.get("ls_ratio")))
    _put(out, "long_pct", _r1(s.get("long_pct")))
    _put(out, "fng", _r2(s.get("fear_greed")))
    _put(out, "tbs", _r2(s.get("taker_buy_sell")))
    return out or None


def _compact_clock(raw):
    """clock -> c{h,dow,ses,ovl,open}."""
    c = raw.get("clock") or {}
    if not _block_has_data(c):
        return None
    out = {}
    _put(out, "h", _rint(c.get("hour_utc")))
    _put(out, "dow", _rint(c.get("dow")))
    _put(out, "ses", c.get("session"))
    _put(out, "ovl", _rint(c.get("london_ny_overlap")))
    _put(out, "open", _rint(c.get("market_open")))
    return out or None


def _compact_macro(raw):
    """macro -> m{us10y,fed,dxy} (2dp, только value)."""
    m = raw.get("macro") or {}
    if not _block_has_data(m):
        return None
    out = {}
    for key, src in (("us10y", "us10y"), ("fed", "fed_rate"),
                     ("dxy", "dxy")):
        item = m.get(src)
        val = _r2(item.get("value")) if isinstance(item, dict) else None
        _put(out, key, val)
    return out or None


def _compact_calendar(raw):
    """calendar -> cal{hi2h,next_min}."""
    c = raw.get("calendar") or {}
    if not _block_has_data(c):
        return None
    out = {}
    _put(out, "hi2h", _rint(c.get("high_impact_in_2h")))
    _put(out, "next_min", _rint(c.get("next_event_in_min")))
    return out or None


def _compact_derivatives(raw):
    """derivatives -> d{oi,oi_chg,fund,basis,top_ls} (fund 4dp, прочее 2dp)."""
    d = raw.get("derivatives") or {}
    if not _block_has_data(d):
        return None
    out = {}
    _put(out, "oi", _r2(d.get("open_interest")))
    _put(out, "oi_chg", _r2(d.get("oi_change_1h_pct")))
    _put(out, "fund", _round(d.get("funding_rate"), 4))
    _put(out, "basis", _r2(d.get("basis_pct")))
    _put(out, "top_ls", _r2(d.get("top_ls_ratio")))
    return out or None


def _compact_news(raw):
    """news_sentiment -> ns{avg,bull,bear} (avg 2dp; счётчики int)."""
    n = raw.get("news_sentiment") or {}
    if not _block_has_data(n):
        return None
    out = {}
    _put(out, "avg", _r2(n.get("avg_sentiment")))
    _put(out, "bull", _rint(n.get("bullish_count")))
    _put(out, "bear", _rint(n.get("bearish_count")))
    return out or None


def _compact_volume(raw):
    """volume -> v{rv,obv,vz,vp,vwd} (2dp)."""
    v = raw.get("volume") or {}
    if not _block_has_data(v):
        return None
    out = {}
    _put(out, "rv", _r2(v.get("rel_vol")))
    _put(out, "obv", _r2(v.get("obv_slope")))
    _put(out, "vz", _r2(v.get("vol_zscore")))
    _put(out, "vp", _r2(v.get("vol_percentile")))
    _put(out, "vwd", _r2(v.get("vwap_dev")))
    return out or None


def _compact_trend(raw):
    """trend -> tr{adx,pdi,mdi,ds,er,rs} (adx/di 1dp; er 3dp; rs 2dp)."""
    t = raw.get("trend") or {}
    if not _block_has_data(t):
        return None
    out = {}
    _put(out, "adx", _r1(t.get("adx")))
    _put(out, "pdi", _r1(t.get("plus_di")))
    _put(out, "mdi", _r1(t.get("minus_di")))
    _put(out, "ds", _r1(t.get("di_spread")))
    _put(out, "er", _round(t.get("ema20_50_ratio"), 3))
    _put(out, "rs", _r2(t.get("reg_slope_20")))
    return out or None


def _compact_momentum(raw):
    """momentum -> mo{macd,ms,mh,mhs,rsi_s} (2dp)."""
    m = raw.get("momentum") or {}
    if not _block_has_data(m):
        return None
    out = {}
    _put(out, "macd", _r2(m.get("macd")))
    _put(out, "ms", _r2(m.get("macd_signal")))
    _put(out, "mh", _r2(m.get("macd_hist")))
    _put(out, "mhs", _r2(m.get("macd_hist_slope")))
    _put(out, "rsi_s", _r2(m.get("rsi_slope")))
    return out or None


def _compact_volatility(raw):
    """volatility -> vl{bw,bwp,hv,atc,atr_r} (bw 3dp; pct/change/ratio 2dp; hv 1dp)."""
    v = raw.get("volatility") or {}
    if not _block_has_data(v):
        return None
    out = {}
    _put(out, "bw", _round(v.get("bb_width"), 3))
    _put(out, "bwp", _r2(v.get("bb_width_pct")))
    _put(out, "hv", _r1(v.get("hv20")))
    _put(out, "atc", _r2(v.get("atr_change_pct")))
    _put(out, "atr_r", _r2(v.get("atr_ratio_short_long")))
    return out or None


def _compact_regime(raw):
    """regime -> rg{h,ac1,er} (2dp)."""
    r = raw.get("regime") or {}
    if not _block_has_data(r):
        return None
    out = {}
    _put(out, "h", _r2(r.get("hurst")))
    _put(out, "ac1", _r2(r.get("autocorr_lag1")))
    _put(out, "er", _r2(r.get("efficiency_ratio")))
    return out or None


def _compact_context(raw):
    """context -> cg{cbtc,ebc,dom,dxy,beta,ir1,irsi} (2dp; заглушки null)."""
    c = raw.get("context") or {}
    if not _block_has_data(c):
        return None
    out = {}
    _put(out, "cbtc", _r2(c.get("corr_btc_30")))
    _put(out, "ebc", _r2(c.get("eth_btc_corr_30")))
    _put(out, "dom", _r2(c.get("btc_dominance")))
    _put(out, "dxy", _r2(c.get("corr_dxy_30")))
    _put(out, "beta", _r2(c.get("beta_index")))
    _put(out, "ir1", _r2(c.get("index_return_1d")))
    _put(out, "irsi", _r2(c.get("index_rsi")))
    return out or None


def _compact_divergence(raw):
    """divergence -> div{rsi,macd,obv} (-1 bear / +1 bull / 0 none, int)."""
    d = raw.get("divergence") or {}
    if not _block_has_data(d):
        return None
    out = {}
    _put(out, "rsi", _rint(d.get("rsi_price")))
    _put(out, "macd", _rint(d.get("macd_price")))
    _put(out, "obv", _rint(d.get("obv_price")))
    return out or None


def _compact_candle(raw):
    """candle -> cnd{br,uw,lw,eng,pin} (ratios 3dp; eng/pin int)."""
    c = raw.get("candle") or {}
    if not _block_has_data(c):
        return None
    out = {}
    _put(out, "br", _round(c.get("body_ratio"), 3))
    _put(out, "uw", _round(c.get("upper_wick_ratio"), 3))
    _put(out, "lw", _round(c.get("lower_wick_ratio"), 3))
    _put(out, "eng", _rint(c.get("engulfing")))
    _put(out, "pin", _rint(c.get("pinbar")))
    return out or None


def _compact_micro(raw):
    """micro -> mcr{sp,obi,bs,as,ltr} (sp 6dp; obi/ltr 4dp; bs/as int)."""
    m = raw.get("micro") or {}
    if not _block_has_data(m):
        return None
    out = {}
    _put(out, "sp", _round(m.get("spread_norm"), 6))
    _put(out, "obi", _round(m.get("ob_imb_10"), 4))
    _put(out, "bs", _rint(m.get("bid_sum_10")))
    _put(out, "as", _rint(m.get("ask_sum_10")))
    _put(out, "ltr", _round(m.get("large_trades_ratio"), 4))
    return out or None


def _mtf_compact(symbol, upto_sec=None):
    """Агрегаты по ТФ вместо сырых свечей: {tf: {tr, trend, rsi, adx}}.

    trend — ФИКСИРОВАННОЕ определение для правила 9 (единое на всех ТФ):
        "up"    если ema20 > ema50*1.001  и close > ema20
        "down"  если ema20 < ema50*0.999  и close < ema20
        "flat"  иначе (мёртвая зона ±0.1% + подтверждение close>ema20 —
                иначе разница H1 vs H4 на «прошивке» EMA путается в шум).
    tr = 1 для up, -1 для down, 0 для flat (совпадает с trend);
    rsi — int (Wilder 14 по срезу с барьером upto_sec);
    adx — int (Wilder 14), None если индикатор не сосчитался.
    """
    out = {}
    for tf in _MTF_TFS:
        try:
            if upto_sec is not None:
                # Реплей: историческое окно до барьера (live-хвост не годится).
                df = get_replay_df(symbol, tf, to_sec=float(upto_sec), limit=80)
            else:
                df = get_series_df(symbol, tf, limit=80,
                                   history_limit=config.TRENDS_HISTORY)
        except Exception:  # noqa: BLE001 — источник недоступен -> пропуск
            continue
        if df is None or df.empty:
            continue
        if upto_sec is not None:
            mask = df["timestamp"] <= pd.to_datetime(
                int(upto_sec), unit="s", utc=True)
            df = df[mask]
        if df is None or len(df) < 50:
            continue
        close = pd.Series(df["close"], dtype="float64")
        last = utils._clean(close.iloc[-1])
        ema20 = utils._clean(close.ewm(span=20, adjust=False).mean().iloc[-1])
        ema50 = utils._clean(close.ewm(span=50, adjust=False).mean().iloc[-1])
        if (last is None or ema20 is None or ema50 is None
                or not ema50):
            tr = 0
        elif ema20 > ema50 * 1.001 and last > ema20:
            tr = 1
        elif ema20 < ema50 * 0.999 and last < ema20:
            tr = -1
        else:
            tr = 0
        trend = "flat" if tr == 0 else ("up" if tr == 1 else "down")
        rsi = utils._clean(rsi_wilder(close, _RSI_PERIOD).iloc[-1])
        try:
            high = pd.Series(df["high"], dtype="float64")
            low = pd.Series(df["low"], dtype="float64")
            adx_last = utils._clean(_di(high, low, close, 14)[2].iloc[-1])
            adx = int(round(adx_last)) if adx_last is not None else None
        except Exception:  # noqa: BLE001 — adx необязателен, не роняем блок
            adx = None
        out[tf.lower()] = {"tr": tr, "trend": trend,
                           "rsi": int(round(rsi)) if rsi is not None else None,
                           "adx": adx}
    return out


def compact_snapshot(symbol, timeframe, upto_sec=None):
    """Максимально сжатый JSON-снимок для LLM (схема v3, без "ok").

    Вызывает get_raw_market_data и упаковывает блоки в короткие ключи с
    округлением. Блоки с ok:false / {} / все-null НЕ попадают в результат
    (missing = no data). trend по ТФ отдан агрегатом mtf, а не свечами.
    """
    raw = get_raw_market_data(symbol, timeframe, upto_sec=upto_sec)
    out = {}
    for key, builder in (
        ("t", _compact_technicals), ("se", _compact_scanner),
        ("s", _compact_sentiment), ("c", _compact_clock),
        ("m", _compact_macro), ("cal", _compact_calendar),
        ("d", _compact_derivatives), ("ns", _compact_news),
        ("v", _compact_volume), ("tr", _compact_trend),
        ("mo", _compact_momentum), ("vl", _compact_volatility),
        ("rg", _compact_regime), ("cg", _compact_context),
        ("div", _compact_divergence), ("cnd", _compact_candle),
        ("mcr", _compact_micro),
    ):
        block = builder(raw)
        if block:
            out[key] = block
    mtf = _mtf_compact(symbol, upto_sec)
    if mtf:
        out["mtf"] = mtf
    return out