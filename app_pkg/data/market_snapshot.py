# -*- coding: utf-8 -*-
"""Сырые рыночные данные для ИИ (Market Snapshot).

Единственный источник правды для нейросети: НИКАКОГО текста, только числа.
Нейросеть не получает «словесное описание рынка» — она получает этот JSON
и делает выводы по формулам системного промпта.

get_raw_market_data(symbol, timeframe, upto_sec=None, ts_override=None) -> dict:
    technicals   — RSI(14), ATR(14), BB %B, SMA20 diff %, ATR-перцентиль,
                   дистанции до 50-барных экстремумов, последняя цена;
    scanner_edge — лучшая комбинация сканера
                   (winrate/sharpe/max_dd/profit_factor/trades/test_sharpe/params);
    sentiment    — crowd: ls_ratio/long_pct/short_pct/taker_buy_sell/fear_greed
                   (крипта) или null-схема {ls_ratio, long_pct, fear_greed: null,
                   ok: false} при ошибке/не-крипте;
    macro        — us10y/fed_rate/dxy (FRED) — форекс/все активы;
    calendar     — экономический календарь (Finnhub) для валюты пары;
    derivatives  — open_interest/oi_change_1h_pct/funding_rate/basis_pct/
                   top_ls_ratio (крипта);
    news_sentiment — lexicon-скоринг заголовков ленты актива (без LLM);
    clock        — сессии по времени ПОСЛЕДНЕЙ свечи среза (реплей-безопасно,
                   не по системным часам), кроме случаев ts_override для тестов.

Единое правило блоков: при ошибке/отсутствии источника поля = null и
"ok": false — без exception и без опасных 0/50-заглушек; "не применимо"
к активу ({} от геттера) -> {"applicable": false, "ok": false}.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pandas as pd

from app_pkg import config, db, utils
from app_pkg.data.derivatives import get_derivatives_snapshot
from app_pkg.data.fetch import get_replay_df, get_series_df
from app_pkg.data.macro import get_econ_calendar, get_macro_snapshot
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
    """Дистанции close до 50-барных экстремумов, в %.

    dist_to_high_pct = (close - max_high_50) / max_high_50 * 100 (обычно <= 0);
    dist_to_low_pct  = (close - min_low_50) / min_low_50 * 100 (обычно >= 0).
    Меньше 50 баров или нулевой экстремум -> pair (None, None).
    """
    if df is None or len(df) < _LEVELS_WINDOW:
        return None, None
    last_close = _round(df["close"].iloc[-1], 8)
    if last_close is None:
        return None, None
    high50 = utils._clean(df["high"].tail(_LEVELS_WINDOW).max())
    low50 = utils._clean(df["low"].tail(_LEVELS_WINDOW).min())
    if not high50 or not low50:
        return None, None
    dist_high = _round((last_close - high50) / high50 * 100.0, 4)
    dist_low = _round((last_close - low50) / low50 * 100.0, 4)
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


def _news_sentiment_blank():
    return {"avg_sentiment": None, "bullish_count": 0, "bearish_count": 0,
            "neutral_count": 0, "sample_size": 0, "worst_headline_score": None,
            "best_headline_score": None, "ok": False, "ts": None}


def _blank_for(block):
    return {"macro": _macro_blank, "calendar": _calendar_blank,
            "derivatives": _derivatives_blank,
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
    blocks["sentiment"] = _resolve_crowd(symbol)

    return {
        "technicals": _technicals_from_df(df),
        "scanner_edge": _scanner_edge(symbol, timeframe),
        "sentiment": blocks["sentiment"],
        "clock": clock,
        "macro": blocks["macro"],
        "calendar": blocks["calendar"],
        "derivatives": blocks["derivatives"],
        "news_sentiment": blocks["news_sentiment"],
    }


# =================================================== токен-диета (схема v3)
# Максимально сжатый снимок для LLM: короткие ключи, округление до значимых
# разрядов, ОТСУТСТВИЕ блоков без данных (missing = no data) и НИКОГДА не
# отдаём "ok": true. Сырые OHLC-массивы по ТФ заменены агрегатами mtf:
# {"1h": {"tr": 1|-1|0, "rsi": int}, "4h": {...}} (tr — знак close - sma50).
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
    """technicals -> t{rsi,atr,bb,sma,ap,dh,dl,close} (2dp; atr/close 1dp)."""
    t = raw.get("technicals") or {}
    if not _block_has_data(t):
        return None
    out = {}
    _put(out, "rsi", _r2(t.get("rsi")))
    _put(out, "atr", _r1(t.get("atr")))
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
    """sentiment -> s{ls,long_pct,fng} (ls 2dp; long_pct 1dp; fng int)."""
    s = raw.get("sentiment") or {}
    if not _block_has_data(s):
        return None
    out = {}
    _put(out, "ls", _r2(s.get("ls_ratio")))
    _put(out, "long_pct", _r1(s.get("long_pct")))
    _put(out, "fng", _rint(s.get("fear_greed")))
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


def _mtf_compact(symbol, upto_sec=None):
    """Агрегаты по ТФ вместо сырых свечей: {tf: {tr, rsi}}.

    tr = знак (close - sma50): 1 выше, -1 ниже, 0 равен/нет данных.
    rsi — int (Wilder 14 по срезу с барьером upto_sec).
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
        sma50 = utils._clean(close.rolling(50).mean().iloc[-1])
        if last is None or sma50 is None or last == sma50:
            tr = 0
        else:
            tr = 1 if last > sma50 else -1
        rsi = utils._clean(rsi_wilder(close, _RSI_PERIOD).iloc[-1])
        out[tf.lower()] = {"tr": tr,
                           "rsi": int(round(rsi)) if rsi is not None else None}
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
    ):
        block = builder(raw)
        if block:
            out[key] = block
    mtf = _mtf_compact(symbol, upto_sec)
    if mtf:
        out["mtf"] = mtf
    return out