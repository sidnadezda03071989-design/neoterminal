"""Единая точка получения OHLCV.

Маршруты источников:
    крипта  -> Binance
    форекс  -> MT5, fallback -> yfinance

fetch_ohlcv() ВСЕГДА возвращает DataFrame (не None): при пустом
результате — пустой DF со схемой timestamp/open/high/low/close/volume.
"""

import logging
import threading
import time
from collections import OrderedDict

import pandas as pd

from app_pkg import config
from app_pkg.data import binance, mt5, yfinance

logger = logging.getLogger(__name__)

_EMPTY_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

data_cache = OrderedDict()  # (symbol, tf) -> {"df": DataFrame, "ts": float, "limit": int}
cache_lock = threading.Lock()


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=_EMPTY_COLUMNS)


def _ttl_for(symbol: str, tf: str) -> float:
    """TTL кеша: крипта 3600с; форекс 120с; дневные форекс-бары 1800с."""
    if symbol in config.CRYPTO_SYMBOLS:
        return config.DATA_CACHE_TTL_CRYPTO
    if tf == "1D":
        return config.DATA_CACHE_TTL_FOREX_1D
    return config.DATA_CACHE_TTL_FOREX


def fetch_ohlcv(symbol: str, tf: str, limit: int = 1000,
                start_sec: float = None, end_sec: float = None) -> pd.DataFrame:
    """Крипта -> binance; форекс -> mt5 (fallback yfinance). Всегда DataFrame."""
    if symbol in config.CRYPTO_SYMBOLS:
        try:
            return binance.fetch_binance_paged(
                symbol, config.BINANCE_TF.get(tf, tf),
                total=limit, start_sec=start_sec, end_sec=end_sec)
        except Exception as exc:  # noqa: BLE001
            logger.warning("binance fetch %s %s failed: %s", symbol, tf, exc)
            return _empty_df()

    df = None
    try:
        df = mt5.fetch_mt5(symbol, tf, limit=limit, start_sec=start_sec,
                           end_sec=end_sec)
    except Exception as exc:  # noqa: BLE001
        logger.warning("mt5 fetch %s %s failed: %s", symbol, tf, exc)
        df = None
    if df is None or df.empty:
        try:
            df = yfinance.fetch_yfinance(
                symbol, tf, limit=limit, start_sec=start_sec, end_sec=end_sec)
        except Exception as exc:  # noqa: BLE001
            logger.warning("yfinance fetch %s %s failed: %s", symbol, tf, exc)
            df = _empty_df()
    return df if df is not None else _empty_df()


def apply_live_merge(symbol: str, tf: str, df: pd.DataFrame) -> pd.DataFrame:
    """Обновляет последний бар данными live_bars, если таймстемпы совпадают.

    ВАЖНО: новые бары НЕ накапливаются. Если live-бар новее последнего
    бара df — df возвращается как есть (актуализирует следующий REST-запрос).
    """
    if df is None or df.empty:
        return df
    try:
        from app_pkg.data.live import live_bars, _live_bars_lock
        with _live_bars_lock:
            bar = live_bars.get((symbol, tf))
    except Exception:  # noqa: BLE001
        return df
    if not bar:
        return df

    last_ts = pd.to_datetime(df["timestamp"].iloc[-1])
    if getattr(last_ts, "tz", None) is None:
        last_ts = last_ts.tz_localize("UTC")
    live_ts = pd.to_datetime(int(bar.get("ts", 0)), unit="s", utc=True)
    if live_ts != last_ts:
        return df

    def _v(key, fallback):
        val = bar.get(key)
        return float(val) if val is not None else float(fallback)

    df = df.copy()
    idx = df.index[-1]
    df.loc[idx, "open"] = _v("open", df.loc[idx, "open"])
    df.loc[idx, "high"] = max(_v("high", df.loc[idx, "high"]),
                              float(df.loc[idx, "high"]))
    df.loc[idx, "low"] = min(_v("low", df.loc[idx, "low"]),
                             float(df.loc[idx, "low"]))
    df.loc[idx, "close"] = _v("close", df.loc[idx, "close"])
    df.loc[idx, "volume"] = _v("volume", df.loc[idx, "volume"])
    return df


def _tail(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Обрезка до запрошенного limit на выходе (кэш остаётся полным)."""
    if limit and len(df) > limit:
        return df.tail(limit).reset_index(drop=True)
    return df


def _store_cache(key, df: pd.DataFrame, history_limit: int) -> None:
    """Запись в LRU data_cache: вставка в конец, вытеснение самого старого."""
    with cache_lock:
        data_cache[key] = {"df": df, "ts": time.time(), "limit": history_limit}
        data_cache.move_to_end(key)  # вставка — ключ в конец очереди LRU
        max_items = config.DATA_CACHE_MAX_ITEMS
        while max_items and len(data_cache) > max_items:
            data_cache.popitem(last=False)  # вытесняем самый старый ключ


def _topup_older(symbol: str, tf: str, cached: pd.DataFrame, need: int,
                 history_limit: int):
    """Докачка СТАРЫХ баров (пагинация назад) + PREPEND к cached.

    end_sec = самый ранний бар кеша минус 1 секунда. Объединение с дедупом
    по timestamp, обрезка до history_limit, обновление кеша.
    Возвращает объединённый df или None, если докачать не удалось
    (история источника кончилась / ошибка) — тогда кеш отдаётся как есть.
    """
    earliest = pd.to_datetime(cached["timestamp"].iloc[0])
    if getattr(earliest, "tz", None) is None:
        earliest = earliest.tz_localize("UTC")
    end_sec = earliest.timestamp() - 1.0
    older = fetch_ohlcv(symbol, tf, limit=need, end_sec=end_sec)
    if older is None or older.empty:
        return None
    older = older[older["timestamp"] < earliest]  # дедуп стыка
    if older.empty:
        return None
    merged = pd.concat([older, cached], ignore_index=True)
    merged = (merged.drop_duplicates(subset=["timestamp"])
                    .sort_values("timestamp")
                    .reset_index(drop=True))
    if len(merged) > history_limit:
        merged = merged.tail(history_limit).reset_index(drop=True)
    merged = apply_live_merge(symbol, tf, merged)
    _store_cache((symbol, tf), merged, history_limit)
    return merged


def get_series_df(symbol: str, tf: str, limit: int = 1000,
                  force: bool = False, history_limit: int = None) -> pd.DataFrame:
    """Кешированный OHLCV с live-мержем и дозагрузкой истории.

    limit — сколько свечей нужно вызывающему СЕЙЧАС; history_limit — до
    какой глубины держать кэш (по умолчанию HISTORY_LIMITS[tf]). Поэтому
    тонкие запросы (alerts limit=2, chat limit=100, ...) берут срез из
    кэша и не инвалидируют его, а запрос limit=20000 после limit=1000
    ДОзагружает только недостающие 19000 баров (пагинация назад),
    а не всю историю заново.

    Логика:
    - живой кеш:
        * len(cached) >= min(limit, history_limit) -> tail(limit);
        * иначе докачка СТАРЫХ баров, PREPEND к существующему df,
          кеш обновляется, возвращается tail(limit);
    - холодный кеш (или force): fetch_ohlcv(symbol, tf, limit=history_limit);
      если источник вернул меньше запрошенного — один проход докачки назад.

    В data_cache["df"] всегда максимум из когда-либо загруженного
    (но не больше HISTORY_LIMITS[tf]); ключ кеша — (symbol, tf).

    ВАЖНО: пустой df в кэш НЕ пишется — при пустом ответе источника
    логируется warning и возвращается пустой DF (следующий вызов снова
    попробует сеть, а не отдаст «мёртвый» кеш).

    TTL: 3600с крипта, 120с форекс, 1800с для форекс 1D.

    Кэш ограничен LRU-объёмом (config.DATA_CACHE_MAX_ITEMS = 60 ключей):
    при чтении ключ перемещается в конец (move_to_end), при вставке нового
    ключа самый старый вытесняется (popitem(last=False)).
    """
    limit = int(limit or 1000)
    history_limit = int(history_limit or config.HISTORY_LIMITS.get(tf, 10000))
    target = min(limit, history_limit)  # потолок полезного объёма кеша
    key = (symbol, tf)

    if not force:
        cached = None
        with cache_lock:
            item = data_cache.get(key)
            if item and (time.time() - item["ts"]) < _ttl_for(symbol, tf):
                df = item["df"]
                if df is not None and not df.empty:  # пустой кеш не валиден
                    data_cache.move_to_end(key)  # LRU: только что прочитанный — самый свежий
                    cached = df
        if cached is not None:
            if len(cached) >= target:
                return _tail(cached, limit)

            # В кэше меньше, чем просят, а история ещё есть — докачиваем.
            merged = _topup_older(symbol, tf, cached, target - len(cached),
                                  history_limit)
            return _tail(merged if merged is not None else cached, limit)

    # Холодный кеш (или force): тянем history_limit, не limit.
    df = fetch_ohlcv(symbol, tf, limit=history_limit)
    df = apply_live_merge(symbol, tf, df)
    if df is None or df.empty:
        logger.warning("empty ohlcv %s %s (limit=%s) — в кэш не пишется",
                       symbol, tf, history_limit)
        return df if df is not None else _empty_df()
    _store_cache(key, df, history_limit)
    if len(df) < target:
        # Источник вернул меньше запрошенного (ограниченное окно) — добираем
        # старые бары, чтобы cold-запрос сразу получил нужный объём.
        merged = _topup_older(symbol, tf, df, target - len(df), history_limit)
        if merged is not None:
            return _tail(merged, limit)
    return _tail(df, limit)


def get_replay_df(symbol: str, tf: str, from_sec: float = None,
                  to_sec: float = None, limit: int = 1200) -> pd.DataFrame:
    """Исторические данные для бэктеста (без live-мержа)."""
    limit = int(limit or config.REPLAY_LIMIT)
    df = fetch_ohlcv(symbol, tf, limit=limit, start_sec=from_sec, end_sec=to_sec)
    if df is None or df.empty:
        return _empty_df()
    df = df.sort_values("timestamp").reset_index(drop=True)
    if len(df) > limit:
        df = df.iloc[-limit:].reset_index(drop=True)
    return df