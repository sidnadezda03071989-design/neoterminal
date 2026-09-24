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

# Потолок «дозаливки» свежих закрытых баров одним заходом (см. _extend_newer).
# Больше — кэш сильно отстал, проще перезалить хвост целиком, чем гнать
# десятки маленьких запросов.
_EXTEND_NEWER_CAP = 1000


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
                start_sec: float | None = None, end_sec: float | None = None) -> pd.DataFrame:
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


def apply_live_merge(symbol: str, tf: str, df: pd.DataFrame,
                     copy: bool = True) -> pd.DataFrame:
    """Обновляет последний бар данными live_bars.

    - Если live-бар того же таймстемпа, что последний бар df — обновляет его OHLCV.
    - Если live-бар НОВЕЕ последнего бара df — добавляет его как новую строку
      (иначе /api/last-bar для 1m отдаёт устаревший бар весь интервал TTL кеша).
    - Если live-бар старее — df возвращается как есть.

    ИДЕМПОТЕНТНОСТЬ И АТОМАРНОСТЬ (регресс «10.07 / 10.12 и лагают»):
    раньше функция мутировала df на месте. При параллельных вызовах из HTTP
    (/api/data) и SSE-push-loop получалась смесь: один поток перезаписывал
    high, другой — close, и на графике последняя свеча «прыгала» между
    значениями. Теперь:
      * все поля берутся из ОДНОГО снимка live-бара (один захват лока);
      * при copy=True (по умолчанию) исходный df не мутируется;
      * поля перезаписываются, а не накапливаются через max/min — повторный
        вызов с тем же live-баром даёт тот же результат.
    copy=False использовать только если df — свежая копия, которой больше
    никто не владеет (экономия аллокации в горячем пути).
    """
    if df is None or df.empty:
        return df
    try:
        from app_pkg.data.live import _live_bars_lock, live_bars
        with _live_bars_lock:
            src = live_bars.get((symbol, tf))
            # Снимок под локом: иначе WS-поток может заменить поля между
            # чтениями, и бар получится из двух разных тиков.
            bar = dict(src) if src else None
    except Exception:  # noqa: BLE001
        return df
    if not bar:
        return df

    last_ts = pd.to_datetime(df["timestamp"].iloc[-1])
    if getattr(last_ts, "tz", None) is None:
        last_ts = last_ts.tz_localize("UTC")
    live_ts = pd.to_datetime(int(bar.get("ts", 0)), unit="s", utc=True)
    if live_ts < last_ts:
        return df  # live-бар старее последнего — не трогаем

    def _v(key, fallback):
        val = bar.get(key)
        return float(val) if val is not None else float(fallback)

    if copy:
        df = df.copy()
    if live_ts == last_ts:
        idx = df.index[-1]
        # Приоритет у live-значений: бар из WS — самый свежий. fallback нужен
        # только если поле в live-баре отсутствует.
        df.loc[idx, "open"] = _v("open", df.loc[idx, "open"])
        df.loc[idx, "high"] = _v("high", df.loc[idx, "high"])
        df.loc[idx, "low"] = _v("low", df.loc[idx, "low"])
        df.loc[idx, "close"] = _v("close", df.loc[idx, "close"])
        df.loc[idx, "volume"] = _v("volume", df.loc[idx, "volume"])
        return df

    # live_ts > last_ts: новый бар — дописываем строку
    new_row = {
        "timestamp": live_ts,
        "open": _v("open", None),
        "high": _v("high", None),
        "low": _v("low", None),
        "close": _v("close", None),
        "volume": _v("volume", None),
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
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

    ВАЖНО: в кэш пишется КОПИЯ df. Раньше сюда шёл сам объект, а затем
    apply_live_merge(..., copy=False) ниже мутировал его in-place — и живой
    бар всё равно оказывался в кэше. Для крипты TTL кеша 3600с, поэтому
    последняя свеча «застывала» на час или расходилась с /api/last-bar
    (симптом «10.07 / 10.12, лагают»).
    apply_live_merge перед _store_cache, и живой бар «запекался» в кеш
    (TTL крипты — 3600с). Дальше merge накладывался ПОВЕРХ запечённого
    бара: при переподключении WS старый бар оставался в кеше, и последняя
    свеча на графике застывала или расходилась с /api/last-bar
    (симптом «10.07 / 10.12, лагают»). Live-мерж делается ровно один раз
    — на выходе из get_series_df / get_cached_last_bar.
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
    # В кэш — КОПИЯ: вызывающий может применить copy=False merge in-place.
    _store_cache((symbol, tf), merged.copy(), history_limit)
    return merged


def _extend_newer(symbol: str, tf: str, df: pd.DataFrame,
                  history_limit: int) -> pd.DataFrame:
    """Дописывает в df ЗАКРЫТЫЕ бары правее правого края кэша.

    Кэш крипты живёт TTL 3600с, и его правая граница сама не двигается:
    REST-ответ застал бары по момент загрузки, а live_bars хранит ТОЛЬКО
    формирующийся бар. Между краем кэша и live-баром поэтому «теряются»
    закрытые свечи: полный reload отдавал «REST-старый + ОДИН живой бар»,
    и фронт после reload терял уже нарисованные закрытые свечи — разрывы
    между барами и «гигантская» хвостовая свеча, прыгающая телом/тенями.

    Закрытые бары (строго > последнего бара df и строго < live-бара)
    добираются из REST и мержатся в df/кэш. Формирующийся live-бар в кэш
    НЕ запекается (timestamp >= live_ts отфильтровывается) — он остаётся в
    live_bars и накладывается live-мержем на выходе (см. apply_live_merge).

    Дёшево: если между краем и live-баром нет закрытых свечей (обычный тик,
    соседние бары) — один проход без сети.
    """
    try:
        from app_pkg.data.live import _live_bars_lock, live_bars
        with _live_bars_lock:
            src = live_bars.get((symbol, tf))
            bar = dict(src) if src else None
    except Exception:  # noqa: BLE001
        return df
    if not bar or df is None or df.empty:
        return df

    last_ts = pd.to_datetime(df["timestamp"].iloc[-1])
    if getattr(last_ts, "tz", None) is None:
        last_ts = last_ts.tz_localize("UTC")
    live_ts = pd.to_datetime(int(bar.get("ts", 0)), unit="s", utc=True)
    if live_ts <= last_ts:
        return df  # live не впереди кэша — свежих закрытых баров нет

    step = config.TF_SECONDS.get(tf, 60)
    gap_bars = int((live_ts - last_ts).total_seconds() / step)
    if gap_bars <= 1:
        return df  # соседние бары — закрытых в промежутке нет
    need = gap_bars + 1
    now = time.time()

    if need > _EXTEND_NEWER_CAP:
        # Кэш сильно отстал (сервер молчал дольше интервалов×CAP) — дешевле
        # перезалить хвост целиком, чем гнать десятки маленьких запросов.
        try:
            rebuilt = fetch_ohlcv(symbol, tf, limit=history_limit, end_sec=now)
        except Exception as exc:  # noqa: BLE001
            logger.warning("extend newer (full) %s %s failed: %s", symbol, tf, exc)
            return df
        if rebuilt is None or rebuilt.empty:
            return df
        rebuilt = rebuilt[rebuilt["timestamp"] < live_ts]  # forming не запекаем
        rebuilt = (rebuilt.drop_duplicates(subset=["timestamp"])
                          .sort_values("timestamp")
                          .reset_index(drop=True))
        if rebuilt.empty:
            return df
        if len(rebuilt) > history_limit:
            rebuilt = rebuilt.tail(history_limit).reset_index(drop=True)
        _store_cache((symbol, tf), rebuilt.copy(), history_limit)
        return rebuilt

    try:
        newer = fetch_ohlcv(symbol, tf, limit=need, end_sec=now)
    except Exception as exc:  # noqa: BLE001
        logger.warning("extend newer %s %s failed: %s", symbol, tf, exc)
        return df
    if newer is None or newer.empty:
        return df
    newer = newer[(newer["timestamp"] > last_ts) &
                  (newer["timestamp"] < live_ts)]
    if newer.empty:
        return df
    merged = pd.concat([df, newer], ignore_index=True)
    merged = (merged.drop_duplicates(subset=["timestamp"])
                    .sort_values("timestamp")
                    .reset_index(drop=True))
    if len(merged) > history_limit:
        merged = merged.tail(history_limit).reset_index(drop=True)
    _store_cache((symbol, tf), merged.copy(), history_limit)
    return merged


def get_series_df(symbol: str, tf: str, limit: int = 1000,
                  force: bool = False, history_limit: int | None = None) -> pd.DataFrame:
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
                # ВАЖНО: даже при попадании в кеш последний бар подтягиваем
                # из live_bars (apply_live_merge). Иначе для 1m крипты с TTL
                # 3600с /api/last-bar и /api/data час отдают один и тот же
                # устаревший бар, и новая свеча на графике не появляется.
                cached = _extend_newer(symbol, tf, cached, history_limit)
                return _tail(apply_live_merge(symbol, tf, cached), limit)

            # В кэше меньше, чем просят, а история ещё есть — докачиваем.
            merged = _topup_older(symbol, tf, cached, target - len(cached),
                                  history_limit)
            base = merged if merged is not None else cached
            # _topup_older отдаёт df без live-мержа (см. её докстроку) —
            # бар накладываем на выходе, как и в остальных ветках. Свежие
            # закрытые бары (правый край кэша) добираем здесь же.
            base = _extend_newer(symbol, tf, base, history_limit)
            return _tail(apply_live_merge(symbol, tf, base), limit)

    # Холодный кеш (или force): тянем history_limit, не limit.
    # В кеш кладём ЧИСТЫЕ данные источника (без live-мержа) — live-бар
    # накладывается только на выходе, иначе он «запекается» в кеш на TTL.
    df = fetch_ohlcv(symbol, tf, limit=history_limit)
    if df is None or df.empty:
        logger.warning("empty ohlcv %s %s (limit=%s) — в кэш не пишется",
                       symbol, tf, history_limit)
        return df if df is not None else _empty_df()
    _store_cache(key, df.copy(), history_limit)
    if len(df) < target:
        # Источник вернул меньше запрошенного (ограниченное окно) — добираем
        # старые бары, чтобы cold-запрос сразу получил нужный объём.
        merged = _topup_older(symbol, tf, df, target - len(df), history_limit)
        if merged is not None:
            df = merged
    # Единственный live-мерж: на выходе. copy=False безопасен, т.к. df здесь
    # — либо копия из _topup_older, либо свежая копия перед записью в кеш
    # (fetch_ohlcv возвращает новый DataFrame, а _tail ниже режет его).
    df = _extend_newer(symbol, tf, df, history_limit)
    return _tail(apply_live_merge(symbol, tf, df, copy=False), limit)


def get_replay_df(symbol: str, tf: str, from_sec: float | None = None,
                  to_sec: float | None = None, limit: int = 1200) -> pd.DataFrame:
    """Исторические данные для бэктеста (без live-мержа)."""
    limit = int(limit or config.REPLAY_LIMIT)
    df = fetch_ohlcv(symbol, tf, limit=limit, start_sec=from_sec, end_sec=to_sec)
    if df is None or df.empty:
        return _empty_df()
    df = df.sort_values("timestamp").reset_index(drop=True)
    if len(df) > limit:
        df = df.iloc[-limit:].reset_index(drop=True)
    return df