"""Загрузка рыночных данных с Binance (REST klines).

Сессия с HTTPAdapter(max_retries=2, pool_connections=4, pool_maxsize=8).
При ошибке основного хоста выполняется fallback на api1.binance.com.
"""

import logging
import time

import pandas as pd
import requests
from requests.adapters import HTTPAdapter

from app_pkg import config

logger = logging.getLogger(__name__)

_BASE_URLS = (
    "https://api.binance.com",
    "https://api1.binance.com",
)

_session = requests.Session()
_session.mount(
    "https://",
    HTTPAdapter(max_retries=2, pool_connections=4, pool_maxsize=8),
)
_session.mount(
    "http://",
    HTTPAdapter(max_retries=2, pool_connections=4, pool_maxsize=8),
)

_EMPTY_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

# Binance REST отдаёт максимум 1000 свечей за один запрос (klines: limit <= 1000).
_KLINES_PAGE_LIMIT = 1000
# Пауза между страницами: удерживает rate-limit Binance при глубокой истории.
_PAGE_SLEEP_SECONDS = 0.1


def _binance_get(path: str, params: dict = None) -> requests.Response:
    """GET на Binance с fallback на api1.binance.com."""
    last_err = None
    for base in _BASE_URLS:
        try:
            resp = _session.get(base + path, params=params,
                                timeout=config.BINANCE_API_TIMEOUT)
            if resp.status_code == 200:
                return resp
            last_err = RuntimeError(f"{base}{path}: HTTP {resp.status_code} "
                                    f"{resp.text[:160]}")
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            logger.warning("Binance %s%s failed: %s", base, path, exc)
    raise last_err or RuntimeError("binance request failed")


def _klines_to_df(rows_json) -> pd.DataFrame:
    """Приводит ответ /api/v3/klines к схеме timestamp/open/high/low/close/volume.

    timestamp — datetime64 в UTC; пустой список даёт пустой DataFrame
    с той же схемой колонок.
    """
    rows = []
    for k in rows_json or []:
        rows.append((
            pd.to_datetime(k[0], unit="ms", utc=True),
            float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]),
        ))
    df = pd.DataFrame(rows, columns=_EMPTY_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def fetch_binance(symbol: str, interval: str, limit: int = 1000,
                  start_sec: float = None, end_sec: float = None) -> pd.DataFrame:
    """Возвращает DataFrame с колонками timestamp/open/high/low/close/volume.

    timestamp — datetime64 в UTC; пустой результат — пустой DataFrame
    с той же схемой колонок.
    """
    params = {"symbol": symbol, "interval": interval, "limit": int(limit)}
    if start_sec:
        params["startTime"] = int(float(start_sec) * 1000)
    if end_sec:
        params["endTime"] = int(float(end_sec) * 1000)

    resp = _binance_get("/api/v3/klines", params)
    return _klines_to_df(resp.json())


def fetch_binance_paged(symbol: str, interval: str, total: int = _KLINES_PAGE_LIMIT,
                        start_sec: float = None, end_sec: float = None) -> pd.DataFrame:
    """Постраничная загрузка до `total` свечей (максимум 1000 за запрос).

    Пагинация идёт назад во времени: первая страница берётся с endTime
    (или без него — самые свежие бары), каждая следующая — с endTime,
    сдвинутым на 1 секунду назад от самого раннего timestamp предыдущей
    пачки. Если пачка вернула меньше 1000 строк — история кончилась, цикл
    прерывается. Пока start_sec не достигнут, пагинация тоже прекращается.

    Возвращает DataFrame со схемой timestamp/open/high/low/close/volume,
    отсортированный по timestamp (не более `total` последних свечей в
    окне [start_sec, end_sec]); при отсутствии данных — пустой DataFrame.
    """
    total = max(1, int(total or _KLINES_PAGE_LIMIT))
    start_ms = int(float(start_sec) * 1000) if start_sec else None
    end_ms = int(float(end_sec) * 1000) if end_sec else None

    pages = []  # страницы в порядке от новых к старым
    fetched = 0
    prev_earliest_ms = None

    while fetched < total:
        params = {"symbol": symbol, "interval": interval,
                  "limit": _KLINES_PAGE_LIMIT}
        if end_ms is not None:
            params["endTime"] = end_ms
        try:
            resp = _binance_get("/api/v3/klines", params)
        except Exception as exc:
            # Первая страница — ошибка наружу; на следующих отдаём то,
            # что уже успели скачать, вместо потери всей истории.
            if not pages:
                raise
            logger.warning("binance paging %s %s stopped: %s", symbol, interval, exc)
            break
        chunk = resp.json() or []
        if not chunk:
            break
        pages.append(chunk)
        fetched += len(chunk)

        earliest_ms = min(int(k[0]) for k in chunk)  # самый ранний timestamp пачки
        # Источник вернул тот же диапазон (endTime проигнорирован) — стоп,
        # иначе бесконечный цикл.
        if prev_earliest_ms is not None and earliest_ms >= prev_earliest_ms:
            break
        prev_earliest_ms = earliest_ms

        if len(chunk) < _KLINES_PAGE_LIMIT:
            break  # пачка короче страницы: раньше данных нет

        next_end_ms = earliest_ms - 1000  # -1 секунда от самого раннего бара
        if start_ms is not None and next_end_ms < start_ms:
            break
        end_ms = next_end_ms
        time.sleep(_PAGE_SLEEP_SECONDS)

    if not pages:
        return _klines_to_df([])

    df = pd.concat([_klines_to_df(page) for page in pages], ignore_index=True)
    df = (df.drop_duplicates(subset=["timestamp"])
            .sort_values("timestamp")
            .reset_index(drop=True))
    if start_ms is not None:
        df = df[df["timestamp"] >= pd.to_datetime(start_ms, unit="ms", utc=True)]
    if len(df) > total:
        df = df.tail(total).reset_index(drop=True)
    return df