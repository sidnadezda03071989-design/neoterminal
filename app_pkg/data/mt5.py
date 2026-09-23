"""Загрузка данных из MetaTrader 5.

Модуль MetaTrader5 импортируется лениво (на Windows с установленным
терминалом). Состояние подключения — в глобальном mt5_state.
"""

import logging
import threading
import time

import pandas as pd

from app_pkg import config

logger = logging.getLogger(__name__)

mt5_state = {"init": False, "ok": False, "detail": "", "symbols": []}
_mt5_lock = threading.Lock()
_mt5 = None  # module MetaTrader5 (лениво)

_EMPTY_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

_TF_MAP = {"1m": "M1", "5m": "M5", "15m": "M15", "1H": "H1", "4H": "H4", "1D": "D1"}

# суффиксы счётов/контрактов: EURUSD.a, EURUSDm, EURUSD.c ...
_SUFFIXES = ["", ".a", ".m", "-m", ".rm", "m", ".c", ".z", "-c"]


def _mt5_ensure() -> bool:
    """Инициализация терминала; при неудаче — автозапуск через MT5_PATH.

    Попытка 1: initialize() без параметров (терминал уже запущен).
    Если не вышло и задан MT5_PATH + MT5_AUTOSTART=1 — цикл попыток
    initialize(path=MT5_PATH): SDK сам запускает терминал, на это нужно
    время, поэтому ретраим до MT5_START_TIMEOUT секунд с паузой 2 сек.
    """
    global _mt5
    with _mt5_lock:
        if mt5_state["init"]:
            return mt5_state["ok"]
        mt5_state["init"] = True
        try:
            import MetaTrader5 as mt5_module
        except Exception as exc:  # noqa: BLE001
            mt5_state.update(ok=False, detail=f"MetaTrader5 не установлен: {exc}")
            return False
        _mt5 = mt5_module

        def _last_error() -> str:
            try:
                return str(mt5_module.last_error())
            except Exception:  # noqa: BLE001
                return ""

        def _try_init(**kwargs) -> bool:
            try:
                return bool(mt5_module.initialize(**kwargs))
            except Exception as exc:  # noqa: BLE001
                logger.warning("MT5 initialize error: %s", exc)
                return False

        # Попытка 1: подключение к уже запущенному терминалу.
        logger.info("MT5 init attempt 1...")
        ok = _try_init()
        logger.info("MT5 init attempt 1... ok: %s", str(ok).lower())
        # Автозапуск: терминал не запущен — SDK стартует его по MT5_PATH.
        if not ok and config.MT5_PATH and config.MT5_AUTOSTART:
            deadline = time.monotonic() + config.MT5_START_TIMEOUT
            attempt = 1
            while not ok and time.monotonic() < deadline:
                attempt += 1
                logger.info("MT5 init attempt %d (autostart via %s)...",
                            attempt, config.MT5_PATH)
                try:
                    mt5_module.shutdown()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("MT5 shutdown before re-init failed: %s", exc)
                ok = _try_init(path=config.MT5_PATH)
                logger.info("MT5 init attempt %d... ok: %s",
                            attempt, str(ok).lower())
                if ok:
                    break
                time.sleep(2)
        if not ok:
            detail = _last_error() or "initialize failed"
            mt5_state.update(ok=False, detail=detail)
            logger.warning("MT5 init failed: %s (fallback — yfinance)", detail)
            return False
        names = []
        try:
            symbols = mt5_module.symbols_get()
            if symbols:
                names = sorted(s.name for s in symbols)
        except Exception as exc:  # noqa: BLE001
            logger.warning("MT5 symbols_get error: %s", exc)
        mt5_state.update(ok=True, detail="connected", symbols=names)
        try:
            t_info = mt5_module.terminal_info()
            if t_info is not None:
                logger.info("MT5 terminal: %s (%s)",
                            t_info.path, t_info.company)
        except Exception as exc:  # noqa: BLE001, RUF100
            logger.warning("MT5 terminal_info error: %s", exc)
        logger.info("MT5 connected, %d symbols", len(names))
        return True


def _mt5_resolve_symbol(symbol: str) -> str:
    """Ищет тикер с суффиксами (.a, -m, m, .c ...), если точного нет."""
    names = mt5_state.get("symbols") or []
    if not names:
        return symbol
    low_to_orig = {s.lower(): s for s in names}
    if symbol in low_to_orig:
        return low_to_orig[symbol]
    low = symbol.lower()
    if low in low_to_orig:
        return low_to_orig[low]
    for suffix in _SUFFIXES:
        cand = (symbol + suffix).lower()
        if cand in low_to_orig:
            return low_to_orig[cand]
    return symbol


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=_EMPTY_COLUMNS)


def fetch_mt5(symbol: str, tf: str, limit: int = 1000,
              start_sec: float | None = None,
              end_sec: float | None = None) -> pd.DataFrame:
    """Копирует котировки из MT5. При недоступности MT5 — пустой DF."""
    if not _mt5_ensure():
        return _empty_df()
    resolved = _mt5_resolve_symbol(symbol)
    mtf = getattr(_mt5, "TIMEFRAME_" + _TF_MAP.get(tf, "M5"), None)
    if mtf is None:
        logger.warning("MT5: неизвестный таймфрейм %s", tf)
        return _empty_df()
    try:
        step = int(config.TF_SECONDS.get(tf, 300))
        # Независимо от варианта окна — tz-aware UTC datetime: без tzinfo SDK
        # трактует его как локальное время ПК и сдвигает период.
        dt_from = dt_to = None
        if start_sec is not None and end_sec is not None:
            # Задан полный период: copy_rates_range отдаёт максимально
            # возможную историю за окно; limit запрос не режет (это
            # предохранитель только для вариантов с одной границей).
            dt_from = pd.to_datetime(int(start_sec), unit="s", utc=True).to_pydatetime()
            dt_to = pd.to_datetime(int(end_sec), unit="s", utc=True).to_pydatetime()
            rates = _mt5.copy_rates_range(resolved, mtf, dt_from, dt_to)
        elif start_sec is not None:
            dt_from = pd.to_datetime(int(start_sec), unit="s", utc=True).to_pydatetime()
            dt_to = pd.to_datetime(
                int(start_sec)
                + int(limit * step * config.FOREX_CALENDAR_PADDING),
                unit="s", utc=True).to_pydatetime()
            logger.debug('MT5 auto-range %s %s: limit=%d -> [%s .. %s]',
                         resolved, tf, limit, dt_from, dt_to)
            rates = _mt5.copy_rates_range(resolved, mtf, dt_from, dt_to)
        elif end_sec is not None:
            dt_from = pd.to_datetime(
                int(end_sec)
                - int(limit * step * config.FOREX_CALENDAR_PADDING),
                unit="s", utc=True).to_pydatetime()
            dt_to = pd.to_datetime(int(end_sec), unit="s", utc=True).to_pydatetime()
            logger.debug('MT5 auto-range %s %s: limit=%d -> [%s .. %s]',
                         resolved, tf, limit, dt_from, dt_to)
            rates = _mt5.copy_rates_range(resolved, mtf, dt_from, dt_to)
        else:
            rates = _mt5.copy_rates_from_pos(resolved, mtf, 0, int(limit))
    except Exception as exc:  # noqa: BLE001
        logger.warning("MT5 copy_rates %s error: %s", resolved, exc)
        return _empty_df()
    if dt_from is not None:
        got = 0 if rates is None else len(rates)
        logger.info("MT5 range %s %s: %s .. %s -> %d bars",
                    resolved, tf, dt_from, dt_to, got)
    if rates is None or len(rates) == 0:
        logger.warning("MT5 fetch %s %s: пустой ответ copy_rates", resolved, tf)
        return _empty_df()
    df = pd.DataFrame(rates)
    df = df.rename(columns={"time": "timestamp"})
    # volume: приоритет real_volume (если есть ненулевые значения)
    # > tick_volume > 0.0. У форекса real_volume обычно нулевой.
    if "real_volume" in df.columns and (df["real_volume"].fillna(0) != 0).any():
        df["volume"] = df["real_volume"]
    elif "tick_volume" in df.columns:
        df["volume"] = df["tick_volume"]
    else:
        df["volume"] = 0.0
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    # Оставляем только нужные колонки (spread и прочее отбрасываем).
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info("MT5 fetch %s %s: %d rows", resolved, tf, len(df))
    return df