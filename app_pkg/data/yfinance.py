"""Загрузка данных форекс/металлов через yfinance.

yfinance импортируется лениво внутри функции, чтобы модуль не тянул
тяжёлые зависимости при импорте пакета.
"""

import logging

import pandas as pd

from app_pkg import config

logger = logging.getLogger(__name__)

_EMPTY_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def yf_symbol(symbol: str) -> str:
    """Тикер Yahoo для символа: форекс получает суффикс =X, XAUUSD — GC=F."""
    if symbol in config.YF_TICKER:
        return config.YF_TICKER[symbol]
    if symbol in config.FOREX_SYMBOLS:
        return symbol + "=X"
    return symbol


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=_EMPTY_COLUMNS)


def fetch_yfinance(symbol: str, interval: str, period: str | None = None,
                   limit: int | None = None, start_sec: float | None = None,
                   end_sec: float | None = None) -> pd.DataFrame:
    """Скачивает данные и приводит к схеме timestamp/open/high/low/close/volume.

    Период выбирается по таймфрейму: 1m->2d, 5m->5d, 15m->1mo,
    1H->1mo, 4H->3mo, 1D->max.
    """
    import yfinance as yf

    if period is None:
        period = config.YF_PERIODS.get(interval, "1mo")
    yint = config.BINANCE_TF.get(interval, interval)
    ticker = yf_symbol(symbol)

    try:
        raw = yf.download(
            ticker,
            period=period,
            interval=yint,
            progress=False,
            auto_adjust=False,
            threads=False,
            group_by="column",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("yfinance download %s (%s) failed: %s", ticker, yint, exc)
        return _empty_df()

    if raw is None or raw.empty:
        return _empty_df()

    def _norm(col):
        if isinstance(col, tuple):
            col = col[0]
        return str(col).lower()

    raw = raw.reset_index()
    raw.columns = [_norm(c) for c in raw.columns]
    raw = raw.rename(columns={"datetime": "timestamp", "date": "timestamp"})

    try:
        ts = pd.to_datetime(raw["timestamp"])
        ts = ts.dt.tz_localize("UTC") if ts.dt.tz is None else ts.dt.tz_convert("UTC")
        out = pd.DataFrame({
            "timestamp": ts,
            "open": raw["open"].astype(float),
            "high": raw["high"].astype(float),
            "low": raw["low"].astype(float),
            "close": raw["close"].astype(float),
            "volume": raw["volume"].astype(float),
        })
    except Exception as exc:  # noqa: BLE001
        logger.warning("yfinance normalize %s failed: %s", ticker, exc)
        return _empty_df()

    if start_sec is not None:
        out = out[out["timestamp"] >= pd.to_datetime(int(start_sec), unit="s", utc=True)]
    if end_sec is not None:
        out = out[out["timestamp"] <= pd.to_datetime(int(end_sec), unit="s", utc=True)]
    if limit is not None and len(out) > int(limit):
        out = out.iloc[-int(limit):]
    out = out.sort_values("timestamp").reset_index(drop=True)
    return out