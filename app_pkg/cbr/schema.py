# -*- coding: utf-8 -*-
"""CBR-база для Харона — схема SQLite и жизненный цикл подключения.

Таблица `snapshots` хранит компактный 44-мерный вектор рыночного снимка
(порядок — app_pkg.cbr.store.FEATURE_ORDER, совпадает с колонками
`labels.charon_features`) плюс аннотации: режим (classify_regime), сессию,
вердикт LLM и уровни. Поля `outcome/exit_*` заполняются асинхронным
backfill'ом (app_pkg.cbr.backfill) по triple-barrier конвенции labels.py
(SL-приоритет). Отдельная БД от основной (data.db), чтобы тяжёлый
бэкфилл не блокировал торговый коннект.

Единственный уникальный ключ — feature_hash (md5 float32-вектора):
повторный снимок того же состояния рынка молча игнорируется (INSERT OR
IGNORE в store_snapshot).
"""

import sqlite3
from pathlib import Path

from app_pkg import config

# Схема — единственный источник правды. Изменения БД -> также обновить
# `.clinerules`-правило и держать init_db() в актуальном виде.
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    ts INTEGER NOT NULL,             -- Unix-секунды бара снимка
    features TEXT NOT NULL,          -- JSON-массив [44] float32
    feature_hash TEXT NOT NULL,      -- md5(features.tobytes()) hex
    regime TEXT,                     -- trend_up|trend_down|flat|reversal
    session TEXT,                    -- asia|london|london_ny|ny|off_hours|weekend
    verdict TEXT,                    -- L|S|F|NULL (сигнал LLM на этом снимке)
    levels_json TEXT,                -- [{"delta": .., "prob": ..}] или NULL
    entry_price REAL,
    atr_pct REAL,                    -- ATR(14)/entry_price (барьеры backfill)
    source TEXT NOT NULL DEFAULT 'backtest',
    -- ---- outcome (заполняется backfill_outcomes; NULL = pending) ----
    outcome INTEGER,                 -- +1/-1/0
    bars_to_outcome INTEGER,
    exit_price REAL,
    max_up REAL,                     -- % экскурсии вверх в окне
    max_dn REAL,                     -- % экскурсии вниз в окне
    pnl_pct REAL,                    -- % по направлению вердикта
    backfilled_at REAL               -- Unix-сек backfill'а
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_snapshots_hash
    ON snapshots(feature_hash);
CREATE INDEX IF NOT EXISTS idx_snapshots_key
    ON snapshots(symbol, timeframe, ts);
CREATE INDEX IF NOT EXISTS idx_snapshots_pending
    ON snapshots(symbol, timeframe) WHERE outcome IS NULL;
CREATE INDEX IF NOT EXISTS idx_snapshots_outcome
    ON snapshots(outcome) WHERE outcome IS NOT NULL;
"""


def init_db(path=None, conn=None):
    """Открывает (или принимает) CBR-подключение: WAL + busy_timeout + схема.

    path — путь к файлу БД (по умолчанию config.CBR_DB_PATH); ":memory:" —
    временная БД теста (WAL-прагма вернёт 'memory'). Если передан готовый
    conn — на нём выполняется схема и он возвращается как есть.
    """
    if conn is not None:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
        return conn
    path = str(path or config.CBR_DB_PATH)
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(path, check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA busy_timeout=5000")
    c.executescript(SCHEMA_SQL)
    c.commit()
    return c


def connect(path=None):
    """Синоним init_db(path): открыть CBR-БД и применить схему."""
    return init_db(path=path)


def migrate(conn=None):
    """Этап 1: миграций нет, схема создаётся init_db. Идемпотентный no-op,
    который гарантирует наличие таблицы на переданном conn."""
    if conn is None:
        return None
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn