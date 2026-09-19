"""SQLite-хранилище NeoTerminal.

WAL-режим, одна глобальная connection с threading.Lock.
Все операции с БД — только через функции этого модуля.
"""

import json
import logging
import sqlite3
import threading

from app_pkg import config, utils

_conn = None
_db_lock = threading.Lock()
_logger = None


def _log():
    global _logger
    if _logger is None:
        _logger = logging.getLogger(__name__)
    return _logger


def _get_db() -> sqlite3.Connection:
    """Ленивая инициализация подключения (WAL + схема)."""
    global _conn
    if _conn is None:
        with _db_lock:
            if _conn is None:
                config.DATA_DIR.mkdir(parents=True, exist_ok=True)
                conn = sqlite3.connect(str(config.DB_PATH), check_same_thread=False)
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA busy_timeout=5000")
                _init_db(conn)
                _conn = conn
                _log().info("SQLite ready: %s", config.DB_PATH)
    return _conn


def _init_db(conn) -> None:
    """Создаёт таблицы, если их ещё нет."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS drawings (
            id TEXT PRIMARY KEY,
            symbol TEXT,
            timeframe TEXT,
            type TEXT,
            points_json TEXT,
            color TEXT,
            label TEXT,
            created_by TEXT,
            created_at TEXT,
            updated_at TEXT
        );
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT,
            content TEXT,
            symbol TEXT,
            ts TEXT
        );
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            condition TEXT,
            value REAL,
            channel TEXT,
            destination TEXT,
            triggered INTEGER DEFAULT 0,
            active INTEGER DEFAULT 1,
            created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            direction TEXT,
            entry_price REAL,
            sl_price REAL,
            tp_price REAL,
            size REAL,
            status TEXT,
            pnl REAL,
            created_at TEXT,
            closed_at TEXT,
            close_price REAL
        );
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            direction TEXT,
            entry_price REAL,
            exit_price REAL,
            size REAL,
            pnl REAL,
            r_ratio REAL,
            created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS scan_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,              -- UUID запуска сканера
            symbol TEXT, timeframe TEXT,
            strategy TEXT,
            params_json TEXT,         -- параметры комбинации
            train_json TEXT,          -- метрики на train (70%)
            test_json TEXT,           -- метрики на test (30%, out-of-sample)
            combined_sharpe REAL,     -- min(train_sharpe, test_sharpe)
            total_trades INTEGER,
            created_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_scan_run ON scan_results(run_id);
        """
    )
    conn.commit()

# ------------------------------------------------------------------ drawings
def _row_to_drawing(row) -> dict:
    d = dict(row)
    d["points"] = json.loads(d.get("points_json") or "[]")
    d.pop("points_json", None)
    return d


def db_add_drawing(d: dict, created_by: str = None) -> None:
    conn = _get_db()
    with _db_lock:
        ts = utils.now_iso()
        conn.execute(
            "INSERT OR REPLACE INTO drawings "
            "(id, symbol, timeframe, type, points_json, color, label, "
            " created_by, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                str(d.get("id") or ""),
                d.get("symbol") or "",
                d.get("timeframe") or "",
                d.get("type") or "",
                json.dumps(d.get("points") or [], ensure_ascii=False),
                d.get("color") or "",
                d.get("label") or "",
                created_by or "",
                ts,
                ts,
            ),
        )
        conn.commit()


def db_update_drawing(drawing_id: str, updates: dict) -> bool:
    """Обновляет набор полей рисунка. Возвращает True, если строки затронуты."""
    if not updates:
        return False
    fields, values = [], []
    for key in ("symbol", "timeframe", "type", "color", "label", "created_by"):
        if key in updates:
            fields.append(f"{key}=?")
            values.append(updates[key])
    if "points" in updates:
        fields.append("points_json=?")
        values.append(json.dumps(updates["points"] or [], ensure_ascii=False))
    if not fields:
        return False
    fields.append("updated_at=?")
    values.append(utils.now_iso())
    values.append(drawing_id)
    conn = _get_db()
    with _db_lock:
        cur = conn.execute(
            f"UPDATE drawings SET {', '.join(fields)} WHERE id=?", values
        )
        conn.commit()
        return cur.rowcount > 0


def db_delete_drawing(drawing_id: str) -> bool:
    conn = _get_db()
    with _db_lock:
        cur = conn.execute("DELETE FROM drawings WHERE id=?", (drawing_id,))
        conn.commit()
        return cur.rowcount > 0


def db_clear_drawings(created_by: str = None) -> int:
    conn = _get_db()
    with _db_lock:
        if created_by:
            cur = conn.execute("DELETE FROM drawings WHERE created_by=?", (created_by,))
        else:
            cur = conn.execute("DELETE FROM drawings")
        conn.commit()
        return cur.rowcount


def db_get_all_drawings(symbol: str = None, timeframe: str = None) -> list:
    """Все рисунки.

    Важно: если задан symbol/timeframe — фильтруем, но глобальные записи
    (symbol='' или timeframe='') возвращаем тоже.
    """
    conn = _get_db()
    sql = "SELECT * FROM drawings"
    clauses, params = [], []
    if symbol:
        clauses.append("(symbol=? OR symbol='')")
        params.append(symbol)
    if timeframe:
        clauses.append("(timeframe=? OR timeframe='')")
        params.append(timeframe)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY created_at ASC, id ASC"
    with _db_lock:
        rows = conn.execute(sql, params).fetchall()
    return [_row_to_drawing(r) for r in rows]

# ---------------------------------------------------------------- чат
def db_append_chat(role: str, content: str, symbol: str = None) -> int:
    conn = _get_db()
    with _db_lock:
        cur = conn.execute(
            "INSERT INTO chat_messages (role, content, symbol, ts) VALUES (?,?,?,?)",
            (role, content, symbol or "", utils.now_iso()),
        )
        conn.commit()
        return cur.lastrowid


def db_get_chat_history(limit: int = 50) -> list:
    """Последние сообщения (старые -> новые)."""
    conn = _get_db()
    limit = max(1, int(limit or 50))
    with _db_lock:
        rows = conn.execute(
            "SELECT * FROM chat_messages ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    rows = list(reversed(rows))
    return [dict(r) for r in rows]


def db_clear_chat() -> int:
    conn = _get_db()
    with _db_lock:
        cur = conn.execute("DELETE FROM chat_messages")
        conn.commit()
        return cur.rowcount

# --------------------------------------------------------------- алерты
def db_get_alerts(active_only: bool = True) -> list:
    conn = _get_db()
    sql = "SELECT * FROM alerts"
    if active_only:
        sql += " WHERE active=1"
    sql += " ORDER BY id DESC"
    with _db_lock:
        rows = conn.execute(sql).fetchall()
    return [dict(r) for r in rows]


def db_add_alert(symbol, condition, value, channel=None, destination=None) -> int:
    conn = _get_db()
    with _db_lock:
        cur = conn.execute(
            "INSERT INTO alerts (symbol, condition, value, channel, destination, "
            "triggered, active, created_at) VALUES (?,?,?,?,?,0,1,?)",
            (symbol, condition, value, channel or "", destination or "", utils.now_iso()),
        )
        conn.commit()
        return cur.lastrowid


def db_delete_alert(alert_id) -> bool:
    conn = _get_db()
    with _db_lock:
        cur = conn.execute("DELETE FROM alerts WHERE id=?", (alert_id,))
        conn.commit()
        return cur.rowcount > 0


def db_mark_alert_triggered(alert_id) -> int:
    """Помечает алерт сработавшим.

    Намеренно БЕЗ conn.commit() в конце — коммит контролирует вызывающий
    код, чтобы десятки алертов не делали пачку отдельных транзакций.
    """
    conn = _get_db()
    with _db_lock:
        cur = conn.execute("UPDATE alerts SET triggered=1 WHERE id=?", (alert_id,))
        return cur.rowcount

# -------------------------------------------------------------- позиции
def db_open_position(symbol, direction, entry_price, sl_price=None,
                     tp_price=None, size=None) -> int:
    conn = _get_db()
    with _db_lock:
        cur = conn.execute(
            "INSERT INTO positions (symbol, direction, entry_price, sl_price, "
            "tp_price, size, status, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (symbol, direction, entry_price, sl_price, tp_price, size, "open", utils.now_iso()),
        )
        conn.commit()
        return cur.lastrowid


def db_close_position(position_id, close_price, pnl=None) -> bool:
    conn = _get_db()
    with _db_lock:
        cur = conn.execute(
            "UPDATE positions SET status='closed', close_price=?, pnl=?, closed_at=? "
            "WHERE id=? AND status='open'",
            (close_price, pnl, utils.now_iso(), position_id),
        )
        conn.commit()
        return cur.rowcount > 0


def db_get_positions(open_only: bool = True) -> list:
    conn = _get_db()
    if open_only:
        sql = "SELECT * FROM positions WHERE status='open'"
    else:
        sql = "SELECT * FROM positions"
    sql += " ORDER BY created_at DESC, id DESC"
    with _db_lock:
        rows = conn.execute(sql).fetchall()
    return [dict(r) for r in rows]


def db_log_trade(symbol, direction, entry_price, exit_price,
                 size=None, pnl=None, r_ratio=None) -> int:
    conn = _get_db()
    with _db_lock:
        cur = conn.execute(
            "INSERT INTO trades (symbol, direction, entry_price, exit_price, "
            "size, pnl, r_ratio, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (symbol, direction, entry_price, exit_price, size, pnl, r_ratio, utils.now_iso()),
        )
        conn.commit()
        return cur.lastrowid


def db_commit() -> None:
    """Коммит текущей транзакции.

    Используется фоновыми циклами, которые выполняют серию записей
    (например _alerts_loop помечает несколько алертов одним коммитом).
    """
    conn = _get_db()
    with _db_lock:
        conn.commit()


# ------------------------------------------------------------------ scan
def _row_to_scan_result(row) -> dict:
    d = dict(row)
    d["params"] = json.loads(d.pop("params_json") or "{}")
    d["train"] = json.loads(d.pop("train_json") or "{}")
    d["test"] = json.loads(d.pop("test_json") or "{}")
    return d


def db_save_scan_result(run_id, symbol, tf, strategy, params, train, test,
                        combined_sharpe, total_trades) -> int:
    """Сохраняет результат одной комбинации grid-search сканера."""
    conn = _get_db()
    with _db_lock:
        cur = conn.execute(
            "INSERT INTO scan_results (run_id, symbol, timeframe, strategy, "
            "params_json, train_json, test_json, combined_sharpe, "
            "total_trades, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                run_id, symbol, tf, strategy,
                json.dumps(params or {}, ensure_ascii=False),
                json.dumps(train or {}, ensure_ascii=False),
                json.dumps(test or {}, ensure_ascii=False),
                combined_sharpe, total_trades, utils.now_iso(),
            ),
        )
        conn.commit()
        return cur.lastrowid


def db_get_scan_results(run_id, limit=None) -> list:
    """Результаты прогона, топ по combined_sharpe (DESC).

    limit по умолчанию — config.SCAN_TOP_N; для экспорта CSV передаётся
    большой limit (все результаты).
    """
    limit = max(1, int(limit or config.SCAN_TOP_N))
    conn = _get_db()
    with _db_lock:
        rows = conn.execute(
            "SELECT * FROM scan_results WHERE run_id=? "
            "ORDER BY combined_sharpe DESC, id ASC LIMIT ?",
            (run_id, limit),
        ).fetchall()
    return [_row_to_scan_result(r) for r in rows]


def db_clear_scan_run(run_id) -> int:
    """Удаляет все результаты прогона (повторный запуск / чистка тестов)."""
    conn = _get_db()
    with _db_lock:
        cur = conn.execute("DELETE FROM scan_results WHERE run_id=?", (run_id,))
        conn.commit()
        return cur.rowcount