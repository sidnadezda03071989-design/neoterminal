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
        CREATE TABLE IF NOT EXISTS ai_backtest_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            symbol TEXT,
            timeframe TEXT,
            params_json TEXT,
            metrics_json TEXT,
            signals_json TEXT,
            trades_json TEXT,
            created_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_ai_backtest_run ON ai_backtest_runs(run_id);
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        );
        CREATE TABLE IF NOT EXISTS asset_notes (
            symbol TEXT PRIMARY KEY,
            content TEXT,
            screenshots_json TEXT,
            created_at TEXT,
            updated_at TEXT
        );
        CREATE TABLE IF NOT EXISTS journal (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            content TEXT,
            screenshots_json TEXT,
            created_at TEXT,
            updated_at TEXT
        );
        """
    )
    conn.commit()

# ------------------------------------------------------------------ settings
def get_setting(key: str, default=None):
    """Значение настройки (settings.key -> value) или default, если её нет."""
    conn = _get_db()
    try:
        row = conn.execute(
            "SELECT value FROM settings WHERE key=?", (key,)
        ).fetchone()
        return row["value"] if row else default
    except sqlite3.Error:
        return default


def set_setting(key: str, value) -> None:
    """Сохранить настройку (upsert). value приводится к строке."""
    conn = _get_db()
    with _db_lock, conn:
        conn.execute(
            "INSERT INTO settings(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, str(value)),
        )


def get_scan_rr() -> float:
    """Базовый R/R скана (настройка UI). Фолбэк — config.BACKTEST_RR."""
    raw = get_setting("scan_rr")
    if raw is None:
        return config.BACKTEST_RR
    try:
        return float(raw)
    except (TypeError, ValueError):
        return config.BACKTEST_RR


def set_scan_rr(value) -> float:
    """Сохранить базовый R/R скана (с clamp по config.SCAN_RR_MIN/MAX).

    Кидает ValueError при нечисловом входе; возвращает сохранённое значение.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError("rr must be a number")
    v = max(config.SCAN_RR_MIN, min(config.SCAN_RR_MAX, v))
    set_setting("scan_rr", v)
    return v


# ------------------------------------------------------------------ drawings
def _row_to_drawing(row) -> dict:
    d = dict(row)
    d["points"] = json.loads(d.get("points_json") or "[]")
    d.pop("points_json", None)
    return d


def db_add_drawing(d: dict, created_by: str | None = None) -> None:
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


def db_clear_drawings(created_by: str | None = None) -> int:
    conn = _get_db()
    with _db_lock:
        if created_by:
            cur = conn.execute("DELETE FROM drawings WHERE created_by=?", (created_by,))
        else:
            cur = conn.execute("DELETE FROM drawings")
        conn.commit()
        return cur.rowcount


def db_get_all_drawings(symbol: str | None = None, timeframe: str | None = None) -> list:
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
def db_append_chat(role: str, content: str, symbol: str | None = None) -> int:
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


# ------------------------------------------------------------------ заметки
def _norm_attachments(raw) -> list:
    """Вложения в единый вид [{name, type, data}].

    Старые записи хранят dataURL-строки (скриншоты) — переводим в объекты.
    """
    out = []
    for it in raw or []:
        if isinstance(it, str):
            out.append({"name": "screenshot.png", "type": "image/png", "data": it})
        elif isinstance(it, dict) and isinstance(it.get("data"), str):
            out.append({
                "name": it.get("name") or "file",
                "type": it.get("type") or "application/octet-stream",
                "data": it["data"],
            })
    return out


def db_get_note(symbol: str):
    """Заметка по активу {symbol, content, screenshots(list), updated_at}
    или None, если заметки ещё нет."""
    conn = _get_db()
    with _db_lock:
        row = conn.execute(
            "SELECT * FROM asset_notes WHERE symbol=?", (symbol,)
        ).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["screenshots"] = _norm_attachments(json.loads(d.get("screenshots_json") or "[]"))
    d.pop("screenshots_json", None)
    return d


def db_save_note(symbol: str, content: str, screenshots: list) -> None:
    """Сохранить/обновить заметку по активу (UPSERT по symbol)."""
    ts = utils.now_iso()
    conn = _get_db()
    with _db_lock, conn:
        conn.execute(
            "INSERT INTO asset_notes (symbol, content, screenshots_json, "
            "created_at, updated_at) VALUES (?,?,?,?,?) "
            "ON CONFLICT(symbol) DO UPDATE SET content=excluded.content, "
            "screenshots_json=excluded.screenshots_json, "
            "updated_at=excluded.updated_at",
            (symbol, content, json.dumps(screenshots or [], ensure_ascii=False),
             ts, ts),
        )


def db_list_notes() -> list:
    """Все заметки, свежие сверху: [{symbol, content_preview,
    screenshots_count, updated_at}]."""
    conn = _get_db()
    with _db_lock:
        rows = conn.execute(
            "SELECT symbol, content, screenshots_json, updated_at "
            "FROM asset_notes ORDER BY updated_at DESC"
        ).fetchall()
    out = []
    for r in rows:
        shots = _norm_attachments(json.loads(r["screenshots_json"] or "[]"))
        text = (r["content"] or "").strip().replace("\n", " ")
        preview = text[:50] + ("…" if len(text) > 50 else "")
        out.append({
            "symbol": r["symbol"],
            "content_preview": preview,
            "screenshots_count": len(shots),
            "updated_at": r["updated_at"],
        })
    return out


def db_get_journal():
    """Одна глобальная запись журнала {content, screenshots, updated_at}
    или None."""
    conn = _get_db()
    with _db_lock:
        row = conn.execute("SELECT * FROM journal WHERE id=1").fetchone()
    if row is None:
        return None
    d = dict(row)
    d["screenshots"] = _norm_attachments(json.loads(d.get("screenshots_json") or "[]"))
    d.pop("screenshots_json", None)
    return d


def db_save_journal(content: str, screenshots: list) -> None:
    """Сохранить/обновить журнал (единственная запись id=1)."""
    ts = utils.now_iso()
    conn = _get_db()
    with _db_lock, conn:
        conn.execute(
            "INSERT INTO journal (id, content, screenshots_json, "
            "created_at, updated_at) VALUES (1,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET content=excluded.content, "
            "screenshots_json=excluded.screenshots_json, "
            "updated_at=excluded.updated_at",
            (content, json.dumps(screenshots or [], ensure_ascii=False),
             ts, ts),
        )


# ------------------------------------------------------------------ scan
def _row_to_scan_result(row) -> dict:
    d = dict(row)
    d["params"] = json.loads(d.pop("params_json") or "{}")
    d["train"] = json.loads(d.pop("train_json") or "{}")
    d["test"] = json.loads(d.pop("test_json") or "{}")
    return d


def db_save_scan_result(run_id, symbol, tf, strategy, params, train, test,
                        combined_sharpe, total_trades) -> int:
    """Сохраняет результат одной комбинации grid-search сканера.

    Для пакетной вставки списка строк (напр. импорт CSV прогона) —
    db_save_scan_results: executemany, одна транзакция на весь список.
    """
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


def _scan_float(value):
    """float() без падений: None/пустая строка/мусор -> None.

    Нужна при восстановлении train/test окон из плоских колонок
    CSV-выгрузки (export.csv), где пустая ячейка = метрика отсутствует.
    """
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def db_save_scan_results(run_id, results_list) -> int:
    """Сохраняет список результатов прогона ОДНОЙ транзакцией (executemany).

    results_list — список словарей; минимально нужны symbol, strategy,
    combined_sharpe. Остальные поля восстанавливаются:
      - params: dict ИЛИ JSON-строка (как в колонке params CSV-выгрузки);
      - train_sharpe/test_sharpe — скаляры из CSV-выгрузки -> заворачиваются
        в train/test окна {"sharpe": ...};
      - winrate/max_dd/profit_factor/trades — колонки test-окна CSV
        (out-of-sample, формат export.csv); если в словаре передан полный
        train/test dict (как из run_scan) — берётся он целиком;
      - total_trades — иначе сумма trades по train+test (0, если нет).
    Строки без symbol или strategy пропускаются (мусор/пустые строки CSV).
    Возвращает число вставленных строк.
    """
    rows = []
    now = utils.now_iso()
    for raw in results_list or []:
        if not isinstance(raw, dict):
            continue
        symbol = str(raw.get("symbol") or "").upper()
        strategy = str(raw.get("strategy") or "")
        if not symbol or not strategy:
            continue
        tf = str(raw.get("timeframe") or config.DEFAULT_TIMEFRAME)
        params = raw.get("params")
        if not isinstance(params, dict):
            try:
                params = json.loads(params) if params else {}
            except (TypeError, ValueError):
                params = {}
        train = raw.get("train") if isinstance(raw.get("train"), dict) else {}
        test = raw.get("test") if isinstance(raw.get("test"), dict) else {}
        if not train:
            train = {"sharpe": _scan_float(raw.get("train_sharpe"))}
        if not test:
            test = {
                "sharpe": _scan_float(raw.get("test_sharpe")),
                "winrate": _scan_float(raw.get("winrate")),
                "max_dd": _scan_float(raw.get("max_dd")),
                "profit_factor": _scan_float(raw.get("profit_factor")),
                "trades": int(_scan_float(raw.get("trades")) or 0),
            }
        total_trades = raw.get("total_trades")
        if total_trades is None:
            total_trades = (int(_scan_float(train.get("trades")) or 0)
                            + int(_scan_float(test.get("trades")) or 0))
        rows.append((
            run_id, symbol, tf, strategy,
            json.dumps(params or {}, ensure_ascii=False),
            json.dumps(train, ensure_ascii=False),
            json.dumps(test, ensure_ascii=False),
            _scan_float(raw.get("combined_sharpe")),
            int(total_trades or 0), now,
        ))
    if not rows:
        return 0
    conn = _get_db()
    with _db_lock:
        cur = conn.executemany(
            "INSERT INTO scan_results (run_id, symbol, timeframe, strategy, "
            "params_json, train_json, test_json, combined_sharpe, "
            "total_trades, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        conn.commit()
        return cur.rowcount


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


# ------------------------------------------------------------- AI backtest
def db_get_best_scan_stats(symbol, tf):
    """Лучшая комбинация сканера для symbol+tf (combined_sharpe DESC).

    Возвращает {strategy, params, combined_sharpe, winrate, sharpe} —
    out-of-sample (test) значения; None, если скана ещё не было.
    """
    conn = _get_db()
    with _db_lock:
        row = conn.execute(
            "SELECT * FROM scan_results WHERE symbol=? AND timeframe=? "
            "ORDER BY combined_sharpe DESC, id ASC LIMIT 1",
            (symbol, tf),
        ).fetchone()
    if row is None:
        return None
    r = _row_to_scan_result(row)
    test = r.get("test") or {}
    return {
        "run_id": r.get("run_id"),
        "symbol": r.get("symbol"),
        "timeframe": r.get("timeframe"),
        "strategy": r.get("strategy"),
        "params": r.get("params") or {},
        # train/test окна целиком (аддитивно): карточке «Статистика стратегий»
        # и ИИ-контексту нужны train_sharpe/test_sharpe для оценки overfit.
        "train": r.get("train") or {},
        "test": r.get("test") or {},
        "combined_sharpe": r.get("combined_sharpe"),
        "winrate": test.get("winrate"),
        "sharpe": test.get("sharpe"),
    }


def db_save_ai_backtest(run_id, symbol, tf, params, metrics, signals, trades) -> int:
    """Сохраняет завершённый прогон AI Backtest в БД.

    signals — уровни вероятностей ({"side","price","probability","diff",...});
    metrics/trades панель больше не показывает (уровни — помощник
    вероятностей, а не стратегия), но колонки схемы остаются для совместимости.
    """
    conn = _get_db()
    with _db_lock:
        cur = conn.execute(
            "INSERT INTO ai_backtest_runs (run_id, symbol, timeframe, "
            "params_json, metrics_json, signals_json, trades_json, created_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (
                run_id, symbol, tf,
                json.dumps(params or {}, ensure_ascii=False),
                json.dumps(metrics or {}, ensure_ascii=False),
                json.dumps(signals or [], ensure_ascii=False),
                json.dumps(trades or [], ensure_ascii=False),
                utils.now_iso(),
            ),
        )
        conn.commit()
        return cur.lastrowid


def db_get_ai_backtest(run_id):
    """Результат прогона AI Backtest из БД; None, если прогона не было."""
    conn = _get_db()
    with _db_lock:
        row = conn.execute(
            "SELECT * FROM ai_backtest_runs WHERE run_id=? ORDER BY id DESC "
            "LIMIT 1", (run_id,),
        ).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["params"] = json.loads(d.pop("params_json") or "{}")
    d["metrics"] = json.loads(d.pop("metrics_json") or "{}")
    d["signals_log"] = json.loads(d.pop("signals_json") or "[]")
    d["trades"] = json.loads(d.pop("trades_json") or "[]")
    # Уровни вероятностей лежат в signals_json (пишем в run_ai_backtest).
    # Отдаём и как "levels" — новая схема ответа (панель рисует только их).
    d["levels"] = d["signals_log"] if isinstance(d["signals_log"], list) else []
    d.setdefault("status", "finished")
    return d