"""Blueprint: заметки по активу + торговый журнал (/api/notes, /api/journal).

GET  /api/notes/<symbol>  → заметка актива {content, screenshots, updated_at}
POST /api/notes/<symbol>  → сохранить/обновить (UPSERT)
GET  /api/notes           → список всех заметок [{symbol, content_preview,
                             screenshots_count, updated_at}]
GET  /api/journal         → блокнот журнала {content, screenshots, updated_at}
POST /api/journal         → сохранить журнал

screenshots — массив вложений: dataURL-строка (скриншот) или объект
{name, type, data}, где data — dataURL (файл или скриншот).
"""

import logging
import re

from flask import Blueprint, jsonify, request

from app_pkg.db import (
    db_get_journal,
    db_get_note,
    db_list_notes,
    db_save_journal,
    db_save_note,
)

bp = Blueprint("notes", __name__)
log = logging.getLogger(__name__)

_SYMBOL_RE = re.compile(r"^[A-Za-z0-9:._-]{1,24}$")
MAX_ATTACH = 12              # вложений на запись
MAX_ATTACH_CHARS = 12_000_000  # ~9 МБ на файл в dataURL (лимит локальной БД)


def _clean_screenshots(raw):
    """Валидация + нормализация вложений. None при невалидном виде."""
    if not isinstance(raw, list):
        return None
    out = []
    for it in raw:
        if isinstance(it, str):
            it = {"name": "screenshot.png", "type": "image/png", "data": it}
        if not isinstance(it, dict):
            return None
        name = it.get("name")
        typ = it.get("type")
        data = it.get("data")
        if not all(isinstance(x, str) for x in (name, typ, data)):
            return None
        if not data or len(data) > MAX_ATTACH_CHARS:
            return None
        out.append({"name": name, "type": typ, "data": data})
    return out[:MAX_ATTACH]


@bp.route("/api/notes", methods=["GET"])
def api_notes_list():
    return jsonify({"notes": db_list_notes()})


@bp.route("/api/notes/<symbol>", methods=["GET"])
def api_note_get(symbol):
    symbol = symbol.upper()
    note = db_get_note(symbol)
    if note is None:
        return jsonify({
            "symbol": symbol, "content": "", "screenshots": [],
            "updated_at": None,
        })
    return jsonify({
        "symbol": symbol,
        "content": note["content"] or "",
        "screenshots": note["screenshots"],
        "updated_at": note["updated_at"],
    })


@bp.route("/api/notes/<symbol>", methods=["POST"])
def api_note_save(symbol):
    symbol = symbol.upper()
    if not _SYMBOL_RE.match(symbol):
        return jsonify({"error": "Invalid symbol"}), 400
    body = request.get_json(silent=True) or {}
    content = body.get("content")
    if not isinstance(content, str):
        content = ""
    screenshots = _clean_screenshots(body.get("screenshots", []))
    if screenshots is None:
        return jsonify({
            "error": "screenshots must be an array of data URLs or {name,type,data}",
        }), 400
    db_save_note(symbol, content, screenshots)
    note = db_get_note(symbol)
    return jsonify({
        "symbol": symbol,
        "content": note["content"] or "",
        "screenshots": note["screenshots"],
        "updated_at": note["updated_at"],
    })


@bp.route("/api/journal", methods=["GET"])
def api_journal_get():
    j = db_get_journal()
    return jsonify({
        "content": j["content"] if j else "",
        "screenshots": j["screenshots"] if j else [],
        "updated_at": j["updated_at"] if j else None,
    })


@bp.route("/api/journal", methods=["POST"])
def api_journal_save():
    body = request.get_json(silent=True) or {}
    content = body.get("content")
    if not isinstance(content, str):
        content = ""
    screenshots = _clean_screenshots(body.get("screenshots", []))
    if screenshots is None:
        return jsonify({
            "error": "screenshots must be an array of data URLs or {name,type,data}",
        }), 400
    db_save_journal(content, screenshots)
    j = db_get_journal()
    return jsonify({
        "content": j["content"] if j else "",
        "screenshots": j["screenshots"] if j else [],
        "updated_at": j["updated_at"] if j else None,
    })