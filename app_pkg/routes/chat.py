# -*- coding: utf-8 -*-
"""Blueprint: /api/chat, /api/chat/history, /api/chat/clear."""

import logging

from flask import Blueprint, jsonify, request

from app_pkg import config
from app_pkg.db import db_get_chat_history, db_clear_chat
from app_pkg.ai.chat import chat_with_model

bp = Blueprint("chat", __name__)
log = logging.getLogger(__name__)


@bp.route("/api/chat", methods=["POST"])
def api_chat():
    body = request.get_json(silent=True) or {}
    message = (body.get("message") or "").strip()
    if not message:
        return jsonify({"error": "Empty message"}), 400
    symbol = str(body.get("symbol", "BTCUSDT")).upper()
    timeframe = str(body.get("timeframe", config.DEFAULT_TIMEFRAME))
    mode = str(body.get("mode", "live"))
    upto_sec = None
    if mode == "replay" and body.get("replay_time") is not None:
        try:
            upto_sec = int(float(body["replay_time"]))
        except (TypeError, ValueError):
            upto_sec = None

    result = chat_with_model(message, symbol, timeframe, mode, upto_sec)
    return jsonify(result)


@bp.route("/api/chat/history")
def api_chat_history():
    limit = request.args.get("limit") or config.CHAT_HISTORY_LIMIT
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = config.CHAT_HISTORY_LIMIT
    return jsonify({"messages": db_get_chat_history(limit=limit)})


@bp.route("/api/chat/clear", methods=["POST"])
def api_chat_clear():
    n = db_clear_chat()
    return jsonify({"cleared": True, "removed": n})