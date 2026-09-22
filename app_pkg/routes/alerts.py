"""Blueprint: /api/alerts, /api/alerts/<id>, /api/positions."""

import logging

from flask import Blueprint, jsonify, request

from app_pkg import config
from app_pkg.db import (
    db_add_alert,
    db_close_position,
    db_delete_alert,
    db_get_alerts,
    db_get_positions,
    db_log_trade,
    db_open_position,
)

bp = Blueprint("alerts", __name__)
log = logging.getLogger(__name__)


# -------------------------------------------------------------- ALERTS
@bp.route("/api/alerts", methods=["GET", "POST"])
def api_alerts():
    if request.method == "GET":
        active_only = request.args.get("active", "1") != "0"
        return jsonify({"alerts": db_get_alerts(active_only=active_only)})

    body = request.get_json(silent=True) or {}
    symbol = (body.get("symbol") or "").upper()
    if symbol not in config.SYMBOLS:
        return jsonify({"error": "Invalid symbol"}), 400
    condition = body.get("condition", "cross")
    try:
        value = float(body.get("value"))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid value"}), 400
    alert = db_add_alert(
        symbol=symbol,
        condition=condition,
        value=value,
        channel=body.get("channel", "browser"),
        destination=body.get("destination", ""),
    )
    return jsonify({"alert": alert, "id": alert}), 201


@bp.route("/api/alerts/<int:alert_id>", methods=["DELETE"])
def api_alert_delete(alert_id):
    ok = db_delete_alert(alert_id)
    return jsonify({"deleted": ok})


# ----------------------------------------------------------- POSITIONS
@bp.route("/api/positions", methods=["GET", "POST"])
def api_positions():
    if request.method == "GET":
        open_only = request.args.get("open", "1") != "0"
        return jsonify({"positions": db_get_positions(open_only=open_only)})

    body = request.get_json(silent=True) or {}
    if not body.get("symbol"):
        return jsonify({"error": "symbol required"}), 400
    try:
        entry_price = float(body["entry_price"])
        size = float(body.get("size", 1.0))
        sl = body.get("sl_price")
        tp = body.get("tp_price")
        sl = float(sl) if sl not in (None, "") else None
        tp = float(tp) if tp not in (None, "") else None
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid numbers"}), 400

    pos_id = db_open_position(
        symbol=body["symbol"],
        direction=str(body.get("direction", "buy")),
        entry_price=entry_price,
        sl_price=sl,
        tp_price=tp,
        size=size,
    )
    return jsonify({"position": {"id": pos_id}, "id": pos_id}), 201


@bp.route("/api/positions/<int:pos_id>/close", methods=["POST"])
def api_position_close(pos_id):
    body = request.get_json(silent=True) or {}
    close_price = body.get("price") or body.get("close_price")
    if close_price is None:
        return jsonify({"error": "price required"}), 400
    try:
        close_price = float(close_price)
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid price"}), 400

    pos = None
    for p in db_get_positions(open_only=True):
        if p["id"] == pos_id:
            pos = p
            break
    if not pos:
        return jsonify({"error": "Position not found"}), 404

    entry = float(pos["entry_price"])
    direction = str(pos.get("direction", "buy")).lower()
    size = float(pos.get("size") or 1.0)
    if direction in ("buy", "long"):
        pnl = (close_price - entry) * size
    else:
        pnl = (entry - close_price) * size

    ok = db_close_position(pos_id, close_price, pnl=round(pnl, 2))
    if not ok:
        return jsonify({"error": "Position already closed"}), 400
    db_log_trade(symbol=pos["symbol"], direction=direction,
                 entry_price=entry, exit_price=close_price,
                 size=size, pnl=round(pnl, 2), r_ratio=0)
    return jsonify({"position": {**pos, "pnl": round(pnl, 2), "closed": True}})