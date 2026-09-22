"""Blueprint: /api/drawings CRUD + /api/ai-draw."""

import logging
import uuid

import pandas as pd
from flask import Blueprint, jsonify, request

from app_pkg import config, utils
from app_pkg.ai.agents import _normalize_model_drawings
from app_pkg.data.fetch import get_series_df
from app_pkg.db import (
    db_add_drawing,
    db_clear_drawings,
    db_delete_drawing,
    db_get_all_drawings,
    db_update_drawing,
)

bp = Blueprint("drawings", __name__)
log = logging.getLogger(__name__)


@bp.route("/api/drawings", methods=["GET", "POST", "DELETE"])
def api_drawings():
    if request.method == "GET":
        symbol = request.args.get("symbol") or None
        timeframe = request.args.get("timeframe") or None
        return jsonify({"drawings": db_get_all_drawings(symbol, timeframe)})

    if request.method == "POST":
        body = request.get_json(silent=True) or {}
        d = body.get("drawing") or body
        d.setdefault("id", str(uuid.uuid4()))
        created_by = d.get("created_by", "user")
        db_add_drawing(d, created_by=created_by)
        return jsonify({"drawings": db_get_all_drawings()}), 201

    # DELETE
    created_by = request.args.get("created_by")
    db_clear_drawings(created_by)
    return jsonify({"cleared": True})


@bp.route("/api/drawings/import", methods=["POST"])
def api_drawings_import():
    body = request.get_json(silent=True) or {}
    raw = body.get("drawings")
    if not isinstance(raw, list):
        return jsonify({"error": "drawings array required"}), 400
    import uuid as _uuid
    for d in raw:
        if not isinstance(d, dict):
            continue
        nd = dict(d)
        nd["id"] = str(nd.get("id") or _uuid.uuid4().hex[:12])
        nd.setdefault("points", [])
        nd.setdefault("color", "#2962ff")
        nd.setdefault("label", "")
        nd.setdefault("type", "trendline")
        db_add_drawing(nd, created_by="user")
    return jsonify({"imported": len(raw), "drawings": db_get_all_drawings()})


@bp.route("/api/drawings/update", methods=["POST"])
def api_drawings_update():
    body = request.get_json(silent=True) or {}
    drawing = body.get("drawing") or body
    did = drawing.get("id")
    if not did:
        return jsonify({"error": "Missing id"}), 400
    ok = db_update_drawing(did, drawing)
    return jsonify({"updated": ok, "drawing": drawing})


@bp.route("/api/drawings/<drawing_id>", methods=["DELETE"])
def api_drawings_delete(drawing_id):
    ok = db_delete_drawing(drawing_id)
    return jsonify({"removed": ok})


@bp.route("/api/ai-draw", methods=["POST"])
def api_ai_draw():
    """Нормализует рисунки от ИИ и сохраняет их с created_by='ai'."""
    body = request.get_json(silent=True) or {}
    symbol = body.get("symbol", "BTCUSDT")
    timeframe = body.get("timeframe", config.DEFAULT_TIMEFRAME)
    raw_drawings = body.get("drawings") or []
    if timeframe not in config.TF_SECONDS:
        return jsonify({"error": "Invalid timeframe"}), 400

    df = get_series_df(symbol, timeframe, limit=100)
    candles = []
    if df is not None and not df.empty:
        for _, r in df.iterrows():
            candles.append({
                "time": int(pd.Timestamp(r["timestamp"]).timestamp()),
                "price": utils._clean(r["close"]),
            })

    normalized = _normalize_model_drawings(raw_drawings, candles)
    for d in normalized[:config.MAX_AI_DRAWABLES]:
        d["id"] = str(uuid.uuid4())
        d["created_by"] = "ai"
        db_add_drawing(d, created_by="ai")
    return jsonify({
        "drawings": normalized,
        "all_drawings": db_get_all_drawings(symbol, timeframe),
    })