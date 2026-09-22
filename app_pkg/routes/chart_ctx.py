"""Blueprint: /api/chart-context — данные графика + контекст для ИИ."""

import logging

from flask import Blueprint, jsonify, request

from app_pkg import config, utils
from app_pkg.ai.context import _compute_multi_tf_trends, build_multi_tf_context
from app_pkg.data.fetch import get_replay_df, get_series_df
from app_pkg.db import db_get_all_drawings
from app_pkg.indicators import slice_payload

bp = Blueprint("chart_ctx", __name__)
log = logging.getLogger(__name__)


def _parse_int(value, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@bp.route("/api/chart-context")
def api_chart_context():
    symbol = request.args.get("symbol", "BTCUSDT").upper()
    timeframe = request.args.get("timeframe", config.DEFAULT_TIMEFRAME)
    mode = request.args.get("mode", "live")
    replay_index = _parse_int(request.args.get("replay_index"))
    limit = _parse_int(request.args.get("limit"), config.DEFAULT_LIMIT)
    visible_from = _parse_int(request.args.get("visible_from"))
    visible_to = _parse_int(request.args.get("visible_to"))

    if symbol not in config.SYMBOLS:
        return jsonify({"error": "Invalid symbol"}), 400
    if timeframe not in config.TF_SECONDS:
        return jsonify({"error": "Invalid timeframe"}), 400

    df = None
    replay_state = {"active": False}
    visible_range = {}

    if mode == "replay":
        from_sec = _parse_int(request.args.get("from"))
        to_sec = _parse_int(request.args.get("to"))
        df = get_replay_df(symbol, timeframe, from_sec, to_sec, limit=limit)
        if df is not None and not df.empty:
            replay_index = min(max(replay_index or 0, 0), len(df) - 1)
            replay_state = {"active": True, "index": replay_index, "total": len(df)}
            visible_range = {
                "from": int(utils.epoch_secs(df["timestamp"])[0]),
                "to": int(utils.epoch_secs(df["timestamp"])[-1]),
            }
    else:
        df = get_series_df(symbol, timeframe, limit=limit, history_limit=limit)
        if df is not None and not df.empty:
            visible_range = {
                "from": int(utils.epoch_secs(df["timestamp"])[0]),
                "to": int(utils.epoch_secs(df["timestamp"])[-1]),
            }

    if df is not None and not df.empty and visible_from is not None:
        mask = utils.epoch_secs(df["timestamp"]) >= visible_from
        df = df[mask]
    if df is not None and not df.empty and visible_to is not None:
        mask = utils.epoch_secs(df["timestamp"]) <= visible_to
        df = df[mask]

    candles, indicators, last_price, total = slice_payload(
        df, replay_index if mode == "replay" else None)

    # Replay: ИИ не должен видеть данные после replay_index. upto_sec —
    # время последней ВИДИМОЙ свечи (после slice_payload), сужается
    # окном пользователя (visible_to), если то задано.
    upto_sec = None
    if mode == "replay" and candles:
        upto_sec = int(candles[-1]["time"])
        if visible_to is not None:
            upto_sec = min(upto_sec, int(visible_to))

    return jsonify({
        "symbol": symbol,
        "timeframe": timeframe,
        "mode": mode,
        "candles": candles,
        "indicators": indicators,
        "visible_range": visible_range,
        "drawings": db_get_all_drawings(symbol, timeframe),
        "replay_state": replay_state,
        "last_price": last_price,
        "multi_tf_trends": _compute_multi_tf_trends(symbol, timeframe, upto_sec),
        "ai_context": build_multi_tf_context(symbol, timeframe, upto_sec)
        if mode == "replay" else None,
        "agent_mode": config.GROQ_AGENT_MODE,
        "generated_at": utils.now_sec(),
    })