# -*- coding: utf-8 -*-
"""Blueprint: /api/backtest, /api/backtest/strategies."""

import logging

from flask import Blueprint, jsonify, request

from app_pkg import config
from app_pkg.ai.backtest import run_backtest, STRATEGY_MAP

bp = Blueprint("backtest", __name__)
log = logging.getLogger(__name__)


@bp.route("/api/backtest", methods=["POST"])
def api_backtest():
    body = request.get_json(silent=True) or {}
    symbol = str(body.get("symbol", "BTCUSDT")).upper()
    tf = str(body.get("timeframe", config.DEFAULT_TIMEFRAME))
    strategy = str(body.get("strategy", "sma_cross"))
    params = body.get("params") or {}
    from_sec = body.get("from")
    to_sec = body.get("to")
    try:
        from_sec = int(from_sec) if from_sec else None
        to_sec = int(to_sec) if to_sec else None
        initial_cash = float(body.get("initial_cash", 10000))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid numeric params"}), 400

    if symbol not in config.SYMBOLS:
        return jsonify({"error": "Invalid symbol"}), 400
    if tf not in config.TF_SECONDS:
        return jsonify({"error": "Invalid timeframe"}), 400
    if strategy not in STRATEGY_MAP:
        return jsonify({"error": f"Unknown strategy: {strategy}"}), 400

    result = run_backtest(symbol, tf, from_sec, to_sec,
                          strategy, params, initial_cash,
                          replay_limit=body.get("limit"))
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)


@bp.route("/api/backtest/strategies")
def api_backtest_strategies():
    return jsonify({
        "strategies": {k: v.__name__ for k, v in STRATEGY_MAP.items()}
    })