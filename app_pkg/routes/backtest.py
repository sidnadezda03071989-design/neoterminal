# -*- coding: utf-8 -*-
"""Blueprint: /api/backtest, /api/backtest/strategies."""

import logging
import time

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
        tp_atr = float(body.get("tp_atr")) if body.get("tp_atr") else None
        sl_atr = float(body.get("sl_atr")) if body.get("sl_atr") else None
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid numeric params"}), 400

    if tp_atr is not None and tp_atr <= 0:
        return jsonify({"error": "tp_atr must be > 0"}), 400
    if sl_atr is not None and sl_atr <= 0:
        return jsonify({"error": "sl_atr must be > 0"}), 400

    if symbol not in config.SYMBOLS:
        return jsonify({"error": "Invalid symbol"}), 400
    if tf not in config.TF_SECONDS:
        return jsonify({"error": "Invalid timeframe"}), 400
    if strategy not in STRATEGY_MAP:
        return jsonify({"error": f"Unknown strategy: {strategy}"}), 400

    result = run_backtest(symbol, tf, from_sec, to_sec,
                          strategy, params, initial_cash,
                          replay_limit=body.get("limit"),
                          tp_atr=tp_atr, sl_atr=sl_atr)
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)


@bp.route("/api/backtest/strategies")
def api_backtest_strategies():
    return jsonify({
        "strategies": {k: v.__name__ for k, v in STRATEGY_MAP.items()}
    })


@bp.route("/api/backtest/trades", methods=["POST"])
def api_backtest_trades():
    """Прогнать backtest для конкретной комбинации и вернуть все сделки
    с TP/SL для отрисовки.

    body: {symbol, timeframe, strategy, params, limit=1000}
    Возвращает: {trades_full: [...], metrics: {...}}
    """
    body = request.get_json(silent=True) or {}
    symbol = str(body.get("symbol", "BTCUSDT")).upper()
    timeframe = str(body.get("timeframe", "15m"))
    strategy = str(body.get("strategy", "sma_cross"))
    params = body.get("params") or {}
    try:
        limit = int(body.get("limit", 1000))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid limit"}), 400
    if limit <= 0:
        return jsonify({"error": "limit must be > 0"}), 400

    # Валидация symbol/timeframe/strategy — как в api_backtest
    if symbol not in config.SYMBOLS:
        return jsonify({"error": "Invalid symbol"}), 400
    if timeframe not in config.TF_SECONDS:
        return jsonify({"error": "Invalid timeframe"}), 400
    if strategy not in STRATEGY_MAP:
        return jsonify({"error": f"Unknown strategy: {strategy}"}), 400

    # Прогон на последних N свечах (limit)
    from_sec = int(time.time()) - limit * config.TF_SECONDS.get(timeframe, 60)
    result = run_backtest(
        symbol, timeframe,
        from_sec=from_sec, to_sec=int(time.time()),
        strategy_name=strategy, params=params,
        initial_cash=10000,
        replay_limit=limit,
        tp_atr=2.0, sl_atr=1.0,
    )
    if "error" in result:
        return jsonify(result), 400

    return jsonify({
        "symbol": symbol,
        "timeframe": timeframe,
        "strategy": strategy,
        "params": params,
        "trades_full": result.get("trades_full", []),
        "metrics": {
            "total_return": result.get("total_return"),
            "sharpe_ratio": result.get("sharpe_ratio"),
            "winrate": result.get("win_rate"),
            "max_drawdown": result.get("max_drawdown"),
            "total_trades": result.get("total_trades"),
        },
    })