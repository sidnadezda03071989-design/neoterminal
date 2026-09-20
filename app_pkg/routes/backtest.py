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
                          replay_limit=None,  # полная история (BACKTEST_MAX_CANDLES)
                          tp_atr=None, sl_atr=None)
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
    для отрисовки.

    body: {symbol, timeframe, strategy, params, limit=20000, dataset=test}
    dataset: "train" | "test" | "full" — какая часть истории гонится
    (BLOCK-33). По умолчанию "test" (последние 30% — out-of-sample, цифра
    совпадает с TEST Trades сканера).

    TP/SL НЕ применяются (BLOCK-36): сканер считает чистые сигнальные выходы,
    и визуализация должна показывать РОВНО те сделки, что в панели сканера
    (95 == 95, а не 95 vs 321). Сделки закрываются по сигналу или в конце
    данных (exit_reason="end").

    Возвращает: {trades_full, metrics, candles_used, dataset, dataset_range,
    bars_from, bars_to, train_range, test_range}
    """
    body = request.get_json(silent=True) or {}
    symbol = str(body.get("symbol", "BTCUSDT")).upper()
    timeframe = str(body.get("timeframe", "15m"))
    strategy = str(body.get("strategy", "sma_cross"))
    params = body.get("params") or {}
    dataset = str(body.get("dataset", config.BACKTEST_DATASET_DEFAULT))
    try:
        limit = int(body.get("limit", config.BACKTEST_MAX_CANDLES))
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
    if dataset not in ("train", "test", "full"):
        return jsonify({"error": f"Invalid dataset: {dataset}"}), 400

    # Прогон на последних N свечах (limit). from_sec — за пределами окна:
    # 365 дней (SCAN_PERIOD_DAYS), как сканер. НЕ limit × TF_SECONDS: при
    # пропусках в данных (форекс) 20000 свечей занимают больше 208 дней, и
    # по limit×tf_sec фетч вернул бы меньше свечей (14295), чем скан —
    # окна разошлись бы, и сделки не совпали бы с панелью (BLOCK-36).
    from_sec = int(time.time()) - config.SCAN_PERIOD_DAYS * 86400
    result = run_backtest(
        symbol, timeframe,
        from_sec=from_sec, to_sec=int(time.time()),
        strategy_name=strategy, params=params,
        initial_cash=10000,
        replay_limit=limit,
        # Без TP/SL (BLOCK-36): те же сигнальные выходы, что в сканере.
        tp_atr=None, sl_atr=None,
        dataset=dataset,
    )
    if "error" in result:
        return jsonify(result), 400

    bars_from = result.get("bars_from")
    bars_to = result.get("bars_to")
    return jsonify({
        "symbol": symbol,
        "timeframe": timeframe,
        "strategy": strategy,
        "params": params,
        "trades_full": result.get("trades_full", []),
        "candles_used": result.get("candles_used"),
        "dataset": dataset,
        "dataset_range": {"from": bars_from, "to": bars_to},
        "bars_from": bars_from,
        "bars_to": bars_to,
        "train_range": result.get("train_range"),
        "test_range": result.get("test_range"),
        "metrics": {
            "total_return": result.get("total_return"),
            "sharpe_ratio": result.get("sharpe_ratio"),
            "winrate": result.get("win_rate"),
            "max_drawdown": result.get("max_drawdown"),
            "total_trades": result.get("total_trades"),
        },
    })