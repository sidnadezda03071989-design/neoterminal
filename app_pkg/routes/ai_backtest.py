"""Blueprint: /api/ai-backtest (AI Backtest «Псевдо-Харон»).

POST /api/ai-backtest           — локальный расчёт уровней вероятностей
                                   -> полный результат
GET  /api/ai-backtest/<run_id>  — результат + levels (in-memory/DB)

Уровни считаются по ТЕКУЩЕМУ срезу данных:

  • mode=replay + upto_sec — контекст обрезан по времени барьера реплея
    (ни одна свеча из будущего в расчёт не попадает);
  • mode=live — последняя доступная свеча.
Данные и контекст — те же, что у локального rules-движка.
"""

import logging

from flask import Blueprint, jsonify, request

from app_pkg import config
from app_pkg.ai.ai_backtest import get_run, run_ai_backtest
from app_pkg.db import db_get_ai_backtest

bp = Blueprint("ai_backtest", __name__)
log = logging.getLogger(__name__)


def _parse_upto_sec(body):
    """upto_sec из body: только для mode=replay и только положительное."""
    raw = body.get("upto_sec")
    if raw in (None, "", False):
        return None
    try:
        val = int(float(raw))
    except (TypeError, ValueError):
        return None
    return val if val > 0 else None


@bp.route("/api/ai-backtest", methods=["POST"])
def api_ai_backtest_start():
    """Запуск локального расчёта уровней вероятностей.

    body: {symbol, timeframe, mode, upto_sec}
      mode      — "live" (по умолчанию) | "replay";
      upto_sec  — время барьера реплея (только для mode=replay).
    Возвращает полный локальный результат сразу.
    """
    body = request.get_json(silent=True) or {}
    symbol = str(body.get("symbol", "BTCUSDT")).upper()
    timeframe = str(body.get("timeframe", config.DEFAULT_TIMEFRAME))
    mode = "replay" if str(body.get("mode", "live")).lower() == "replay" \
        else "live"
    upto_sec = _parse_upto_sec(body) if mode == "replay" else None

    if symbol not in config.SYMBOLS:
        return jsonify({"error": "Invalid symbol"}), 400
    if timeframe not in config.TF_SECONDS:
        return jsonify({"error": "Invalid timeframe"}), 400

    result = run_ai_backtest(symbol, timeframe, upto_sec=upto_sec,
                             mode=mode)
    return jsonify(result)


@bp.route("/api/ai-backtest/<run_id>")
def api_ai_backtest_result(run_id):
    """Результат расчёта: in-memory (живой run), иначе — из БД."""
    run = get_run(run_id)
    if not run:
        run = db_get_ai_backtest(run_id)
    if not run:
        return jsonify({"error": "AI backtest run not found"}), 404
    return jsonify(run)
