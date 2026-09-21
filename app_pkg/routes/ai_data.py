# -*- coding: utf-8 -*-
"""Blueprint: /api/ai-data (вкладка «🧠 Данные для ИИ»).

GET  /api/ai-data/raw?symbol=BTCUSDT&timeframe=15m
     — чистый JSON с цифрами (Market Snapshot): технические индикаторы,
       статистика сканера, сентимент. Только числа, без текста.
GET  /api/ai-data/prompt
     — текущий системный промпт AI Backtest (config/charon_prompt.txt);
       создаётся с дефолтными правилами, если файла ещё нет.
POST /api/ai-data/prompt
     — body {"prompt": "..."} — сохранить промпт в config/charon_prompt.txt.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from flask import Blueprint, jsonify, request

from app_pkg import config
from app_pkg.ai.prompts import charon_prompt_text, save_charon_prompt
from app_pkg.data.market_snapshot import get_raw_market_data

bp = Blueprint("ai_data", __name__)
log = logging.getLogger(__name__)

# Расчёт сырых данных уходит в отдельный пул с жёстким таймаутом: медленный
# форекс-источник (MT5/yfinance) НЕ должен держать thread Waitress и вешать
# панель. По таймауту отдаётся пустой снимок (числа None) — контракт для ИИ
# сохраняется, сервер не «зависает».
_RAW_DATA_TIMEOUT_SEC = 20.0
_raw_exec = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ai-data-raw")


def _empty_snapshot():
    """Пустой снимок Market Snapshot (числа None) при таймауте/ошибке."""
    return {
        "technicals": {
            "rsi": None, "atr": None, "bb_pct_b": None,
            "sma20_diff_pct": None, "close": None,
        },
        "scanner_edge": {
            "winrate": None, "sharpe": None, "max_dd": None, "params": {},
        },
        "sentiment": {"ls_ratio": 0, "long_pct": 0, "fear_greed": 50},
    }


def _valid_pair(symbol, timeframe):
    """True, если пара входит в конфигурацию терминала."""
    return symbol in config.SYMBOLS and timeframe in config.TF_SECONDS


@bp.route("/api/ai-data/raw", methods=["GET"])
def api_ai_data_raw():
    """Сырые данные (только цифры) для ИИ по текущей паре symbol+timeframe."""
    symbol = str(request.args.get("symbol") or "").upper()
    timeframe = str(request.args.get("timeframe") or "").strip()
    if not symbol:
        return jsonify({"error": "symbol is required"}), 400
    if timeframe not in config.TF_SECONDS:
        return jsonify({"error": f"Invalid timeframe: {timeframe}"}), 400
    future = _raw_exec.submit(get_raw_market_data, symbol, timeframe)
    try:
        data = future.result(timeout=_RAW_DATA_TIMEOUT_SEC)
    except Exception:  # noqa: BLE001 — медленный/битый источник (TimeoutError в т.ч.)
        log.warning("ai-data raw timeout/error for %s %s (limit %.0fs)",
                    symbol, timeframe, _RAW_DATA_TIMEOUT_SEC)
        future.cancel()
        data = _empty_snapshot()
    return jsonify(data)


@bp.route("/api/ai-data/prompt", methods=["GET"])
def api_ai_data_prompt_get():
    """Текущий системный промпт AI Backtest."""
    return jsonify({"prompt": charon_prompt_text()})


@bp.route("/api/ai-data/prompt", methods=["POST"])
def api_ai_data_prompt_post():
    """Сохранить системный промпт: body {"prompt": "..."}."""
    body = request.get_json(silent=True) or {}
    try:
        saved = save_charon_prompt(body.get("prompt"))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"ok": True, "prompt": saved})