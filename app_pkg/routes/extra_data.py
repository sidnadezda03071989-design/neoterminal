"""Blueprint'ы доп. блоков Market Snapshot.

GET /api/sentiment?symbol=BTCUSDT      — crowd (только крипта)
GET /api/macro?symbol=EURUSD          — макро-срез FRED
GET /api/calendar?symbol=EURUSD       — экономический календарь (по валюте)
GET /api/derivatives?symbol=BTCUSDT   — деривативы (только крипта)
GET /api/news-sentiment?symbol=...    — lexicon-сентимент ленты новостей

Единый контракт: без symbol -> 400; результат геттера наружу; если геттер
сказал «не применимо к активу» ({}), наружу идёт {"applicable": false,
"ok": false}. Внутри геттеров свои кеши/деградация — роуты их не трогают.
"""

import logging

from flask import Blueprint, jsonify, request

from app_pkg.data.derivatives import get_derivatives_snapshot
from app_pkg.data.macro import get_econ_calendar, get_macro_snapshot
from app_pkg.data.news_sentiment import get_news_sentiment
from app_pkg.data.sentiment import get_crowd_snapshot

log = logging.getLogger(__name__)

sentiment_bp = Blueprint("sentiment", __name__)
macro_bp = Blueprint("macro", __name__)
derivatives_bp = Blueprint("derivatives", __name__)
news_sentiment_bp = Blueprint("news_sentiment", __name__)


def _symbol_arg():
    """symbol из query-параметров (нормализованный) или None."""
    symbol = str(request.args.get("symbol") or "").upper().strip()
    return symbol or None


def _apply_not_applicable(payload):
    """{} от геттера -> «не применимо»; иначе данные как есть."""
    if not payload:
        return {"applicable": False, "ok": False}
    return payload


@sentiment_bp.route("/api/sentiment", methods=["GET"])
def api_sentiment():
    symbol = _symbol_arg()
    if not symbol:
        return jsonify({"error": "symbol is required"}), 400
    return jsonify(_apply_not_applicable(get_crowd_snapshot(symbol)))


@macro_bp.route("/api/macro", methods=["GET"])
def api_macro():
    symbol = _symbol_arg()
    if not symbol:
        return jsonify({"error": "symbol is required"}), 400
    return jsonify(get_macro_snapshot(symbol))


@macro_bp.route("/api/calendar", methods=["GET"])
def api_calendar():
    symbol = _symbol_arg()
    if not symbol:
        return jsonify({"error": "symbol is required"}), 400
    ccy = symbol[:3] if (len(symbol) == 6 and symbol.isalpha()) else ""
    return jsonify(_apply_not_applicable(get_econ_calendar(ccy)))


@derivatives_bp.route("/api/derivatives", methods=["GET"])
def api_derivatives():
    symbol = _symbol_arg()
    if not symbol:
        return jsonify({"error": "symbol is required"}), 400
    return jsonify(_apply_not_applicable(get_derivatives_snapshot(symbol)))


@news_sentiment_bp.route("/api/news-sentiment", methods=["GET"])
def api_news_sentiment():
    symbol = _symbol_arg()
    if not symbol:
        return jsonify({"error": "symbol is required"}), 400
    limit = request.args.get("limit", default=30, type=int)
    return jsonify(get_news_sentiment(symbol, limit=limit))