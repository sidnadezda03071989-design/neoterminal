"""Blueprint: новости по активу (/api/news/<symbol>)."""

import logging

from flask import Blueprint, jsonify, request

from app_pkg.data.news import get_symbol_news
from app_pkg.routes.scanner import api_scanner_stats

bp = Blueprint("news", __name__)
log = logging.getLogger(__name__)


@bp.route("/api/scanner/stats", methods=["GET"])
def scanner_stats_alias():
    """Алиас GET /api/scanner/stats (см. app_pkg/routes/scanner.py).

    Единая точка истины — scanner-блюпринт; здесь только проброс, чтобы
    «статистика стратегий» оставалась доступной и из news-модуля.
    """
    return api_scanner_stats()


@bp.route("/api/news/<symbol>", methods=["GET"])
def api_news(symbol):
    """Свежие новости по активу (Finnhub + CoinGecko, кеш NEWS_CACHE_TTL).

    ?refresh=1 — игнорировать кеш и сходить в сеть принудительно (кнопка ⟳).
    """
    symbol = (symbol or "").upper()
    refresh = request.args.get("refresh") == "1"
    resp = get_symbol_news(symbol, refresh=refresh)
    if not resp.get("news"):
        payload = dict(resp)
        payload["error"] = (
            "Новости получить не удалось: проверьте FINNHUB_API_KEY в .env "
            "и подключение к сети (Finnhub может лимитировать).")
        return jsonify(payload), 200
    return jsonify(resp)
