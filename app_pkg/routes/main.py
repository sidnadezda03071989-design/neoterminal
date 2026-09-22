"""Blueprint: корневые страницы, healthz, версия, статус MT5."""

from flask import Blueprint, jsonify, render_template

from app_pkg import config, utils

bp = Blueprint("main", __name__)


@bp.route("/")
def index():
    return render_template("index.html")


@bp.route("/healthz")
def healthz():
    return jsonify({"status": "ok", "ts": utils.now_iso()})


@bp.route("/api/version")
def api_version():
    return jsonify({"app": config.APP_NAME, "version": config.APP_VERSION})


@bp.route("/api/mt5-status")
def api_mt5_status():
    from app_pkg.data import mt5
    state = dict(mt5.mt5_state)
    state["resolved_symbols"] = {
        sym: mt5._mt5_resolve_symbol(sym) for sym in sorted(config.FOREX_SYMBOLS)
    }
    # список никогда не сериализуется целиком — оставляем компактную копию
    state.pop("symbols", None)
    return jsonify(state)