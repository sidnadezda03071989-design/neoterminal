"""NeoTerminal — пакет бэкенда, фабрика create_app()."""

# ВАЖНО: .env должен быть загружен ДО любых импортов, читающих os.getenv
# на уровне модуля (app_pkg.config читает getenv при импорте). Если
# config импортируется раньше load_dotenv() — значения зафиксированы
# пустыми и всё уходит в heuristic fallback.
from pathlib import Path

from dotenv import load_dotenv

_env_file = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_file)

# только после этого — остальные импорты:
from flask import Flask, jsonify


def create_app():
    from app_pkg.logging_setup import setup_logging
    setup_logging()

    base = Path(__file__).resolve().parent.parent
    app = Flask(
        __name__,
        template_folder=str(base / "templates"),
        static_folder=str(base / "static"),
    )
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    # Статика всегда перепроверяется (300/not-modified), чтобы браузер не
    # держал старые ES-модули после правок JS (кнопки/панели «не нажимались»).
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

    from app_pkg.routes import register_routes
    register_routes(app)

    # Доп. блоки Market Snapshot: crowd/macro/calendar/derivatives/news_sentiment
    # (см. app_pkg/routes/extra_data.py — имена blueprint'ов оттуда).
    from app_pkg.routes.extra_data import (
        derivatives_bp,
        macro_bp,
        news_sentiment_bp,
        sentiment_bp,
    )
    for bp in (sentiment_bp, macro_bp, derivatives_bp, news_sentiment_bp):
        app.register_blueprint(bp)

    @app.errorhandler(404)
    def not_found(_e):
        return jsonify({"error": "Not found"}), 404

    return app