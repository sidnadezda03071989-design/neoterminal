"""Регистрация blueprint'ов и замер latency запросов.

Все маршруты объявлены отдельными blueprint-модулями в этом пакете.
"""

import time

from flask import request

from app_pkg import metrics as metrics_core


def register_routes(app):
    from app_pkg.routes.ai import bp as ai_bp
    from app_pkg.routes.ai_backtest import bp as ai_backtest_bp
    from app_pkg.routes.ai_data import bp as ai_data_bp
    from app_pkg.routes.alerts import bp as alerts_bp
    from app_pkg.routes.backtest import bp as backtest_bp
    from app_pkg.routes.charon import bp as charon_bp
    from app_pkg.routes.chart_ctx import bp as chart_ctx_bp
    from app_pkg.routes.chat import bp as chat_bp
    from app_pkg.routes.data import bp as data_bp
    from app_pkg.routes.drawings import bp as drawings_bp
    from app_pkg.routes.main import bp as main_bp
    from app_pkg.routes.metrics import bp as infra_bp
    from app_pkg.routes.news import bp as news_bp
    from app_pkg.routes.notes import bp as notes_bp
    from app_pkg.routes.scanner import bp as scanner_bp
    from app_pkg.routes.scripts import bp as scripts_bp

    for bp in (main_bp, data_bp, chart_ctx_bp, drawings_bp, ai_bp,
               chat_bp, alerts_bp, backtest_bp, infra_bp, scanner_bp,
               notes_bp, ai_backtest_bp, news_bp, ai_data_bp, charon_bp,
               scripts_bp):
        app.register_blueprint(bp)

    @app.before_request
    def _start_timer():
        request.environ["_nt_start"] = time.time()

    @app.after_request
    def _stop_timer(resp):
        start = request.environ.pop("_nt_start", None)
        if start is not None:
            metrics_core._record_metric("latency", time.time() - start,
                                        {"path": request.path})
        return resp

    return app