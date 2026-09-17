# -*- coding: utf-8 -*-
"""Blueprint: /metrics (Prometheus text format) и /ws (SSE event-stream)."""

from flask import Blueprint, Response

from app_pkg.metrics import render_metrics
from app_pkg.ws import event_stream

bp = Blueprint("metrics", __name__)


@bp.route("/metrics")
def prometheus_metrics():
    """Метрики в формате Prometheus."""
    return Response(render_metrics(), mimetype="text/plain; charset=utf-8")


@bp.route("/ws")
def sse_stream():
    """SSE-поток событий для браузера."""
    return Response(
        event_stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
