# -*- coding: utf-8 -*-
"""Prometheus-совместимые метрики NeoTerminal.

- latency      — гистограмма-заменитель с квантилями 0.5/0.95;
- cache_hits / cache_misses — счётчики обращений к кешам ИИ.
"""

import os
import platform
import threading
import time

from app_pkg.config import METRICS_QUANTILES

_METRICS = {}
_METRICS_LOCK = threading.Lock()

_START_TIME = time.time()


def _record_metric(metric_type, value, labels=None):
    """Записать точку (ts, value) в бакет (metric_type, labels)."""
    key = (metric_type, tuple(sorted((labels or {}).items())))
    with _METRICS_LOCK:
        bucket = _METRICS.setdefault(key, [])
        bucket.append((time.time(), float(value)))
        if len(bucket) > 1000:
            _METRICS[key] = bucket[-500:]


def _quantile(values, q):
    """Простой квантиль по отсортированным значениям (nearest rank)."""
    if not values:
        return 0.0
    sv = sorted(values)
    pos = min(len(sv) - 1, max(0, int(round(q * (len(sv) - 1)))))
    return sv[pos]


def render_metrics():
    """Текст в формате Prometheus."""
    lines = []
    with _METRICS_LOCK:
        for (mtype, lbls), points in sorted(_METRICS.items()):
            if not points:
                continue
            values = [p[1] for p in points]
            labels_str = ",".join(f'{k}="{v}"' for k, v in lbls)
            lbl = f"{{{labels_str}}}" if labels_str else ""
            name = f"neo_{mtype}"
            lines.append(f"# HELP {name} {mtype} values")
            lines.append(f"# TYPE {name} summary")
            for q in METRICS_QUANTILES:
                lines.append(
                    f'{name}{lbl},quantile="{q}" {_quantile(values, q):.6f}')
            lines.append(
                f'{name}{lbl},quantile="0.99" {_quantile(values, 0.99):.6f}')
            lines.append(f"{name}_count{lbl} {len(values)}")
            lines.append(f"{name}_sum{lbl} {sum(values):.6f}")

    lines.append('# HELP neo_info NeoTerminal info')
    lines.append(
        f'neo_info{{python="{platform.python_version()}",pid="{os.getpid()}"}} 1')
    lines.append("# HELP neo_uptime_seconds Uptime, seconds")
    lines.append(f"neo_uptime_seconds {time.time() - _START_TIME:.0f}")
    return "\n".join(lines) + "\n"