"""Изотоническая калибровка вероятностей Charon (детерминированный путь).

LLM убрана из сигнального контура; сырые вероятности apply_all_rules /
structure_levels калибруются по историческим hit-rate'ам из SQLite
(таблица charon_calibration_history). Кривая строится как монотонная
(изотоническая) piecewise-функция raw_prob -> empirical hit rate по
бакетам; сглаживание Лапласа не даёт кривой уйти за [0, 1].

Контракт:
  * calibrate(raw_prob, side, context_features) — без истории (<50
    сэмплов на сторону) возвращает raw_prob (fallback, как в спеке);
  * update_history(verdict, actual_outcome) — дозапись (raw_prob, hit)
    для переобучения;
  * export_curves() — отладочный экспорт {side: [(prob, hit_rate), ...]}.

Ни сетевых вызовов, ни случайности: один и тот же снимок БД даёт одну
и ту же кривую.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

from app_pkg import db

log = logging.getLogger(__name__)

# Минимум сэмплов на сторону, чтобы калибровка вступила в силу.
MIN_SAMPLES = 50
# Кэш кривых перечитывается не чаще, чем раз в CACHE_TTL секунд.
CACHE_TTL = 5.0

_CACHE_LOCK = threading.Lock()
_CACHE: dict[str, tuple[float, Any]] = {}
# side -> raw_prob -> эмпирический hit-rate.
_CURVES: dict[str, dict[int, float]] = {}


def _raw(raw_prob: Any) -> float:
    """Сырая вероятность в [0, 1]; None/нечисло -> 0.5 (нейтрально)."""
    try:
        return min(1.0, max(0.0, float(raw_prob)))
    except (TypeError, ValueError):
        return 0.5


def _normalize_side(side: Any) -> str:
    s = str(side or "").strip().upper()
    if s in ("UP", "U", "LONG", "L"):
        return "UP"
    if s in ("DOWN", "D", "SHORT", "S"):
        return "DOWN"
    return s or "UP"


def _interp(curve: dict[int, float], raw_prob: float) -> float:
    """Piecewise-линейная интерполяция по узлам кривой (sorted)."""
    if not curve:
        return raw_prob
    xs = sorted(curve)
    if raw_prob <= xs[0]:
        return curve[xs[0]]
    if raw_prob >= xs[-1]:
        return curve[xs[-1]]
    for i in range(1, len(xs)):
        x0, x1 = xs[i - 1], xs[i]
        if x0 <= raw_prob <= x1:
            if x1 == x0:
                return curve[x1]
            t = (raw_prob - x0) / (x1 - x0)
            return curve[x0] + t * (curve[x1] - curve[x0])
    return raw_prob


def _build_curve(rows: list[tuple[float, float]]) -> dict[int, float]:
    """[(prob, hit_rate), ...] -> изотоническая кривая raw_prob -> hit_rate.

    Узлы уже отбакетированы в db.db_get_calibration_curve; здесь только
    монотонизация (running-maximum) — hit-rate не падает с ростом raw_prob.
    """
    curve: dict[int, float] = {}
    for raw_prob, hit_rate in rows:
        node = round(_raw(raw_prob), 4)
        curve[node] = min(1.0, max(0.0, float(hit_rate)))
    if curve:
        xs = sorted(curve)
        run_max = curve[xs[0]]
        for x in xs:
            run_max = max(run_max, curve[x])
            curve[x] = run_max
    return curve


def _load_curves(force: bool = None) -> None:
    """Перечитать кривые из БД (с кэшем по TTL)."""
    if force is None:
        force = False
    now = time.time()
    with _CACHE_LOCK:
        stamp = _CACHE.get("_stamp", (0.0,))[0]
        if not force and now - stamp < CACHE_TTL:
            return
        curves: dict[str, dict[int, float]] = {}
        for side in ("UP", "DOWN"):
            rows = db.db_get_calibration_curve(side, min_samples=MIN_SAMPLES)
            if rows:
                curves[side] = _build_curve(rows)
        _CURVES.clear()
        _CURVES.update(curves)
        _CACHE["_stamp"] = (now,)


class IsotonicCalibrator:
    """Изотонический калибровщик вероятностей по истории SQLite."""

    def __init__(self, db_path: Any = None):
        # db_path принят для совместимости со спекой; подключение
        # управляется единым app_pkg.db (WAL + thread-local lock).
        self._db_path = db_path
        try:
            _load_curves(force=True)
        except Exception:  # noqa: BLE001 — инициализация без БД допустима
            log.debug("calibration curves not loaded yet", exc_info=True)

    def calibrate(self, raw_prob: float, side: str,
                  context_features: dict | None = None) -> float:
        """Калибровать сырую вероятность по кривой стороны.

        context_features ({adx, trend_strength, volume_ratio,
        mtf_confluence}) зарезервирован для будущих контекстных кривых;
        сейчас используется только история по стороне. Fallback на
        raw_prob при нехватке истории (<MIN_SAMPLES).
        """
        del context_features  # пока не используется (см. докстринг)
        p = _raw(raw_prob)
        side_key = _normalize_side(side)
        try:
            _load_curves()
        except Exception:  # noqa: BLE001
            return p
        curve = _CURVES.get(side_key) or {}
        if not curve:
            return p
        return _interp(curve, p)

    def update_history(self, verdict: dict, actual_outcome: int,
                       symbol: str = "", timeframe: str = "",
                       context: dict | None = None) -> None:
        """Дозаписать исход (raw_prob, hit) для переобучения кривой.

        verdict — {pu, pd, ...} или {levels: [{side, probability}, ...]};
        actual_outcome — 1 (TP reached) / 0 (SL hit or timeout).
        """
        hit = 1 if int(actual_outcome) else 0
        entries: list[tuple[float, str]] = []
        if isinstance(verdict, dict):
            levels = verdict.get("levels")
            if isinstance(levels, list):
                for lv in levels:
                    if isinstance(lv, dict) and "probability" in lv:
                        entries.append((_raw(lv.get("probability")),
                                        _normalize_side(lv.get("side"))))
            else:
                pu = verdict.get("pu")
                pd = verdict.get("pd")
                if pu is not None:
                    entries.append((_raw(pu), "UP"))
                if pd is not None:
                    entries.append((_raw(pd), "DOWN"))
        ts = time.time()
        try:
            ctx_json = json.dumps(context or {}, ensure_ascii=False)
        except (TypeError, ValueError):
            ctx_json = "{}"
        for raw_prob, side in entries:
            db.db_save_calibration_entry(
                timestamp=ts, symbol=symbol, timeframe=timeframe,
                raw_prob=raw_prob, side=side, hit=hit, context_json=ctx_json)
        if entries:
            _load_curves(force=True)

    def export_curves(self) -> dict:
        """Отладочный экспорт кривых {side: [(prob, hit_rate), ...]}."""
        try:
            _load_curves()
        except Exception:  # noqa: BLE001
            pass
        return {
            side: sorted((float(p), float(v))
                         for p, v in curve.items())
            for side, curve in _CURVES.items()
        }


# Единый singleton для сигнального пути.
CALIBRATOR = IsotonicCalibrator()
