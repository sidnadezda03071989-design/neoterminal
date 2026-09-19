# -*- coding: utf-8 -*-
"""Мультиагентный анализ рынка через Groq.

Порядок в модуле: header, imports, log, def.
Агент 1 (Price Structure) получает только свечи.
Агент 2 (Indicators & Statistics) — только индикаторы и мульти-ТФ тренды.
"""

import logging
import re
import statistics
from concurrent.futures import ThreadPoolExecutor

from app_pkg import config
from app_pkg.ai.prompts import (
    QWEN_SYSTEM_PROMPT, AGENT1_SYSTEM_PROMPT, AGENT2_SYSTEM_PROMPT,
)
from app_pkg.ai.llm import _llm_request, _extract_json

log = logging.getLogger(__name__)

_VALID_DRAWING_TYPES = {"trendline", "ray", "h_line", "rectangle", "fib", "pen"}


# ---------------------------------------------------------------- рисунки
def _nearest_time(t, candles):
    """Ближайшее время свечи; при одинаковой дистанции — более поздняя."""
    if not candles:
        return t
    times = [c["time"] for c in candles]
    return min(times, key=lambda x: (abs(x - t), -x))


def _normalize_model_drawings(raw_drawings, candles, visible_range=None):
    """Нормализует рисунки модели в формат фронта.

    Снап времени к ближайшей свече, лимит 10 точек, label до 40 символов.
    visible_range = {"from": sec, "to": sec} или None: time точек клампится
    в диапазон ДО снапа, price — к границам рынка окна
    [min(low), max(high)] (с допуском ±1%), иначе VL-модель уводит
    рисунки за пределы графика.
    """
    if not raw_drawings or not isinstance(raw_drawings, list):
        return []
    result = []
    times = [c["time"] for c in candles]

    # Границы рынка по свечам видимого окна (fallback — все свечи).
    window = list(candles)
    vr_from = vr_to = None
    if isinstance(visible_range, dict):
        try:
            vr_from = int(visible_range.get("from"))
            vr_to = int(visible_range.get("to"))
            in_window = [c for c in candles if vr_from <= c["time"] <= vr_to]
            if in_window:
                window = in_window
        except (TypeError, ValueError):
            vr_from = vr_to = None
    price_min = min((float(c["low"]) for c in window), default=None)
    price_max = max((float(c["high"]) for c in window), default=None)

    def _clamp_time(t):
        if vr_from is None:
            return t
        if t < vr_from:
            return vr_from
        if t > vr_to:
            return vr_to
        return t

    def _clamp_price(pr):
        if price_min is None or price_max is None:
            return pr
        if pr < price_min * 0.99:
            return price_min
        if pr > price_max * 1.01:
            return price_max
        return pr

    def _next_time(t):
        later = [x for x in times if x > t]
        return min(later) if later else None

    for d in raw_drawings:
        if not isinstance(d, dict):
            continue
        dtype = d.get("type", "trendline")
        if dtype not in _VALID_DRAWING_TYPES:
            dtype = "trendline"
        pts = d.get("points")
        if not pts or not isinstance(pts, list):
            continue
        pts_clean = []
        for p in pts:
            if not isinstance(p, dict):
                continue
            t, pr = p.get("time"), p.get("price")
            if t is None or pr is None:
                continue
            try:
                pts_clean.append({
                    "time": _nearest_time(_clamp_time(int(t)), candles),
                    "price": round(_clamp_price(float(pr)), 10),
                })
            except (TypeError, ValueError):
                continue
        if dtype in ("trendline", "ray", "rectangle") and len(pts_clean) < 2:
            continue
        if dtype == "h_line" and len(pts_clean) < 1:
            continue
        if len(pts_clean) > 10:
            pts_clean = pts_clean[:10]
        # trendline/ray: точки на одной свече не рисуются — сдвигаем вторую
        # на следующую свечу, развести некуда (данные кончились) — отбрасываем.
        if dtype in ("trendline", "ray") and len(pts_clean) >= 2:
            if pts_clean[0]["time"] == pts_clean[1]["time"]:
                nxt = _next_time(pts_clean[0]["time"])
                if nxt is None:
                    continue
                pts_clean[1] = {**pts_clean[1], "time": nxt}
        # fib: совпали по времени — разносим по границам видимого окна.
        if dtype == "fib" and len(pts_clean) >= 2:
            if pts_clean[0]["time"] == pts_clean[1]["time"]:
                if not window:
                    continue
                pts_clean[0] = {**pts_clean[0],
                                "time": _nearest_time(int(window[0]["time"]), candles)}
                pts_clean[1] = {**pts_clean[1],
                                "time": _nearest_time(int(window[-1]["time"]), candles)}
        label = (d.get("label") or "")[:40]
        result.append({
            "type": dtype,
            "points": pts_clean,
            "color": d.get("color") or "#ff9800",
            "label": label,
        })
    return result


def _merge_drawings(a, b):
    """Слияние рисунков без дублей (ключ: type + points)."""
    merged = []
    seen = set()
    for d in list(a or []) + list(b or []):
        if not isinstance(d, dict):
            continue
        key = (d.get("type"), repr(d.get("points")))
        if key in seen:
            continue
        seen.add(key)
        merged.append(d)
    return merged[:10]

# ------------------------------------------------------------- нормализация
def _normalize_ai_result(parsed, candles):
    """Приводит сырой ответ модели к единому dict.

    NEUTRAL -> HOLD; confidence обрезается 0..1; reason до
    config.LLM_REASON_MAX_CHARS (токен-диета, было 500).
    Возвращает None, если parsed отсутствует.
    """
    if not parsed:
        return None
    signal = str(parsed.get("signal", "HOLD")).upper().strip()
    if signal == "NEUTRAL":
        signal = "HOLD"
    if signal not in ("BUY", "SELL", "HOLD"):
        signal = "HOLD"
    try:
        confidence = float(parsed.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = min(1.0, max(0.0, confidence))
    return {
        "signal": signal,
        "confidence": round(confidence, 2),
        "reason": str(parsed.get("reason", ""))[:config.LLM_REASON_MAX_CHARS],
        "indicator_signals": parsed.get("indicator_signals") or {},
        "price_levels": parsed.get("price_levels") or {},
        "suggested_drawings": _normalize_model_drawings(
            parsed.get("suggested_drawings", []), candles),
    }


# --------------------------------------------------------------- эвристика
def _heuristic_analysis(context):
    """Фолбэк без API-ключей: оценка по ценам и RSI из текста контекста."""
    text = str(context)
    if "close" not in text.lower() and "price" not in text.lower():
        return {
            "signal": "HOLD", "confidence": 0.3,
            "reason": "Heuristic: недостаточно данных",
            "indicator_signals": {}, "price_levels": {},
            "suggested_drawings": [],
        }
    lines = [l for l in text.split("\n") if l.strip()]
    closes = []
    for l in lines[-30:]:
        nums = re.findall(r"[-+]?\d*\.?\d+", l)
        if nums:
            closes.append(float(nums[-1]))
    if len(closes) < 2:
        return {
            "signal": "HOLD", "confidence": 0.4,
            "reason": "Heuristic: мало цен",
            "indicator_signals": {}, "price_levels": {},
            "suggested_drawings": [],
        }
    avg = statistics.mean(closes)
    last = closes[-1]
    change = (last - avg) / avg
    rsi_val = None
    m = re.search(r"rsi14?=([\d.]+)", text, re.IGNORECASE)
    if m:
        rsi_val = float(m.group(1))
    if rsi_val is not None:
        if rsi_val > 70:
            return {
                "signal": "SELL", "confidence": 0.55,
                "reason": f"Heuristic: RSI={rsi_val:.0f} >70",
                "indicator_signals": {"rsi": "overbought"},
                "price_levels": {}, "suggested_drawings": [],
            }
        if rsi_val < 30:
            return {
                "signal": "BUY", "confidence": 0.55,
                "reason": f"Heuristic: RSI={rsi_val:.0f} <30",
                "indicator_signals": {"rsi": "oversold"},
                "price_levels": {}, "suggested_drawings": [],
            }
    if change > 0.05:
        return {
            "signal": "BUY", "confidence": 0.50,
            "reason": f"Heuristic: цена выше средней на {change:.1%}",
            "indicator_signals": {}, "price_levels": {},
            "suggested_drawings": [],
        }
    if change < -0.05:
        return {
            "signal": "SELL", "confidence": 0.50,
            "reason": f"Heuristic: цена ниже средней на {change:.1%}",
            "indicator_signals": {}, "price_levels": {},
            "suggested_drawings": [],
        }
    return {
        "signal": "HOLD", "confidence": 0.5,
        "reason": "Heuristic: тренд не выражен",
        "indicator_signals": {}, "price_levels": {},
        "suggested_drawings": [],
    }


# ------------------------------------------------------- фильтрация контекста
def _filter_candles_only(context):
    """Свечи без индикаторов-осцилляторов и без блока Trends."""
    lines = context.split("\n")
    filtered = []
    block = None
    for l in lines:
        if l.startswith("=== Indicators") or l.startswith("=== Trends ==="):
            block = "skip"
            continue
        if l.startswith("=== "):
            block = None
        if block is None:
            filtered.append(l)
    return "\n".join(filtered)


def _filter_indicators_only(context):
    """Только блоки Indicators и Trends."""
    lines = context.split("\n")
    filtered = []
    capture = False
    for l in lines:
        if l.startswith("=== Indicators") or l.startswith("=== Trends ==="):
            capture = True
        elif l.startswith("=== Candles"):
            capture = False
        if capture:
            filtered.append(l)
    return "\n".join(filtered) if filtered else context

# ------------------------------------------------------------------ агенты
def _run_agent(system, base_url, api_key, model, timeout, user_text):
    """Один агент: запрос Groq -> JSON -> нормализованный dict."""
    try:
        raw = _llm_request(
            system,
            [{"role": "user", "content": user_text or "Анализ недоступен"}],
            api_key=api_key, base_url=base_url, model=model, timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("Agent request error: %s", exc)
        return None
    if not raw:
        return None
    parsed = _extract_json(raw)
    if not parsed:
        return None
    return _normalize_ai_result(parsed, [])


def _run_agent1(context):
    """Agent1 — Price Structure: только свечи, без индикаторов."""
    user_text = _filter_candles_only(str(context))
    return _run_agent(
        AGENT1_SYSTEM_PROMPT,
        config.DEEPSEEK_BASE_URL, config.DEEPSEEK_API_KEY,
        config.DEEPSEEK_MODEL, config.DEEPSEEK_TIMEOUT, user_text,
    )


def _run_agent2(context):
    """Agent2 — Indicators & Statistics: только индикаторы и тренды."""
    user_text = _filter_indicators_only(str(context))
    return _run_agent(
        AGENT2_SYSTEM_PROMPT,
        config.DEEPSEEK_BASE_URL, config.DEEPSEEK_API_KEY,
        config.DEEPSEEK_MODEL, config.DEEPSEEK_TIMEOUT, user_text,
    )


# -------------------------------------------------------------- агрегация
def _finalize_verdict(r):
    return {
        "signal": r["signal"],
        "confidence": round(r.get("confidence", 0.0), 2),
        "reason": (r.get("reason") or "")[:500],
        "indicator_signals": r.get("indicator_signals") or {},
        "price_levels": r.get("price_levels") or {},
        "suggested_drawings": r.get("suggested_drawings") or [],
    }


def _aggregate_agent_results(r1, r2):
    """Объединение двух агентов.

    Совпали сигналы: confidence = min(1, (c1+c2)/2 + 0.05).
    Разошлись: берётся более уверенный с -0.10.
    """
    if r1 is None and r2 is None:
        return {
            "signal": "HOLD", "confidence": 0.3,
            "reason": "Агенты не вернули результат",
            "indicator_signals": {}, "price_levels": {},
            "suggested_drawings": [],
        }
    if r1 is None:
        return _finalize_verdict(r2)
    if r2 is None:
        return _finalize_verdict(r1)

    if r1["signal"] == r2["signal"]:
        signal = r1["signal"]
        confidence = min(1.0, (r1["confidence"] + r2["confidence"]) / 2.0 + 0.05)
        reason = f"[Price] {r1['reason']} | [Indicators] {r2['reason']}"
    else:
        high = r1 if r1["confidence"] >= r2["confidence"] else r2
        signal = high["signal"]
        confidence = max(0.1, high["confidence"] - 0.10)
        reason = (f"Агенты разошлись: {r1['signal']} vs {r2['signal']}. "
                  f"Принят более уверенный: {signal}")

    return {
        "signal": signal,
        "confidence": round(confidence, 2),
        "reason": reason[:500],
        "indicator_signals": r2.get("indicator_signals") or {},
        "price_levels": r1.get("price_levels") or {},
        "suggested_drawings": _merge_drawings(
            r1.get("suggested_drawings", []), r2.get("suggested_drawings", [])),
    }


# -------------------------------------------------------------- точки входа
def analyze_with_ai(context):
    """Один агент (GROQ_AGENT_MODE=1)."""
    if not config.any_llm_enabled():
        return _heuristic_analysis(context)
    try:
        raw = _llm_request(
            QWEN_SYSTEM_PROMPT, [{"role": "user", "content": str(context)}])
    except Exception as exc:  # noqa: BLE001
        log.warning("analyze_with_ai error: %s", exc)
        return _heuristic_analysis(context)
    if not raw:
        return _heuristic_analysis(context)
    parsed = _extract_json(raw)
    result = _normalize_ai_result(parsed, [])
    return result if result else _heuristic_analysis(context)


def analyze_with_agents(context):
    """Два агента параллельно (GROQ_AGENT_MODE=2)."""
    if not config.any_llm_enabled():
        return _heuristic_analysis(context)
    with ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(_run_agent1, context)
        f2 = pool.submit(_run_agent2, context)
        r1 = f1.result()
        r2 = f2.result()
    return _aggregate_agent_results(r1, r2)