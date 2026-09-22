"""Blueprint: /api/ai-analysis.

Ответ — ПЛОСКИЙ dict (без обёртки {"analysis": ...}).
"""

import logging
import uuid

import pandas as pd
from flask import Blueprint, jsonify, request

from app_pkg import config, utils
from app_pkg.ai.agents import (
    _normalize_ai_result,
    _normalize_model_drawings,
    analyze_with_agents,
    analyze_with_ai,
)
from app_pkg.ai.context import build_multi_tf_context, format_tf_section
from app_pkg.ai.llm import _extract_json, _llm_vision_request
from app_pkg.ai.prompts import QWEN_SYSTEM_PROMPT
from app_pkg.cache import get_cached_verdict, set_cached_verdict
from app_pkg.data.fetch import get_replay_df, get_series_df
from app_pkg.db import db_add_drawing, db_get_all_drawings

bp = Blueprint("ai", __name__)
log = logging.getLogger(__name__)


def _replay_upto_sec(symbol, timeframe, body):
    """upto_sec для replay: время последней видимой свечи (<= replay_index).

    Порядок: 1) body.candles (уже обрезаны slice_payload по replay_index),
    2) replay_index -> время через get_replay_df, затем cap по body.visible_to
    (пользователь мог сузить окно вручную).
    """
    upto = None

    candles = body.get("candles")
    if isinstance(candles, list) and candles:
        t = candles[-1].get("time")
        try:
            upto = int(t)
        except (TypeError, ValueError):
            upto = None

    if upto is None:
        idx = body.get("replay_index")
        if idx is None:
            idx = (body.get("replay_state") or {}).get("index")
        if idx is not None:
            try:
                rdf = get_replay_df(symbol, timeframe,
                                    from_sec=body.get("from"),
                                    to_sec=body.get("to"))
            except Exception:  # noqa: BLE001
                rdf = None
            if rdf is not None and not rdf.empty:
                i = min(max(int(idx), 0), len(rdf) - 1)
                upto = int(utils.epoch_secs(rdf["timestamp"])[i])

    try:
        visible_to = int(body.get("visible_to"))
    except (TypeError, ValueError):
        visible_to = None
    if visible_to is not None:
        upto = visible_to if upto is None else min(upto, visible_to)
    return upto


@bp.route("/api/ai-analysis", methods=["POST"])
def api_ai_analysis():
    body = request.get_json(silent=True) or {}
    symbol = str(body.get("symbol", "BTCUSDT")).upper()
    timeframe = str(body.get("timeframe", config.DEFAULT_TIMEFRAME))
    mode = str(body.get("mode", "live"))
    replay_index = body.get("replay_index")

    # Скриншот графика (base64 PNG без префикса data:) — опционален.
    screenshot = body.get("screenshot")
    if not isinstance(screenshot, str) or len(screenshot) < 100:
        screenshot = None

    if symbol not in config.SYMBOLS:
        return jsonify({"error": "Invalid symbol"}), 400
    if timeframe not in config.TF_SECONDS:
        return jsonify({"error": "Invalid timeframe"}), 400

    upto_sec = _replay_upto_sec(symbol, timeframe, body) \
        if mode == "replay" else None

    # mode И upto_sec в ключе: replay-вердикты не должны попадать в live
    # и наоборот; upto_sec фиксирует срез, чтобы сдвиг окна не отдавал
    # старый (свежий) кеш. has_screenshot разделяет vision и текстовый
    # вердикты: они по-разному анализируют одно и то же окно.
    has_screenshot = screenshot is not None
    cache_key = (symbol, timeframe, mode, replay_index, upto_sec,
                 has_screenshot)
    cached = get_cached_verdict(cache_key)
    if cached:
        return jsonify({**cached, "cached": True})

    df = get_series_df(symbol, timeframe, limit=200)
    if upto_sec is not None and df is not None and not df.empty:
        df = df[utils.epoch_secs(df["timestamp"]) <= upto_sec]
    candles = []
    if df is not None and not df.empty:
        for _, r in df.iterrows():
            candles.append({
                "time": int(pd.Timestamp(r["timestamp"]).timestamp()),
                "open": utils._clean(r["open"]),
                "high": utils._clean(r["high"]),
                "low": utils._clean(r["low"]),
                "close": utils._clean(r["close"]),
                "volume": utils._clean(r["volume"]),
            })

    # Видимый диапазон для нормализации рисунков ИИ: из body (фронт
    # присылает его из /api/chart-context), иначе — последние 100 свечей.
    visible_range = body.get("visible_range")
    if not (isinstance(visible_range, dict)
            and isinstance(visible_range.get("from"), int)
            and isinstance(visible_range.get("to"), int)):
        visible_range = None
    if visible_range is None and candles:
        tail = candles[-100:]
        visible_range = {"from": tail[0]["time"], "to": tail[-1]["time"]}
    # Replay: данные после upto_sec ИИ не видит — обрезаем и диапазон.
    if upto_sec is not None and visible_range and visible_range["to"] > upto_sec:
        visible_range = {**visible_range, "to": int(upto_sec)}

    # Контекст с маркерами (агенты фильтруют его по секциям).
    parts = [f"Symbol: {symbol}  Timeframe: {timeframe}  Mode: {mode}"]
    section, _ = format_tf_section(symbol, timeframe, timeframe,
                                   limit=None, upto_sec=upto_sec)
    if section:
        parts.append(section)
    drawings = db_get_all_drawings(symbol, timeframe)
    user_drawings = [d for d in drawings if d.get("created_by") != "ai"]
    if user_drawings:
        parts.append("=== Drawings ===")
        for d in user_drawings[:5]:
            pts = d.get("points", [])[:2]
            pts_str = ";".join(f"t={p.get('time')}p={p.get('price')}" for p in pts)
            parts.append(f"{d.get('type')}:{pts_str} label={d.get('label', '')}")
    mf = build_multi_tf_context(symbol, timeframe, upto_sec)
    if mf:
        parts.append(mf)
    context = "\n".join(parts)

    # Vision: скриншот + текстовый контекст в Groq vision-модель.
    # Недоступна — безшумный fallback на обычный текстовый анализ.
    vision_used = False
    if screenshot and config.VISION_ENABLED:
        try:
            raw = _llm_vision_request(QWEN_SYSTEM_PROMPT, context, screenshot)
            result = _normalize_ai_result(_extract_json(raw), candles)
            if result:
                vision_used = True
                log.info("vision request ok (screenshot %d chars)",
                         len(screenshot))
        except Exception as exc:  # noqa: BLE001
            log.warning("vision unavailable, fallback to text: %s", exc)

    fallback = False
    if vision_used:
        agents_used = ["vision"]
        model = config.DEEPSEEK_VISION_MODEL
    elif config.GROQ_AGENT_MODE == 2:
        result = analyze_with_agents(context)
        agents_used = ["price_structure", "indicators"]
        model = config.DEEPSEEK_MODEL
        fallback = not config.any_llm_enabled()
        if fallback:
            model = "heuristic"
    else:
        result = analyze_with_ai(context)
        agents_used = ["single"]
        model = config.DEEPSEEK_MODEL
        fallback = not config.any_llm_enabled()
        if fallback:
            model = "heuristic"

    saved_drawings = _normalize_model_drawings(
        result.get("suggested_drawings", []), candles, visible_range)
    for d in saved_drawings[:config.MAX_AI_DRAWABLES]:
        d["id"] = str(uuid.uuid4())
        d["created_by"] = "ai"
        db_add_drawing(d, created_by="ai")

    result.update({
        "model": model,
        "fallback": fallback,
        "agents": agents_used,
        "vision_used": vision_used,
        "timestamp": utils.now_sec(),
        "drawings": saved_drawings,
        "agent_mode": config.GROQ_AGENT_MODE,
    })

    set_cached_verdict(cache_key, dict(result))
    return jsonify(result)