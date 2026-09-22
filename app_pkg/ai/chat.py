"""Чат с ИИ: определение интента, анализ рынка или обычное общение."""

import logging
import re
import uuid

import pandas as pd

from app_pkg import config
from app_pkg.ai.agents import (
    _normalize_model_drawings,
    analyze_with_agents,
    analyze_with_ai,
)
from app_pkg.ai.context import build_multi_tf_context
from app_pkg.ai.llm import _extract_json, _llm_request
from app_pkg.ai.prompts import (
    _ANALYZE_INTENT_RE,
    CHAT_SYSTEM_PROMPT_SMALLTALK,
)
from app_pkg.data.fetch import get_replay_df, get_series_df
from app_pkg.db import (
    db_add_drawing,
    db_append_chat,
    db_get_all_drawings,
    db_get_chat_history,
)
from app_pkg.utils import _clean, epoch_secs

log = logging.getLogger(__name__)


def append_chat_message(role: str, content: str, symbol: str = None) -> int:
    """Хелпер записи сообщения в историю чата."""
    return db_append_chat(role, content, symbol)


def _detect_analysis_intent(message: str) -> bool:
    """True, если пользователь просит анализ рынка (не болтовню)."""
    if not message:
        return False
    # _ANALYZE_INTENT_RE уже скомпилирован с re.IGNORECASE — флаги
    # передавать нельзя: ValueError «cannot process flags argument».
    return bool(_ANALYZE_INTENT_RE.search(message))


def _candles_from_df(df):
    """DataFrame -> список компактных свечей для снапа рисунков."""
    if df is None or df.empty:
        return []
    result = []
    for _, r in df.iterrows():
        result.append({
            "time": int(pd.Timestamp(r["timestamp"]).timestamp()),
            "open": _clean(r["open"]),
            "high": _clean(r["high"]),
            "low": _clean(r["low"]),
            "close": _clean(r["close"]),
            "volume": _clean(r["volume"]),
        })
    return result


def _shorten(text, n=120):
    """Однострочный укороченный текст для контекстной вставки в промпт."""
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) > n:
        text = text[: n - 1].rstrip() + "…"
    return text


def _recent_context_note(symbol):
    """Короткая строка «Предыдущий контекст: …» для analysis-промпта.

    Полная история в analysis не идёт (промпт и так тяжёлый) — только
    упоминание последних 3 сообщений. Включается, только если предыдущее
    сообщение пользователя было анализом того же символа.
    """
    history = db_get_chat_history(limit=config.CHAT_MEMORY_MESSAGES)
    if not history:
        return None
    users = [m for m in history if m.get("role") == "user"]
    # текущее сообщение уже записано в БД (append до flow),
    # поэтому «предыдущее» — это users[-2]
    if len(users) < 2:
        return None
    prev_user = users[-2]
    if (prev_user.get("symbol") or "").upper() != (symbol or "").upper():
        return None
    if not _detect_analysis_intent(prev_user.get("content") or ""):
        return None
    recent = history[:-1][-3:]  # последние 3, без текущего сообщения
    parts = []
    for m in recent:
        who = "user спросил" if m.get("role") == "user" else "AI ответил"
        parts.append(who + ": " + _shorten(m.get("content")))
    return "Предыдущий контекст: " + "; ".join(parts) + "."


def _analysis_flow(message, symbol, timeframe, upto_sec=None, mode="live"):
    """Полный анализ: контекст -> агенты -> рисунки -> reply.

    upto_sec (replay): свечи берутся из get_replay_df(to_sec=upto_sec),
    иначе из свежего get_series_df и дополнительно обрезаются по upto_sec —
    снапшот рисунков и "Price:" не должны видеть данные после upto_sec.
    """
    if mode == "replay" and upto_sec is not None:
        df = get_replay_df(symbol, timeframe, to_sec=float(upto_sec))
    else:
        df = get_series_df(symbol, timeframe, limit=100)
    if upto_sec is not None and df is not None and not df.empty:
        df = df[epoch_secs(df["timestamp"]) <= int(upto_sec)]
    candles = _candles_from_df(df)

    mf_context = build_multi_tf_context(symbol, timeframe, upto_sec)
    context_parts = [f"Symbol: {symbol} Timeframe: {timeframe} Mode: analysis"]
    if candles:
        last = candles[-1]
        context_parts.append(
            f"Price: close={last['close']} high={last['high']} low={last['low']}")
    if mf_context:
        context_parts.append(f"Multi-TF:\n{mf_context}")
    prev_note = _recent_context_note(symbol)
    if prev_note:
        context_parts.append(prev_note)
    context_parts.append(f"User: {message}")
    user_text = "\n".join(context_parts)

    if config.GROQ_AGENT_MODE == 2:
        result = analyze_with_agents(user_text)
        model = config.DEEPSEEK_MODEL
    else:
        result = analyze_with_ai(user_text)
        model = config.DEEPSEEK_MODEL

    fallback = not config.any_llm_enabled()
    drawings = []
    if result:
        drawings = _normalize_model_drawings(
            result.get("suggested_drawings", []), candles)
        for d in drawings[:config.MAX_AI_DRAWABLES]:
            d["id"] = str(uuid.uuid4())
            d["created_by"] = "ai"
            db_add_drawing(d, created_by="ai")
        reply = f"**{result['signal']}** (уверенность {result['confidence']:.0%})\n{result['reason']}"
    else:
        reply = "Не удалось выполнить анализ."
    all_drawings = db_get_all_drawings(symbol, timeframe)
    return reply, drawings[:config.MAX_AI_DRAWABLES], all_drawings, model, fallback


def _truncate_chat_message(text):
    """Обрезка старого сообщения чата (токен-диета): head + "..." + tail."""
    text = str(text or "")
    limit = config.CHAT_MESSAGE_MAX_CHARS
    if len(text) <= limit:
        return text
    return text[:150] + "..." + text[-40:]


def _truncate_chat_message_dict(m):
    """Обрезка content у сообщения-словаря истории (роль сохраняется)."""
    return {"role": m.get("role"),
            "content": _truncate_chat_message(m.get("content"))}


def _chat_flow(message, symbol, timeframe):
    """Обычное общение через chat-промпт (или заглушку без API).

    Помнит контекст беседы: последние CHAT_MEMORY_MESSAGES сообщений
    (старые → новые) уходят в messages между system-промптом и текущим
    вопросом пользователя. Каждое старое сообщение обрезается до
    CHAT_MESSAGE_MAX_CHARS (токен-диета), текущее — целиком.
    """
    history = db_get_chat_history(limit=config.CHAT_MEMORY_MESSAGES)
    # Текущее сообщение уже записано в БД (chat_with_model делает append
    # до вызова flow) — исключаем его из истории: ниже оно добавляется
    # отдельно, иначе модель видит вопрос дважды.
    if history and history[-1].get("role") == "user":
        history = history[:-1]
    # system в msgs запрещён: его добавляет _llm_request ровно один
    # (двойной system ломает и Groq: HTTP 400 InvalidParameter).
    msgs = [
        {"role": str(m.get("role")), "content": str(m.get("content") or "")}
        for m in history
        if isinstance(m, dict) and m.get("role") in ("user", "assistant")
    ]
    current = {"role": "user", "content": message}
    # Защита от дубля текущего вопроса: если последним в истории оказался
    # тот же user-месседж — заменяем его, а не добавляем второй.
    if msgs and msgs[-1]["role"] == "user" and msgs[-1]["content"] == message:
        msgs[-1] = current
    else:
        msgs.append(current)
    # Токен-диета: старые сообщения обрезаем, последнее (текущий вопрос)
    # оставляем целиком.
    msgs = [_truncate_chat_message_dict(m) if i < len(msgs) - 1 else m
            for i, m in enumerate(msgs)]
    # Дебаг: видно дубли ролей/двойной system, если регрессия вернётся.
    log.debug("chat_flow msgs: roles=%s count=%d",
              [m["role"] for m in msgs], len(msgs))
    try:
        resp = _llm_request(CHAT_SYSTEM_PROMPT_SMALLTALK, msgs,
                            purpose="chat")
    except Exception as exc:  # noqa: BLE001
        log.warning("chat reply error: %s", exc)
        resp = None
    if resp:
        # Промпт просит json {"reply": ...} — вынимаем reply, чтобы
        # в чат не уезжала сырая JSON-обёртка. Если модель вернула
        # plain text (парсинг не удался / нет ключа reply) — отдаём как есть.
        try:
            parsed = _extract_json(resp)
        except Exception as exc:  # noqa: BLE001
            log.warning("chat reply parse error: %s", exc)
            parsed = None
        if isinstance(parsed, dict) and parsed.get("reply") is not None:
            return str(parsed["reply"]), config.DEEPSEEK_MODEL, False
        return resp, config.DEEPSEEK_MODEL, False
    return ("Извините, LLM-провайдер недоступен. "
            "Попробуйте позже или попросите анализ рынка."), "heuristic", True


def chat_with_model(message, symbol, timeframe, mode="live", upto_sec=None):
    """Обработать сообщение чата; вернуть dict с ответом и историей."""
    message = (message or "").strip()
    intent = _detect_analysis_intent(message)
    append_chat_message("user", message, symbol)

    if intent:
        reply, drawings, all_drawings, model, fallback = _analysis_flow(
            message, symbol, timeframe, upto_sec=upto_sec, mode=mode)
        intent_tag = "analysis"
    else:
        reply, model, fallback = _chat_flow(message, symbol, timeframe)
        drawings, all_drawings = [], db_get_all_drawings()

    append_chat_message("assistant", reply, symbol)

    return {
        "reply": reply,
        "drawings": drawings,
        "all_drawings": all_drawings,
        "model": model,
        "fallback": fallback,
        "intent": intent_tag if intent else "chat",
        "history": db_get_chat_history(limit=config.CHAT_HISTORY_LIMIT),
    }