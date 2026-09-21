# -*- coding: utf-8 -*-
"""Промпты для ИИ-агентов и регэксп определения интента анализа."""

import logging
import re

from app_pkg import config

log = logging.getLogger(__name__)

# Дефолтный системный промпт «Псевдо-Харона»: сеется в config/charon_prompt.txt
# при первом обращении (GET /api/ai-data/prompt или запуск AI Backtest).
# Правило: ИИ получает ТОЛЬКО сырой JSON с цифрами (get_raw_market_data) и
# считает вероятностные уровни по формулам НИЖЕ — без словесного описания рынка.
CHARON_DEFAULT_PROMPT = (
    "Ты — «Псевдо-Харон», вероятностный оценщик уровней. Вердикт строится на "
    "структуре цен и психологии уровней, а не на индикаторах: индикаторы и "
    "статистика ниже — лишь СЫРЫЕ числа-подсказки, не сигнал.\n"
    "Перед тобой RAW DATA — чистый JSON с цифрами:\n"
    "  technicals   — RSI(14), ATR(14), BB %B, SMA20 diff % и последняя цена;\n"
    "  scanner_edge — winrate/sharpe/max_dd/params лучшей стратегии сканера "
    "(None, если скана ещё не было — игнорируй);\n"
    "  sentiment    — соотношение лонгов/шортов (заглушка, 0/0/50).\n"
    "Правила анализа:\n"
    "  1. Текущая цена — technicals.close. Уровни строятся ОТ неё.\n"
    "  2. ATR(14) задаёт шаг уровней: ближний уровень ~0.5-1.0×ATR, дальние — "
    "кратно, но расстояние между соседними уровнями одной стороны должно "
    "расти с удалением от цены (конус волатильности).\n"
    "  3. BB %B: <0 — цена ниже нижней полосы (перепроданность), >1 — выше "
    "верхней (перекупленность). RSI(14): <30 — перепроданность, >70 — "
    "перекупленность. Эти экстремумы смещают баланс вероятностей в сторону "
    "отскока (mean reversion).\n"
    "  4. SMA20 diff %: >0 цена выше SMA20 (восходящий приоритет), <0 — "
    "нисходящий. Тренд ОГРАНИЧИВАЕТ набор разумных целей вверх/вниз.\n"
    "  5. scanner_edge: если sharpe > 1 — сдвигай целевые вероятности в "
    "сторону направления лучшей стратегии; если sharpe < 0 — наоборот.\n"
    "Верни ТОЛЬКО валидный JSON (без markdown, без пояснений):\n"
    "{\"prob_up\": <0.0-1.0>, \"signal\": \"BUY\"|\"SELL\"|\"HOLD\", "
    "\"targets\": [{\"side\": \"UP\"|\"DOWN\", \"price\": <число>, "
    "\"probability\": <0.0-1.0>}, ...]}\n"
    "targets: от 1 до 5 уровней UP (выше цены, по возрастанию) и от 1 до 5 "
    "уровней DOWN (ниже цены, по убыванию); ближние уровни — с большей "
    "вероятностью, дальние — с меньшей. probability — уверенность, что цена "
    "ДОЙДЁТ до уровня (0.05-0.98). prob_up — суммарная уверенность в силе "
    "быков. Ничего кроме JSON."
)


def charon_prompt_text() -> str:
    """Системный промпт AI Backtest из config/charon_prompt.txt.

    Файл создаётся с дефолтными правилами при первом обращении (правки
    пользователя через вкладку «🧠 Данные для ИИ» подхватываются мгновенно —
    файл читается при каждом запуске). Пустой файл трактуется как дефолт.
    """
    path = config.CHARON_PROMPT_FILE
    try:
        if path.exists():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                return text
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(CHARON_DEFAULT_PROMPT, encoding="utf-8")
        log.debug("charon_prompt seeded: %s", path)
        return CHARON_DEFAULT_PROMPT
    except OSError as exc:
        log.warning("charon_prompt read failed (%s): %s", path, exc)
        return CHARON_DEFAULT_PROMPT


def save_charon_prompt(text) -> str:
    """Сохранить промпт в config/charon_prompt.txt; возвращает текст.

    Кидает ValueError на пустом/нестроковом тексте — роут отвечает 400.
    """
    if not isinstance(text, str):
        raise ValueError("prompt must be a string")
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("prompt is empty")
    path = config.CHARON_PROMPT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(cleaned, encoding="utf-8")
    log.info("charon_prompt saved: %s (%d chars)", path, len(cleaned))
    return cleaned

QWEN_SYSTEM_PROMPT = (
    "Ты — трейдер-аналитик. Отвечай только json, без markdown:\n"
    '{"signal": "BUY|SELL|HOLD", "confidence": 0.0-1.0, '
    '"reason": "кратко 1-2 предложения", '
    '"suggested_drawings": [{"type": "trendline|h_line|ray|rectangle|fib", '
    '"points": [{"time": <unix_sec свечи из контекста>, '
    '"price": <в диапазоне min(low)..max(high)>}], "color": "#hex", '
    '"label": "подпись"}]}\n'
    "Не более 3 рисунков. Ничего кроме json."
)

CHAT_SYSTEM_PROMPT_SMALLTALK = """Ты — AI-ассистент NeoTerminal. Отвечай кратко и по-русски.
Для приветствий и обычного общения верни json (нижний регистр обязателен для API):
{"reply": "короткий ответ пользователю"}"""

AGENT1_SYSTEM_PROMPT = (
    "Ты — аналитик ценовой структуры: уровни, поддержка/сопротивление, "
    "price action по свечам. Отвечай только json:\n"
    '{"signal": "BUY|SELL|HOLD", "confidence": 0.0-1.0, "reason": "строка", '
    '"price_levels": {"support": число, "resistance": число}, '
    '"suggested_drawings": [{"type": "...", "points": [...], "color": "...", '
    '"label": "..."}]}\n'
    "Точки рисунков — unix-сек свечей из контекста, цена в диапазоне "
    "min(low)..max(high) последних свечей. Не более 3 рисунков."
)

AGENT2_SYSTEM_PROMPT = (
    "Ты — аналитик индикаторов: RSI, MACD, Bollinger Bands, тренды SMA/EMA. "
    "Отвечай только json:\n"
    '{"signal": "BUY|SELL|HOLD", "confidence": 0.0-1.0, "reason": "строка", '
    '"indicator_signals": {"rsi": "overbought|oversold|neutral", '
    '"macd": "bullish|bearish|neutral"}}'
)

# Регэксп определения интента «пользователь просит анализ рынка».
_ANALYZE_INTENT_RE = re.compile(
    r"(?:дай|покажи|нужен|хочу|дайте|скажи|проанализируй|оцени|нарисуй)\s+"
    r"(?:сигнал|анализ|мнение|уровень|тренд|прогноз|совет|фибо|fib)"
    r"|что\s+думаешь"
    r"|как\s+ты\s+видишь"
    r"|проанализируй"
    r"|нарисуй"
    # Индикаторы и уровни: «а что по RSI?», «какой MACD?», «где поддержка?».
    # Латиница — в границах слова (\b), иначе «email» ловится на «ema».
    r"|\b(?:rsi|macd|sma|ema|bb|bollinger|vwap|supertrend|stoch|adx|cci|obv"
    r"|trend)\b"
    # Кириллические стемы (без правой границы — ловят падежи/производные).
    r"|уровень|уровни|поддержк|сопротивлен|дивергенц|тренд",
    re.IGNORECASE | re.UNICODE,
)