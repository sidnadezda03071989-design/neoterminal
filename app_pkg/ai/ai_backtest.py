# -*- coding: utf-8 -*-
"""AI Backtest «Псевдо-Харон»: вероятностные уровни через LLM.

По методу Харона опасность рынка оценивается не индикаторами, а уровнями
поддержки/сопротивления рядом с ценой. Здесь вместо сети предсказания — LLM:
она получает ТОТ ЖЕ контекст, что и Харон (build_multi_tf_context + статистика
сканера), и возвращает НЕСКОЛЬКО уровней вверх и вниз от текущей цены, у
каждого — вероятность того, что цена дойдёт до уровня.

Ключевые отличия от прежней версии (панель «сделок/винрейта» убрана):

  • ОДИН запрос к LLM на текущий срез данных (а не прогон по истории шагами):
    в replay — контекст обрезан по времени барьера (upto_sec), в live —
    последняя доступная свеча. Данные — ровно те же, что у Харона.
  • Результат — только уровни с вероятностями: никаких сделок, winrate,
    sharpe. Это «помощник вероятностей», а не стратегия.
  • Работает и в реплее, и в реальном времени: движок stateless, вызывается
    сколько угодно раз (кнопка «Запустить», авто-пересчёт при сдвиге барьера).

Зоны и пилюли «+{diff} {prob}%» рисует AIProbZonesRenderer.
"""

import json
import logging
import threading
import uuid

import pandas as pd

from app_pkg import config, db, utils
from app_pkg.ai.context import build_multi_tf_context
from app_pkg.ai.llm import _llm_request, _extract_json
from app_pkg.ai.prompts import charon_prompt_text
from app_pkg.data.fetch import get_series_df
from app_pkg.data.market_snapshot import get_raw_market_data
from app_pkg.ws import _ws_push

log = logging.getLogger(__name__)

_RUNS = {}
_RUNS_MAX = 50

# Сколько уровней запрашиваем/принимаем в каждую сторону (UP/DOWN).
PROB_LEVELS_MAX = 5


def get_run(run_id):
    return dict(_RUNS.get(run_id) or {})


def start_run(run_id, symbol, timeframe, upto_sec=None, mode="live",
              model=None):
    """Сеет запись прогона ДО старта фонового потока: GET <run_id> сразу
    отвечает status='running', а не 404 (гонка с потоком в routes)."""
    _RUNS[run_id] = {
        "run_id": run_id, "status": "running", "done": 0, "total": 1,
        "upto_sec": upto_sec, "mode": mode, "levels": [], "error": None,
        "symbol": symbol, "timeframe": timeframe, "model": model,
    }
    _prune_runs()


def _prune_runs():
    if len(_RUNS) > _RUNS_MAX:
        for rid in list(_RUNS)[:len(_RUNS) - _RUNS_MAX]:
            _RUNS.pop(rid, None)


def _push_progress(run_id, done, total, upto_sec=None, levels=0,
                   finished=False, error=None):
    try:
        data = {"run_id": run_id, "done": done, "total": total,
                "current_bar": upto_sec, "levels": levels}
        if finished:
            data["finished"] = True
        if error:
            data["error"] = error
        _ws_push("ai_backtest_progress", data)
    except Exception:  # noqa: BLE001
        log.exception("ai_backtest_progress push failed")


def _normalize_side(raw):
    if not isinstance(raw, str):
        return ""
    return raw.strip().upper()


def _clamp_prob(raw):
    try:
        prob = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if prob > 1.0:
        # Модель могла ответить в процентах (61.3 вместо 0.613).
        prob = prob / 100.0
    return min(1.0, max(0.0, prob))


def parse_probability_levels(raw_text, current_price=None):
    """Ответ LLM -> отсортированный список уровней с вероятностями.

    Поддерживает оба формата ответа:
      гибридный (новый, system-prompt из charon_prompt.txt):
        {"prob_up": 0-1, "signal": "...", "targets": [{side, price,
                                                       probability}, ...]}
      legacy:     {"levels": [{side, price, probability}, ...]}
    Формат элемента на выходе: {"side": "UP"|"DOWN", "price": float,
    "probability": float, "diff": float, "diff_pct": float}.
    Уровни на «неправильной» стороне цены отбрасываются (UP всегда выше
    current_price, DOWN — ниже): так пилюля всегда со знаком +, как на макете.
    """
    if not raw_text:
        return None
    parsed = _extract_json(raw_text)
    if not isinstance(parsed, dict):
        return None
    raw_levels = parsed.get("targets")
    if raw_levels is None:
        raw_levels = parsed.get("levels")
    if isinstance(raw_levels, dict):
        # Модель вернула {"UP": {...}, "DOWN": {...}} вместо списка.
        expanded = []
        for side, item in raw_levels.items():
            if not isinstance(item, dict):
                continue
            copy = dict(item)
            copy.setdefault("side", side)
            expanded.append(copy)
        raw_levels = expanded
    if not isinstance(raw_levels, list):
        return None

    cur = utils._clean(current_price)
    up, down = [], []
    for item in raw_levels:
        if not isinstance(item, dict):
            continue
        side = _normalize_side(item.get("side") or item.get("direction"))
        price = utils._clean(item.get("price"))
        if price is None or price <= 0:
            continue
        # Цель без side: сторона выводится из цены относительно текущей.
        if side not in ("UP", "DOWN") and cur is not None:
            side = "UP" if price > cur else "DOWN"
        if side not in ("UP", "DOWN"):
            continue
        if cur is not None:
            if side == "UP" and price <= cur:
                continue
            if side == "DOWN" and price >= cur:
                continue
        entry = {
            "side": side,
            "price": round(price, 8),
            "probability": round(_clamp_prob(item.get("probability")), 4),
        }
        if cur:
            entry["diff"] = round(price - cur, 8)
            entry["diff_pct"] = round((price - cur) / cur * 100.0, 4)
        (up if side == "UP" else down).append(entry)

    # UP — по возрастанию цены (ближний уровень первым), DOWN — по убыванию.
    up.sort(key=lambda x: x["price"])
    down.sort(key=lambda x: -x["price"])
    levels = (up[:PROB_LEVELS_MAX]
              + sorted(down, key=lambda x: -x["price"])[:PROB_LEVELS_MAX])
    if not levels:
        return None
    return levels


def _scanner_block(symbol, tf):
    """Статистика лучшей комбинации сканера для symbol+tf (AI Backtest).

    Ровно те же данные, что в build_multi_tf_context (context.py) и в
    карточке «Статистика стратегий» вкладки «Новости»
    (GET /api/scanner/stats) + явная вербальная рекомендация: доверять /
    не доверять сигналам стратегии.
    """
    s = db.db_get_best_scan_stats(symbol, tf)
    if not s:
        return ""
    params = s.get("params") or {}
    params_str = (", ".join(f"{k}={v}" for k, v in sorted(params.items()))
                  if params else "default")
    combined = utils._clean(s.get("combined_sharpe")) or 0.0
    test_sharpe = utils._clean(s.get("sharpe")) or 0.0
    winrate = utils._clean(s.get("winrate"))
    if combined > 3:
        advice = (f"Доверяй сигналам {s.get('strategy')} "
                  f"(исторически очень надежная стратегия на этом активе).")
    elif test_sharpe < 0:
        advice = (f"НЕ доверяй сигналам {s.get('strategy')} — ищи уровень "
                  f"поддержки/сопротивления (стратегия убыточна на истории).")
    else:
        advice = (f"Сигналы {s.get('strategy')} учитывай как второстепенные: "
                  f"подтверждай структурой рынка.")
    lines = [f"=== Scanner stats ({symbol} {tf}) ==="]
    lines.append(
        f"Статистика сканера для {symbol} {tf}: "
        f"{s.get('strategy')} ({params_str}) даёт combined Sharpe "
        f"{round(float(combined), 2)}")
    if winrate is not None:
        lines.append(f"out-of-sample winrate={round(float(winrate) * 100, 1)}%")
    lines.append(advice)
    return "\n".join(lines)


def build_levels_context(symbol, timeframe, upto_sec=None, current_price=None):
    """Контекст ТОГО ЖЕ состава, что у Харона (analysis / ai-backtest).

    upto_sec: обрезает ВСЕ таймфреймы по времени (replay-барьер) — ни одна
    свеча из будущего в промпт не попадает. None (live) — свежий ряд.
    """
    parts = [f"Symbol: {symbol} Timeframe: {timeframe} Mode: ai-backtest"]
    mf = build_multi_tf_context(symbol, timeframe, upto_sec)
    if mf:
        parts.append(mf)
    scan = _scanner_block(symbol, timeframe)
    if scan:
        parts.append(scan)
    parts.append(f"=== Current price ===\n{current_price}")
    return "\n".join(parts)


def levels_for_slice(symbol, timeframe, upto_sec=None, model=None,
                     current_price=None):
    """Один запрос к LLM -> (уровни, цена среза, текст ошибки).

    Чистая функция без побочных эффектов на реестр прогонов: её же
    используют синхронный роут и фоновый поток.

    Возвращает (levels, price, error): error — человекочитаемая причина
    (лимит квоты провайдера, rate limit, нет JSON в ответе), None при успехе.
    Причину НЕ глотаем: раньше любая ошибка LLM превращалась в пустой список
    и панель показывала бесполезное «уровни не получены» без деталей.
    """
    if current_price is None:
        current_price = _slice_price(symbol, timeframe, upto_sec)
    if current_price is None:
        return [], None, ("Нет данных для среза: не удалось получить цену "
                          "(проверьте символ/таймфрейм)")
    # Гибридная архитектура: системный промпт с правилами — из файла
    # (config/charon_prompt.txt, редактируется во вкладке «🧠 Данные для ИИ»),
    # а user-сообщение — чистый JSON с цифрами (get_raw_market_data).
    # Никаких текстовых описаний рынка: нейросеть считает уровни по формулам
    # промпта, а не по пересказу свечей.
    system_prompt = charon_prompt_text()
    raw_data = get_raw_market_data(symbol, timeframe, upto_sec=upto_sec)
    user_message = json.dumps(
        {"current_price": current_price, **raw_data},
        ensure_ascii=False, separators=(",", ":"),
    )
    try:
        raw = _llm_request(
            system_prompt,
            [{"role": "user", "content": user_message}],
            model=model,
            purpose=None,
        )
    except Exception as exc:  # noqa: BLE001
        msg = _llm_error_hint(exc)
        log.warning("ai_backtest llm step failed: %s (%s)", exc, msg)
        return [], current_price, msg
    if not raw:
        return [], current_price, ("LLM не вернула ответ (все провайдеры "
                                   "цепочки недоступны — проверьте ключи/лимиты)")
    levels = parse_probability_levels(raw, current_price)
    if not levels:
        return [], current_price, ("LLM вернула ответ без валидных уровней: "
                                   + str(raw)[:200])
    return levels, current_price, None


def _llm_error_hint(exc):
    """Текст исключения LLM -> понятная пользователю причина."""
    text = str(exc) or exc.__class__.__name__
    low = text.lower()
    if "rate limit" in low:
        return "Лимит запросов LLM (rate limit) — подождите минуту"
    if "quota" in low or "free tier" in low:
        return "Квота LLM исчерпана (free tier) — пополните или смените провайдера"
    if "no space" in low or "context" in low and "length" in low:
        return "Слишком большой контекст для модели"
    if "timeout" in low or "timed out" in low:
        return "Таймаут запроса к LLM"
    return f"Ошибка LLM: {text[:200]}"


def _slice_price(symbol, timeframe, upto_sec=None):
    """Цена последнего бара среза: <= upto_sec (replay) или последняя (live).

    Источник — тот же get_series_df, что и у контекста Харона: цена «отсюда»
    для уровней всегда совпадает с последней свечой в промпте.
    """
    try:
        df = get_series_df(symbol, timeframe, limit=config.DEFAULT_LIMIT)
        if df is None or df.empty:
            return None
        if upto_sec is not None:
            mask = df["timestamp"] <= pd.to_datetime(int(upto_sec), unit="s",
                                                     utc=True)
            df = df[mask]
            if df.empty:
                return None
        return utils._clean(df["close"].iloc[-1])
    except Exception:  # noqa: BLE001
        log.exception("ai_backtest slice price failed")
        return None


def run_ai_backtest(symbol, timeframe, upto_sec=None, mode="live",
                    model=None, run_id=None):
    """Расчёт уровней вероятностей для ТЕКУЩЕГО среза данных.

    Возвращает полный результат (dict) со списком levels. Никаких сделок и
    метрик стратегии — только уровни и вероятности (панель-помощник).
    """
    run_id = run_id or uuid.uuid4().hex
    started = utils.now_sec()
    _RUNS[run_id] = {"run_id": run_id, "status": "running", "done": 0,
                     "total": 1, "levels": [], "error": None,
                     "symbol": symbol, "timeframe": timeframe,
                     "upto_sec": upto_sec, "mode": mode, "model": model}
    _push_progress(run_id, 0, 1, upto_sec=upto_sec, levels=0)

    levels, price, error = levels_for_slice(symbol, timeframe,
                                            upto_sec=upto_sec, model=model)

    elapsed = round(utils.now_sec() - started, 2)
    result = {
        "run_id": run_id,
        "status": "finished",
        "symbol": symbol,
        "timeframe": timeframe,
        "mode": mode,
        "upto_sec": upto_sec,
        "price": round(price, 8) if price is not None else None,
        "levels": levels,
        "model": model,
        "error": error,
        "elapsed_seconds": elapsed,
        "created_at": utils.now_iso(),
    }
    if levels:
        try:
            db.db_save_ai_backtest(
                run_id, symbol, timeframe,
                {"mode": mode, "upto_sec": upto_sec, "model": model,
                 "price": result["price"]},
                {}, levels, [])
        except Exception:  # noqa: BLE001
            log.exception("ai_backtest db save failed")

    _RUNS[run_id] = result
    _prune_runs()
    _push_progress(run_id, 1, 1, upto_sec=upto_sec, levels=len(levels),
                   finished=True, error=error)
    log.info("ai_backtest %s done: %d levels, %.2fs", run_id[:8],
             len(levels), elapsed)
    return result


def run_ai_backtest_async(symbol, timeframe, upto_sec=None, mode="live",
                          model=None, run_id=None):
    """Запуск в фоновом потоке: роут отвечает сразу, клиент опрашивает GET."""
    run_id = run_id or uuid.uuid4().hex
    start_run(run_id, symbol, timeframe, upto_sec=upto_sec, mode=mode,
              model=model)

    def _bg():
        try:
            run_ai_backtest(symbol, timeframe, upto_sec=upto_sec, mode=mode,
                            model=model, run_id=run_id)
        except Exception:  # noqa: BLE001
            log.exception("ai_backtest %s failed", run_id[:8])
            _RUNS[run_id] = {"run_id": run_id, "status": "finished",
                             "error": "Внутренняя ошибка расчёта уровней",
                             "levels": [], "symbol": symbol,
                             "timeframe": timeframe}
            _push_progress(run_id, 1, 1, upto_sec=upto_sec, levels=0,
                           finished=True,
                           error="Внутренняя ошибка расчёта уровней")

    threading.Thread(target=_bg, daemon=True,
                     name=f"ai-bt-{run_id[:8]}").start()
    return run_id
