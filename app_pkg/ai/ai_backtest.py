"""AI Backtest «Псевдо-Харон»: детерминированные вероятностные уровни.

По методу Харона опасность рынка оценивается не индикаторами, а уровнями
поддержки/сопротивления рядом с ценой. LLM REMOVED: вместо сети и модели
работает детерминированный движок в процессе:

  • вердикт/направление — app_pkg.ai.apply_rules.apply_all_rules (правила
    Charon 1-20, перенесённые из config/charon_prompt.txt, чистый Python);
  • уровни — свечная структура (structure_levels_from_df) как подложка плюс
    вероятности того же вердикта; пост-фильтр — app_pkg.ai.signal_filter
    (MTF/ADX/VWAP+OBV правила) и изотоническая калибровка
    (app_pkg.ai.calibration по истории charon_calibration_history).

_llm_request сохранён как сигнатурный шов (тесты/диагностика), но внутри
считает правила и НЕ ходит в сеть: сигнальный путь полностью offline.

Ключевые отличия от прежней версии (панель «сделок/винрейта» убрана):

  • ОДИН расчёт на текущий срез данных (а не прогон по истории шагами):
    в replay — контекст обрезан по времени барьера (upto_sec), в live —
    последняя доступная свеча. Данные — ровно те же, что у Харона.
  • Результат — только уровни с вероятностями: никаких сделок, winrate,
    sharpe. Это «помощник вероятностей», а не стратегия.
  • Работает и в реплее, и в реальном времени: движок stateless, считается
    строго по явному запросу (кнопка «Запустить» в панели).

Зоны и пилюли «+{diff} {prob}%» рисует AIProbZonesRenderer.
"""

import hashlib
import json
import logging
import threading
import uuid

import pandas as pd

from app_pkg import config, db, utils
from app_pkg.ai.apply_rules import apply_all_rules
from app_pkg.ai.context import build_multi_tf_context
from app_pkg.ai.llm import _extract_json
from app_pkg.ai.prompts import charon_prompt_text
from app_pkg.ai.signal_filter import filter_levels, filter_verdict
from app_pkg.ai.structure_levels import structure_levels_from_df
from app_pkg.data.fetch import get_replay_df, get_series_df
from app_pkg.data.market_snapshot import compact_snapshot
from app_pkg.ws import _ws_push

log = logging.getLogger(__name__)

_RUNS = {}
_RUNS_MAX = 50

# Сколько уровней запрашиваем/принимаем в каждую сторону (UP/DOWN).
PROB_LEVELS_MAX = 5

# ---------------------------------------------- токен-диета: вердикты (v3)
# LEGACY (LLM REMOVED): ключи ниже сохранены, т.к. _llm_request/_llm_verdict
# держат прежнюю сигнатуру (тесты и панель читают их для tooltips). Сеть не
# вызывается — вердикт считает apply_all_rules, токенов не тратится.
# Компактный ответ: короткие ключи, без reason. Детерминизм — T=0.
VERDICT_MAX_TOKENS = 120
VERDICT_TEMPERATURE = 0.0
# Историческая заметка: deepseek-v4-flash тратил лимит на chain-of-thought
# (finish_reason="length", пустой content). Теперь неактуально, но значение
# осталось в сигнатуре вызова для совместимости с monkeypatch в тестах.
VERDICT_REASONING_EFFORT = "none"
# Триггеры адаптивного шага: расчёт зовём только если сработал хотя бы один.
# LEGACY: сохранены для обратной совместимости планировщика (адаптивный шаг).
ADAPTIVE_RSI_NEAR = 5.0     # (a) rsi в пределах 5 от oversold/overbought
ADAPTIVE_RSI_DELTA = 7.0    # (b) |rsi - rsi на прошлом вызове| > 7
ADAPTIVE_CLOSE_ATR = 0.7    # (c) |close - close на прошлом вызове| > .7*atr
DEFAULT_OVERSOLD = 30.0
DEFAULT_OVERBOUGHT = 70.0
# Флаг пакетной обработки: 1 — по одному снимку, 5 — пять за один вызов.
BATCH_SIZE_DEFAULT = 1


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
    except Exception:
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


def _dict_prob(item):
    """Вероятность из dict-элемента: probability | prob (модель иногда пишет
    сокращённо) | None."""
    if not isinstance(item, dict):
        return None
    if "probability" in item:
        return item.get("probability")
    if "prob" in item:
        return item.get("prob")
    return None


def _dedupe_levels(items):
    """Убрать уровни с одинаковой ценой в одной стороне: берём max(prob).

    Модель иногда отдаёт один и тот же уровень дважды (напр. две пары tg с
    одной ценой) — на панели это выглядело дублем строки.
    """
    best = {}
    for lv in items:
        key = round(lv["price"], 8)
        prev = best.get(key)
        if prev is None or lv["probability"] > prev["probability"]:
            best[key] = lv
    return list(best.values())


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
    from_tg = False
    if raw_levels is None:
        raw_levels = parsed.get("levels")
    if raw_levels is None:
        # v3-вердикт: tg = [[price, prob], ...]
        raw_levels = parsed.get("tg")
        from_tg = True
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
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            # v3 tg-пара [price, prob]: сторону выведем из цены.
            item = {"price": item[0], "probability": item[1]}
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
            "probability": round(_clamp_prob(_dict_prob(item)), 4),
        }
        if cur:
            entry["diff"] = round(price - cur, 8)
            entry["diff_pct"] = round((price - cur) / cur * 100.0, 4)
        (up if side == "UP" else down).append(entry)

    # Дубли по цене в одной стороне (модель иногда повторяет уровень):
    # оставляем вариант с большей вероятностью.
    up = _dedupe_levels(up)
    down = _dedupe_levels(down)
    # UP — по возрастанию цены (ближний уровень первым), DOWN — по убыванию.
    up.sort(key=lambda x: x["price"])
    down.sort(key=lambda x: -x["price"])
    levels = (up[:PROB_LEVELS_MAX]
              + sorted(down, key=lambda x: -x["price"])[:PROB_LEVELS_MAX])
    if not levels:
        return None
    if from_tg:
        # v3: суммарная вероятность по всем уровням = 1.00 (100%).
        total = sum(lv["probability"] for lv in levels)
        if total > 0:
            for lv in levels:
                lv["probability"] = round(lv["probability"] / total, 4)
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


def _structure_levels(symbol, timeframe, upto_sec, current_price):
    """Структурные уровни из свечей (свинги/пивоты/VWAP/ATR-якоря).

    Детерминированная база уровней «по логике»: при T=0 и ~120 токенах LLM
    рисует формульную сетку (равный шаг ~ATR с линейно убывающим шансом),
    одинаковую для каждого прогона. Цены берём из реальной структуры рынка,
    а модель используем лишь для уточнения вероятностей совпавших уровней.
    Любой сбой здесь не роняет бэктест: возвращаем [] и живём по LLM-уровням.
    """
    try:
        if upto_sec is not None:
            df = get_replay_df(symbol, timeframe, to_sec=float(upto_sec),
                               limit=config.DEFAULT_LIMIT)
            if df is not None and not df.empty:
                df = df[df["timestamp"] <= pd.to_datetime(
                    int(upto_sec), unit="s", utc=True)]
        else:
            df = get_series_df(symbol, timeframe, limit=config.DEFAULT_LIMIT)
        if df is None or df.empty:
            return []
        return structure_levels_from_df(df, current_price)
    except Exception:
        log.exception("ai_backtest structure levels failed")
        return []


def _blend_levels(struct, llm, current_price):
    """Структурная база; LLM лишь подтверждает уровни, что совпали по цене.

    Цены всегда рисуются из реальной структуры рынка (свинги/пивоты/VWAP):
    равномерные сетки LLM (равный шаг ~ATR) не попадают на график. Уровень
    LLM «подтверждает» структурный, если его цена близка (толерантность
    ~0.3% от цены) к структурному той же стороны — вероятность такого уровня
    чуть повышается (+0.02, кап 0.5, порядок ближний->дальний сохраняется).
    Всё остальное LLM отдаётся на откуп детерминированному filter_levels.
    """
    tol = abs(current_price) * 0.003 if current_price else 0.0
    base = [dict(sl) for sl in struct]
    for lv in llm:
        match = None
        for sl in base:
            if sl["side"] == lv["side"] and abs(sl["price"] - lv["price"]) <= tol:
                match = sl
                break
        if match is None:
            continue
        match["probability"] = round(min(0.5, match["probability"] + 0.02), 4)
    ups = sorted((x for x in base if x["side"] == "UP"),
                 key=lambda x: x["price"])
    downs = sorted((x for x in base if x["side"] == "DOWN"),
                   key=lambda x: -x["price"])
    return (ups[:PROB_LEVELS_MAX] + downs[:PROB_LEVELS_MAX])


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
    # а user-сообщение — КОМПАКТНЫЙ JSON-снимок (compact_snapshot, схема v3:
    # t/se/s/c/m/cal/d/ns). Ключи снимка совпадают с описанием в промпте —
    # никаких текстовых пересказов рынка.
    system_prompt = charon_prompt_text()
    snapshot = compact_snapshot(symbol, timeframe, upto_sec=upto_sec)
    user_message = json.dumps(
        {"current_price": current_price, **snapshot},
        ensure_ascii=False, separators=(",", ":"),
    )
    try:
        raw = _llm_request(
            system_prompt,
            [{"role": "user", "content": user_message}],
            model=model,
            purpose=None,
            max_tokens=VERDICT_MAX_TOKENS,
            temperature=VERDICT_TEMPERATURE,
            reasoning_effort=VERDICT_REASONING_EFFORT,
        )
    except Exception as exc:  # noqa: BLE001
        msg = _llm_error_hint(exc)
        log.warning("ai_backtest llm step failed: %s (%s)", exc, msg)
        return [], current_price, msg
    if not raw:
        return [], current_price, ("LLM не вернула ответ (все провайдеры "
                                   "цепочки недоступны — проверьте ключи/лимиты)")
    struct = _structure_levels(symbol, timeframe, upto_sec, current_price)
    ts = upto_sec if upto_sec is not None else int(utils.now_sec())
    source = "backtest" if upto_sec is not None else "live"
    levels = parse_probability_levels(raw, current_price)
    if levels is None:
        # Валидный вердикт без выразимых уровней — обычно sig=F / плоский
        # рынок {"pu":0,"pd":0,"pf":0,"sig":"F","tg":[]}: это НЕ ошибка LLM,
        # а честный ответ «сделки нет». Если структура свечей есть — рисуем
        # уровни из логики рынка вместо пустоты; иначе пусто без error.
        # Мусор/не-JSON/ответ без sig по-прежнему даёт диагностику.
        verdict = parse_verdict(raw)
        if verdict is not None:
            log.info("ai_backtest flat verdict %s/%s: %s",
                     symbol, timeframe, verdict)
            chosen = struct if struct else []
            if chosen:
                chosen = filter_levels(chosen, snapshot)
            _cbr_store_snapshot(
                symbol, timeframe, ts, snapshot, verdict=verdict,
                levels=chosen, price=current_price, source=source)
            return chosen, current_price, None
        return [], current_price, ("LLM вернула ответ без валидных уровней: "
                                   + str(raw)[:200])
    if struct:
        # Рисуем структурные цены, вероятности LLM — только для совпавших.
        levels = _blend_levels(struct, levels, current_price)
    # Детерминированный пост-фильтр: MTF/ADX/VWAP+OBV правила поверх
    # вероятностей уровней (тот же compact_snapshot, что ушёл в промпт).
    levels = filter_levels(levels, snapshot)
    _cbr_store_snapshot(
        symbol, timeframe, ts, snapshot, levels=levels, price=current_price,
        source=source)
    return levels, current_price, None


def _cbr_store_snapshot(symbol, timeframe, ts, snap, verdict=None, levels=None,
                        price=None, source="backtest"):
    """Хук CBR: записать снимок в БД (гейт config.CBR_ENABLED, ошибки глушим).

    Снимок не должен ронять бэктест: любая ошибка CBR -> warning + return.
    levels -> [{delta_pct, prob}] (delta — % от текущей цены).
    """
    if not config.CBR_ENABLED or not isinstance(snap, dict) or not snap:
        return
    try:
        from app_pkg.cbr.store import get_cbr_conn, store_snapshot
        sn = dict(snap)
        sn.setdefault("symbol", symbol)
        sn.setdefault("timeframe", timeframe)
        if "ts" not in sn:
            sn["ts"] = int(ts)
        sn["source"] = source
        if verdict:
            sn["verdict"] = verdict.get("sig")
        if levels:
            cur = price if price is not None else utils._clean(
                (snap.get("t") or {}).get("close"))
            items = []
            for lv in levels:
                lprice = utils._clean(lv.get("price"))
                if lprice is None:
                    continue
                if cur:
                    delta = round((lprice - cur) / cur * 100.0, 4)
                else:
                    delta = round(lprice, 4)
                items.append({"delta": delta,
                              "prob": _clamp_prob(lv.get("probability"))})
            sn["levels"] = items
        if price is not None:
            sn["entry_price"] = price
        store_snapshot(get_cbr_conn(), sn)
    except Exception as exc:  # noqa: BLE001 — CBR не должен валить бэктест
        log.warning("cbr store failed: %s", exc)


def _llm_request(system, messages, model=None, purpose=None, max_tokens=None,
                 temperature=None, usage_out=None, reasoning_effort=None,
                 **kwargs):
    """Детерминированная замена сетевого вызова LLM (сигнатурный шов).

    LLM REMOVED: сеть не используется. Функция сохранена с прежней сигнатурой,
    потому что на неё опираются levels_for_slice/_llm_verdict, а тесты
    подменяют её через monkeypatch. Ответ считается правилами Charon 1-20
    (apply_all_rules) по снимку из messages и возвращается в том же
    JSON-формате, что отдавала модель ({sig, pu, pd, pf, targets}).
    usage_out заполняется нулями — токенов нет, token-диета вырождается.
    """
    if isinstance(usage_out, dict):
        usage_out.update({"prompt_tokens": 0, "completion_tokens": 0,
                          "total_tokens": 0, "deterministic": True})
    snap = None
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            candidate = _extract_json(content)
            if isinstance(candidate, dict):
                snap = candidate
                break
    if snap is None:
        return None
    verdict = _verdict_from_obj(apply_all_rules(snap))
    if verdict is None:
        return None
    return json.dumps(
        {"sig": verdict.get("sig"), "pu": verdict.get("pu"),
         "pd": verdict.get("pd"), "pf": verdict.get("pf"),
         "targets": verdict.get("tg") or []},
        ensure_ascii=False, separators=(",", ":"))


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

    Источник — тот же, что и у контекста Харона: цена «отсюда» для уровней
    всегда совпадает с последней свечой в промпте. В реплее берём ИСТОРИЧЕСКОЕ
    окно, заканчивающееся на барьере (get_replay_df), а не live-хвост
    get_series_df: иначе барьер старше последних N баров давал пустой срез
    («нет данных для среза»).
    """
    try:
        if upto_sec is not None:
            df = get_replay_df(symbol, timeframe, to_sec=float(upto_sec),
                               limit=config.DEFAULT_LIMIT)
        else:
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
    except Exception:
        log.exception("ai_backtest slice price failed")
        return None


def direction_probability(levels):
    """Вероятности простого ЛОНГ/ШОРТ по уровням (после filter_levels).

    ЛОНГ = сумма вероятностей UP-уровней, ШОРТ = сумма DOWN. Каждая сторона
    не выше 1.0; если лонг + шорт > 1.0 — нормируем на сумму (гарантия
    «общая вероятность не больше 100%»), иначе оставляем как есть (остаток
    до 100% — «без сделки»). Вероятности уровней уже прошли filter_levels
    (MTF/ADX/VWAP+OBV), поэтому результат объединяет и уровни, и метрики
    снимка. Пустой список -> {long:0, short:0}.
    """
    long_p = short_p = 0.0
    for lv in levels or []:
        if not isinstance(lv, dict):
            continue
        side = str(lv.get("side") or "").upper()
        try:
            prob = float(lv.get("probability") or 0.0)
        except (TypeError, ValueError):
            continue
        prob = min(1.0, max(0.0, prob))
        if side == "UP":
            long_p += prob
        elif side == "DOWN":
            short_p += prob
    long_p = min(1.0, long_p)
    short_p = min(1.0, short_p)
    total = long_p + short_p
    if total > 1.0:
        scale = 1.0 / total
        long_p *= scale
        short_p *= scale
    return {"long": round(long_p, 4), "short": round(short_p, 4)}


def run_ai_backtest(symbol, timeframe, upto_sec=None, mode="live",
                    model=None, run_id=None):
    """Расчёт уровней вероятностей для ТЕКУЩЕГО среза данных.

    Возвращает полный результат (dict) со списком levels и сводкой
    direction ({long, short} — вероятности простого ЛОНГ/ШОРТ). Никаких
    сделок и метрик стратегии — только уровни и вероятности (панель-
    помощник).
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
    # Вероятность простого ЛОНГ/ШОРТ: агрегат вероятностей уровней каждой
    # стороны (уровни уже прошли filter_levels с метриками MTF/ADX/VWAP+OBV).
    direction = direction_probability(levels)
    result = {
        "run_id": run_id,
        "status": "finished",
        "symbol": symbol,
        "timeframe": timeframe,
        "mode": mode,
        "upto_sec": upto_sec,
        "price": round(price, 8) if price is not None else None,
        "levels": levels,
        "direction": direction,
        "model": model,
        "error": error,
        "elapsed_seconds": elapsed,
        "created_at": utils.now_iso(),
    }
    if not levels and not error:
        # Пусто без ошибки = валидный вердикт «рынок плоский». Не ошибка —
        # поясняем нейтральной заметкой вместо красного ⚠.
        result["notice"] = ("Уровней не найдено: рынок без явной сделки "
                            "(сигнал FLAT) — уровни не рисуем")
    try:
        db.db_save_ai_backtest(
            run_id, symbol, timeframe,
            {"mode": mode, "upto_sec": upto_sec, "model": model,
             "price": result["price"]},
            {"direction": direction}, levels, [])
    except Exception:
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
        except Exception:
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


# ======================================== адаптивный прогон (токен-диета v3)

def _verdict_params(compact, override=None):
    """oversold/overbought для триггера (a): из se.p, иначе дефолт 30/70."""
    if override:
        try:
            return {"oversold": float(override.get("oversold",
                                                   DEFAULT_OVERSOLD)),
                    "overbought": float(override.get("overbought",
                                                     DEFAULT_OVERBOUGHT))}
        except (TypeError, ValueError):
            pass
    params = (compact.get("se") or {}).get("p") or {}

    def _f(key, default):
        try:
            return float(params.get(key))
        except (TypeError, ValueError):
            return default

    return {"oversold": _f("oversold", DEFAULT_OVERSOLD),
            "overbought": _f("overbought", DEFAULT_OVERBOUGHT)}


def _snap_t(compact):
    return compact.get("t") or {}


def _snap_clock(compact):
    c = compact.get("c") or {}
    return c.get("ses"), c.get("open")


def _snap_hi2h(compact):
    return (compact.get("cal") or {}).get("hi2h")


def _should_call(cur, prev, last_call, params):
    """Триггеры адаптивного шага: True — звать LLM.

    last_call — {rsi, close} на прошлом РЕАЛЬНОМ вызове (None — первый шаг
    окна, триггер e). prev — предыдущий шаг (для смены ses/open/hi2h).
    Триггеры: (a) rsi у зон; (b) |Δrsi| > 7 от прошлого вызова;
    (c) |Δclose| > 0.7*atr от прошлого вызова; (d) смена ses/open/hi2h.
    """
    if last_call is None:
        return True  # (e) первый шаг окна
    t = _snap_t(cur)
    rsi = t.get("rsi")
    close = t.get("close")
    atr = t.get("atr")
    os_, ob = params["oversold"], params["overbought"]
    if rsi is not None and (abs(rsi - os_) <= ADAPTIVE_RSI_NEAR
                            or abs(rsi - ob) <= ADAPTIVE_RSI_NEAR):
        return True  # (a)
    last_rsi = last_call.get("rsi")
    if (rsi is not None and last_rsi is not None
            and abs(rsi - last_rsi) > ADAPTIVE_RSI_DELTA):
        return True  # (b)
    last_close = last_call.get("close")
    if (close is not None and last_close is not None and atr
            and abs(close - last_close) > ADAPTIVE_CLOSE_ATR * atr):
        return True  # (c)
    if prev is not None:
        if _snap_clock(cur) != _snap_clock(prev):
            return True  # (d)
        if _snap_hi2h(cur) != _snap_hi2h(prev):
            return True  # (d)
    return False


def _verdict_cache_key(compact):
    """md5(round(rsi//5), round(close//(atr/2)), флаги блоков, ses)."""
    t = _snap_t(compact)
    rsi = t.get("rsi")
    close = t.get("close")
    atr = t.get("atr")
    ses = (compact.get("c") or {}).get("ses")
    rsi_b = int(rsi // 5) if rsi is not None else None
    close_b = int(close // (atr / 2.0)) if (close is not None and atr) else None
    flags = ",".join(sorted(compact.keys()))
    raw = repr((rsi_b, close_b, flags, ses))
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _verdict_from_obj(obj):
    """Объект вердикта v3 -> нормализованный dict (или None при ошибке)."""
    if not isinstance(obj, dict):
        return None
    sig = str(obj.get("sig") or "").strip().upper()
    if sig not in ("L", "S", "F"):
        word = str(obj.get("signal") or obj.get("sig") or "").strip().upper()
        if word in ("BUY", "LONG", "L"):
            sig = "L"
        elif word in ("SELL", "SHORT", "S"):
            sig = "S"
        elif word in ("HOLD", "FLAT", "F", "WAIT"):
            sig = "F"
    if sig not in ("L", "S", "F"):
        return None
    pu = _clamp_prob(obj.get("pu"))
    pd_ = _clamp_prob(obj.get("pd"))
    pf = _clamp_prob(obj.get("pf"))
    targets = []
    raw_tg = obj.get("tg") or obj.get("targets") or []
    if isinstance(raw_tg, list):
        for item in raw_tg:
            if isinstance(item, dict):
                price = utils._clean(item.get("price"))
                prob = _clamp_prob(_dict_prob(item))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                price = utils._clean(item[0])
                prob = _clamp_prob(item[1])
            else:
                continue
            if price is not None:
                targets.append([round(price, 2), prob])
    conf = pu if sig == "L" else pd_ if sig == "S" else pf
    return {"sig": sig, "pu": pu, "pd": pd_, "pf": pf,
            "tg": targets, "conf": round(conf, 4)}


def parse_verdict(raw_text):
    """Ответ LLM -> компактный вердикт {sig,pu,pd,pf,tg,conf} или None."""
    if not raw_text:
        return None
    return _verdict_from_obj(_extract_json(raw_text))


def parse_verdict_batch(raw_text, n):
    """Ответ LLM -> список из n вердиктов или None (тогда fallback по одному)."""
    if not raw_text:
        return None
    parsed = _extract_json(raw_text)
    if isinstance(parsed, dict):
        for key in ("v", "verdicts", "batch", "results", "targets"):
            if isinstance(parsed.get(key), list):
                parsed = parsed[key]
                break
    if not isinstance(parsed, list) or len(parsed) != n:
        return None
    return [_verdict_from_obj(item) for item in parsed]


def _bar_timestamps(df, bars, upto_sec=None):
    """Unix-секунды последних bars баров среза (<= upto_sec при replay)."""
    if df is None or df.empty:
        return []
    if upto_sec is not None:
        mask = df["timestamp"] <= pd.to_datetime(int(upto_sec), unit="s",
                                                 utc=True)
        df = df[mask]
    if df is None or df.empty:
        return []
    tail = df.tail(int(bars)) if bars else df
    out = []
    for value in tail["timestamp"]:
        ts = pd.Timestamp(value)
        if getattr(ts, "tz", None) is None:
            ts = ts.tz_localize("UTC")
        out.append(int(ts.timestamp()))
    return out


def _llm_verdict(system, payload_text, model, usage):
    """Один вызов LLM с вердикт-настройками (120 токенов, T=0) и учётом usage."""
    return _llm_request(
        system,
        [{"role": "user", "content": payload_text}],
        model=model,
        purpose=None,
        max_tokens=VERDICT_MAX_TOKENS,
        temperature=VERDICT_TEMPERATURE,
        usage_out=usage,
        reasoning_effort=VERDICT_REASONING_EFFORT,
    )


def _add_usage(totals, usage):
    """Суммировать usage.{prompt,completion,total}_tokens в totals токен-диеты."""
    mapping = (("prompt_tokens", "tokens_prompt"),
               ("completion_tokens", "tokens_completion"),
               ("total_tokens", "tokens_total"))
    for src, dst in mapping:
        try:
            totals[dst] = totals.get(dst, 0) + int(usage.get(src) or 0)
        except (TypeError, ValueError):
            continue


# Кэш артефактов Этапа 2 (индекс + статистика нормализации) в рамках процесса.
_CBR_KNN_ASSETS = {"index": None, "stats": None}


def _cbr_knn_assets():
    """(index, stats) — ленивая загрузка артефактов CBR Этапа 2 (или None,None)."""
    if _CBR_KNN_ASSETS["index"] is not None:
        return _CBR_KNN_ASSETS["index"], _CBR_KNN_ASSETS["stats"]
    from app_pkg.cbr import index as cbr_index
    from app_pkg.cbr import normalize as cbr_norm
    index = cbr_index.load_index(str(config.CBR_FAISS_INDEX_PATH))
    stats = cbr_norm.load_stats(str(config.CBR_NORMALIZE_STATS_PATH))
    _CBR_KNN_ASSETS["index"] = index
    _CBR_KNN_ASSETS["stats"] = stats
    return index, stats


def _cbr_blend_verdict(verdict, snap, symbol, timeframe, ts=None):
    """Смешивание Charon + CBR (Этап 2): добавляет final_pu/final_pd/cbr.

    Гейт config.CBR_STAGE2_ENABLED (default False). Любая ошибка (нет
    артефактов, БД, данных) -> вердикт без изменений: хук не должен ронять
    бэктест. ts переопределяется на бар (иначе live => 'сейчас').
    """
    if not config.CBR_STAGE2_ENABLED or not isinstance(verdict, dict):
        return verdict
    try:
        index, stats = _cbr_knn_assets()
        if index is None or not (isinstance(stats, dict) and "mean" in stats):
            return verdict
        from app_pkg.cbr import blender, query_api
        from app_pkg.cbr.store import get_cbr_conn
        snap2 = dict(snap)
        if ts is not None:
            snap2["ts"] = int(ts)
        cbr = query_api.get_similar_summary(
            get_cbr_conn(), index, snap2, symbol, timeframe,
            bar_ts=int(ts) if ts is not None else None,
            stats=stats)
        if not isinstance(cbr, dict) or cbr.get("insufficient_data"):
            return verdict  # CBR без данных — остаётся чистый Charon
        out = dict(verdict)
        b = blender.blend(out.get("pu", 0.0), out.get("pd", 0.0), cbr,
                          cbr.get("regime"))
        out["final_pu"] = b["final_pu"]
        out["final_pd"] = b["final_pd"]
        out["cbr"] = {
            "n": cbr.get("n"), "confidence": cbr.get("confidence"),
            "winrate_up": cbr.get("winrate_up"),
            "winrate_down": cbr.get("winrate_down"),
            "regime": cbr.get("regime"),
            "w1": b["w1"], "w2": b["w2"], "source": b["source"],
        }
        return out
    except Exception as exc:  # noqa: BLE001
        log.warning("cbr blend skipped: %s", exc)
        return verdict


def _flush_pending(pending, results, cache, totals, model, system,
                   symbol=None, timeframe=None):
    """Отправить накопленные снимки: batch>1 — пачкой, иначе по одному.

    При невалидном/неполном массиве пакет разбирается по одному (fallback).
    Заполняет results/cache и копит totals.llm_calls + tokens. Если заданы
    symbol/timeframe и включён CBR-гейт — каждый размеченный снимок
    записывается в CBR-базу (вердикт-хук).
    """
    if not pending:
        return
    if len(pending) > 1:
        usage = {}
        payload = json.dumps([snap for _, _, snap, _ in pending],
                             ensure_ascii=False, separators=(",", ":"))
        raw = _llm_verdict(system, payload, model, usage)
        totals["llm_calls"] += 1
        _add_usage(totals, usage)
        verdicts = parse_verdict_batch(raw, len(pending))
        if verdicts is not None and all(verdicts):
            for (idx, ts, snap, key), verdict in zip(pending, verdicts):
                filtered = filter_verdict(verdict, snap, generated_at=ts)
                filtered = _cbr_blend_verdict(filtered, snap, symbol,
                                              timeframe, ts=ts)
                results[idx] = dict(filtered)
                cache[key] = dict(filtered)
                _cbr_store_snapshot(symbol, timeframe, ts, snap,
                                    verdict=filtered)
            return
    for idx, ts, snap, key in pending:
        usage = {}
        payload = json.dumps(snap, ensure_ascii=False, separators=(",", ":"))
        raw = _llm_verdict(system, payload, model, usage)
        totals["llm_calls"] += 1
        _add_usage(totals, usage)
        verdict = parse_verdict(raw)
        if verdict is not None:
            filtered = filter_verdict(verdict, snap, generated_at=ts)
            filtered = _cbr_blend_verdict(filtered, snap, symbol,
                                          timeframe, ts=ts)
            results[idx] = filtered
            cache[key] = dict(filtered)
            _cbr_store_snapshot(symbol, timeframe, ts, snap, verdict=filtered)


def _carry_forward(prev_verdict):
    """Пропущенный шаг: сигнал прежний, confidence *= 0.9."""
    if not prev_verdict:
        return None
    out = dict(prev_verdict)
    out["conf"] = round(out.get("conf", 0.0) * 0.9, 4)
    out["carried"] = True
    return out


def run_verdict_backtest(symbol, timeframe, bars=50, batch=BATCH_SIZE_DEFAULT,
                         adaptive=True, model=None, upto_sec=None, params=None):
    """Прогон по барам с адаптивным шагом и пакетированием (токен-диета v3).

    Каждый бар -> compact_snapshot(symbol, tf, upto_sec=ts). При adaptive=True
    LLM зовётся только по триггеру (_should_call), иначе carry-forward
    (сигнал прежний, confidence *= 0.9). Кэш вердиктов живёт в рамках прогона.
    batch>1 пакует столько снимков в один вызов (fallback — по одному).
    Метрики: tokens_prompt/tokens_completion/total, llm_calls, cache_hits.
    """
    batch = max(1, int(batch or 1))
    run_id = uuid.uuid4().hex
    df = get_series_df(symbol, timeframe, limit=bars,
                       history_limit=config.TRENDS_HISTORY)
    stamps = _bar_timestamps(df, bars, upto_sec)
    system = charon_prompt_text()
    totals = {"tokens_prompt": 0, "tokens_completion": 0, "tokens_total": 0,
              "llm_calls": 0, "cache_hits": 0}
    cache = {}
    results = {}
    signals = []
    steps = []
    pending = []
    last_call = None
    last_verdict = None
    prev = None
    use_cache = bool(adaptive)
    for idx, ts in enumerate(stamps):
        snap = compact_snapshot(symbol, timeframe, ts)
        step_params = _verdict_params(snap, params)
        trigger = True if not adaptive else _should_call(
            snap, prev, last_call, step_params)
        key = _verdict_cache_key(snap)
        if trigger and use_cache and key in cache:
            last_verdict = dict(cache[key])
            results[idx] = last_verdict
            totals["cache_hits"] += 1
            last_call = {"rsi": _snap_t(snap).get("rsi"),
                         "close": _snap_t(snap).get("close")}
        elif trigger:
            pending.append((idx, ts, snap, key))
        else:
            last_verdict = _carry_forward(last_verdict)
            results[idx] = last_verdict
        if len(pending) >= batch:
            _flush_pending(pending, results, cache, totals, model, system,
                           symbol=symbol, timeframe=timeframe)
            for pidx, _pts, psnap, _pk in pending:
                if results.get(pidx):
                    last_verdict = results[pidx]
                last_call = {"rsi": _snap_t(psnap).get("rsi"),
                             "close": _snap_t(psnap).get("close")}
            pending = []
        prev = snap
    if pending:
        _flush_pending(pending, results, cache, totals, model, system,
                       symbol=symbol, timeframe=timeframe)
        for pidx, _pts, psnap, _pk in pending:
            if results.get(pidx):
                last_verdict = results[pidx]
            last_call = {"rsi": _snap_t(psnap).get("rsi"),
                         "close": _snap_t(psnap).get("close")}
    for idx, ts in enumerate(stamps):
        verdict = results.get(idx) or {}
        sig = verdict.get("sig")
        signals.append(sig)
        steps.append({"i": idx, "upto_sec": ts, "sig": sig,
                      "conf": verdict.get("conf"),
                      "carried": bool(verdict.get("carried")),
                      "filter": verdict.get("filter")})
    n = len(stamps)
    metrics = dict(totals)
    metrics["steps"] = n
    metrics["llm_calls_ratio"] = round(totals["llm_calls"] / n, 4) if n else 0.0
    metrics["adaptive"] = adaptive
    metrics["batch"] = batch
    metrics["cache_size"] = len(cache)
    return {
        "run_id": run_id, "status": "finished", "symbol": symbol,
        "timeframe": timeframe, "bars": n, "adaptive": adaptive, "batch": batch,
        "signals": signals, "steps": steps, "metrics": metrics,
        "tokens_prompt": totals["tokens_prompt"],
        "tokens_completion": totals["tokens_completion"],
        "llm_calls": totals["llm_calls"], "cache_hits": totals["cache_hits"],
    }
