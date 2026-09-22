"""Blueprint: /api/scan (grid-search сканер стратегий).

POST /api/scan                   — запуск скана (опц. body.custom_grids)
GET  /api/scan/grids             — доступные стратегии/гриды/ТФ для UI
GET  /api/scan/<run_id>          — топ SCAN_TOP_N + verdict по каждой
POST /api/scan/<run_id>/cancel   — остановить прогон (кнопка «Отменить»)
GET  /api/scan/<run_id>/summary  — человекочитаемая сводка + рекомендации
GET  /api/scan/<run_id>/export.csv — CSV со всеми результатами прогона
GET  /api/scanner/stats          — лучшая стратегия symbol+tf (панель новостей)
"""

import csv
import io
import json
import logging
import threading
import uuid

from flask import Blueprint, Response, jsonify, request

from app_pkg import config, db
from app_pkg.ai import scanner as scanner_mod
from app_pkg.ai.scanner import _normalize_timeframes

bp = Blueprint("scanner", __name__)
log = logging.getLogger(__name__)




# ------------------------------------------------------------- verdict/label
# Формат подписи комбинации: strategy_id -> (ожидаемые ключи params, шаблон).
# Если params содержат ключи вне ожидаемого набора — подпись уходит в
# fallback «strategy_id key=val,...» (см. _strategy_label).
_LABEL_FORMATS = {
    "sma_cross": (("fast", "slow"), "{fast}/{slow}"),
    "macd_cross": (("fast", "slow", "signal"), "{fast}-{signal}-{slow}"),
    "rsi_reversal": (("period", "oversold", "overbought"),
                     "{period}/{oversold}/{overbought}"),
    "ema_cross": (("fast", "slow"), "{fast}/{slow}"),
    "bb_reversal": (("period", "std"), "{period}/{std}"),
    "bb_breakout": (("period", "std"), "{period}/{std}"),
    "supertrend": (("period", "multiplier"), "{period}-{multiplier}"),
    "stoch_reversal": (("k_period", "d_period", "oversold", "overbought"),
                       "{k_period}/{d_period}/{oversold}/{overbought}"),
    "stoch_cross": (("k_period", "d_period"), "{k_period}/{d_period}"),
    "cci_reversal": (("period", "oversold", "overbought"),
                     "{period}/{oversold}/{overbought}"),
    "vwap_reversal": (("vwap_threshold",), "{vwap_threshold}"),
    "adx_trend": (("period", "adx_threshold"), "{period}/{adx_threshold}"),
}


def _strategy_label(result):
    """Человекочитаемое название: 'SMA Cross 15/26', 'Supertrend 10-3.0'.

    Порядок: params без незнакомых ключей форматируются шаблоном стратегии
    (отсутствующие значения -> '?'); при незнакомых ключах — fallback
    'strategy_id key1=val1,key2=val2', чтобы в UI ничего не потерялось.
    """
    sid = str(result.get("strategy") or "")
    meta = config.SCAN_ALLOWED_STRATEGIES.get(sid) or {}
    label = meta.get("label", sid)
    p = result.get("params") or {}
    spec = _LABEL_FORMATS.get(sid)
    if spec:
        keys, fmt = spec
        if set(p) <= set(keys):
            values = {k: p.get(k, "?") for k in keys}
            return f"{label} " + fmt.format(**values)
    if p:
        kv = ", ".join(f"{k}={p[k]}" for k in sorted(p))
        return f"{sid or '?'} {kv}"
    return label or sid


def _verdict(result):
    """Verdict комбинации: (verdict, verdict_text, quality).

    Порядок проверок важен: сначала грубые проблемы (losing/overfit/noise),
    потом оценка стабильности по combined_sharpe.
    """
    train = result.get("train") or {}
    test = result.get("test") or {}
    train_sharpe = float(train.get("sharpe") or 0)
    test_sharpe = float(test.get("sharpe") or 0)
    combined = float(result.get("combined_sharpe") or 0)
    trades = int(test.get("trades") or 0)
    if test_sharpe <= -0.1:
        return "losing", "Убыточна на новых данных", "poor"
    if train_sharpe - test_sharpe > 0.4:
        return "overfit", "Переоптимизация (train >> test)", "poor"
    if trades < 50:
        return "noise", "Мало сделок (< 50) — недостаточно данных", "fair"
    if combined >= 0.5:
        quality = "excellent" if combined >= 1.0 else "good"
        return "stable", "Стабильна и на train, и на test", quality
    if combined >= 0.2:
        return "stable", "Работает умеренно", "fair"
    return "weak", "Слабый edge", "poor"


def _annotate(result):
    """Копия результата + verdict/verdict_text/quality/strategy_label."""
    r = dict(result)
    r["verdict"], r["verdict_text"], r["quality"] = _verdict(r)
    r["strategy_label"] = _strategy_label(r)
    return r


# --------------------------------------------------------------- custom grids
def _validate_custom_grids(custom_grids):
    """Валидация custom_grids -> (merged_grids, None) | (None, error_text).

    Каждый указанный параметр мержится ПОВЕРХ дефолтного грида: параметры,
    которых нет в custom grid, берутся из SCAN_GRIDS. Проверяем: стратегия
    в SCAN_ALLOWED_STRATEGIES, параметр в allowed list, значения — int в
    разумных пределах, 1..SCAN_MAX_CUSTOM_VALUES значений на параметр,
    общее число комбинаций ≤ SCAN_MAX_COMBINATIONS (та же формулировка
    ошибки, что и в api_scan_start — см. «Too many combinations»).
    """
    if not isinstance(custom_grids, dict) or not custom_grids:
        return None, ("custom_grids должен быть непустым объектом "
                      "{стратегия: {param: [значения]}}")
    grids = {}
    total = 0
    for sid, grid in custom_grids.items():
        meta = config.SCAN_ALLOWED_STRATEGIES.get(sid)
        if not meta:
            return None, f"Неизвестная стратегия: {sid}"
        if not isinstance(grid, dict) or not grid:
            return None, f"{sid}: грид должен быть непустым объектом"
        clean = {}
        for pname, values in grid.items():
            if pname not in meta["params"]:
                return None, (f"{sid}: параметр '{pname}' не разрешён "
                              f"(разрешены: {', '.join(meta['params'])})")
            if not isinstance(values, list):
                return None, f"{sid}.{pname}: нужен список значений"
            if not 1 <= len(values) <= config.SCAN_MAX_CUSTOM_VALUES:
                return None, (f"{sid}.{pname}: от 1 до "
                              f"{config.SCAN_MAX_CUSTOM_VALUES} значений, "
                              f"получено {len(values)}")
            lo, hi = config.SCAN_PARAM_LIMITS.get(pname, (2, 300))
            # Границы с плавающей точкой (std / multiplier / vwap_threshold)
            # разрешают дробные значения, остальные параметры — только int.
            float_ok = any(isinstance(b, float) for b in (lo, hi))
            clean_values = []
            for v in values:
                if isinstance(v, bool) or not isinstance(v, (int, float)) \
                        or (not float_ok and not isinstance(v, int)) \
                        or not lo <= v <= hi:
                    kind = "число" if float_ok else "целое"
                    return None, (f"{sid}.{pname}: значение {v!r} — "
                                  f"нужно {kind} в диапазоне {lo}–{hi}")
                clean_values.append(v)
            clean[pname] = clean_values
        merged = {**(config.SCAN_GRIDS.get(sid) or {}), **clean}
        combos = 1
        for vals in merged.values():
            combos *= len(vals)
        total += combos
        grids[sid] = merged
    if total > config.SCAN_MAX_COMBINATIONS:
        # Тот же текст, что и в api_scan_start (проверка по символам):
        # дешёвая проверка произведения длин ДО материализации комбинаций.
        return None, (f"Too many combinations: {total} > "
                      f"{config.SCAN_MAX_COMBINATIONS}. Сократите "
                      f"символы, ТФ или стратегии.")
    return grids, None


def _human_seconds(seconds):
    """Секунды -> '45 сек' / '12 мин' / '2 ч 15 мин' (для warning и UI)."""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds} сек"
    minutes = int(round(seconds / 60))
    if minutes < 60:
        return f"{minutes} мин"
    hours, mins = divmod(minutes, 60)
    return f"{hours} ч {mins} мин" if mins else f"{hours} ч"


def _estimate_seconds(total_combinations, symbols_count, timeframes_count=1):
    """Оценка длительности прогона (сек).

    combos × SCAN_SECONDS_PER_COMBINATION / SCAN_WORKERS — параллельные
    бэктесты в ThreadPoolExecutor; fetch — symbols × timeframes ×
    SCAN_FETCH_SECONDS_PER_SYMBOL (данные тянутся по каждому символу
    отдельно на КАЖДЫЙ ТФ, см. scanner.run_scan).

    combos_per_tf считаем для наглядности: total размазан по ТФ, но
    стоимость бэктестов от разбивки не зависит — важен общий объём.
    """
    timeframes = max(1, int(timeframes_count))
    combos_per_tf = total_combinations / timeframes
    workers = max(1, int(config.SCAN_WORKERS))
    backtests = (total_combinations * config.SCAN_SECONDS_PER_COMBINATION
                 / workers)
    fetch = (symbols_count * timeframes
             * config.SCAN_FETCH_SECONDS_PER_SYMBOL)
    return int(round(backtests + fetch))
@bp.route("/api/scan", methods=["POST"])
def api_scan_start():
    """Запуск grid-search скана по массиву ТФ. Возвращает {run_id, status: "started"}.

    body.timeframes — список ТФ (["15m", "1H"]); body.timeframe (single str)
    принимается для обратной совместимости и нормализуется в [_normalize_timeframes].
    Каждый ТФ валидируется против config.SCAN_TIMEFRAMES.

    Опционально body.custom_grids — свои вариации параметров:
      {"sma_cross": {"fast": [5, 10, 15], "slow": [20, 30]}, ...}
    Параметры, не указанные в custom grid, берутся из дефолтного грида.

    Защита от перегрузки: total > SCAN_MAX_COMBINATIONS -> 400 (скан не
    стартует); оценка > SCAN_HARD_LIMIT_SECONDS (4 часа) -> 400; оценка >
    SCAN_WARN_SECONDS (30 минут) -> 200 + warning (UI спрашивает подтверждение).
    В ответе всегда масштаб прогона: estimated_seconds, estimated_human,
    total_combinations, total_symbols, total_timeframes.
    """
    body = request.get_json(silent=True) or {}
    symbols = body.get("symbols") or []
    strategies = body.get("strategies") or []
    custom_grids = body.get("custom_grids") or None
    # Опциональный выбор режима истории: True — полная (HISTORY_LIMITS[tf]),
    # False — быстрая (SCAN_REPLAY_LIMIT). Не задан — дефолт из config.
    use_full_history = body.get("use_full_history")
    if use_full_history is not None and not isinstance(use_full_history, bool):
        return jsonify({"error": "use_full_history must be a boolean"}), 400
    effective_full = (config.SCAN_USE_FULL_HISTORY
                      if use_full_history is None else bool(use_full_history))

    if not isinstance(symbols, list) or not symbols:
        return jsonify({"error": "symbols must be a non-empty list"}), 400
    if not isinstance(strategies, list) or not strategies:
        return jsonify({"error": "strategies must be a non-empty list"}), 400

    symbols = [str(s).upper() for s in symbols]
    strategies = [str(s) for s in strategies]
    bad_symbols = [s for s in symbols if s not in config.SYMBOLS]
    if bad_symbols:
        return jsonify({"error": f"Invalid symbols: {bad_symbols}"}), 400
    bad_strategies = [s for s in strategies
                      if s not in config.SCAN_ALLOWED_STRATEGIES]
    if bad_strategies:
        return jsonify({"error": f"Unknown strategies: {bad_strategies}"}), 400

    # Таймфреймы: массив или fallback на single timeframe (обратная совместимость).
    timeframes = _normalize_timeframes(
        body.get("timeframes") or body.get("timeframe"))
    if not timeframes:
        return jsonify({"error": "At least one timeframe required"}), 400

    # Валидация каждого ТФ против SCAN_TIMEFRAMES.
    bad_tf = [tf for tf in timeframes if tf not in config.SCAN_TIMEFRAMES]
    if bad_tf:
        return jsonify({"error": f"Invalid timeframes: {bad_tf}"}), 400

    grids = None
    if custom_grids:
        grids, err = _validate_custom_grids(custom_grids)
        if err:
            return jsonify({
                "error": err,
                "max_combinations": config.SCAN_MAX_COMBINATIONS,
            }), 400

    # Объём скана (symbols × timeframes × Σ combos) считаем до старта:
    # UI показывает масштаб — сколько всего комбинаций будет прогнано.
    total_combinations = scanner_mod.plan_totals(
        symbols, strategies, grids=grids, timeframes=timeframes)
    if total_combinations > config.SCAN_MAX_COMBINATIONS:
        return jsonify({
            "error": (f"Too many combinations: {total_combinations} > "
                      f"{config.SCAN_MAX_COMBINATIONS}. "
                      "Сократите символы, ТФ или стратегии."),
            "total_combinations": total_combinations,
            "total_symbols": len(symbols),
            "total_timeframes": len(timeframes),
            "max_combinations": config.SCAN_MAX_COMBINATIONS,
        }), 400

    estimated_seconds = _estimate_seconds(
        total_combinations, len(symbols), len(timeframes))
    if estimated_seconds > config.SCAN_HARD_LIMIT_SECONDS:
        return jsonify({
            "error": (f"Too long: прогон займёт "
                      f"{_human_seconds(estimated_seconds)} > 4 часов. "
                      "Сократите объём или запустите по частям."),
            "estimated_seconds": estimated_seconds,
            "estimated_human": _human_seconds(estimated_seconds),
            "total_combinations": total_combinations,
            "total_symbols": len(symbols),
            "total_timeframes": len(timeframes),
            "max_combinations": config.SCAN_MAX_COMBINATIONS,
            "hard_limit_seconds": config.SCAN_HARD_LIMIT_SECONDS,
        }), 400

    warning = None
    if estimated_seconds > config.SCAN_WARN_SECONDS:
        warning = (f"Прогон займёт {_human_seconds(estimated_seconds)} "
                   f"(больше 30 минут). Продолжить?")

    run_id = uuid.uuid4().hex

    def _bg():
        try:
            scanner_mod.run_scan(
                symbols, timeframes, strategies,
                run_id=run_id, grids=grids, use_full_history=use_full_history)
        except Exception:
            log.exception("scan run %s failed", run_id)

    threading.Thread(target=_bg, daemon=True,
                     name=f"scan-{run_id[:8]}").start()
    payload = {
        "run_id": run_id, "status": "started",
        "use_full_history": effective_full,
        "total_combinations": total_combinations,
        "estimated_seconds": estimated_seconds,
        "estimated_human": _human_seconds(estimated_seconds),
        "total_symbols": len(symbols),
        "total_timeframes": len(timeframes),
        "warn_seconds": config.SCAN_WARN_SECONDS,
        "hard_limit_seconds": config.SCAN_HARD_LIMIT_SECONDS,
        "filtered_count": scanner_mod.get_run_stats(run_id).get("filtered", 0),
    }
    if warning:
        payload["warning"] = warning
    return jsonify(payload)


@bp.route("/api/scan/<run_id>/cancel", methods=["POST"])
def api_scan_cancel(run_id):
    """Остановить прогон скана (кнопка «Отменить» в UI).

    Не убивает воркер жёстко, а помечает run_id на отмену: run_scan
    проверяет флаг на границах пар/стратегий/комбинаций и завершает прогон
    досрочно. Успевшие записаться результаты остаются (GET /api/scan/<run_id>
    отдаст частичные данные; RL stats получают "cancelled": True, SSE шлёт
    финальное scan_progress с cancelled=True).
    """
    scanner_mod.cancel_run(str(run_id))
    return jsonify({"status": "cancelled", "run_id": str(run_id)})


@bp.route("/api/scan/<run_id>")
def api_scan_results(run_id):
    """Топ-результаты прогона, отсортированные по combined_sharpe DESC.

    ?min_sharpe=0 (default) — показывать только прибыльные комбинации;
    ?min_sharpe=-999 — показать всё. Если ни один результат не прошёл
    фильтр — возвращаем warning, а не молчаливый пустой список.
    """
    try:
        min_sharpe = float(request.args.get("min_sharpe", 0))
    except (TypeError, ValueError):
        min_sharpe = 0.0
    results = [_annotate(r) for r in db.db_get_scan_results(
        run_id, limit=config.SCAN_TOP_N)
        if (r.get("combined_sharpe") or 0) >= min_sharpe]
    stats = scanner_mod.get_run_stats(run_id)
    if not results:
        return jsonify({
            "run_id": run_id, "results": [], "count": 0,
            "warning": "Все комбинации убыточны. Попробуйте другую "
                       "стратегию или timeframe.",
            "stats": stats,
        })
    return jsonify({"run_id": run_id, "results": results,
                    "count": len(results), "stats": stats})


@bp.route("/api/scan/grids")
def api_scan_grids():
    """Доступные стратегии (параметры + дефолтные гриды) и ТФ — для UI.

    param_limits и константы оценки нужны фронту, чтобы считать число
    комбинаций (и время прогона) теми же цифрами, что и POST /api/scan.
    default_timeframes — ТФ, отмеченные в UI по умолчанию (чекбоксы
    #scan-timeframes): берутся из config.SCAN_DEFAULT_TIMEFRAMES.

    rr / sl_atr — базовый R/R скана (настройка UI, db.get_scan_rr) и
    стоп в ATR: те самые множители, которыми run_backtest строит TP/SL.
    Фронт показывает поле «Базовый R/R» с этими же границами.
    """
    return jsonify({
        "strategies": config.SCAN_ALLOWED_STRATEGIES,
        "timeframes": config.SCAN_TIMEFRAMES,
        "default_timeframes": list(config.SCAN_DEFAULT_TIMEFRAMES),
        "default_use_full_history": config.SCAN_USE_FULL_HISTORY,
        "max_combinations": config.SCAN_MAX_COMBINATIONS,
        "max_combinations_full": config.SCAN_MAX_COMBINATIONS_FULL,
        "max_custom_values": config.SCAN_MAX_CUSTOM_VALUES,
        "param_limits": config.SCAN_PARAM_LIMITS,
        "workers": config.SCAN_WORKERS,
        "seconds_per_combination": config.SCAN_SECONDS_PER_COMBINATION,
        "fetch_seconds_per_symbol": config.SCAN_FETCH_SECONDS_PER_SYMBOL,
        "warn_seconds": config.SCAN_WARN_SECONDS,
        "hard_limit_seconds": config.SCAN_HARD_LIMIT_SECONDS,
        "rr": db.get_scan_rr(),
        "rr_default": config.SCAN_RR_DEFAULT,
        "rr_min": config.SCAN_RR_MIN,
        "rr_max": config.SCAN_RR_MAX,
        "rr_step": config.SCAN_RR_STEP,
        "sl_atr": config.BACKTEST_SL_ATR,
    })


@bp.route("/api/scan/settings", methods=["POST"])
def api_scan_settings():
    """Сохранить настройку скана по ключу body {rr: <число>}.

    Пока поддерживается один ключ — «базовый R/R» (body.rr), которым
    run_backtest строит TP от стопа (TP = rr × SL-дистанция). Значение
    валидируется и клампится в [rr_min, rr_max]. Применяется к следующему
    прогону скана, а также к /api/backtest и /api/backtest/trades — чтобы
    блоки на графике и панель оставались 1:1.
    """
    body = request.get_json(silent=True) or {}
    if "rr" not in body:
        return jsonify({"error": "rr is required"}), 400
    try:
        saved = db.set_scan_rr(body.get("rr"))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"status": "saved", "rr": saved})


@bp.route("/api/scan/<run_id>/summary")
def api_scan_summary(run_id):
    """Человекочитаемая сводка прогона: вердикты, лучшие по символам,
    рекомендации (для UI-блока сверху и копирования отчёта)."""
    annotated = [_annotate(r) for r in
                 db.db_get_scan_results(run_id, limit=10 ** 9)]
    kept = len(annotated)
    stats = scanner_mod.get_run_stats(run_id)
    total = int(stats.get("total") or 0) or kept

    by_verdict = {}
    best_by_symbol = {}
    verdicts_by_symbol = {}
    for r in annotated:
        by_verdict[r["verdict"]] = by_verdict.get(r["verdict"], 0) + 1
        sym = r.get("symbol") or "?"
        verdicts_by_symbol.setdefault(sym, []).append(r["verdict"])
        cur = best_by_symbol.get(sym)
        if cur is None or (r.get("combined_sharpe") or 0) \
                > (cur.get("combined_sharpe") or 0):
            best_by_symbol[sym] = {
                "symbol": sym,
                "strategy": r.get("strategy"),
                "strategy_label": r["strategy_label"],
                "params": r.get("params") or {},
                "combined_sharpe": r.get("combined_sharpe"),
                "verdict": r["verdict"],
            }

    stable = [r for r in annotated if r["verdict"] == "stable"]
    stable.sort(key=lambda r: r.get("combined_sharpe") or 0, reverse=True)
    recommendations = []
    for r in stable[:3]:
        recommendations.append(
            f"✅ {r.get('symbol')} + {r['strategy_label']} — стабильный edge,"
            f" sharpe {round(r.get('combined_sharpe') or 0, 2)}")
    if not stable:
        recommendations.append("Ни одна стратегия не показала стабильный edge")
    for sym, vs in verdicts_by_symbol.items():
        # Символ, где ВСЕ комбинации убыточны — «проблемный».
        if vs and all(v == "losing" for v in vs):
            recommendations.append(
                f"⚠️ {sym} — ни одна комбинация не работает стабильно")
    if stable:
        best = stable[0]
        recommendations.append(
            f"🎯 Лучший вариант: {best.get('symbol')}, "
            f"{best['strategy_label']}")

    return jsonify({
        "run_id": run_id,
        "total_combinations": total,
        "kept": kept,
        "by_verdict": by_verdict,
        "best_by_symbol": best_by_symbol,
        "recommendations": recommendations,
    })


@bp.route("/api/scanner/stats", methods=["GET"])
def api_scanner_stats():
    """Статистика лучшей стратегии сканера для symbol+timeframe.

    ?symbol=BTCUSDT&timeframe=15m -> лучший результат по combined_sharpe из
    scan_results (SQLite, общий регистр сканов: каждый прогоночный CSV
    scan_<run_id>.csv — просто выгрузка тех же строк). {} если скана по этой
    паре ещё не было (200, а не 404: пустая карточка — нормальное состояние).

    Каждая строка train/test: winrate/max_dd/profit_factor/trades —
    out-of-sample окно (test) и train-окно соответственно.
    """
    symbol = str(request.args.get("symbol") or "").upper()
    tf = str(request.args.get("timeframe") or "").strip()
    if not symbol:
        return jsonify({"error": "symbol is required"}), 400
    if symbol not in config.SYMBOLS:
        return jsonify({"error": f"Invalid symbol: {symbol}"}), 400
    best = db.db_get_best_scan_stats(symbol, tf)
    if not best:
        return jsonify({})
    annotated = _annotate(best)
    train = best.get("train") or {}
    test = best.get("test") or {}
    return jsonify({
        "symbol": best.get("symbol"),
        "timeframe": best.get("timeframe"),
        "strategy": best.get("strategy"),
        "strategy_label": annotated["strategy_label"],
        "params": best.get("params") or {},
        "train_sharpe": train.get("sharpe"),
        "test_sharpe": test.get("sharpe"),
        "combined_sharpe": best.get("combined_sharpe"),
        "winrate": test.get("winrate"),
        "max_dd": test.get("max_dd"),
        "verdict": annotated["verdict"],
        "verdict_text": annotated["verdict_text"],
    })


@bp.route("/api/scan/<run_id>/export.csv")
def api_scan_export_csv(run_id):
    """CSV со ВСЕМИ результатами прогона (не только топ-N).

    winrate/max_dd/profit_factor/trades — out-of-sample (test) значения.
    """
    results = db.db_get_scan_results(run_id, limit=10 ** 9)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["symbol", "timeframe", "strategy", "params", "train_sharpe",
                     "test_sharpe", "combined_sharpe", "winrate", "max_dd",
                     "profit_factor", "trades"])
    for r in results:
        train = r.get("train") or {}
        test = r.get("test") or {}
        writer.writerow([
            r.get("symbol", ""),
            r.get("timeframe", ""),
            r.get("strategy", ""),
            json.dumps(r.get("params") or {}, ensure_ascii=False),
            train.get("sharpe", ""),
            test.get("sharpe", ""),
            r.get("combined_sharpe", ""),
            test.get("winrate", ""),
            test.get("max_dd", ""),
            test.get("profit_factor", ""),
            test.get("trades", ""),
        ])
    return Response(
        buf.getvalue(), mimetype="text/csv",
        headers={"Content-Disposition":
                 f"attachment; filename=scan_{run_id}.csv"})
