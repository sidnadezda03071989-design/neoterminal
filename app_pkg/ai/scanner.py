# -*- coding: utf-8 -*-
"""Grid-search сканер стратегий.

Перебирает декартово произведение параметров SCAN_GRIDS по каждой стратегии,
прогоняет бэктест на train (70%) и out-of-sample test (30%) окнах, считает
консервативную оценку combined_sharpe = min(train_sharpe, test_sharpe) и
пишет результаты в SQLite. Прогресс уходит клиентам по SSE (scan_progress).
"""

import itertools
import logging
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from app_pkg import config, db
from app_pkg.ai.backtest import run_backtest
from app_pkg.data.fetch import get_replay_df
from app_pkg.indicators import compute_indicators
from app_pkg.ws import _ws_push

log = logging.getLogger(__name__)

# Статистика последних прогонов: run_id -> {"total", "kept", "filtered",
# "finished"}. Нужна API (/api/scan), чтобы UI знал масштаб скана: сколько
# комбинаций сгенерировано и сколько отсеялось по SCAN_MIN_TRADES.
RUN_STATS = {}
RUN_STATS_MAX = 50  # сколько последних прогонов держим в памяти


def get_run_stats(run_id):
    """Статистика прогона для API; {} если прогон неизвестен."""
    return dict(RUN_STATS.get(run_id) or {})


def _prune_run_stats():
    """Чистим старые прогоны, чтобы RUN_STATS не рос бесконечно."""
    if len(RUN_STATS) <= RUN_STATS_MAX:
        return
    for run_id in list(RUN_STATS)[:len(RUN_STATS) - RUN_STATS_MAX]:
        RUN_STATS.pop(run_id, None)


# Отмена сканов по кнопке «Отменить» (POST /api/scan/<run_id>/cancel).
# run_scan проверяет флаг на границах пар/стратегий/комбинаций и завершает
# прогон досрочно; статистика помечается "cancelled": True, а записавшиеся
# результаты остаются доступны через GET /api/scan/<run_id>.
_CANCELLED = set()
_CANCELLED_MAX = 50


def cancel_run(run_id):
    """Пометить прогон на остановку. Воркер остановится на ближайшей
    границе (пара/комбинация) и закроет RUN_STATS с cancelled=True."""
    log.info("scan: %s cancel requested", run_id)
    _CANCELLED.add(run_id)
    if len(_CANCELLED) > _CANCELLED_MAX:
        for rid in list(_CANCELLED)[:len(_CANCELLED) - _CANCELLED_MAX]:
            _CANCELLED.discard(rid)


def is_cancelled(run_id):
    """True, если прогон помечен на отмену (в т.ч. уже завершённый)."""
    return run_id in _CANCELLED


def _generate_combinations(strategy, grid):
    """Декартово произведение параметров грида -> list[dict].

    Если комбинаций больше SCAN_MAX_COMBINATIONS — случайная подвыборка
    с seed=42 (воспроизводимый набор при одинаковом гриде).
    """
    keys = sorted(grid.keys())  # детерминированный порядок ключей
    combos = [
        dict(zip(keys, values))
        for values in itertools.product(*(grid[k] for k in keys))
    ]
    if len(combos) > config.SCAN_MAX_COMBINATIONS:
        combos = random.Random(42).sample(combos, config.SCAN_MAX_COMBINATIONS)
    log.info("scan grid %s: %d combinations", strategy, len(combos))
    return combos


def _normalize_timeframes(timeframes):
    """ТФ скана -> список. str (обратная совместимость) -> [str]; None -> дефолт.

    Пустой список остаётся пустым: роут POST /api/scan отдаёт на него 400
    («At least one timeframe required»), а run_scan просто ничего не считает.
    """
    if timeframes is None:
        return [config.DEFAULT_TIMEFRAME]
    if isinstance(timeframes, str):
        return [timeframes]
    return [str(tf) for tf in timeframes]


def build_plan(strategies, grids=None, symbols=None, timeframes=None):
    """План скана.

    Без symbols/timeframes — раскладка гридов [(strategy, combos)] (объём
    скана, перебор стратегий). С ними — плоский список ЗАДАЧ
    [(symbol, tf, strategy, params)] в порядке прогона:
    symbol -> tf -> strategy -> params (тот же порядок обходит run_scan).

    grids — опциональное переопределение дефолтных гридов (custom_grids
    из POST /api/scan): {strategy: {param: [values]}}; для стратегий без
    переопределения используется config.SCAN_GRIDS.
    """
    custom = grids or {}
    plan = []
    for strategy in strategies:
        grid = custom.get(strategy) or config.SCAN_GRIDS.get(strategy)
        if not grid:
            log.warning("scan: unknown strategy %s — пропуск", strategy)
            continue
        plan.append((strategy, _generate_combinations(strategy, grid)))
    if symbols is None or timeframes is None:
        return plan
    # params — те же dict-объекты, что и в plan: плоский список задач не
    # дублирует комбинации, только ссылки на них.
    return [
        (symbol, tf, strategy, params)
        for symbol in symbols
        for tf in timeframes
        for strategy, combos in plan
        for params in combos
    ]


def plan_totals(symbols, strategies, grids=None, timeframes=None):
    """Сколько всего комбинаций будет прогнано (объём скана до старта).

    total = symbols × timeframes × Σ combos_per_strategy. Без timeframes
    (старые вызовы) считаем один ТФ.
    """
    return len(build_plan(strategies, grids, symbols=symbols,
                          timeframes=_normalize_timeframes(timeframes)))


def _metrics(res):
    """Метрики прогона run_backtest -> плоский dict для JSON."""
    trades = res.get("trades") or []
    gross = sum(t.get("pnl", 0) for t in trades if t.get("pnl", 0) > 0)
    loss = abs(sum(t.get("pnl", 0) for t in trades if t.get("pnl", 0) < 0))
    if loss > 0:
        profit_factor = round(gross / loss, 2)
    else:
        # Убыточных сделок нет; бесконечность в JSON/SQLite не кладём, каппим.
        profit_factor = 999.0 if gross > 0 else 0.0
    return {
        "sharpe": res.get("sharpe_ratio", 0),
        "winrate": res.get("win_rate", 0),
        # BLOCK-38: expectancy (% за сделку) и R/R — главные метрики панели:
        # один winrate не отличает 36% выигрышей при R/R 2:1 (плюс) от 70% при
        # R/R 1:3 (минус). Считает run_backtest по pnl_pct сделок.
        "expectancy": res.get("expectancy", 0),
        "rr_ratio": res.get("rr_ratio"),
        "avg_win_pct": res.get("avg_win_pct", 0),
        "avg_loss_pct": res.get("avg_loss_pct", 0),
        "max_dd": res.get("max_drawdown", 0),
        "total_return": res.get("total_return", 0),
        "profit_factor": profit_factor,
        "trades": res.get("total_trades", 0),
    }


def _run_single(symbol, tf, strategy_name, params, df_train, df_test,
                ind_train=None, ind_test=None):
    """Один прогон комбинации: бэктест на train и на test.

    df_train/df_test — готовые окна (уже нарезаны в run_scan): передаются
    прямо в run_backtest(df=...), который НЕ ходит за данными. Так 90
    комбинаций × 10 символов дают 10 запросов get_replay_df (по одному
    на символ), а не 900. combined_sharpe = min(train, test) —
    консервативная оценка: стратегия должна быть хороша на обоих окнах.

    TP/SL (BLOCK-37): оба окна считаются с уровнями — SL = BACKTEST_SL_ATR×ATR,
    TP строится от стопа с базовым R/R из настройки скана (db.get_scan_rr(),
    UI-поле «Базовый R/R»; фолбэк config.BACKTEST_RR). Панель и «📊 Показать
    на графике» гонят один и тот же вариант (routes/backtest.py читает ту же
    настройку), поэтому trades в панели и блоки на графике совпадают 1:1.

    ind_train/ind_test — готовые индикаторы окон (compute_indicators),
    посчитанные ОДИН раз на пару (symbol, tf) в run_scan; если не переданы
    (внешние вызовы _run_single напрямую) — считаются здесь.

    Отсев: меньше SCAN_MIN_TRADES сделок хотя бы на одном окне.
    Возвращает dict или None.
    """
    if df_train is None or df_test is None \
            or len(df_train) < 2 or len(df_test) < 2:
        return None

    if ind_train is None:
        ind_train = compute_indicators(df_train)
    if ind_test is None:
        ind_test = compute_indicators(df_test)

        # BLOCK-42: сканер оценивает стратегию с ВЫХОДОМ ТОЛЬКО по линиям TP/SL
        # (config.BACKTEST_SL_ATR × ATR для риска, config.BACKTEST_RR для R/R).
        # Тот же самый прогон отдаёт /api/backtest/trades — поэтому число
        # сделок в панели и число блоков на графике совпадают 1:1, а у каждой
        # сделки есть tp_price/sl_price (IN/OUT/TP/SL на графике).
    res_train = run_backtest(symbol, tf, None, None, strategy_name, params,
                             df=df_train, ind=ind_train,
                             sl_atr=config.BACKTEST_SL_ATR,
                             rr=db.get_scan_rr())
    res_test = run_backtest(symbol, tf, None, None, strategy_name, params,
                            df=df_test, ind=ind_test,
                            sl_atr=config.BACKTEST_SL_ATR,
                            rr=db.get_scan_rr())
    if "error" in res_train or "error" in res_test:
        log.warning("scan %s %s %s: бэктест вернул ошибку",
                    symbol, strategy_name, params)
        return None

    m_train, m_test = _metrics(res_train), _metrics(res_test)
    if min(m_train["trades"], m_test["trades"]) < config.SCAN_MIN_TRADES:
        return None

    return {
        "params": params,
        "train": m_train,
        "test": m_test,
        "combined_sharpe": min(m_train["sharpe"], m_test["sharpe"]),
        "total_trades": m_train["trades"] + m_test["trades"],
    }


def _eta_seconds(start, done, total, elapsed_factor=1.0):
    """Оценка оставшегося времени (сек) по средней скорости прогона.

    rate = done / elapsed (комбинаций в секунду), eta = (total - done) / rate.
    None, если ещё ничего не обработано или прогон уже закончился.

    elapsed_factor — поправка для полной истории (SCAN_USE_FULL_HISTORY):
    бэктест на 20k свечей идёт ~4× дольше, чем на 5k, поэтому «эффективный»
    прошедший отсчёт умножается на фактор (rate становится в разы ниже).
    """
    if not done or done >= total:
        return None
    elapsed = (time.time() - start) * max(float(elapsed_factor), 1e-9)
    if elapsed <= 0:
        return None
    rate = done / elapsed
    if rate <= 0:
        return None
    return int(round((total - done) / rate))


def _task_key(task):
    """Ключ группировки задач скана (symbol, tf, strategy) — см. build_plan."""
    return task[:3]


def _push_progress(run_id, done, total, current, eta_seconds=None,
                   tf_index=None, tf_total=None, current_tf=None,
                   current_symbol=None, last_error=None, failed_at=None,
                   cancelled=False):
    """SSE-событие прогресса скана (прогресс не должен ломать сам скан).

    eta_seconds добавляется не в каждое событие, а раз в
    SCAN_ETA_PUSH_EVERY комбинаций (см. run_scan): UI показывает
    «Осталось ~M мин».

    ТФ-поля (при прогоне по нескольким ТФ): tf_index/tf_total — номер и
    всего ТФ, current_tf/current_symbol — где скан сейчас:
    {..., "tf_index": 2, "tf_total": 5, "current_tf": "1H",
     "current_symbol": "USDCHF", "eta_seconds": 1800}.

    Поля ошибки (пара (symbol, tf) упала): last_error — текст исключения,
    failed_at — "SYMBOL/TF". Ошибка не останавливает скан — следующая пара
    обрабатывается.

    cancelled=True — финальное событие после остановки по кнопке «Отменить»
    (done может быть < total): UI завершает прогон и показывает частичные
    результаты.
    """
    try:
        data = {"run_id": run_id, "done": done, "total": total,
                "current": current}
        if eta_seconds is not None:
            data["eta_seconds"] = eta_seconds
        if tf_total is not None:
            data.update(tf_index=tf_index, tf_total=tf_total,
                        current_tf=current_tf, current_symbol=current_symbol)
        if last_error is not None:
            data["last_error"] = last_error
            data["failed_at"] = failed_at
        if cancelled:
            data["cancelled"] = True
        _ws_push("scan_progress", data)
    except Exception:  # noqa: BLE001
        log.exception("scan_progress push failed")


def run_scan(symbols, timeframes, strategies, run_id=None, grids=None,
             use_full_history=None):
    """Запуск скана: символы × ТФ × стратегии, перебор комбинаций.

    timeframes — список ТФ (["15m", "1H"]); str принимается для обратной
    совместимости (оборачивается в [str]) — см. _normalize_timeframes.

    grids — опциональное переопределение дефолтных гридов (custom_grids).

    use_full_history — опционально: переопределяет config.SCAN_USE_FULL_HISTORY
    для ЭТОГО прогона (фетч берёт HISTORY_LIMITS[tf] вместо SCAN_REPLAY_LIMIT,
    ETA прогресса умножается на ~4; полная история = 20k крипта / 13k форекс).

    Для каждой пары (symbol, tf): get_replay_df за SCAN_PERIOD_DAYS — ОДИН
    запрос на пару (df переиспользуется всеми комбинациями через df=), сплит
    70/30 по времени, далее ThreadPoolExecutor(SCAN_WORKERS) -> _run_single
    параллельно; прошедшие отсев результаты -> db_save_scan_result (пишется
    tf). Прогресс в SSE: {event: "scan_progress", data: {run_id, done, total,
    current, tf_index, tf_total, current_tf, current_symbol}}; раз в
    SCAN_ETA_PUSH_EVERY комбинаций в событие добавляется eta_seconds
    (остаток прогона, сек). Возвращает run_id.
    """
    full = config.SCAN_USE_FULL_HISTORY if use_full_history is None \
        else bool(use_full_history)
    eta_factor = 4.0 if full else 1.0
    run_id = run_id or uuid.uuid4().hex
    start = time.time()  # отсчёт для ETA прогресса (см. _eta_seconds)
    to_sec = int(time.time())
    from_sec = to_sec - config.SCAN_PERIOD_DAYS * 86400

    # Объём для прогресса считаем заранее (генерация комбинаций дешёвая).
    # Задачи — плоский план: symbol -> tf -> strategy -> params.
    tfs = _normalize_timeframes(timeframes)
    tasks = build_plan(strategies, grids, symbols=symbols, timeframes=tfs)
    total = len(tasks)
    tf_index_by_name = {tf: i for i, tf in enumerate(tfs, start=1)}
    done = 0
    kept = 0  # комбинаций прошло отсев по SCAN_MIN_TRADES
    RUN_STATS[run_id] = {"total": total, "kept": 0, "filtered": 0,
                         "finished": False, "tf_index": 0,
                         "tf_total": len(tfs), "current_tf": None,
                         "current_symbol": None}

    df = None
    df_key = None  # (symbol, tf) текущего df
    # Задачи — плоский план symbol -> tf -> strategy -> params (build_plan):
    # группируем их по паре (symbol, tf) -> по стратегии -> [params], чтобы
    # данные и индикаторы считались ОДИН раз на пару, а не на каждую
    # комбинацию из сотен.
    plan_by_pair = {}
    for task in tasks:
        key = (task[0], task[1])
        plan_by_pair.setdefault(key, {}).setdefault(task[2], []).append(task[3])
    timeout = config.SCAN_SYMBOL_TIMEOUT_SECONDS

    for (symbol, tf), strategies in plan_by_pair.items():
        if is_cancelled(run_id):
            break
        tf_index = tf_index_by_name.get(tf, 1)
        pair_start = time.time()
        log.info("scan: start %s (tf=%s)", symbol, tf)
        try:
            # Один запрос данных на пару (symbol, tf): готовый df
            # переиспользуется во всех комбинациях (см. _run_single).
            if df_key != (symbol, tf):
                # Полная история: максимум доступных свечей из HISTORY_LIMITS[tf]
                # (НЕ SCAN_REPLAY_LIMIT — иначе упрёмся в 20000-потолок только
                # по булеву режима); быстрый режим — SCAN_REPLAY_LIMIT.
                fetch_limit = (
                    config.HISTORY_LIMITS.get(tf, 20000)
                    if full else config.SCAN_REPLAY_LIMIT
                )
                df = get_replay_df(symbol, tf, from_sec, to_sec,
                                   limit=fetch_limit)
                df_key = (symbol, tf)
                rows = 0 if df is None else len(df)
                log.info("scan: %s/%s fetch limit=%d, got %d rows",
                         symbol, tf, fetch_limit, rows)
                if rows < 1000:
                    log.warning("scan: %s/%s too few rows (%d) — данные почти "
                                "не покрывают историю", symbol, tf, rows)
            if df is None or df.empty:
                log.warning("scan: %s/%s empty df, skip", symbol, tf)
                done += sum(len(combs) for combs in strategies.values())
                RUN_STATS[run_id].update(
                    tf_index=tf_index, tf_total=len(tfs), current_tf=tf,
                    current_symbol=symbol)
                _push_progress(run_id, done, total, f"{symbol}:{tf}:no-data",
                               tf_index=tf_index, tf_total=len(tfs),
                               current_tf=tf, current_symbol=symbol)
                continue
            # Жёсткая обрезка: MT5 может вернуть куда больше запрошенного
            # лимита (напр. 23115 свечей XAUUSD при limit=5000) — обрезаем до
            # fetch_limit (20000 в полной истории / SCAN_REPLAY_LIMIT в быстрой).
            if len(df) > fetch_limit:
                df = df.iloc[-fetch_limit:].reset_index(drop=True)
            log.info("scan: %s/%s df rows=%d", symbol, tf, len(df))

            # Сплит 70/30 по времени (df отсортирован по timestamp) и
            # индикаторы окон — ОДИН раз на пару: run_backtest больше не
            # пересчитывает compute_indicators на каждую комбинацию.
            split = int(len(df) * config.SCAN_TRAIN_SPLIT)
            df_train = df.iloc[:split].reset_index(drop=True)
            df_test = df.iloc[split:].reset_index(drop=True)
            ind_train = compute_indicators(df_train)
            ind_test = compute_indicators(df_test)

            for strategy, params_list in strategies.items():
                if is_cancelled(run_id):
                    break
                if time.time() - pair_start > timeout:
                    log.warning("scan: %s/%s timeout — пропуск стратегии %s "
                                "(%d комбинаций)", symbol, tf, strategy,
                                len(params_list))
                    continue
                log.info("scan: %s/%s strategy=%s — %d combos",
                         symbol, tf, strategy, len(params_list))
                strategy_kept = 0
                executor = ThreadPoolExecutor(
                    max_workers=config.SCAN_WORKERS)
                try:
                    futures = {
                        executor.submit(_run_single, symbol, tf, strategy,
                                        params, df_train, df_test,
                                        ind_train, ind_test): params
                        for params in params_list
                    }
                    for fut in as_completed(futures):
                        if is_cancelled(run_id):
                            for pending in futures:
                                pending.cancel()
                            break
                        if time.time() - pair_start > timeout:
                            log.warning("scan: %s/%s timeout (%ds)",
                                        symbol, tf, timeout)
                            break
                        params = futures[fut]
                        done += 1
                        try:
                            res = fut.result()
                        except Exception:  # noqa: BLE001 — не роняем весь скан
                            log.exception("scan: комбинация %s упала", params)
                            res = None
                        if res:
                            kept += 1
                            strategy_kept += 1
                            db.db_save_scan_result(
                                run_id, symbol, tf, strategy,
                                res["params"], res["train"], res["test"],
                                res["combined_sharpe"], res["total_trades"])
                        eta = None
                        if config.SCAN_ETA_PUSH_EVERY \
                                and done % config.SCAN_ETA_PUSH_EVERY == 0:
                            eta = _eta_seconds(start, done, total,
                                               elapsed_factor=eta_factor)
                        if eta is not None:
                            # Дублируем ETA в RUN_STATS: поллинг-фолбэк UI
                            # (GET /api/scan/<run_id>) видит его без SSE.
                            RUN_STATS[run_id]["done"] = done
                            RUN_STATS[run_id]["eta_seconds"] = eta
                        # ТФ-поля тоже в RUN_STATS: без SSE поллинг рисует
                        # тот же «ТФ i/n: tf · symbol».
                        RUN_STATS[run_id].update(
                            tf_index=tf_index, tf_total=len(tfs),
                            current_tf=tf, current_symbol=symbol)
                        _push_progress(run_id, done, total,
                                       f"{symbol}:{tf}:{strategy}:{params}",
                                       eta_seconds=eta, tf_index=tf_index,
                                       tf_total=len(tfs), current_tf=tf,
                                       current_symbol=symbol)
                finally:
                    # Отмена: не ждём незаконченные воркеры (wait=False,
                    # cancel_futures=True) — финальное скан-progress событие
                    # уходит сразу, а дописывающиеся результаты сохраняются
                    # в фоне.
                    executor.shutdown(wait=False, cancel_futures=True)
                log.info("scan: %s/%s strategy=%s done (%d saved)",
                         symbol, tf, strategy, strategy_kept)
        except Exception as exc:  # noqa: BLE001 — не валим весь скан
            log.exception("scan: %s/%s crashed: %s", symbol, tf, exc)
            RUN_STATS[run_id].update(
                tf_index=tf_index, tf_total=len(tfs), current_tf=tf,
                current_symbol=symbol)
            _push_progress(run_id, done, total, f"{symbol}:{tf}:error",
                           last_error=str(exc), failed_at=f"{symbol}/{tf}",
                           tf_index=tf_index, tf_total=len(tfs),
                           current_tf=tf, current_symbol=symbol)
            continue

    # Финальное событие: done — фактическое число обработанных комбинаций
    # (при таймаутах/ошибках/отмене может быть меньше total).
    cancelled = is_cancelled(run_id)
    if cancelled:
        log.info("scan: %s cancelled (done=%d/%d)", run_id, done, total)
    _push_progress(run_id, done, total, None,  # финальное событие
                   cancelled=cancelled,
                   tf_index=len(tfs), tf_total=len(tfs),
                   current_tf=tfs[-1] if tfs else None, current_symbol=None)
    # update, а не новый dict: сохраняем done/eta_seconds, накопленные в
    # прогрессе (их читает поллинг-фолбэк UI, GET /api/scan/<run_id> -> stats).
    stats = RUN_STATS.get(run_id) or {}
    stats.update({"total": total, "kept": kept, "filtered": total - kept,
                  "finished": True, "cancelled": cancelled})
    RUN_STATS[run_id] = stats
    _prune_run_stats()
    _CANCELLED.discard(run_id)  # прогон завершён — флаг отмены больше не нужен
    return run_id
