"""
Charon-Only Validation Script
Agent: 1/6
Dependencies: app_pkg.ai.ai_backtest.run_verdict_backtest (READ ONLY)

ОТВЕТ НА ОДИН ВОПРОС:
    «Есть ли у Charon-only стратегии (без CBR) положительный Sharpe
    на d днях исторических данных?»

Метод:
    - сигналы L/S/F на каждый бар окна берутся ИЗ движка
        app_pkg.ai.ai_backtest.run_verdict_backtest (READ ONLY, движок
        НЕ модифицируется; CBR-гейты выключены): каждый шаг движка
        строит СВОЙ compact_snapshot(symbol, tf, ts) с барьером ts
        (get_replay_df to_sec=ts) — «point-in-time», без будущего;
    - сделка = бар с сигналом != F, вход по close сигнального бара,
        выход по касанию SL/TP (SL-приоритет), горизонт 20 баров,
        таймаут -> close; комиссия 0.10 п.п. (конвенция A/B/C-движка
        scripts/cbr_backtest_abc.py);
    - Sharpe годовой = mean/std*sqrt(252) (конвенция A/B/C-движка);
    - walk-forward: n_splits равных подпериодов окна, метрики по
        point-in-time сигналам каждого периода;
    - shuffle test: 100 случайных перестановок сигналов по барам
        (seed=42), p-value = доля перестановок с Sharpe >= наблюдаемого;
    - look-ahead audit: каждый сигнал обязан иметь барьер == собственному
        таймстемпу бара (движок даёт барьер шага в steps[]).

Вердикт (таблица ТЗ):
    Sharpe >= 1.5  И  p < 0.05  -> SUCCESS
    0.5 <= Sharpe < 1.5         -> MARGINAL
    Sharpe < 0.5  ИЛИ p >= 0.05 -> FAILED

Запуск:
    python scripts/validate_charon_only.py
    python scripts/validate_charon_only.py --symbol BTCUSDT --timeframe 1h --days 700
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['CBR_STAGE2_ENABLED'] = 'False'
# Charon-only = без CBR вообще: не только без смешивания (STAGE2), но и без
# записи снимков прогона в CBR-базу (CBR_ENABLED). Выставляем ДО импорта
# app_pkg (load_dotenv не перезапишет уже заданные process-переменные).
os.environ['CBR_ENABLED'] = '0'
from app_pkg.ai.ai_backtest import run_verdict_backtest
from app_pkg.data.fetch import get_series_df
import pandas as pd
import numpy as np
import json
from datetime import datetime
from datetime import timezone

import argparse
import math
from pathlib import Path

import logging
# Движок пишет предупреждение на КАЖДЫЙ бар при деградации внешних блоков
# снимка (news_sentiment/calendar отключены нами выше). Подавляем копию на
# консоль, сохраняя работу движка; ошибки уровня ERROR всё равно видны.
logging.getLogger().setLevel(logging.ERROR)

from app_pkg import config
from app_pkg.indicators import _atr

# ----------------------------------------------------------------------------
# Charon-only: внешние СЕТЕВЫЕ блоки снимка (news_sentiment, econ_calendar) в
# этом окружении гарантированно деградируют до null-блоков (finnhub 403/
# read-timeout). Движок каждый бар ждёт ~15s сетевое исключение, чтобы
# получить пустой блок. Чтобы не жечь часы на заведомо мёртвые запросы,
# перехватываем их мгновенным исключением ДО сети: run_verdict_backtest
# проходит ровно тот же путь «ошибка -> _blank_for(name)», что и сейчас, но
# без ожидания. Файлы движка НЕ модифицируются; снимок остаётся чисто
# ценовым (это и есть «Charon-only»). macro — пустой сразу (нет FRED_API_KEY);
# derivatives/micro — мгновенный {} для не-крипты.
# ----------------------------------------------------------------------------
from app_pkg.data import market_snapshot as _market_snapshot


class _FastSnapBlank(Exception):
    """Внутренний маркер отключённого внешнего блока снимка."""


def _snap_blank(*_args, **_kwargs):
    raise _FastSnapBlank("external snapshot block disabled (charon-only)")


for _bname in ("get_news_sentiment", "get_econ_calendar"):
    if hasattr(_market_snapshot, _bname):
        setattr(_market_snapshot, _bname, _snap_blank)

# ----------------------------------------------------------------------------
# Параметры валидации (консервативные; конвенция A/B/C-движка — cbr_backtest_abc)
# ----------------------------------------------------------------------------
TRADE_HORIZON = 20          # баров, как в backfill CBR (labels.py)
TP_PCT = 0.20               # тейк-профит 20%
SL_FLOOR = 0.10             # нижняя граница стопа
SL_ATR_K = 2.0              # множитель ATR для стопа
FEE_PP = 0.10               # 0.05%/сторона = 0.10 п.п. всего
PPY = 252.0                 # годовая нормализация Sharpe (конвенция движка)
SEED = 42                   # воспроизводимость shuffle-теста

# Стандарт валидации (перекрывается argparse в main()).
N_SPLITS = 3                # периодов walk-forward
N_ITERATIONS = 100          # итераций shuffle-теста
ADAPTIVE = True             # адаптивный шаг движка run_verdict_backtest

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

# ----------------------------------------------------------------------------
# Модульное состояние окна валидации. validate_charon() наполняет его ДО
# вызовов walk_forward_analysis / shuffle_test / look_ahead_audit — так
# сохраняются ТОЧНЫЕ сигнатуры функций из ТЗ (в функции не прокидывается df).
# ----------------------------------------------------------------------------
_DF = None          # DataFrame окна (см. _load_window)
_TS = []            # unix-секунды баров окна (соответствие строкам _DF)
_O = np.array([])   # open
_H = np.array([])   # high
_L = np.array([])   # low
_C = np.array([])   # close
_A = np.array([])   # ATR(14), None -> nan
_SIGNAL_MAP = {}    # ts -> sig по point-in-time прогону движка
_STEP_TS = []       # барьеры шагов движка (для look-ahead аудита)
_LK_CHECKS = {}     # детали look-ahead аудита


def _canonical_tf(tf):
    """1h -> 1H (канон config.TF_SECONDS), как в cbr_lookahead_audit."""
    tf = str(tf or "").strip()
    for canon in config.TF_SECONDS:
        if canon.lower() == tf.lower():
            return canon
    return tf


def _bts(value):
    """Timestamp/ар -> unix-секунды int (tz-aware)."""
    ts = pd.Timestamp(value)
    if getattr(ts, "tz", None) is None:
        ts = ts.tz_localize("UTC")
    return int(ts.timestamp())


def _load_window(symbol, tf, days):
    """Полный кеш + окно последних `days` суток. Наполняет модульные глобалы.

    РАСЧЁТ на движок: run_verdict_backtest внутри вызывает
    get_series_df(limit=bars, history_limit=TRENDS_HISTORY=200) и берёт
    tail(bars) из КЕША. Чтобы окно целиком попало в tail(bars), кеш должен
    быть наполнен глубже окна — тянем до MAX_DATA_LIMIT ДО прогона движка,
    затем режем окно по времени.
    """
    global _DF, _TS, _O, _H, _L, _C, _A
    days = max(1, int(days))
    # Запас коэффициента ~1.6 на не-торговые окна (выходные форекс и т.п.)
    # + 1000 баров на прогрев индикаторов, потолок 20000.
    need = min(int(config.MAX_DATA_LIMIT), int(days * 24 * 1.6) + 1000)
    all_df = get_series_df(symbol, tf, limit=need,
                           history_limit=config.MAX_DATA_LIMIT)
    if all_df is None or all_df.empty:
        raise RuntimeError(f"Нет данных для {symbol} {tf}")
    last_ts = _bts(all_df["timestamp"].iloc[-1])
    start_bound = pd.to_datetime(last_ts - days * 86400, unit="s", utc=True)
    df = (all_df[all_df["timestamp"] >= start_bound]
          .reset_index(drop=True))
    if all_df["timestamp"].iloc[0] < start_bound and len(df) < 30:
        raise RuntimeError(
            f"История источника короче окна: {len(all_df)} баров всего, "
            f"{len(df)} баров за {days} суток")
    min_bars = max(200, 3 * TRADE_HORIZON)
    if len(df) < min_bars:
        raise RuntimeError(
            f"Мало баров в окне ({len(df)} < {min_bars}) — увеличьте days "
            f"или возьмите другой таймфрейм")
    df = df.reset_index(drop=True)
    _DF = df
    _TS = [_bts(t) for t in df["timestamp"]]
    _O = df["open"].to_numpy(dtype="float64")
    _H = df["high"].to_numpy(dtype="float64")
    _L = df["low"].to_numpy(dtype="float64")
    _C = df["close"].to_numpy(dtype="float64")
    atr = _atr(df["high"], df["low"], df["close"], 14).to_numpy(dtype="float64")
    atr = np.where(np.isfinite(atr), atr, np.nan)
    _A = atr
    return df, last_ts


def _simulate_trade(ts_q, direction):
    """Сделка по бару ts_q (бар-индекс берётся из _TS).

    Mirror scripts/cbr_backtest_abc.simulate_trade: вход по close сигнального
    бара, SL = max(SL_FLOOR, SL_ATR_K*atr_pct), TP = TP_PCT, SL-приоритет
    касаний, горизонт TRADE_HORIZON, таймаут -> close последнего бара.
    PnL в % от входа за вычетом FEE_PP. None при F/нехватке баров.
    """
    if direction == "F":
        return None
    j0 = _TS.index(int(ts_q)) if int(ts_q) in _TS else None
    if j0 is None:
        return None
    entry = float(_C[j0])
    if not np.isfinite(entry) or entry <= 0:
        return None
    atr_pct = float(_A[j0]) / entry if np.isfinite(_A[j0]) else 0.0
    sl_pct = max(SL_FLOOR, SL_ATR_K * max(atr_pct, 0.0))
    lower = entry * (1.0 - sl_pct)
    upper = entry * (1.0 + TP_PCT)
    j_end = min(j0 + TRADE_HORIZON, len(_C))
    if j_end <= j0 + 1:
        return None
    exit_price = None
    exit_j = None
    for j in range(j0 + 1, j_end):
        if _L[j] <= lower:
            exit_price, exit_j = lower, j
            break
        if _H[j] >= upper:
            exit_price, exit_j = upper, j
            break
    timeout = exit_price is None
    if timeout:
        exit_price = float(_C[j_end - 1])
        exit_j = j_end - 1
    sign = 1.0 if direction == "L" else -1.0
    gross = sign * (exit_price - entry) / entry * 100.0 - FEE_PP
    return {"ts": int(ts_q), "sig": direction,
            "pnl_pct": round(float(gross), 4),
            "entry_price": round(entry, 8), "exit_price": round(exit_price, 8),
            "bars_held": int(exit_j - j0), "timeout": bool(timeout)}


def _trades_from_signals(signals):
    """Список сделок по сигналам баров окна (None -> F, не торгуем).

    Сигнал i соответствует бару _TS[i]; лишние сигналы (длиннее окна)
    игнорируются, недостающие — F.
    """
    if _DF is None:
        return []
    trades = []
    n = min(len(signals), len(_TS))
    for i in range(n):
        sig = signals[i]
        sig = "F" if sig is None else str(sig).strip().upper()
        if sig not in ("L", "S"):
            sig = "F"
        t = _simulate_trade(_TS[i], sig)
        if t is not None:
            trades.append(t)
    return trades


# ------------------------------------------------------------------ STEP 1-3
# Требуемые функции ТЗ (сигнатуры точные).

def calculate_sharpe(returns, risk_free_rate=0.0) -> float:
    """Годовой Sharpe: mean/std*sqrt(252) (конвенция A/B/C-движка).

    len < 2 или std ~ 0 -> 0.0; risk_free_rate вычитается из mean.
    """
    arr = np.asarray([float(r) for r in returns if r is not None],
                     dtype="float64")
    if arr.size < 2:
        return 0.0
    sd = float(arr.std())
    if sd < 1e-12:
        return 0.0
    mean = float(arr.mean())
    if risk_free_rate:
        mean -= float(risk_free_rate)
    return mean / sd * math.sqrt(PPY)


def calculate_max_drawdown(equity_curve) -> float:
    """Максимальный peak-to-trough провал equity (положительная величина)."""
    eq = np.asarray([float(e) for e in equity_curve if e is not None],
                    dtype="float64")
    if eq.size == 0:
        return 0.0
    peak = np.maximum.accumulate(eq)
    return float(-(eq - peak).min())


def calculate_profit_factor(trades) -> float:
    """gross_profit / gross_loss. Нет проигрышей -> inf (при прибыли)."""
    def _pnl(t):
        if isinstance(t, dict):
            v = t.get("pnl_pct")
            if v is None:
                v = t.get("pnl")
            return v
        return t

    pnls = []
    for t in trades:
        v = _pnl(t)
        if v is None:
            continue
        try:
            pnls.append(float(v))
        except (TypeError, ValueError):
            continue
    if not pnls:
        return 0.0
    gp = sum(p for p in pnls if p > 0)
    gl = abs(sum(p for p in pnls if p < 0))
    if gl > 1e-12:
        return float(gp / gl)
    return float("inf") if gp > 0 else 0.0


def _metric_block(trades):
    """Метрики по списку сделок: sharpe/winrate/pf/mdd/sum/avg."""
    if _DF is None:
        return {"n_trades": 0, "sharpe": None, "win_rate": None,
                "profit_factor": None, "max_drawdown": None,
                "sum_pnl_pct": 0.0, "avg_pnl_pct": None}
    pnls = np.asarray([t["pnl_pct"] for t in trades], dtype="float64")
    if pnls.size == 0:
        return {"n_trades": 0, "sharpe": None, "win_rate": None,
                "profit_factor": None, "max_drawdown": None,
                "sum_pnl_pct": 0.0, "avg_pnl_pct": None}
    eq = np.cumsum(pnls)
    pf = calculate_profit_factor(trades)
    return {
        "n_trades": int(pnls.size),
        "sharpe": round(calculate_sharpe(pnls.tolist()), 4),
        "win_rate": round(float(np.mean(pnls > 0)), 4),
        "profit_factor": (round(float(pf), 4) if np.isfinite(pf) else None),
        "max_drawdown": round(calculate_max_drawdown(eq.tolist()), 4),
        "sum_pnl_pct": round(float(pnls.sum()), 4),
        "avg_pnl_pct": round(float(pnls.mean()), 4),
    }


def walk_forward_analysis(symbol, timeframe, n_splits=3) -> dict:
    """WALK-FORWARD: n раВНЫХ подпериодов окна, метрики по каждому.

    Сигналы ДВИЖКА point-in-time (сигнал бара t построен по данным <= t),
    поэтому нарезка периодов = честная проверка стабильности edge вне
    выборки без повторного прогона тяжелого движка (прогон детерминирован:
    перезапуск движка на подпериоде дал бы те же сигналы).
    """
    symbol = str(symbol or "EURUSD").upper()
    tf = _canonical_tf(timeframe)
    n_splits = max(1, int(n_splits))
    if _DF is None or not _TS:
        return {"n_splits": n_splits, "periods": [], "window_bars": 0,
                "note": "окно не загружено (сначала validate_charon)"}
    n = len(_TS)
    bounds = [int(i * n / n_splits) for i in range(n_splits + 1)]
    periods = []
    for k in range(n_splits):
        i0, i1 = bounds[k], bounds[k + 1]
        period_ts = _TS[i0:i1]
        sigs = [_SIGNAL_MAP.get(t, "F") for t in period_ts]
        trades = _trades_from_signals(sigs)
        mb = _metric_block(trades)
        periods.append({
            "period": k + 1,
            "start_ts": int(period_ts[0]),
            "end_ts": int(period_ts[-1]),
            "n_bars": int(len(period_ts)),
            "n_trades": mb["n_trades"],
            "sharpe": mb["sharpe"],
            "win_rate": mb["win_rate"],
            "profit_factor": mb["profit_factor"],
            "max_drawdown": mb["max_drawdown"],
            "sum_pnl_pct": mb["sum_pnl_pct"],
        })
    return {"n_splits": n_splits, "window_bars": int(n), "periods": periods}


def shuffle_test(signals, n_iterations=100) -> dict:
    """PERMUTATION TEST на сигналах (seed=42): p-value односторонний.

    Нулевая гипотеза «сигнала нет»: случайная перестановка сигналов по
    барам окна даёт на этом же рыночном ряде Sharpe >= наблюдаемого.
    p_value = (count_ge + 1) / (n_iterations + 1) — сглаживание 0.
    """
    n_iterations = max(1, int(n_iterations))
    if _DF is None or not _TS:
        return {"p_value": 1.0, "note": "окно не загружено (validate_charon)",
                "n_iterations": n_iterations}
    sigs = []
    for s in signals[:len(_TS)]:
        s = "F" if s is None else str(s).strip().upper()
        sigs.append(s if s in ("L", "S") else "F")
    while len(sigs) < len(_TS):
        sigs.append("F")
    obs_trades = _trades_from_signals(sigs)
    obs_sharpe = _metric_block(obs_trades)["sharpe"] or 0.0
    sigs_arr = np.asarray(sigs, dtype=object)
    rng = np.random.RandomState(int(SEED))
    perm_sharpes = []
    count_ge = 0
    for _ in range(n_iterations):
        perm = rng.permutation(len(sigs_arr))
        sh = _metric_block(_trades_from_signals(list(sigs_arr[perm])))["sharpe"] \
            or 0.0
        perm_sharpes.append(float(sh))
        if sh >= obs_sharpe:
            count_ge += 1
    p_value = (count_ge + 1) / (n_iterations + 1)
    return {
        "p_value": round(float(p_value), 4),
        "observed_sharpe": round(float(obs_sharpe), 4),
        "n_iterations": int(n_iterations),
        "count_ge": int(count_ge),
        "mean_shuffled_sharpe": round(float(np.mean(perm_sharpes)), 4),
        "std_shuffled_sharpe": round(float(np.std(perm_sharpes)), 4),
        "seed": int(SEED),
    }


def look_ahead_audit(symbol, timeframe) -> bool:
    """LOOK-AHEAD AUDIT: True, если утечек будущего нет.

    Проверки по point-in-time шагам движка (_STEP_TS — барьеры шагов):
      1) каждый бар окна покрыт сигналом;
      2) барьер шага == собственному таймстемпу бара (сигнал бара t НЕ
         использует данные после t);
      3) барьеры строго монотонны;
      4) ни один барьер не позже последнего бара окна.
    """
    if _DF is None or not _TS or not _STEP_TS:
        _LK_CHECKS["error"] = "нет данных прогона движка"
        return False
    steps = [int(t) for t in _STEP_TS]
    window = [int(t) for t in _TS]
    checks = {
        "n_steps": int(len(steps)),
        "n_window_bars": int(len(window)),
        "monotonic": bool(all(a < b for a, b in zip(steps, steps[1:]))),
        "barrier_is_own_bar": bool(len(steps) == len(window)
                                   and all(a == b
                                            for a, b in zip(steps, window))),
        "every_bar_covered": bool(set(steps) == set(window)),
        "no_barrier_after_window": bool(
            (not steps) or (int(steps[-1]) <= int(window[-1]))),
        "max_barrier": int(steps[-1]) if steps else None,
        "window_last_ts": int(window[-1]) if window else None,
    }
    clean = bool(checks["monotonic"] and checks["barrier_is_own_bar"]
                 and checks["every_bar_covered"]
                 and checks["no_barrier_after_window"])
    checks["clean"] = clean
    _LK_CHECKS.update(checks)
    return clean


def _decide(sharpe, p_value):
    """Таблица ТЗ: SUCCESS / MARGINAL / FAILED."""
    sharpe = float(sharpe or 0.0)
    if sharpe < 0.5 or p_value is None or p_value >= 0.05:
        return "FAILED"
    if sharpe >= 1.5:
        return "SUCCESS"
    return "MARGINAL"


def _json_default(o):
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return str(o)


def _big_verdict(verdict):
    line = "#" * 40
    text = verdict
    pad = max(0, (40 - len(text) - 2) // 2)
    inner = " " * pad + text + " " * (40 - len(text) - pad - 2)
    return f"{line}\n#{inner}#\n{line}"


def validate_charon(symbol='EURUSD', timeframe='1h', days=700):
    """Полная валидация Charon-only: ответ на вопрос ТЗ.

    Возвращает dict с ключами (обязательными по ТЗ):
        sharpe, win_rate, profit_factor, max_drawdown, walk_forward,
        shuffle_p_value, look_ahead_clean, verdict
    + служебные (params, shuff_test детали, look_ahead, engine, metrics).
    """
    symbol = str(symbol or "EURUSD").upper()
    tf = _canonical_tf(timeframe)
    days = max(1, int(days))

    print("=== Charon-only validation (без CBR) ===")
    print(f"{symbol} {tf}: days={days} "
          f"splits={N_SPLITS} shuffle_iter={N_ITERATIONS} "
          f"adaptive={ADAPTIVE} seed={SEED}")
    print(f"Загрузка окна ({days} суток) ...")
    df, last_ts = _load_window(symbol, tf, days)
    window_bars = len(df)
    print(f"Окно: {window_bars} баров, "
          f"{datetime.fromtimestamp(_TS[0], tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
          f" .. {datetime.fromtimestamp(_TS[-1], tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    print(f"Прогон движка run_verdict_backtest ({window_bars} баров, "
          f"CBR off) ...")
    res = run_verdict_backtest(symbol, tf, bars=window_bars,
                               adaptive=bool(ADAPTIVE),
                               upto_sec=int(_TS[-1]))
    if res.get("status") != "finished":
        raise RuntimeError(f"Движок вернул status={res.get('status')} "
                           f"({res.get('error')})")

    # Сигналы по времени шагов (point-in-time, барьер шага = его бар).
    global _STEP_TS, _SIGNAL_MAP, _LK_CHECKS
    signals = []
    _STEP_TS = []
    _SIGNAL_MAP = {}
    _LK_CHECKS = {}
    for st in res.get("steps") or []:
        t = int(st["upto_sec"])
        _STEP_TS.append(t)
        _SIGNAL_MAP[t] = st.get("sig") if st.get("sig") in ("L", "S", "F") \
            else "F"
    signal_resolved = len(_SIGNAL_MAP)
    for t in _TS:
        signals.append(_SIGNAL_MAP.get(t, "F"))

    # Контекст: часть баров окна могла остаться без сигнала (пустой ответ
    # движка на старых барах с холодным источником) — не торгуем их.
    n_uncovered = len(_TS) - signal_resolved
    if n_uncovered > 0:
        print(f"ВНИМАНИЕ: {n_uncovered} баров без сигнала движка (F/нет "
              f"данных) — не торгуются")

    trades = _trades_from_signals(signals)
    mb = _metric_block(trades)

    wf = walk_forward_analysis(symbol, tf, n_splits=int(N_SPLITS))
    shuf = shuffle_test(signals, n_iterations=int(N_ITERATIONS))
    la_clean = look_ahead_audit(symbol, tf)

    sharpe = mb["sharpe"] or 0.0
    p_value = shuf["p_value"]
    verdict = _decide(sharpe, p_value)

    trade_count = sum(1 for s in signals if s in ("L", "S"))
    coverage = (trade_count / len(signals)) if signals else 0.0

    result = {
        "sharpe": sharpe,
        "win_rate": mb["win_rate"],
        "profit_factor": mb["profit_factor"],
        "max_drawdown": mb["max_drawdown"],
        "walk_forward": wf,
        "shuffle_p_value": p_value,
        "look_ahead_clean": la_clean,
        "verdict": verdict,
        # --- служебное ---
        "params": {"symbol": symbol, "timeframe": tf, "days": days,
                   "window_bars": int(window_bars),
                   "first_bar_ts": int(_TS[0]), "last_bar_ts": int(_TS[-1]),
                   "n_splits": int(N_SPLITS),
                   "n_iterations": int(N_ITERATIONS),
                   "seed": int(SEED), "adaptive": bool(ADAPTIVE),
                   "trade_horizon": TRADE_HORIZON, "tp_pct": TP_PCT,
                   "sl_atr_k": SL_ATR_K, "sl_floor": SL_FLOOR,
                   "fee_pp": FEE_PP, "sharpe_ppy": PPY,
                   "cbr_stage2": False, "cbr_enabled": False,
                   "cbr_store_hook": False},
        "metrics": {
            "n_bars": int(len(signals)), "n_trades": mb["n_trades"],
            "n_signal_bars": int(trade_count), "coverage": round(coverage, 4),
            "sum_pnl_pct": mb["sum_pnl_pct"], "avg_pnl_pct": mb["avg_pnl_pct"],
            "max_drawdown": mb["max_drawdown"],
        },
        "shuffle_test": shuf,
        "look_ahead": dict(_LK_CHECKS),
        "engine": {
            "llm_calls": int((res.get("metrics") or {}).get("llm_calls", 0) or 0),
            "cache_hits": int((res.get("metrics") or {}).get("cache_hits", 0) or 0),
            "tokens_total": int((res.get("metrics") or {}).get("tokens_total", 0) or 0),
            "bars": int(res.get("bars", 0) or 0),
            "run_id": str(res.get("run_id", "")),
        },
        "decision": {
            "sharpe": sharpe, "p_value": p_value,
            "verdict": verdict,
            "rule": ("SUCCESS: Sharpe>=1.5 И p<0.05; "
                     "MARGINAL: 0.5<=Sharpe<1.5 (и p<0.05); "
                     "FAILED: Sharpe<0.5 ИЛИ p>=0.05"),
        },
    }

    # ---------- артефакт: reports/charon_validation_*.json ----------
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = REPORTS_DIR / f"charon_validation_{stamp}.json"
    report = {"generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
              "script": "scripts/validate_charon_only.py", "agent": "1/6",
              **result}
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                   default=_json_default), encoding="utf-8")

    # ---------- консоль: таблица ----------
    print()
    print("=" * 62)
    print(f" CHARON-ONLY VALIDATION — {symbol} {tf} — Agent 1/6")
    print("=" * 62)

    def _pfmt(v):
        return "-" if v is None else f"{v:.3f}"

    def _pfmt2(v):
        return "-" if v is None else f"{v:.4f}"

    def _wfmt(v):
        return "-" if v is None else f"{100.0 * v:.2f}%"

    wf_str = " | ".join(
        f"P{p['period']} {_pfmt(p['sharpe'])}" for p in wf["periods"])
    rows = [
        ("Sharpe (год., sqrt 252)", _pfmt2(sharpe)),
        ("Win rate", _wfmt(mb["win_rate"])),
        ("Profit factor", _pfmt2(mb["profit_factor"])),
        ("Max drawdown", "-" if mb["max_drawdown"] is None
         else f"{mb['max_drawdown']:.4f}"),
        ("Trades / signal bars", f"{mb['n_trades']} / "
         f"{result['metrics']['n_signal_bars']}"),
        ("Coverage (сигнальных баров)", f"{coverage:.2%}"),
        (f"Walk-forward ({wf['n_splits']} периодов)", wf_str),
        (f"Shuffle p-value ({N_ITERATIONS} итер.)", f"{p_value:.4f}"),
        ("Mean shuffled Sharpe", _pfmt2(shuf["mean_shuffled_sharpe"])),
        ("Look-ahead audit", "CLEAN" if la_clean else "FAILED"),
        ("LLM calls / cache hits",
         f"{result['engine']['llm_calls']} / {result['engine']['cache_hits']}"),
    ]
    for name, val in rows:
        print(f"  {name:<34s} {val}")
    print("-" * 62)
    print(f" ВЕРДИКТ: {verdict}")
    print("=" * 62)
    print()
    print(_big_verdict(verdict))
    print()
    print(f"JSON: {out_path}")
    return result


def main():
    global N_SPLITS, N_ITERATIONS, ADAPTIVE, SEED
    parser = argparse.ArgumentParser(
        description="Charon-only (без CBR) валидация: Sharpe на истории.")
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--timeframe", "--tf", dest="timeframe", default="1h")
    parser.add_argument("--days", type=int, default=700)
    parser.add_argument("--splits", type=int, default=N_SPLITS)
    parser.add_argument("--shuffle-iterations", type=int, dest="shuffle_iter",
                        default=N_ITERATIONS)
    parser.add_argument("--no-adaptive", action="store_true", dest="no_adaptive")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    N_SPLITS = max(1, args.splits)
    N_ITERATIONS = max(1, args.shuffle_iter)
    ADAPTIVE = not args.no_adaptive
    SEED = args.seed

    result = validate_charon(symbol=args.symbol, timeframe=args.timeframe,
                             days=args.days)
    return 0 if result["verdict"] in ("SUCCESS", "MARGINAL", "FAILED") else 1


if __name__ == "__main__":
    sys.exit(main())