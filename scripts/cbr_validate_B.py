# -*- coding: utf-8 -*-
"""Валидация CBR-only (B): пять проверок достоверности claimed Sharpe.

После A/B/C (Этап 2) blended (C) стабильно обгоняет Charon (A) по winrate,
но pure CBR (B) показывает аномально высокий Sharpe (~6-8) — для retail
реалистичен диапазон 1.5-3.0. Этот скрипт ПРОВЕРЯЕТ, реален ли этот
risk-adjusted edge: K/min_distance фиксируются (по умолчанию K=10, md=1.0),
blender не трогается, БД не пересобирается — только диагностика B.

Проверки:
  1. Sample size и базовые метрики (n_trades >= 100/200)
  2. Комиссии + slippage: gross -> net Sharpe (FEE 0.05% + SLIP 0.02%/сторону)
  3. Walk-forward: весь диапазон на 3 РАВНЫХ подпериода
  4. Out-of-sample 70/30 по времени с ЗАФИКСИРОВАННЫМИ K/md
  5. Permutation-test: перемешивание outcome в копии БД (seed=42),
     ratio real/suffled Sharpe (по gross — постоянные издержки не влияют)

Вердикт по таблице ТЗ: GO / WARN / NO-GO с причиной.
Артефакты: reports/cbr_validate_B.json + reports/cbr_validate_B.png.

Запуск: python scripts/cbr_validate_B.py --csv scripts/charon_data_cache/BTCUSDT_1h_700d.csv --K 10 --md 1.0
"""

import argparse
import datetime
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

_BASE_DIR = Path(__file__).resolve().parent.parent
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

from app_pkg import config  # noqa: E402
from app_pkg.cbr import index as cbr_index  # noqa: E402
from app_pkg.cbr import normalize as cbr_norm  # noqa: E402
from app_pkg.cbr import query_api, schema, store  # noqa: E402
from scripts.cbr_backtest_abc import (  # noqa: E402
    DEFAULT_CSV, TRADE_HORIZON, TP_PCT, SL_FLOOR, SL_ATR_K,
    direction, load_csv, load_db, snap_from_features,
)

REPORTS_DIR = _BASE_DIR / "reports"

# Параметры валидации (консервативные, Binance taker + slippage).
FEE_PER_SIDE = 0.0005      # 0.05% комиссия на сторону
SLIPPAGE = 0.0002          # 0.02% проскальзывание на сторону
SHARPE_PPY = 252.0         # та же годовая нормализация, что в A/B/C-движке

# Пороги вердикта (таблица ТЗ).
N_MIN_OK = 200             # >= 200 сделок — норма
N_MIN_WARN = 100           # 100-200 — предупреждение
NET_SHARPE_OK = 2.5        # >= 2.5 после комиссий — edge реален
NET_SHARPE_LOW = 1.5       # < 1.5 — комиссии съедают edge
SHUFFLE_RATIO_OK = 2.0     # real/suffled > 2 — сигнал реальный
SHUFFLE_RATIO_WARN = 1.2   # 1.2-2.0 — слабый сигнал
OOS_RATIO_OK = 0.7         # test >= 0.7*train — стабильно
OOS_RATIO_WARN = 0.4       # 0.4-0.7 — частичный overfit
WF_SHARPE_OK = 1.5         # все подпериоды > 1.5 — стабильно
CONCENTRATION_LIMIT = 0.9  # > 90% прибыли в одном периоде — режим-зависимо


# ---------------------------------------------------------------- helpers


def annualized_sharpe(pnls, ppy=SHARPE_PPY):
    """Годовой Sharpe из списка PnL (mean/std * sqrt(ppy)).

    ppy=252 — та же конвенция, что в метриках A/B/C-движка (сопоставимо с
    claimed 6.6). len<2 или std==0 -> 0.0.
    """
    arr = np.asarray([float(p) for p in pnls if p is not None], dtype="float64")
    if arr.size < 2:
        return 0.0
    sd = float(arr.std())
    if sd < 1e-12:
        return 0.0
    return float(arr.mean()) / sd * math.sqrt(ppy)


def net_pnl_pct(gross_pct, entry_price, exit_price,
                fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE):
    """Net pnl в % от входа: gross минус комиссия+слиппадж на обе стороны.

    Формула ТЗ: pnl_net = pnl_gross - entry*(FEE+SLIP) - exit*(FEE+SLIP),
    выраженная в % от цены входа.
    """
    entry = float(entry_price)
    if entry <= 0:
        return float(gross_pct)
    c = (float(fee_per_side) + float(slippage)) * (entry + float(exit_price)) \
        / entry * 100.0
    return float(gross_pct) - c


def simulate_trade_info(ts_arr, o_arr, h_arr, l_arr, c_arr, pos_map, ts_q,
                        direction_sign, atr_pct):
    """Барьеры сделки (та же конвенция, что в backtest_abc) + gross PnL.

    Возвращает dict c полями entry_price/exit_price/gross_pct/bars_held/timeout
    или None, если сделки нет (F/нет баров/нет данных). SL-приоритет касаний,
    горизонт TRADE_HORIZON, таймаут -> close последнего бара.
    """
    if direction_sign == "F":
        return None
    j0 = pos_map.get(int(ts_q))
    if j0 is None:
        return None
    entry = float(c_arr[j0])
    if entry <= 0:
        return None
    sl_pct = max(SL_FLOOR, SL_ATR_K * (atr_pct if atr_pct is not None
                                       else SL_FLOOR))
    lower = entry * (1.0 - sl_pct)
    upper = entry * (1.0 + TP_PCT)
    j_end = min(j0 + TRADE_HORIZON, len(c_arr))
    if j_end <= j0 + 1:
        return None
    exit_price = None
    exit_j = None
    for j in range(j0 + 1, j_end):
        if l_arr[j] <= lower:
            exit_price, exit_j = lower, j
            break
        if h_arr[j] >= upper:
            exit_price, exit_j = upper, j
            break
    timeout = exit_price is None
    if timeout:
        exit_price = float(c_arr[j_end - 1])
        exit_j = j_end - 1
    sign = 1.0 if direction_sign == "L" else -1.0
    gross = sign * (exit_price - entry) / entry * 100.0
    return {"entry_price": float(entry), "exit_price": float(exit_price),
            "gross_pct": float(gross), "bars_held": int(exit_j - j0),
            "timeout": bool(timeout)}


def run_b_only(conn, db_rows, index, stats, csv_data, symbol, timeframe,
               K, min_distance, regime_match, window_days,
               start_fraction=0.0, max_bars=0):
    """CBR-only (B) на временном окне: список сделок с gross/net PnL.

    db_rows = {ts: (vec, regime)} из load_db(conn). Сделка = бар, где
    direction(winrate_up, winrate_down) != F. Возвращает список dict:
    ts/regime/d/entry_price/exit_price/gross_pct/net_pct/bars_held/timeout.
    """
    ts_arr, o_arr, h_arr, l_arr, c_arr, pos_map = csv_data
    all_ts = sorted(db_rows)
    n = len(all_ts)
    if n == 0:
        return []
    start_pos = int(n * start_fraction)
    query_ts = all_ts[start_pos:]
    if max_bars > 0:
        query_ts = query_ts[-int(max_bars):]
    trades = []
    for ts in query_ts:
        vec, regime = db_rows[ts]
        close = c_arr[pos_map[ts]] if pos_map.get(ts) is not None else 0.0
        atr_pct = (float(vec[store.ATR_INDEX]) / float(close)
                   if close > 0 else 0.0)
        snap = snap_from_features(vec, symbol, timeframe, ts)
        cbr = query_api.get_similar_summary(
            conn, index, snap, symbol, timeframe, bar_ts=int(ts),
            K=K, min_distance=min_distance,
            regime_match=regime_match, window_days=window_days, stats=stats)
        insuff = not isinstance(cbr, dict) or cbr.get("insufficient_data")
        up = 0.0 if insuff else float(cbr.get("winrate_up", 0.0))
        dn = 0.0 if insuff else float(cbr.get("winrate_down", 0.0))
        d = direction(up, dn)
        info = simulate_trade_info(ts_arr, o_arr, h_arr, l_arr, c_arr, pos_map,
                                   ts, d, atr_pct)
        if info is None:
            continue
        info["ts"] = int(ts)
        info["regime"] = regime
        info["d"] = d
        info["net_pct"] = net_pnl_pct(info["gross_pct"],
                                      info["entry_price"], info["exit_price"])
        trades.append(info)
    return trades


def trade_stats(trades, key="net_pct"):
    """Метрики по списку сделок: n/winrate/avg_pnl/std/sharpe/dd/pf/bars."""
    out = {"n": int(len(trades))}
    if len(trades) == 0:
        out.update({"winrate": None, "avg_pnl_pct": None, "std_pnl_pct": None,
                    "sharpe": None, "max_dd_pct": None, "profit_factor": None,
                    "sum_pnl_pct": 0.0, "avg_bars_held": None})
        return out
    pnl = np.asarray([float(t[key]) for t in trades], dtype="float64")
    out["winrate"] = round(float(np.mean(pnl > 0.0)), 4)
    out["avg_pnl_pct"] = round(float(pnl.mean()), 4)
    out["std_pnl_pct"] = round(float(pnl.std()), 4)
    out["sharpe"] = round(annualized_sharpe(pnl), 4)
    out["sum_pnl_pct"] = round(float(pnl.sum()), 4)
    eq = np.cumsum(pnl)
    out["max_dd_pct"] = round(float((eq - np.maximum.accumulate(eq)).min()), 4)
    wins = pnl[pnl > 0].sum()
    losses = abs(pnl[pnl < 0].sum())
    pf = wins / losses if losses > 1e-12 else (float("inf") if wins > 0 else 0.0)
    out["profit_factor"] = round(pf, 4) if np.isfinite(pf) else None
    held = [float(t["bars_held"]) for t in trades]
    out["avg_bars_held"] = round(float(np.mean(held)), 2)
    return out


# ------------------------------------------------- splits & status helpers


def split_into_periods(ts_min, ts_max, n_periods=3):
    """Весь диапазон [ts_min, ts_max] на n РАВНЫХ по времени подпериодов.

    Возвращает [(start_ts, end_ts), ...] — end эксклюзивен, последний = ts_max.
    """
    n_periods = max(1, int(n_periods))
    width = (int(ts_max) - int(ts_min)) / float(n_periods)
    out = []
    for i in range(n_periods):
        s = int(ts_min) + i * width
        e = int(ts_max) if i == n_periods - 1 else int(ts_min) + (i + 1) * width
        out.append((s, e))
    return out


def split_train_test(ts_min, ts_max, train_fraction=0.70):
    """Time-based 70/30: (train (s,e), test (s,e)), end эксклюзивен."""
    ts_min = int(ts_min)
    ts_max = int(ts_max)
    span = (ts_max - ts_min) * float(train_fraction)
    split = ts_min + span
    return (ts_min, split), (split, ts_max)


def filter_trades(trades, start_ts, end_ts):
    """Сделки с entry ts в [start_ts, end_ts)."""
    return [t for t in trades if int(t["ts"]) >= int(start_ts)
            and int(t["ts"]) < int(end_ts)]


def sample_status(n_trades, n_ok=N_MIN_OK, n_warn=N_MIN_WARN):
    """Достаточно ли сделок для статистической достоверности."""
    if n_trades >= n_ok:
        return "ok"
    if n_trades >= n_warn:
        return "warn"
    return "fail"


def fees_status(net_sharpe, ok=NET_SHARPE_OK, low=NET_SHARPE_LOW):
    """Пережил ли edge комиссии + slippage."""
    if net_sharpe is None:
        return "warn"
    if net_sharpe >= ok:
        return "ok"
    if net_sharpe >= low:
        return "warn"
    return "fail"


def wf_status(sharpes, contribs, limit=CONCENTRATION_LIMIT,
              ok_thr=WF_SHARPE_OK):
    """Walk-forward: отрицательный период или концентрация прибыли -> fail."""
    sharpes = [float(s) if s is not None else 0.0 for s in sharpes]
    contribs = [float(c) for c in contribs]
    # Прибыль сконцентрирована в одном периоде (больше 90%).
    total_pos = sum(max(c, 0.0) for c in contribs)
    if total_pos > 0.0 and max(contribs) > limit * total_pos:
        return "fail"
    # Отрицательный период среди положительных — режим-зависимость.
    if min(sharpes) <= 0.0:
        return "fail"
    if all(s > ok_thr for s in sharpes):
        return "ok"
    return "warn"


def oos_status(train_sharpe, test_sharpe, ok_ratio=OOS_RATIO_OK,
               warn_ratio=OOS_RATIO_WARN):
    """Overfit-проверка: test должен держать >= 70% (0.4-0.7 — WARN) train."""
    if train_sharpe is None or test_sharpe is None:
        return "warn"
    if train_sharpe <= 0.0:
        return "fail" if test_sharpe <= 0.0 else "warn"
    ratio = test_sharpe / train_sharpe
    if ratio >= ok_ratio:
        return "ok"
    if ratio >= warn_ratio:
        return "warn"
    return "fail"


def shuffle_ratio(real_sharpe, shuffled_sharpe):
    """real/suffled; None, если real <= 0 (сигнала нет в принципе)."""
    if real_sharpe is None or real_sharpe <= 0.0:
        return None
    if shuffled_sharpe is None or shuffled_sharpe <= 0.0:
        return float("inf")
    return real_sharpe / shuffled_sharpe


def shuffle_status(ratio, ok_ratio=SHUFFLE_RATIO_OK,
                   warn_ratio=SHUFFLE_RATIO_WARN):
    """Permutation-test: ratio ~1 — все результаты = шум лейблов."""
    if ratio is None:
        return "fail"
    if ratio > ok_ratio:
        return "ok"
    if ratio >= warn_ratio:
        return "warn"
    return "fail"


_REASONS = {
    "sample": {"fail": "no-go: мало сделок (n<100) — увеличьте K / ослабьте "
                        "min_distance, чтобы набрать >=200 сделок",
               "warn": "warn: сделок 100-200 — Sharpe статистически неустойчив"},
    "fees": {"fail": "no-go: комиссии и slippage съедают весь edge "
                     "(net Sharpe < 1.5)",
             "warn": "warn: net Sharpe 1.5-2.5 — edge существенно съеден "
                     "издержками"},
    "walk_forward": {"fail": "no-go: стратегия нестабильна по периодам "
                             "(отрицательный период или >90% прибыли в одном) "
                             "— режим-зависима",
                     "warn": "warn: walk-forward разброс повышен (не все "
                             "периоды >1.5)"},
    "oos": {"fail": "no-go: out-of-sample Sharpe < 40% от train — "
                    "переобучение на обучающем отрезке",
            "warn": "warn: out-of-sample даёт 40-70% от train — частичный "
                    "overfit"},
    "shuffle": {"fail": "no-go: shuffled Sharpe близок к реальному — сигнала "
                        "нет, результаты объясняются шумом лейблов",
                "warn": "warn: shuffled Sharpe 50-83% от реального — сигнал "
                        "слабый"},
}


def assemble_verdict(checks):
    """GO / WARN / NO-GO по таблице ТЗ (приоритет fail: sample>fees>wf>oos>sh)."""
    by_id = {c["id"]: c for c in checks}
    order = ("sample", "fees", "walk_forward", "oos", "shuffle")
    for cid in order:
        if by_id[cid]["status"] == "fail":
            return "NO-GO", _REASONS[cid]["fail"]
    warns = [cid for cid in order if by_id[cid]["status"] == "warn"]
    if warns:
        return "WARN", "; ".join(_REASONS[cid]["warn"] for cid in warns)
    return "GO", ("все пять проверок пройдены: сделок достаточно, "
                  "edge держится после комиссий, по периодам и вне выборки, "
                  "shuffle-тест подтверждает сигнал")


# ------------------------------------------------------- permutation test


def build_shuffled_db(conn, tmp_dir, seed=42):
    """Копия БД (temp) с перемешанными outcome/bars_to_outcome/pnl_pct.

    Порядок строк ts ASC — тот же, что в индексе и _load_knn_meta, поэтому
    позиция faiss-соседа указывает на ту же строку, но с чужим outcome
    (нулей-гипотеза: метка и фичи независимы). Исходная БД не трогается.
    """
    rows = conn.execute(
        "SELECT ts, symbol, timeframe, features, feature_hash, regime, session, "
        "source, outcome, bars_to_outcome, pnl_pct FROM snapshots "
        "WHERE outcome IS NOT NULL ORDER BY ts ASC").fetchall()
    if not rows:
        raise ValueError("no labeled rows to shuffle")
    triples = [(r["outcome"], r["bars_to_outcome"], r["pnl_pct"]) for r in rows]
    rng = np.random.RandomState(int(seed))
    shuffled = list(triples)
    rng.shuffle(shuffled)
    path = str(Path(tmp_dir) / "cbr_shuffled.db")
    c2 = schema.init_db(path)
    data = []
    for r, (o, b, p) in zip(rows, shuffled):
        data.append((r["symbol"], r["timeframe"], int(r["ts"]), r["features"],
                     r["feature_hash"], r["regime"], r["session"], r["source"],
                     o, b, p))
    c2.executemany(
        "INSERT OR IGNORE INTO snapshots (symbol,timeframe,ts,features,"
        "feature_hash,regime,session,source,outcome,bars_to_outcome,pnl_pct) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)", data)
    c2.commit()
    return c2, path


# --------------------------------------------------------------- reporting


def _fnum(v, digits=3):
    if v is None:
        return None
    if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
        return None
    return round(float(v), digits)


def save_png(pargs, trades, wf_periods, out_path):
    """3 графика: equity curve (net), распределение PnL, Sharpe по периодам."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    pnls = [float(t["net_pct"]) for t in trades]

    ts_days = [datetime.datetime.fromtimestamp(int(t["ts"]),
                                               datetime.timezone.utc)
               for t in trades]
    axes[0].plot(ts_days, np.cumsum(pnls), lw=1.2)
    axes[0].set_title(f"Equity curve, net % (n={len(trades)})")
    axes[0].grid(alpha=0.3)
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%m-%Y"))

    axes[1].hist(pnls, bins=60, alpha=0.75)
    axes[1].axvline(0.0, color="r", lw=0.8)
    axes[1].set_title(f"PnL distribution, net % (mean={np.mean(pnls):.3f})")

    names = [p["name"] for p in wf_periods]
    sh = [p["sharpe"] if p["sharpe"] is not None else 0.0
          for p in wf_periods]
    axes[2].bar(names, sh, alpha=0.75)
    for i, v in enumerate(sh):
        axes[2].text(i, v, f"{v:.2f}", ha="center", va="bottom")
    axes[2].axhline(0.0, color="k", lw=0.7)
    axes[2].set_title("Sharpe by period (net)")

    fig.suptitle(f"{pargs['symbol']} {pargs['timeframe']} K={pargs['K']} "
                 f"md={pargs['min_distance']} — CBR-only (B) validation",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(str(out_path), dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Валидация CBR-only (B): Sharpe по 5 проверкам "
                    "(sample/fees/walk-forward/OOS/permutation).")
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--db", default=str(config.CBR_DB_PATH))
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--tf", dest="timeframe", default="1H")
    parser.add_argument("--timeframe", dest="timeframe")
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--md", "--min-distance", type=float, dest="min_distance",
                        default=1.0)
    parser.add_argument("--no-regime-match", action="store_false",
                        dest="regime_match", default=config.CBR_REGIME_MATCH)
    parser.add_argument("--window-days", type=int, dest="window_days",
                        default=config.CBR_QUERY_WINDOW_DAYS)
    parser.add_argument("--start-fraction", type=float, dest="start_fraction",
                        default=0.0, help="0.0 = весь диапазон (для 3 периодов)")
    parser.add_argument("--max-bars", type=int, default=0,
                        help="ограничить число баров (ускорение)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(REPORTS_DIR / "cbr_validate_B.json"))
    args = parser.parse_args()

    timeframe = str(args.timeframe or "1H").upper()
    symbol = str(args.symbol or "BTCUSDT").upper()
    pargs = {"symbol": symbol, "timeframe": timeframe, "db": str(args.db),
             "csv": str(args.csv), "K": args.K, "min_distance": args.min_distance,
             "regime_match": args.regime_match, "window_days": args.window_days,
             "start_fraction": args.start_fraction, "seed": args.seed,
             "fee_per_side": FEE_PER_SIDE, "slippage": SLIPPAGE,
             "sharpe_ppy": SHARPE_PPY}

    conn = schema.init_db(args.db)
    db_rows = load_db(conn, symbol, timeframe)
    if not db_rows:
        raise SystemExit("Нет размеченных строк в БД")
    csv_data = load_csv(args.csv)
    index = cbr_index.load_index(str(config.CBR_FAISS_INDEX_PATH))
    stats = cbr_norm.load_stats(str(config.CBR_NORMALIZE_STATS_PATH))
    if index is None or not (isinstance(stats, dict) and "mean" in stats):
        raise SystemExit("Нет артефактов Этапа 2: python scripts/cbr_build.py "
                         "--artifacts")

    print("=== CBR-only (B) validation ===")
    print(f"{symbol} {timeframe}: K={args.K} md={args.min_distance} "
          f"regime_match={args.regime_match} window={args.window_days}d "
          f"start_fraction={args.start_fraction} seed={args.seed}")

    # ---- прогон B на реальных метках.
    trades = run_b_only(conn, db_rows, index, stats, csv_data, symbol,
                        timeframe, args.K, args.min_distance,
                        args.regime_match, args.window_days,
                        args.start_fraction, args.max_bars)
    s_gross = trade_stats(trades, "gross_pct")
    s_net = trade_stats(trades, "net_pct")

    check_sample = sample_status(s_gross["n"])
    check_fees = fees_status(s_net["sharpe"])
    avg_fee_pp = (float(s_gross["avg_pnl_pct"]) - float(s_net["avg_pnl_pct"])
                  if s_gross["n"] else 0.0)

    # ---- 1-2: sample + fees.
    checks = [
        {"id": "sample", "title": "Sample size", "status": check_sample,
         "n_trades": s_gross["n"],
         "detail": (f"n={s_gross['n']} сделок "
                    f"({N_MIN_OK}+: OK, {N_MIN_WARN}-{N_MIN_OK}: WARN, "
                    f"<{N_MIN_WARN}: FAIL)")},
        {"id": "fees", "title": "Sharpe after fees", "status": check_fees,
         "sharpe_gross": _fnum(s_gross["sharpe"]),
         "sharpe_net": _fnum(s_net["sharpe"]),
         "avg_fee_pp": round(avg_fee_pp, 4),
         "detail": (f"gross={s_gross['sharpe']:.2f} net={s_net['sharpe']:.2f} "
                    f"(fees {FEE_PER_SIDE*100:.2f}% + slippage "
                    f"{SLIPPAGE*100:.2f}% на сторону)")},
    ]

    # ---- 3: walk-forward (net) по 3 равным подпериодам всего диапазона.
    all_ts = sorted(db_rows)
    start_pos = int(len(all_ts) * args.start_fraction)
    ts_min, ts_max = all_ts[start_pos], all_ts[-1]
    periods = split_into_periods(ts_min, ts_max, 3)
    wf_periods = []
    for i, (s, e) in enumerate(periods):
        sub = filter_trades(trades, s, e)
        st = trade_stats(sub, "net_pct")
        wf_periods.append({"name": f"P{i + 1}", "start_ts": int(s),
                           "end_ts": int(e), "n": st["n"],
                           "winrate": st["winrate"], "sharpe": st["sharpe"],
                           "sum_pnl_pct": st["sum_pnl_pct"]})
    check_wf = wf_status([p["sharpe"] for p in wf_periods],
                         [p["sum_pnl_pct"] for p in wf_periods])
    checks.append({"id": "walk_forward", "title": "Walk-forward (3 periods)",
                   "status": check_wf, "periods": wf_periods,
                   "detail": " / ".join(
                       f"P{i+1}: n={p['n']} sharpe={_fmt(p['sharpe'])}"
                       for i, p in enumerate(wf_periods))})

    # ---- 4: out-of-sample 70/30 (net).
    (tr_s, tr_e), (te_s, te_e) = split_train_test(ts_min, ts_max, 0.70)
    train_t = filter_trades(trades, tr_s, tr_e)
    test_t = filter_trades(trades, te_s, te_e)
    st_train = trade_stats(train_t, "net_pct")
    st_test = trade_stats(test_t, "net_pct")
    oos_ratio = (st_test["sharpe"] / st_train["sharpe"]
                 if st_train["sharpe"] and st_train["sharpe"] > 0 else None)
    check_oos = oos_status(st_train["sharpe"], st_test["sharpe"])
    checks.append({"id": "oos", "title": "Out-of-sample 70/30", "status": check_oos,
                   "train": {"n": st_train["n"], "sharpe": _fnum(st_train["sharpe"])},
                   "test": {"n": st_test["n"], "sharpe": _fnum(st_test["sharpe"])},
                   "ratio": _fnum(oos_ratio),
                   "detail": f"train={_fmt(st_train['sharpe'])} "
                             f"test={_fmt(st_test['sharpe'])} "
                             f"ratio={_fmt(oos_ratio)}"})

    # ---- 5: permutation (gross; издержки постоянны, на ratio не влияют).
    with tempfile.TemporaryDirectory(prefix="cbr_perm_") as tmp:
        conn2, tmp_db_path = build_shuffled_db(conn, tmp, args.seed)
        try:
            trades_sh = run_b_only(conn2, db_rows, index, stats, csv_data,
                                   symbol, timeframe, args.K, args.min_distance,
                                   args.regime_match, args.window_days,
                                   args.start_fraction, args.max_bars)
        finally:
            conn2.close()
        st_sh = trade_stats(trades_sh, "gross_pct")
    ratio_sh = shuffle_ratio(s_gross["sharpe"], st_sh["sharpe"])
    check_sh = shuffle_status(ratio_sh)
    checks.append({"id": "shuffle", "title": "Permutation (labels shuffled)",
                   "status": check_sh,
                   "real_sharpe": _fnum(s_gross["sharpe"]),
                   "shuffled_sharpe": _fnum(st_sh["sharpe"]),
                   "shuffled_n": st_sh["n"], "ratio": _fnum(ratio_sh),
                   "detail": f"real={_fmt(s_gross['sharpe'])} "
                             f"shuffled={_fmt(st_sh['sharpe'])} "
                             f"ratio={_fmt(ratio_sh)}"})

    verdict, reason = assemble_verdict(checks)
    if verdict == "GO":
        recommendation = (f"использовать B (CBR-only) в проде: K={args.K}, "
                          f"min_distance={args.min_distance}")
    elif verdict == "NO-GO" and check_sample == "fail":
        recommendation = (f"набрать >= {N_MIN_OK} сделок: увеличить K, "
                          f"ослабить min_distance, затем повторить валидацию")
    else:
        recommendation = (f"пересмотреть параметры K/min_distance при "
                          f"фиксированных правилах и повторить валидацию; "
                          f"blender/Этап 2 при этом не менять")

    report = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "params": pargs,
        "sample": {"n_trades": s_gross["n"], "winrate": s_gross["winrate"],
                   "avg_pnl_pct": s_gross["avg_pnl_pct"],
                   "std_pnl_pct": s_gross["std_pnl_pct"],
                   "sharpe_gross": _fnum(s_gross["sharpe"]),
                   "max_dd_pct": s_gross["max_dd_pct"],
                   "profit_factor": _fnum(s_gross["profit_factor"]),
                   "avg_bars_held": s_gross["avg_bars_held"]},
        "checks": checks,
        "verdict": {"value": verdict, "reason": reason,
                    "recommendation": recommendation},
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                              default=str), encoding="utf-8")
    png_path = out.with_suffix(".png")
    if trades:
        save_png(pargs, trades, wf_periods, png_path)
    else:
        png_path.write_text("no trades", encoding="utf-8")

    # ---- печать отчёта (формат ТЗ).
    def line(name, s):
        print(f"  {name:<58s}{s}" if len(name) < 58 else f"  {name}: {s}")

    print("Валидация 1 — Sample size:")
    line(f"n_trades = {s_gross['n']}", _status(check_sample))
    line(f"winrate = {100 * (s_gross['winrate'] or 0):.2f}%" , "")
    line(f"avg_pnl = {(s_gross['avg_pnl_pct'] or 0):.2f}% (gross)", "")
    print("Валидация 2 — Sharpe after fees:", _status(check_fees))
    line(f"gross = {s_gross['sharpe']:.2f}  net = {s_net['sharpe']:.2f}",
         "(fees %.2f%% + slippage %.2f%% per side)" %
         (FEE_PER_SIDE * 100, SLIPPAGE * 100))
    print("Валидация 3 — Walk-forward (3 periods):",
          _status(check_wf))
    for p in wf_periods:
        line(f"P{p['name'][1]}: n={p['n']} sharpe={_fmt(p['sharpe'])} "
             f"winrate={100 * (p['winrate'] or 0):.2f}%", "")
    print("Валидация 4 — Out-of-sample 70/30:", _status(check_oos))
    line(f"train: n={st_train['n']} sharpe={_fmt(st_train['sharpe'])}", "")
    line(f"test:  n={st_test['n']} sharpe={_fmt(st_test['sharpe'])}", "")
    line(f"ratio: {_fmt(oos_ratio)}", "")
    print("Валидация 5 — Permutation test:", _status(check_sh))
    line(f"real:     sharpe={_fmt(s_gross['sharpe'])}", "")
    line(f"shuffled: sharpe={_fmt(st_sh['sharpe'])}", "")
    line(f"ratio:    {_fmt(ratio_sh)}", "")
    print()
    print(f"ВЕРДИКТ: {verdict}   ({reason})")
    print(f"Рекомендация: {recommendation}")
    print(f"Saved: {out} / {png_path}")


def _status(s):
    return {"ok": "OK", "warn": "WARN", "fail": "FAIL"}.get(s, s)


def _fmt(v, digits=3):
    if v is None:
        return "-"
    return f"{v:.{digits}f}"


if __name__ == "__main__":
    main()