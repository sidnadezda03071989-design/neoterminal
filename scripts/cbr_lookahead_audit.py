"""Аудит look-ahead bias в CBR k-NN (Этап 2): scripts/cbr_lookahead_audit.py.

Проверяет, что get_similar_summary возвращает в соседи только бары, чей
outcome заведомо известен на момент бара запроса: ts_соседа + horizon <
bar_ts. Без этого фильтра k-NN «протекает» из будущего — winrate/Sharpe
становятся невозможными для retail.

Проверки (для K соседей по N случайным барам, seed=42):
  A. future neighbors       — ts_соседа >= ts_бара (сосед из будущего);
  B. outcome-not-yet-known  — ts_соседа + horizon > ts_бара (исход соседа
                              ещё не состоялся на момент бара);
  C. time delta             — распределение (ts_бара - ts_соседа): min должен
                              быть > horizon (20 баров), negative_count = 0;
  D. trade coverage         — доля баров, открывающих сделку (direction L/S)
                              по спреду вероятностей; > 0.20 → сигнал
                              не селективный.

Вердикт: NO LOOK-AHEAD, если A==0, B==0, min_delta > horizon и нет
отрицательных дельт; иначе LOOK-AHEAD CONFIRMED.

Чистые функции (check_neighbor_pairs, direction_from_summary,
trade_coverage, verdict) держатся на модульном уровне — они покрываются
tests/test_cbr_lookahead.py без прогона тяжёлого бэктеста.

Запуск:
    python scripts/cbr_lookahead_audit.py --db data/cbr.db --symbol BTCUSDT --tf 1h
"""

import argparse
import sys
from pathlib import Path

import numpy as np

_BASE_DIR = Path(__file__).resolve().parent.parent
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

from app_pkg import config
from app_pkg.cbr import index as cbr_index
from app_pkg.cbr import normalize as cbr_norm
from app_pkg.cbr import query_api, schema
from scripts.cbr_backtest_abc import load_db, snap_from_features

# Порог спреда вероятностей для направления (та же конвенция, что в
# cbr_backtest_abc.SIG_SPREAD): |up - down| > 0.15 -> сделка L/S.
SIG_SPREAD = 0.15
# Неселективность: > 20% баров открывают сделку.
COVERAGE_WARN = 0.20


def canonical_tf(tf):
    """Приводит таймфрейм к канонической форме config.TF_SECONDS (1h->1H)."""
    tf = str(tf or "").strip()
    for canon in config.TF_SECONDS:
        if canon.lower() == tf.lower():
            return canon
    return tf


# ------------------------------------------------------------ чистые функции


def check_neighbor_pairs(pairs, horizon_sec):
    """А/B/C по парам (query_ts, [neighbor_ts, ...]).

    Возвращает (future_count, not_known_count, deltas_days):
      future_count      — соседей с query_ts - n_ts <= 0 (из будущего);
      not_known_count   — соседей с n_ts + horizon_sec > query_ts
                          (исход соседа неизвестен на момент бара);
      deltas_days       — список (query_ts - n_ts) / 86400 по всем парам.
    """
    future = 0
    not_known = 0
    deltas = []
    for bar_ts, nbrs in pairs:
        for n_ts in nbrs or []:
            d = int(bar_ts) - int(n_ts)
            deltas.append(d / 86400.0)
            if d <= 0:
                future += 1
            if int(n_ts) + int(horizon_sec) > int(bar_ts):
                not_known += 1
    return future, not_known, deltas


def direction_from_summary(res, spread=SIG_SPREAD):
    """L/S/F по сводке get_similar_summary (insufficient -> F)."""
    if not isinstance(res, dict) or res.get("insufficient_data"):
        return "F"
    up = float(res.get("winrate_up", 0.0))
    dn = float(res.get("winrate_down", 0.0))
    if up - dn > spread:
        return "L"
    if dn - up > spread:
        return "S"
    return "F"


def trade_coverage(directions):
    """Доля баров, открывающих сделку; None на пустом списке."""
    if not directions:
        return None
    traded = sum(1 for d in directions if d != "F")
    return traded / float(len(directions))


def audit_verdict(future, not_known, negative_count, min_delta_days,
                  horizon_days):
    """NO if все проверки чистые, иначе LOOK-AHEAD CONFIRMED."""
    clean = (future == 0 and not_known == 0 and negative_count == 0
             and min_delta_days is not None and min_delta_days > horizon_days)
    if clean:
        return "NO LOOK-AHEAD — утечка в другом месте"
    return "LOOK-AHEAD CONFIRMED — исправить get_similar_summary"


# ------------------------------------------------------------ прогон аудита


def run_audit(conn, index, stats, symbol, tf, K=10, min_distance=1.0,
              regime_match=True, window_days=180, seed=42, sample_size=200,
              horizon=None, coverage_sample=0, spread=SIG_SPREAD):
    """Полный прогон проверок A/B/C/D по реальной БД.

    Запрашивает get_similar_summary с include_neighbors=True для баров из
    (случайная выборка для A/B/C) ∪ (выборка для D). Возвращает dict с
    результатами: a_count/b_count/deltas_stats/coverage/verdict и т.п.
    """
    symbol = str(symbol or "").upper()
    tf = canonical_tf(tf)
    bar_sec = config.TF_SECONDS.get(tf, 3600)
    horizon_bars = int(horizon if horizon is not None
                       else config.CBR_BACKFILL_HORIZON)
    horizon_sec = int(horizon_bars) * int(bar_sec)
    horizon_days = horizon_sec / 86400.0

    bars = load_db(conn, symbol, tf)
    if not bars:
        raise ValueError(f"no labeled bars in DB for {symbol} {tf}")
    all_ts = sorted(bars)
    n = len(all_ts)

    rng = np.random.RandomState(int(seed))
    sample_size = max(1, min(int(sample_size), n))
    sample_ts = {all_ts[i] for i in rng.choice(
        n, size=sample_size, replace=False)}

    if int(coverage_sample) > 0:
        cov_n = max(1, min(int(coverage_sample), n))
        cov_ts = {all_ts[i] for i in rng.choice(
            n, size=cov_n, replace=False)}
    else:
        cov_ts = set(all_ts)
    wanted = sample_ts | cov_ts

    pairs = []
    cov_directions = []
    for ts in wanted:
        vec, _regime = bars[ts]
        snap = snap_from_features(vec, symbol, tf, ts)
        res = query_api.get_similar_summary(
            conn, index, snap, symbol, tf, bar_ts=int(ts),
            K=int(K), min_distance=float(min_distance),
            regime_match=bool(regime_match), window_days=int(window_days),
            stats=stats, include_neighbors=True)
        if ts in cov_ts:
            cov_directions.append(direction_from_summary(res, spread))
        if ts in sample_ts and not (not isinstance(res, dict)
                                    or res.get("insufficient_data")):
            pairs.append((int(ts), [int(t) for t in
                                    res.get("neighbors_ts", [])]))

    future, not_known, deltas = check_neighbor_pairs(pairs, horizon_sec)
    deltas_arr = np.asarray([float(d) for d in deltas], dtype="float64")
    if deltas_arr.size:
        deltas_stats = {
            "median_days": float(np.median(deltas_arr)),
            "p10_days": float(np.percentile(deltas_arr, 10)),
            "min_days": float(deltas_arr.min()),
            "negative_count": int((deltas_arr < 0).sum()),
        }
    else:
        deltas_stats = {"median_days": None, "p10_days": None,
                        "min_days": None, "negative_count": 0}

    cov = trade_coverage(cov_directions)
    verdict = audit_verdict(future, not_known,
                            deltas_stats["negative_count"],
                            deltas_stats["min_days"], horizon_days)

    return {
        "symbol": symbol,
        "timeframe": tf,
        "K": int(K),
        "min_distance": float(min_distance),
        "regime_match": bool(regime_match),
        "window_days": int(window_days),
        "horizon_bars": horizon_bars,
        "horizon_days": round(horizon_days, 3),
        "sample_bars": len(sample_ts),
        "neighbor_pairs": len(pairs),
        "a_future_neighbors": future,
        "b_outcome_not_known": not_known,
        "deltas": deltas_stats,
        "coverage": (round(cov, 4) if cov is not None else None),
        "coverage_n": len(cov_directions),
        "coverage_traded": round((cov or 0.0) * len(cov_directions)),
        "coverage_selective": bool(cov is not None and cov < COVERAGE_WARN),
        "verdict": verdict,
    }


# ---------------------------------------------------------------- CLI выход


def print_report(r):
    """Печать отчёта в формате ТЗ."""
    d = r["deltas"]
    a = r["a_future_neighbors"]
    b = r["b_outcome_not_known"]
    nd = d["negative_count"]
    mnd = d["min_days"]
    cov = r["coverage"]
    cov_txt = "n/a" if cov is None else f"{100.0 * cov:.1f}%"
    print(f"LOOK-AHEAD AUDIT — {r['symbol']} {r['timeframe']}")
    print("=" * 50)
    print(f"A. Future neighbors:            {a} / {r['neighbor_pairs']}  "
          f"(0 = OK)")
    print(f"B. Outcome-not-yet-known:       {b} / {r['neighbor_pairs']}  "
          f"(0 = OK)")
    print("C. Time delta:")
    med = d["median_days"]
    med_txt = "n/a" if med is None else f"{med:.3f}"
    print(f"   median: {med_txt} days")
    fmt = "n/a" if mnd is None else f"{mnd:.3f}"
    print(f"   min:    {fmt} days   (должно быть >= "
          f"{r['horizon_days']:.2f} = {r['horizon_bars']} баров)")
    print(f"   negative_count: {nd}")
    print(f"D. Trade coverage: {cov_txt} ({r['coverage_traded']}/"
          f"{r['coverage_n']})")
    print()
    print("ВЕРДИКТ: ")
    mark = "✅" if r["verdict"].startswith("NO") else "❌"
    print(f"  {mark} {r['verdict']}")


def main():
    parser = argparse.ArgumentParser(
        description="Аудит look-ahead bias в CBR k-NN (get_similar_summary).")
    parser.add_argument("--db", default=str(config.CBR_DB_PATH))
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--tf", dest="tf", default="1H")
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--min-distance", type=float, dest="min_distance",
                        default=1.0)
    parser.add_argument("--no-regime-match", action="store_false",
                        dest="regime_match", default=True)
    parser.add_argument("--window-days", type=int, dest="window_days",
                        default=config.CBR_QUERY_WINDOW_DAYS)
    parser.add_argument("--horizon", type=int, dest="horizon", default=None,
                        help="барьеры backfill (по умолчанию "
                             "config.CBR_BACKFILL_HORIZON)")
    parser.add_argument("--sample-size", type=int, dest="sample_size",
                        default=200)
    parser.add_argument("--coverage-sample", type=int, dest="coverage_sample",
                        default=0, help="0 = все бары (медленно, но точно)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    conn = schema.init_db(args.db)
    index = cbr_index.load_index(str(config.CBR_FAISS_INDEX_PATH))
    stats = cbr_norm.load_stats(str(config.CBR_NORMALIZE_STATS_PATH))
    if index is None or not (isinstance(stats, dict) and "mean" in stats):
        raise SystemExit("Нет артефактов Этапа 2: python scripts/cbr_build.py "
                         "--artifacts")

    r = run_audit(conn, index, stats, args.symbol, args.tf, K=args.K,
                  min_distance=args.min_distance,
                  regime_match=args.regime_match,
                  window_days=args.window_days, seed=args.seed,
                  sample_size=args.sample_size, horizon=args.horizon,
                  coverage_sample=args.coverage_sample)
    print_report(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())