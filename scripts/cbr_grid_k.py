# -*- coding: utf-8 -*-
"""Сетка K x min_distance для CBR Этапа 2 (scripts/cbr_backtest_abc run-backbone).

Прогоняет A/B/C-бэктест на 12 комбинациях (K, min_distance) с ОСТАЛЬНЫМИ
параметрами ТЗ (окно 180d, regime_match=True, SL/TP/горизонт/комиссия те же).
Для каждой строки K — метрики A/B/C; лучшая комбинация выбирается по
максимальному запасу C над обоими одиночными двигателями:
    score = Sharpe(C) - max(Sharpe(A), Sharpe(B)).

Артефакты: reports/cbr_grid_k.csv (+ .json рядом). Печатает таблицу и победителя.
Запуск:    python scripts/cbr_grid_k.py
"""

import csv
import datetime
import json
import sys
from pathlib import Path

import numpy as np

_BASE_DIR = Path(__file__).resolve().parent.parent
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

from app_pkg import config  # noqa: E402
from app_pkg.cbr import schema  # noqa: E402

from scripts.cbr_backtest_abc import (  # noqa: E402
    REPORTS_DIR, DEFAULT_CSV, build_verdict, load_csv, load_db, run_backtest)

# 12 комбинаций (K x min_distance).
GRID_COMBOS = [
    (10, 1.0), (10, 2.0), (10, 3.0),
    (30, 1.0), (30, 2.0), (30, 3.0),
    (50, 1.0), (50, 2.0), (50, 3.0),
    (100, 1.0), (100, 2.0), (200, 2.0),
]


def _sharpe(m):
    return (float(m.get("sharpe")) if m.get("sharpe") is not None else -1e9)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Сетка K x min_distance CBR Этапа 2.")
    parser.add_argument("--db", default=str(config.CBR_DB_PATH))
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", default="1H")
    parser.add_argument("--window-days", type=int, dest="window_days",
                        default=config.CBR_QUERY_WINDOW_DAYS)
    parser.add_argument("--start-fraction", type=float, dest="start_fraction",
                        default=0.70)
    parser.add_argument("--max-bars", type=int, default=0,
                        help="ограничить тестируемых баров (ускорение демо)")
    args = parser.parse_args()

    conn = schema.init_db(args.db)
    sym = args.symbol.upper()
    db_rows = load_db(conn, sym, args.timeframe)
    csv_data = load_csv(args.csv)

    results = []
    for i, (K, min_dist) in enumerate(GRID_COMBOS):
        rows, engines, by_regime, coverage, info = run_backtest(
            conn, db_rows, csv_data, K=K, min_distance=min_dist,
            regime_match=True, window_days=args.window_days,
            start_fraction=args.start_fraction, max_bars=args.max_bars)
        verdict = build_verdict(engines)
        score = (_sharpe(engines["C_blended"])
                 - max(_sharpe(engines["A_charon"]),
                       _sharpe(engines["B_cbr"])))
        rec = {
            "K": K, "min_distance": min_dist,
            "wr_A": engines["A_charon"]["winrate"],
            "wr_B": engines["B_cbr"]["winrate"],
            "wr_C": engines["C_blended"]["winrate"],
            "sh_A": engines["A_charon"]["sharpe"],
            "sh_B": engines["B_cbr"]["sharpe"],
            "sh_C": engines["C_blended"]["sharpe"],
            "trades_A": engines["A_charon"]["trades"],
            "trades_B": engines["B_cbr"]["trades"],
            "trades_C": engines["C_blended"]["trades"],
            "avg_pnl_C": engines["C_blended"]["avg_pnl_pct"],
            "score": round(score, 4),
            "is_pass": verdict["is_pass"],
        }
        results.append(rec)
        print(f"[grid {i + 1}/{len(GRID_COMBOS)}] K={K:3d} md={min_dist:.1f} "
              f"wrA={rec['wr_A']} wrB={rec['wr_B']} wrC={rec['wr_C']} "
              f"shA={rec['sh_A']} shB={rec['sh_B']} shC={rec['sh_C']} "
              f"score={rec['score']} pass={rec['is_pass']}")

    best = max(results, key=lambda r: r["score"])
    print()
    print(f"BEST: K={best['K']} min_distance={best['min_distance']} "
          f"score={best['score']} "
          f"wrC={best['wr_C']} shC={best['sh_C']} pass={best['is_pass']}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS_DIR / "cbr_grid_k.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    json_path = REPORTS_DIR / "cbr_grid_k.json"
    json_path.write_text(json.dumps({
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "symbol": sym, "timeframe": args.timeframe,
        "window_days": args.window_days,
        "pick_criterion": "score = Sharpe(C) - max(Sharpe(A), Sharpe(B))",
        "best": best,
        "rows": results,
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"Saved: {csv_path} / {json_path}")


if __name__ == "__main__":
    main()