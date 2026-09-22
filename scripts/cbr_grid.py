# -*- coding: utf-8 -*-
"""Grid по (k, horizon) для triple-barrier CBR: поиск сбалансированных классов.

Для каждой пары (k, horizon) ставится временная CBR-БД из одних и тех же
снимков. features НЕ зависят от барьеров — store_snapshot прогоняется один раз
в base.db, для каждой пары меняется только backfill_outcomes. Лучшая пара по
balance_score (≈35/35/30 раскладка up/down/flat) копируется в
config.CBR_DB_PATH.

Запуск из корня проекта:
    python scripts/cbr_grid.py \
        --csv scripts/charon_data_cache/BTCUSDT_1h_700d.csv \
        --symbol BTCUSDT --tf 1h --db-base data/cbr_grid \
        --k 1.5 2.5 3.5 4.5 --horizon 10 20 40
"""

import argparse
import shutil
import sqlite3
import sys
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent.parent
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

from app_pkg import config  # noqa: E402
from app_pkg.cbr import backfill, schema, store  # noqa: E402
from app_pkg.ml import labels  # noqa: E402
from scripts import cbr_build  # noqa: E402

TARGET_UP, TARGET_DOWN, TARGET_FLAT = 0.35, 0.35, 0.30
BALANCE_OK = 0.85


def canonical_tf(tf):
    """Приводит таймфрейм к канонической форме config.TF_SECONDS
    (1h/1H->1H, 5m->5m), иначе возвращает как есть."""
    tf = str(tf or "").strip()
    for canon in config.TF_SECONDS:
        if canon.lower() == tf.lower():
            return canon
    return tf


def _checkpoint(conn):
    """Транкейт WAL, чтобы главный .db-файл был самодостаточным."""
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except sqlite3.Error:
        pass


def _remove_db(path):
    """Удаляет .db и его WAL-сайдкары (-wal/-shm)."""
    for p in (Path(path), Path(str(path) + "-wal"), Path(str(path) + "-shm")):
        p.unlink(missing_ok=True)


def commit_copy(src_path, dst_path):
    """WAL-safe копирование SQLite-БД через backup API (сайдкары не болят)."""
    dst_path = Path(dst_path)
    _remove_db(dst_path)
    src = sqlite3.connect(src_path)
    try:
        dst = sqlite3.connect(dst_path)
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()


def balance_score(pct_up, pct_down, pct_flat):
    """Баланс классов: 1 при ровно 35/35/30, падает при отклонении доли."""
    return 1.0 - max(abs(float(pct_up) - TARGET_UP),
                     abs(float(pct_down) - TARGET_DOWN),
                     abs(float(pct_flat) - TARGET_FLAT))


def compute_metrics(conn, symbol, timeframe):
    """(pct_up, pct_down, pct_flat, avg_bars, balance) по размеченным строкам."""
    symbol = str(symbol).upper()
    timeframe = str(timeframe).strip()
    rows = conn.execute(
        "SELECT outcome, COUNT(*) AS c FROM snapshots "
        "WHERE symbol=? AND timeframe=? AND outcome IS NOT NULL "
        "GROUP BY outcome", (symbol, timeframe)).fetchall()
    total = float(sum(r["c"] for r in rows)) or 1.0
    counts = {int(r["outcome"]): int(r["c"]) for r in rows}
    pct_up = counts.get(1, 0) / total
    pct_down = counts.get(-1, 0) / total
    pct_flat = counts.get(0, 0) / total
    avg_bars = conn.execute(
        "SELECT AVG(bars_to_outcome) AS b FROM snapshots "
        "WHERE symbol=? AND timeframe=? AND outcome IS NOT NULL",
        (symbol, timeframe)).fetchone()["b"]
    avg_bars = None if avg_bars is None else float(avg_bars)
    return pct_up, pct_down, pct_flat, avg_bars, balance_score(
        pct_up, pct_down, pct_flat)


def build_base_db(csv_path, symbol, timeframe, db_path, source="build"):
    """store_snapshot по всем барам один раз -> base.db.

    Возвращает (bars, price_fn, stored, dup): бары + price_fn для backfill'а
    каждой пары (features не зависят от k/horizon).
    """
    db_path = Path(db_path)
    _remove_db(db_path)  # пересоздаём с нуля: минувшие прогоны не накапливать
    df = cbr_build.load_candle_df(csv_path, symbol, timeframe)
    if df is None or df.empty:
        raise SystemExit(f"Нет данных: csv={csv_path}")
    n = len(df)
    symbol = str(symbol).upper()
    timeframe = canonical_tf(timeframe)
    closes = df["close"].astype(float).to_numpy(dtype="float64")
    bars = [{
        "ts": cbr_build._bar_ts_sec(row["timestamp"]),
        "open": float(row["open"]), "high": float(row["high"]),
        "low": float(row["low"]), "close": float(row["close"]),
    } for _, row in df.iterrows()]

    def price_fn(sym, tf, from_ts, to_ts):
        return [b for b in bars if b["ts"] >= from_ts
                and (to_ts is None or b["ts"] <= to_ts)]

    features = labels.charon_features(df, symbol=symbol)
    start = cbr_build._first_finite_index(features)
    conn = schema.init_db(db_path)
    stored = dup = 0
    for i in range(start, n):
        ts = bars[i]["ts"]
        entry = float(closes[i])
        if entry <= 0:
            continue
        row = {name: (float(v) if v is not None else None)
               for name, v in features.iloc[i].items()}
        rid = store.store_snapshot(conn, {
            "symbol": symbol, "timeframe": timeframe, "ts": ts,
            "source": source, **row,
        }, entry_price=entry)
        if rid is None:
            dup += 1
        else:
            stored += 1
    _checkpoint(conn)
    conn.close()
    return bars, price_fn, stored, dup


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grid по k*horizon CBR: выбор сбалансированных барьеров.")
    parser.add_argument("--csv", required=True, help="путь к CSV OHLCV")
    parser.add_argument("--symbol", default="BTCUSDT", help="символ")
    parser.add_argument("--tf", default="1h", help="таймфрейм")
    parser.add_argument("--db-base", default=None,
                        help="каталог временных БД (по умолчанию "
                             "data/cbr_grid)")
    parser.add_argument("--out", default=None,
                        help="куда копировать БД лучшей пары "
                             "(по умолчанию config.CBR_DB_PATH)")
    parser.add_argument("--k", type=float, nargs="+",
                        default=[1.5, 2.5, 3.5, 4.5],
                        help="множители ATR барьеров")
    parser.add_argument("--horizon", type=int, nargs="+",
                        default=[10, 20, 40], help="горизонты (баров)")
    parser.add_argument("--quiet", action="store_true", help="без прогресса")
    args = parser.parse_args()

    base_dir = Path(args.db_base or config.DATA_DIR / "cbr_grid")
    base_dir.mkdir(parents=True, exist_ok=True)
    target = Path(args.out or config.CBR_DB_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)

    base_db = base_dir / "base.db"
    if not args.quiet:
        print(f"[cbr_grid] строю base.db из {args.csv} ...")
    bars, price_fn, stored, dup = build_base_db(
        args.csv, args.symbol, args.tf, base_db)
    symbol = str(args.symbol).upper()
    timeframe = canonical_tf(args.tf)
    if not args.quiet:
        print(f"[cbr_grid] snapshots: stored={stored} dup={dup}")

    header = (f"{'k':>5s} {'h':>5s} {'up%':>6s} {'dn%':>6s} {'flat%':>7s} "
              f"{'avg_bars':>9s} {'balance':>8s}")
    print(header)
    best = None
    best_tmp = None
    tmps = []
    for k in args.k:
        for h in args.horizon:
            tmp = base_dir / f"k{k}_h{h}.db"
            shutil.copyfile(base_db, tmp)
            tmps.append(tmp)
            conn = schema.init_db(tmp)
            try:
                backfill.backfill_outcomes(conn, symbol, timeframe, price_fn,
                                           horizon=h, atr_k=k)
                pct_up, pct_down, pct_flat, avg_bars, balance = \
                    compute_metrics(conn, symbol, timeframe)
            finally:
                _checkpoint(conn)
                conn.close()
            avg_s = f"{avg_bars:.1f}" if avg_bars is not None else "-"
            print(f"{k:>5} {h:>5} {pct_up * 100:>5.1f}% {pct_down * 100:>5.1f}% "
                  f"{pct_flat * 100:>6.1f}% {avg_s:>9} {balance:>8.2f}")
            if best is None or balance > best[6]:
                best = (k, h, pct_up, pct_down, pct_flat, avg_bars, balance)
                best_tmp = tmp

    commit_copy(best_tmp, target)
    for tmp in tmps:
        _remove_db(tmp)

    k, h, pct_up, pct_down, pct_flat, avg_bars, balance = best
    print()
    print(f"BEST: k={k}, horizon={h}")
    print(f"balance_score={balance:.2f}")
    print(f"up={pct_up * 100:.1f}% dn={pct_down * 100:.1f}% "
          f"flat={pct_flat * 100:.1f}%")
    print(f"avg_bars={avg_bars:.1f}" if avg_bars is not None
          else "avg_bars=-")
    print(f"DB скопирован: {target}")
    if balance < BALANCE_OK:
        print(f"WARNING: balance_score={balance:.2f} < {BALANCE_OK} — "
              "расширь диапазон k до 5.0+ и горизонт")


if __name__ == "__main__":
    main()