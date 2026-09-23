"""Оркестратор CBR-базы Харона (Этап 1+): запись снимков + backfill + отчёт.

Читает OHLCV-источник (CSV в формате timestamp,open,high,low,close,volume
или фетч по symbol/timeframe), считает charon_features по ВСЕМ барам
(labels: те же числа, что видит нейросеть), записывает «сырые» снимки
(source=build) в CBR-БД с entry_price=close бара, затем заполняет
triple-barrier outcome через price_fn по тому же датасету и печатает отчёт.

Запуск из корня проекта:
    python scripts/cbr_build.py --csv scripts/charon_data_cache/BTCUSDT_1h_700d.csv
    python scripts/cbr_build.py --symbol BTCUSDT --timeframe 1H --fetch

Прогревные бары (NaN в фичах из-за окон индикаторов) пропускаются; фичи
каждого бара хранятся ровно в порядке labels.charon_features (см. check в
store.FEATURE_NAMES).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_BASE_DIR = Path(__file__).resolve().parent.parent
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

from app_pkg import config
from app_pkg.cbr import backfill, query_api, schema, store
from app_pkg.cbr import index as cbr_index
from app_pkg.cbr import normalize as cbr_norm
from app_pkg.ml import labels


def build_stage2_artifacts(conn, symbol, timeframe):
    """Пересборка артефактов Этапа 2 для (symbol, timeframe).

    stats  = rolling-нормализация размеченных фич (data/cbr_norm_stats.pkl);
    index  = FAISS HNSW32 по нормализованным векторам (data/cbr.faiss).

    Возвращает {"stats_path": str, "index_path": str, "rows": int}.
    """
    import datetime
    symbol = str(symbol).upper()
    timeframe = str(timeframe).strip()
    X, ts, _regimes, _y = cbr_index.load_feature_matrix(conn, symbol, timeframe)
    stats = cbr_norm.compute_rolling_stats(X, ts=ts)
    stats_path = str(config.CBR_NORMALIZE_STATS_PATH)
    index_path = str(config.CBR_FAISS_INDEX_PATH)
    cbr_norm.save_stats(stats, stats_path)
    Xn = cbr_norm.normalize_matrix(X, ts, stats)
    index = cbr_index.build_faiss_index(Xn)
    cbr_index.save_index(index, index_path)
    print(f"[cbr_build stage2] {symbol} {timeframe}: rows={len(X)} "
          f"index={index_path} stats={stats_path} "
          f"built_at={datetime.datetime.now(datetime.timezone.utc).isoformat()}")
    return {"stats_path": stats_path, "index_path": index_path,
            "rows": len(X)}


def load_candle_df(csv_path=None, symbol=None, timeframe=None):
    """DataFrame свечей: CSV или живой фетч (get_series_df)."""
    if csv_path:
        df = pd.read_csv(csv_path)
        df.columns = [str(col).strip() for col in df.columns]
        return df
    if not symbol or not timeframe:
        raise SystemExit("--csv или --symbol+--timeframe (обязательно)")
    from app_pkg.data.fetch import get_series_df
    dsymbol = str(symbol).upper()
    tf = str(timeframe).strip()
    df = get_series_df(dsymbol, tf, limit=config.HISTORY_LIMITS.get(
        tf, config.MAX_DATA_LIMIT), history_limit=config.MAX_DATA_LIMIT)
    df = df.reset_index(drop=True)
    df["symbol"] = dsymbol
    df["timeframe"] = tf
    return df


def _bar_ts_sec(ts):
    """Значение timestamp -> Unix-секунды (tz-safe)."""
    ts = pd.to_datetime(ts)
    if getattr(ts, "tz", None) is None:
        ts = ts.tz_localize("UTC")
    return int(ts.timestamp())


def _first_finite_index(features):
    """Первый индекс строки с ВСЕМИ конечными фичами (прогрев пропускаем)."""
    arr = features.to_numpy(dtype="float64")
    valid = np.isfinite(arr)
    good = valid.all(axis=1)
    idx = int(np.argmax(good)) if good.any() else len(arr)
    return idx if good[idx] else len(arr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Запись + backfill CBR-базы (Этап 1+), отчёт по прогону.")
    parser.add_argument("--csv", default=None, help="путь к CSV OHLCV")
    parser.add_argument("--symbol", default="BTCUSDT", help="символ (live-fetch)")
    parser.add_argument("--timeframe", default="1H", help="таймфрейм")
    parser.add_argument("--fetch", action="store_true",
                        help="тянуть данные с биржи вместо CSV")
    parser.add_argument("--db", default=None, help="путь к CBR-БД "
                        "(по умолчанию config.CBR_DB_PATH)")
    parser.add_argument("--horizon", type=int,
                        default=config.CBR_BACKFILL_HORIZON)
    parser.add_argument("--k", type=float, dest="atr_k", default=2.5,
                        help="множитель ATR для барьеров (default: 2.5)")
    parser.add_argument("--source", default="build",
                        help="source хранимых снимков (default: build)")
    parser.add_argument("--max-bars", type=int, default=0,
                        help="ограничить число последних баров (0 = все; "
                             "ускоряет демо на больших датасетах)")
    parser.add_argument("--quiet", action="store_true", help="без прогресса")
    parser.add_argument("--artifacts", action="store_true",
                        help="только пересборка артефактов Этапа 2 "
                             "(индекс+статистика, данные не трогаем)")
    args = parser.parse_args()

    db_path = args.db or config.CBR_DB_PATH
    conn = schema.init_db(db_path)

    symbol = str(args.symbol).upper()
    timeframe = str(args.timeframe).strip()

    if args.artifacts:
        build_stage2_artifacts(conn, symbol, timeframe)
        return

    df = load_candle_df(args.csv, symbol if args.fetch else args.symbol,
                        timeframe if args.fetch else None)
    if df is None or df.empty:
        raise SystemExit(f"Нет данных: csv={args.csv} symbol={symbol} "
                         f"tf={args.fetch and timeframe or '-'}")
    if "timestamp" not in df.columns or "high" not in df.columns:
        raise SystemExit("CSV должен содержать колонки "
                         "timestamp,open,high,low,close,volume")
    if args.csv:
        symbol = str(df["symbol"][0] if "symbol" in df.columns else symbol)
        symbol = symbol.upper()
    if args.max_bars > 0:
        df = df.tail(args.max_bars).reset_index(drop=True)

    n = len(df)
    closes = df["close"].astype(float).to_numpy(dtype="float64")
    bars = [{
        "ts": _bar_ts_sec(row["timestamp"]),
        "open": float(row["open"]), "high": float(row["high"]),
        "low": float(row["low"]), "close": float(row["close"]),
    } for _, row in df.iterrows()]

    def price_fn(sym, tf, from_ts, to_ts):
        out = [b for b in bars if b["ts"] >= from_ts
               and (to_ts is None or b["ts"] <= to_ts)]
        return out

    features = labels.charon_features(df, symbol=symbol)
    start = _first_finite_index(features)
    if not args.quiet:
        print(f"[cbr_build] {symbol} {timeframe}: {n} баров, "
              f"warmup={start} (NaN-фичи пропущены)")

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
            "source": args.source, **row,
        }, entry_price=entry)
        if rid is None:
            dup += 1
        else:
            stored += 1
    if not args.quiet:
        print(f"[cbr_build] записано={stored} дубликатов={dup}")

    conn.execute(
        "SELECT COUNT(*) c FROM snapshots WHERE symbol=? AND timeframe=? "
        "AND outcome IS NULL", (symbol, timeframe)).fetchone()["c"]
    backfilled = backfill.backfill_outcomes(
        conn, symbol, timeframe, price_fn, horizon=args.horizon,
        atr_k=args.atr_k)
    pending_after = conn.execute(
        "SELECT COUNT(*) c FROM snapshots WHERE symbol=? AND timeframe=? "
        "AND outcome IS NULL", (symbol, timeframe)).fetchone()["c"]

    mix = {r["outcome"]: int(r["c"]) for r in conn.execute(
        "SELECT outcome, COUNT(*) c FROM snapshots WHERE symbol=? "
        "AND timeframe=? AND outcome IS NOT NULL GROUP BY outcome",
        (symbol, timeframe))}
    agg = conn.execute(
        "SELECT AVG(pnl_pct) p, AVG(bars_to_outcome) b FROM snapshots "
        "WHERE symbol=? AND timeframe=? AND outcome IS NOT NULL",
        (symbol, timeframe)).fetchone()

    stats = query_api.get_stats_for_llm(
        conn, symbol, timeframe, window_days=99999,
        min_samples=1, max_rows=3)

    print()
    print("=== CBR build report ===")
    print(f"Symbol/Timeframe: {symbol} {timeframe}")
    print(f"DB: {db_path}")
    print(f"Bars processed: {n}")
    print(f"Stored (source={args.source}): {stored}")
    print(f"Deduplicated: {dup}")
    print(f"Backfilled: {backfilled} "
          f"(осталось pending: {max(0, pending_after)})")
    print(f"Outcome balance: +1: {mix.get(1, 0)} | -1: {mix.get(-1, 0)} "
          f"| 0: {mix.get(0, 0)}")
    if agg and agg["p"] is not None:
        print(f"Avg pnl_pct: {round(float(agg['p']), 4)}%")
        print(f"Avg bars_to_outcome: {round(float(agg['b']), 2)}")
    if isinstance(stats, dict) and not stats.get("insufficient_data"):
        f = query_api.format_for_prompt(stats, max_chars=100000)
        print(f"Prompt block ({len(f)} chars): {f}")
    print()


if __name__ == "__main__":
    main()