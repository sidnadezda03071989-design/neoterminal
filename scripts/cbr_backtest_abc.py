"""A/B/C-бэктест CBR Этапа 2: Charon-only vs CBR-only vs blended.

Сравнение трёх двигателей на ОДНОМ и том же окне истории:
  A = Charon-only  (offline_charon_verdict по ЛОКАЛЬНЫМ 44 фичам правил 4-8/11,
                    без LLM/сети — детерминированный аналог промпта);
  B = CBR-only     (сигнал k-NN: winrate_up/winrate_down соседей, FAISS Этапа 2);
  C = blended      (blender.blend = взвешенное смешивание A и B по режимам).

Каждый бар окна -> сигнал двигателя -> (L|S|F) -> simulate_trade на OHLC CSV:
  SL = max(10%, 2*atr_pct), TP = 20%, горизонт 20 баров, SL-приоритет касаний,
  комиссия 0.05%/сторона (0.10 п.п. всего). FLAT -> сделки нет.

Метрики: сделки/винрейт/avg_pnl/profit_factor/sharpe(252)/max_dd/coverage,
плюс разбивка по режиму сигнального бара. Критерий приёмки по ТЗ:
  C обязан превзойти A минимум на +3 п.п. winrate ИЛИ +0.3 Sharpe.

Данные:
  DB  — features/режимы те же, что легли в FAISS-индекс (data/cbr.db);
  CSV — OHLCV для цены входа/барьеров (scripts/charon_data_cache/...).

Артефакты: reports/cbr_backtest_abc.json + reports/cbr_backtest_abc.csv
Запуск:     python scripts/cbr_backtest_abc.py
"""

import argparse
import csv
import datetime
import json
import math
import sys
from pathlib import Path

import numpy as np

_BASE_DIR = Path(__file__).resolve().parent.parent
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

from app_pkg import config
from app_pkg.cbr import (
    blender,
    query_api,
    schema,
    store,
)
from app_pkg.cbr import index as cbr_index
from app_pkg.cbr import normalize as cbr_norm

REPORTS_DIR = _BASE_DIR / "reports"
DEFAULT_CSV = _BASE_DIR / "scripts" / "charon_data_cache" / "BTCUSDT_1h_700d.csv"

TRADE_HORIZON = 20        # баров (как backfill)
TP_PCT = 0.20             # тейк-профит 20%
SL_FLOOR = 0.10           # нижняя граница стопа
SL_ATR_K = 2.0            # множитель ATR для стопа
FEE_PP = 0.10             # 0.05%/сторона = 0.10 п.п. всего
SIG_SPREAD = 0.15         # порог разницы вероятностей для направления


def _dt_from_ts(ts):
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc)


def load_csv(csv_path):
    """CSV OHLCV -> сортированные массивы (ts, open, high, low, close) + индекс."""
    ts_list, o, h, l, c = [], [], [], [], []
    with open(str(csv_path), encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        pos = {name: i for i, name in enumerate(h.strip() for h in header)}
        for row in reader:
            if len(row) < 6:
                continue
            raw = row[pos["timestamp"]].strip()
            try:
                dt = datetime.datetime.fromisoformat(raw.replace("+00:00", "+00:00"))
                t = int(dt.timestamp())
            except ValueError:
                continue
            ts_list.append(t)
            o.append(float(row[pos["open"]]))
            h.append(float(row[pos["high"]]))
            l.append(float(row[pos["low"]]))
            c.append(float(row[pos["close"]]))
    idx = np.searchsorted(np.asarray(ts_list, dtype="int64"), ts_list)
    order = np.argsort(np.asarray(ts_list, dtype="int64"))
    ts = np.asarray(ts_list, dtype="int64")[order]
    o = np.asarray(o)[order]
    h = np.asarray(h)[order]
    l = np.asarray(l)[order]
    c = np.asarray(c)[order]
    pos_map = {int(t): int(i) for i, t in enumerate(ts)}
    return ts, o, h, l, c, pos_map


def load_db(conn, symbol, timeframe):
    """(ts -> (features_arr, regime)) для размеченных строк (ts ASC)."""
    rows = conn.execute(
        "SELECT ts, features, regime FROM snapshots "
        "WHERE symbol=? AND timeframe=? AND outcome IS NOT NULL "
        "ORDER BY ts ASC", (symbol, timeframe)).fetchall()
    out = {}
    for r in rows:
        vec = cbr_index._decode_features(r["features"])
        if vec is None:
            continue
        out[int(r["ts"])] = (vec, r["regime"])
    return out


def clamp01(v):
    v = float(v)
    return 0.0 if math.isnan(v) else max(0.0, min(1.0, v))


def offline_charon_verdict(fv):
    """Детерминированный Charon по локальным 44 фичам (правила 4-8, 11).

    Поля берутся из плоского словаря FEATURE_NAMES (те же числа, что видит
    сеть). Отсутствующее поле = no data = правило игнорируется (v3).
    Вероятности нормализуются к сумме 1 (контракт промпта "sum=1.00").
    """
    def f(name):
        return fv.get(name)

    pu, pd = 0.0, 0.0
    conf, k = 1.0, 1.5

    rv = f("volume.rel_vol")
    vz = f("volume.vol_zscore")
    if rv is not None and vz is not None and rv > 1.5 and vz > 1.0:
        pu += 0.1
    if rv is not None and rv < 0.5:
        conf -= 0.1

    adx = f("trend.adx")
    pdi = f("trend.plus_di")
    mdi = f("trend.minus_di")
    if adx is not None and adx > 25.0:
        if pdi is not None and mdi is not None:
            if pdi > mdi:
                pu += 0.1
            elif mdi > pdi:
                pd += 0.1
    ds = f("trend.di_spread")
    if ds is not None and ds > 10.0:
        k *= 1.2

    mh = f("momentum.macd_hist")
    mhs = f("momentum.macd_hist_slope")
    if mh is not None and mhs is not None:
        if mh > 0.0 and mhs > 0.0:
            pu += 0.1
        elif mh < 0.0 and mhs < 0.0:
            pd += 0.1

    ap = f("technicals.atr_percentile")
    bwp = f("volatility.bb_width_pct")
    hv = f("volatility.hv20")
    if ap is not None and ap > 0.8:
        k *= 1.5
        conf -= 0.1
    if bwp is not None and bwp > 0.8:
        k *= 1.3
    if hv is not None and hv > 50.0:
        k *= 1.5

    hurst = f("regime.hurst")
    erf = f("regime.efficiency_ratio")
    ac1 = f("regime.autocorr_lag1")
    if hurst is not None and erf is not None and hurst > 0.5 and erf > 0.5:
        k *= 1.3
    if hurst is not None and ac1 is not None and hurst < 0.5 and ac1 < 0:
        conf -= 0.1

    market_open = f("clock.market_open")
    flat_forced = market_open is not None and float(market_open) == 0.0

    # Нормализация к сумме 1 (контракт промпта).
    total = max(pu + pd, 0.0)
    if total > 1.0:
        pu /= total
        pd /= total
        pf = 0.0
    else:
        pf = 1.0 - total
    if flat_forced:
        sig = "F"
    elif pu - pd > SIG_SPREAD:
        sig = "L"
    elif pd - pu > SIG_SPREAD:
        sig = "S"
    else:
        sig = "F"
    return {"pu": clamp01(pu), "pd": clamp01(pd), "pf": clamp01(pf),
            "sig": sig, "conf": clamp01(conf), "k": k}


def snap_from_features(vec, symbol, timeframe, ts):
    """flat-снимок по вектору (схема store.snapshot_to_vector) + ts/указатели."""
    snap = {name: float(v) for name, v in zip(store.FEATURE_NAMES, vec)}
    snap["symbol"] = symbol
    snap["timeframe"] = timeframe
    snap["ts"] = int(ts)
    return snap


def direction(p_up, p_down):
    """Направление из (вероятность вверх, вниз) с порогом SIG_SPREAD."""
    if p_up - p_down > SIG_SPREAD:
        return "L"
    if p_down - p_up > SIG_SPREAD:
        return "S"
    return "F"


def simulate_trade(ts_arr, o_arr, h_arr, l_arr, c_arr, pos_map, ts_q,
                   direction, atr_pct):
    """PnL сделки (в % после комиссии) или None (нет сделки/данных).

    Вход по close сигнального бара. SL/TP по правилам ТЗ, SL-приоритет при
    касании обоих на одном баре (конвенция backfill). Таймаут -> close
    последнего бара окна. None при direction=F или нехватке будущих баров.
    """
    if direction == "F":
        return None
    j0 = pos_map.get(int(ts_q))
    if j0 is None:
        return None
    entry = float(c_arr[j0])
    if entry <= 0:
        return None
    sl_pct = max(SL_FLOOR, SL_ATR_K * (atr_pct if atr_pct is not None
                                       else SL_FLOOR))
    tp_pct = TP_PCT
    lower = entry * (1.0 - sl_pct)
    upper = entry * (1.0 + tp_pct)
    j_end = min(j0 + TRADE_HORIZON, len(c_arr))
    if j_end <= j0 + 1:
        return None
    exit_price = None
    for j in range(j0 + 1, j_end):
        if l_arr[j] <= lower:
            exit_price = lower
            break
        if h_arr[j] >= upper:
            exit_price = upper
            break
    if exit_price is None:
        exit_price = float(c_arr[j_end - 1])
    sign = 1.0 if direction == "L" else -1.0
    return round(sign * (exit_price - entry) / entry * 100.0 - FEE_PP, 4)


def metrics(pnl_list):
    """Метрики двигателя по списку PnL сделок (None = сделки не было)."""
    pnl = np.asarray([p for p in pnl_list if p is not None], dtype="float64")
    base = {"trades": len(pnl)}
    if len(pnl) == 0:
        base.update({"winrate": None, "avg_pnl_pct": None, "profit_factor": None,
                     "sharpe": None, "max_dd": None, "sum_pnl_pct": 0.0})
        return base
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_w = float(wins.sum())
    gross_l = float(abs(losses.sum()))
    pf = gross_w / gross_l if gross_l > 0 else (
        float("inf") if gross_w > 0 else 0.0)
    eq = np.cumsum(pnl)
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    sharpe = (float(pnl.mean()) / float(pnl.std()) * math.sqrt(252.0)
              if pnl.std() > 1e-12 else 0.0)
    base.update({
        "winrate": round(float(np.mean(pnl > 0)), 4),
        "avg_pnl_pct": round(float(pnl.mean()), 4),
        "profit_factor": round(pf, 4) if np.isfinite(pf) else None,
        "sharpe": round(sharpe, 4),
        "max_dd": round(float(dd.min()), 4),
        "sum_pnl_pct": round(float(pnl.sum()), 4),
    })
    return base


def _cold(rows):
    """cold-по последнему n кэша (если надо — тут не используется)."""
    return


def run_backtest(conn, db_rows, csv_name, K, min_distance, regime_match,
                 window_days, start_fraction=0.70, max_bars=0):
    """Прогон A/B/C. Возвращает (rows, engines, by_regime, stats)."""
    ts_arr, o_arr, h_arr, l_arr, c_arr, pos_map = csv_name
    index = cbr_index.load_index(str(config.CBR_FAISS_INDEX_PATH))
    stats = cbr_norm.load_stats(str(config.CBR_NORMALIZE_STATS_PATH))
    if index is None or not (isinstance(stats, dict) and "mean" in stats):
        raise SystemExit("Нет артефактов Этапа 2: запустите сначала "
                         "python scripts/cbr_build.py --artifacts")

    all_ts = sorted(db_rows)
    n = len(all_ts)
    if n == 0:
        raise SystemExit("Пусто в DB-окне для бэктеста")
    start_pos = int(n * start_fraction)
    query_ts = all_ts[start_pos:]
    if max_bars > 0:
        query_ts = query_ts[-int(max_bars):]
    rows = []
    for ts in query_ts:
        vec, regime = db_rows[ts]
        atr_pct = float(vec[store.ATR_INDEX]) / float(c_arr[pos_map[ts]]) \
            if pos_map.get(ts) is not None and c_arr[pos_map[ts]] > 0 else 0.0
        charon = offline_charon_verdict(
            {name: float(v) for name, v in zip(store.FEATURE_NAMES, vec)})
        snap = snap_from_features(vec, "BTCUSDT", "1H", ts)
        cbr = query_api.get_similar_summary(
            conn, index, snap, "BTCUSDT", "1H", bar_ts=int(ts),
            K=K, min_distance=min_distance,
            regime_match=regime_match, window_days=window_days, stats=stats)
        insuff = not isinstance(cbr, dict) or cbr.get("insufficient_data")
        n_cbr = 0 if insuff else int(cbr.get("n", 0))
        up_c = 0.0 if insuff else float(cbr.get("winrate_up", 0.0))
        dn_c = 0.0 if insuff else float(cbr.get("winrate_down", 0.0))
        conf_c = 0.0 if insuff else float(cbr.get("confidence", 0.0))
        blend = blender.blend(charon["pu"], charon["pd"],
                              None if insuff else cbr, regime)
        dA = charon["sig"]
        dB = direction(up_c, dn_c)
        dC = direction(blend["final_pu"], blend["final_pd"])
        srcA, srcB, srcC = "charon_only", "cbr_only", blend["source"]
        rows.append({
            "ts": ts, "regime": regime,
            "A_charon": simulate_trade(
                ts_arr, o_arr, h_arr, l_arr, c_arr, pos_map, ts, dA, atr_pct),
            "B_cbr": simulate_trade(
                ts_arr, o_arr, h_arr, l_arr, c_arr, pos_map, ts, dB, atr_pct),
            "C_blend": simulate_trade(
                ts_arr, o_arr, h_arr, l_arr, c_arr, pos_map, ts, dC, atr_pct),
            "A_sig": dA, "B_sig": dB, "C_sig": dC,
            "srcA": srcA, "srcB": srcB, "srcC": srcC,
            "n": n_cbr, "conf": round(conf_c, 4),
        })

    engines = {
        "A_charon": metrics([r["A_charon"] for r in rows]),
        "B_cbr": metrics([r["B_cbr"] for r in rows]),
        "C_blended": metrics([r["C_blend"] for r in rows]),
    }
    coverage = {"A_charon": _coverage(rows, "A_charon"),
                "B_cbr": _coverage(rows, "B_cbr"),
                "C_blended": _coverage(rows, "C_blend")}
    by_regime = {}
    for regime in sorted(set(r["regime"] for r in rows)):
        sub = [r for r in rows if r["regime"] == regime]
        by_regime[regime] = {
            "A_charon": metrics([r["A_charon"] for r in sub]),
            "B_cbr": metrics([r["B_cbr"] for r in sub]),
            "C_blended": metrics([r["C_blend"] for r in sub]),
        }
    info = {"K": K, "min_distance": min_distance,
            "regime_match": regime_match, "window_days": window_days,
            "query_bars": len(rows), "start_fraction": start_fraction}
    return rows, engines, by_regime, coverage, info


def _coverage(rows, key):
    total = len(rows)
    traded = sum(1 for r in rows if r.get(key) is not None)
    return round(traded / total, 4) if total else 0.0


def build_verdict(engines):
    """Критерий ТЗ: C против A (+3 п.п. winrate ИЛИ +0.3 Sharpe)."""
    a, c = engines["A_charon"], engines["C_blended"]
    d_wr = (c.get("winrate") or 0.0) - (a.get("winrate") or 0.0)
    d_sh = ((c.get("sharpe") or 0.0) - (a.get("sharpe") or 0.0)
            if a.get("sharpe") is not None and c.get("sharpe") is not None
            else 0.0)
    any_winrate = a.get("winrate") is not None and c.get("winrate") is not None
    is_pass = any_winrate and (d_wr >= 0.03 or d_sh >= 0.3)
    note = ("C превосходит A: winrate+%.2f, sharpe+%.2f"
            % (d_wr, d_sh))
    if not any_winrate:
        note += " (A или C без сделок — метрика не определена)"
    return {
        "is_pass": bool(is_pass),
        "delta_winrate": round(d_wr, 4),
        "delta_sharpe": round(d_sh, 4),
        "note": note,
    }


def main():
    parser = argparse.ArgumentParser(
        description="A/B/C-бэктест CBR Этапа 2 (Charon vs CBR vs blended).")
    parser.add_argument("--db", default=str(config.CBR_DB_PATH))
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", default="1H")
    parser.add_argument("--K", type=int, default=config.CBR_K_DEFAULT)
    parser.add_argument("--min-distance", type=float, dest="min_distance",
                        default=config.CBR_MIN_DISTANCE)
    parser.add_argument("--no-regime-match", action="store_false",
                        dest="regime_match", default=config.CBR_REGIME_MATCH)
    parser.add_argument("--window-days", type=int, dest="window_days",
                        default=config.CBR_QUERY_WINDOW_DAYS)
    parser.add_argument("--start-fraction", type=float, dest="start_fraction",
                        default=0.70)
    parser.add_argument("--max-bars", type=int, default=0,
                        help="ограничить число тестируемых баров (ускорение)")
    args = parser.parse_args()

    conn = schema.init_db(args.db)
    sym = args.symbol.upper()
    db_rows = load_db(conn, sym, args.timeframe)
    csv_data = load_csv(args.csv)

    rows, engines, by_regime, coverage, info = run_backtest(
        conn, db_rows, csv_data, args.K, args.min_distance,
        args.regime_match, args.window_days, args.start_fraction,
        args.max_bars)
    verdict = build_verdict(engines)

    report = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "symbol": sym, "timeframe": args.timeframe,
        "db": str(args.db), "csv": str(args.csv),
        **info, "engines": engines,
        "coverage": coverage, "by_regime": by_regime,
        "verdict": verdict,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / "cbr_backtest_abc.json"
    csv_path = REPORTS_DIR / "cbr_backtest_abc.csv"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ts", "regime", "A_sig", "A_pnl", "B_sig", "B_pnl",
                    "C_sig", "C_pnl", "srcA", "srcB", "srcC", "n", "conf"])
        for r in rows:
            w.writerow([r["ts"], r["regime"], r["A_sig"], r["A_charon"],
                        r["B_sig"], r["B_cbr"], r["C_sig"], r["C_blend"],
                        r["srcA"], r["srcB"], r["srcC"], r["n"], r["conf"]])
    print("=== CBR A/B/C backtest ===")
    print(f"{sym} {args.timeframe}: query_bars={info['query_bars']} "
          f"K={args.K} min_dist={args.min_distance} "
          f"regime_match={args.regime_match} window={args.window_days}d")
    for name in ("A_charon", "B_cbr", "C_blended"):
        m = engines[name]
        print(f"{name:9s} trades={m['trades']:5d} wr={m['winrate']} "
              f"sharpe={m['sharpe']} avg_pnl={m['avg_pnl_pct']} "
              f"pf={m['profit_factor']} dd={m['max_dd']} "
              f"cov={coverage[name]}")
    print("Verdict:", verdict["note"], "| PASS" if verdict["is_pass"]
          else "| FAIL")
    print(f"Saved: {json_path} / {csv_path}")


if __name__ == "__main__":
    main()