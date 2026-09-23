"""Асинхронная разметка outcome по triple-barrier (конвенция labels.py).

backfill_outcomes берёт PENDING-строки таблицы snapshots (outcome IS NULL)
и выставляет метку {-1, 0, +1} по будущим барам за окно t+1..t+horizon.

Конвенция совпадает с labels.triple_barrier_labels и бэктестером:
  upper = entry*(1 + atr_k*atr_pct), lower = entry*(1 - atr_k*atr_pct);
  первое касание в окне определяет исход, ПРИОРИТЕТ НИЖНЕГО барьера (обе
  линии на одном баре -> outcome = -1, тот же порядок, что в _exit_on_bar);
  ни одного касания за horizon -> outcome = 0 (timeout), exit = close
  последнего бара окна.
pnl_pct — прибыль по направлению вердикта: sign(L)=+1, sign(S)=-1, иначе 0;
exit_price = задетый барьер (или close при таймауте). Экскурсии max_up/
max_dn считаются по ВСЕМ барам окна (в % от entry).

Идемпотентность: UPDATE только строк с outcome IS NULL; повторный запуск
не затирает уже размеченные строки. Строки, для которых не хватает
horizon будущих баров, пропускаются (остаются pending).
"""

import bisect
import logging

import numpy as np

from app_pkg import utils

log = logging.getLogger(__name__)


def _bars_arrays(price_rows):
    """[(ts, open, high, low, close), ...] -> сортированные numpy-массивы."""
    rows = sorted(price_rows or [], key=lambda r: int(r.get("ts", 0)))
    if not rows:
        return None
    ts = np.array([int(r["ts"]) for r in rows], dtype="int64")
    high = np.array([float(r["high"]) for r in rows], dtype="float64")
    low = np.array([float(r["low"]) for r in rows], dtype="float64")
    close = np.array([float(r["close"]) for r in rows], dtype="float64")
    return ts, high, low, close


def backfill_outcomes(conn, symbol, timeframe, price_fn, horizon=20,
                      atr_k=1.5):
    """Размечает outcome для pending-строк (symbol, timeframe).

    price_fn(symbol, timeframe, from_ts, to_ts) -> iterable dict'ов
      {"ts": int, "open": float, "high": float, "low": float, "close": float}
    — вызывается ОДИН раз с (symbol, timeframe, min_pending_ts, None).

    Возвращает число обновлённых строк. Кидает ValueError при вырожденном
    atr_pct (<=0) у pending-строки: барьер из таймаута нельзя построить.
    """
    symbol = str(symbol or "").upper()
    timeframe = str(timeframe or "").strip()
    horizon = max(1, int(horizon or 20))
    atr_k = float(atr_k)
    if atr_k <= 0:
        raise ValueError("atr_k must be > 0")

    rows = conn.execute(
        "SELECT id, ts, entry_price, atr_pct, verdict FROM snapshots "
        "WHERE symbol=? AND timeframe=? AND outcome IS NULL "
        "  AND entry_price IS NOT NULL AND atr_pct IS NOT NULL "
        "ORDER BY ts ASC",
        (symbol, timeframe),
    ).fetchall()
    if not rows:
        return 0

    min_ts = int(rows[0]["ts"])
    bars = price_fn(symbol, timeframe, min_ts, None)
    arrays = _bars_arrays(bars)
    if arrays is None:
        log.warning("cbr backfill %s %s: price_fn вернул пусто", symbol,
                    timeframe)
        return 0
    ts_arr, high_arr, low_arr, close_arr = arrays

    now = utils.now_sec()
    updated = 0
    for row in rows:
        entry = float(row["entry_price"])
        atr_pct = float(row["atr_pct"])
        if entry <= 0 or atr_pct <= 0:
            raise ValueError(
                "cannot backfill: degenerate atr_pct on ts={} ({} {})".format(int(row["ts"]), symbol, timeframe))
        upper = entry * (1.0 + atr_k * atr_pct)
        lower = entry * (1.0 - atr_k * atr_pct)

        j0 = bisect.bisect_right(ts_arr, int(row["ts"]))
        j_end = j0 + horizon
        if j_end > len(ts_arr):
            continue  # не хватает будущих баров — строка остаётся pending

        win_high = high_arr[j0:j_end]
        win_low = low_arr[j0:j_end]
        win_close = close_arr[j0:j_end]

        outcome, hit_j, exit_price = None, None, None
        for k in range(horizon):
            # SL-приоритет: если на баре задеты обе линии — outcome = -1
            if win_low[k] <= lower:
                outcome, hit_j, exit_price = -1, k + 1, lower
                break
            if win_high[k] >= upper:
                outcome, hit_j, exit_price = 1, k + 1, upper
                break
        if outcome is None:  # таймаут: вертикальный барьер
            outcome, hit_j, exit_price = 0, horizon, float(win_close[-1])

        max_up = float(np.max((win_high - entry) / entry * 100.0)) \
            if len(win_high) else 0.0
        max_dn = float(np.min((win_low - entry) / entry * 100.0)) \
            if len(win_low) else 0.0

        sig = str(row["verdict"] or "").strip().upper()
        sign = 1 if sig == "L" else (-1 if sig == "S" else 0)
        pnl_pct = sign * (exit_price - entry) / entry * 100.0

        conn.execute(
            "UPDATE snapshots SET outcome=?, bars_to_outcome=?, exit_price=?, "
            " max_up=?, max_dn=?, pnl_pct=?, backfilled_at=? "
            "WHERE id=? AND outcome IS NULL",
            (outcome, hit_j, round(exit_price, 8), round(max_up, 4),
             round(max_dn, 4), round(pnl_pct, 4), now, row["id"]),
        )
        updated += 1
    conn.commit()
    return updated