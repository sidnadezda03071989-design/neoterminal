"""Token-diet API: статистика CBR-базы для промпта LLM вместо сырых чисел.

Три публичных входа:
  get_stats_for_llm(conn, symbol, timeframe, regime=None, ...) -> dict
      — статистика outcome по окну ts >= MAX(ts) - window_days (окно
        относительно ПОСЛЕДНЕГО бара базы, а не системных часов: база
        могла строиться по истории). Схема ответа ФИКСИРОВАНА (см. ниже):
        никаких сырых векторов, всё в %, 2 знака, максимум max_rows
        примеров. Если размеченных строк в окне меньше min_samples ->
        возвращается ровно {"n": N, "insufficient_data": True}.
      by_regime (сводка по режимам) включается ТОЛЬКО когда regime=None;
      при заданном regime выборка фильтруется этим режимом.
  get_similar_summary(conn, index, current_snapshot, symbol, tf,
                      bar_ts, ...) -> dict
      — Этап 2: настоящий k-NN по FAISS-индексу нормализованных фич.
        bar_ts ОБЯЗАТЕЛЕН: в соседи попадают только бары, чей outcome
        заведомо известен на момент бара запроса (ts_соседа + horizon <
        bar_ts) — без него k-NN «протекает» из будущего (look-ahead).
        (См. докстринг функции.) Старая сигнатура Этапа 1
        (conn, symbol, timeframe, current_snapshot, window_days=...) тоже
        поддерживается: без переданного index она делегирует в
        get_stats_for_llm по классифицированному режиму.
  format_for_prompt(stats, max_chars=250) -> компактная строка <= max_chars
      (целевая длина ~250 символов) — готовый блок CBR для user-сообщения.
      Понимает И старую схему get_stats_for_llm, И новую схему
      get_similar_summary (winrate_*). Схемы покрыты тестами на < 300 символов.
"""

import json
import logging

import numpy as np

from app_pkg import config
from app_pkg.cbr import index as cbr_index
from app_pkg.cbr import normalize as cbr_norm
from app_pkg.cbr.store import classify_regime, snapshot_to_vector

log = logging.getLogger(__name__)

# Схема ответа get_stats_for_llm (фиксирована; тесты сверяют ключи).
_STATS_KEYS = ("symbol", "timeframe", "n", "window_days", "regime", "ratio",
               "avg_pnl", "avg_bars", "max_up_avg", "max_dn_avg",
               "by_regime", "samples")

# Схема ответа get_similar_summary (Этап 2, фиксирована; тесты сверяют ключи).
_KNN_KEYS = ("symbol", "timeframe", "n", "n_filled", "winrate_up",
             "winrate_down", "winrate_flat", "avg_pnl_pct",
             "median_bars_to_outcome", "confidence", "avg_distance",
             "regime_match_rate", "regime", "min_distance", "K",
             "window_days", "recent_examples")


def _r2(v):
    return None if v is None else round(float(v), 2)


def _insufficient(symbol, timeframe, n):
    """Минимальный ответ при недостатке данных (ровно эти ключи, см. ТЗ)."""
    return {"n": int(n), "insufficient_data": True}


def get_stats_for_llm(conn, symbol, timeframe, regime=None, window_days=90,
                      min_samples=10, max_rows=3):
    """Статистика CBR-базы в промпт (см. докстринг модуля)."""
    symbol = str(symbol or "").upper()
    timeframe = str(timeframe or "").strip()
    window_sec = max(1, int(window_days or 90)) * 86400
    max_rows = max(1, int(max_rows or 3))

    max_row = conn.execute(
        "SELECT MAX(ts) AS m FROM snapshots WHERE symbol=? AND timeframe=?",
        (symbol, timeframe)).fetchone()
    if max_row is None or max_row["m"] is None:
        return _insufficient(symbol, timeframe, 0)
    lo = int(max_row["m"]) - window_sec

    where = ["symbol=? AND timeframe=? AND outcome IS NOT NULL AND ts>=?"]
    params = [symbol, timeframe, lo]
    if regime is not None:
        where.append("regime=?")
        params.append(str(regime))

    out = {}
    cur = conn.execute(
        f"SELECT outcome, COUNT(*) AS c FROM snapshots WHERE "
        f"{' AND '.join(where)} GROUP BY outcome", params).fetchall()
    n = int(sum(r["c"] for r in cur))
    if n < min_samples or n <= 0:
        return _insufficient(symbol, timeframe, n)

    counts = {int(r["outcome"]): int(r["c"]) for r in cur}
    ratio = {
        "up": _r2(counts.get(1, 0) / n),
        "down": _r2(counts.get(-1, 0) / n),
        "flat": _r2(counts.get(0, 0) / n),
    }

    agg = conn.execute(
        f"SELECT AVG(pnl_pct) AS p, AVG(bars_to_outcome) AS b, "
        f"AVG(max_up) AS u, AVG(max_dn) AS d FROM snapshots WHERE "
        f"{' AND '.join(where)}", params).fetchone()

    samples = conn.execute(
        f"SELECT ts, outcome, pnl_pct, bars_to_outcome, entry_price, "
        f"exit_price, regime, session FROM snapshots WHERE "
        f"{' AND '.join(where)} ORDER BY ts DESC, id DESC LIMIT ?",
        params + [max_rows]).fetchall()
    sample_rows = [{
        "ts": int(s["ts"]),
        "outcome": int(s["outcome"]),
        "pnl_pct": _r2(s["pnl_pct"]),
        "bars_to_outcome": int(s["bars_to_outcome"]) if s["bars_to_outcome"] is not None else None,
        "entry_price": _r2(s["entry_price"]),
        "exit_price": _r2(s["exit_price"]),
        "regime": s["regime"],
        "session": s["session"],
    } for s in samples]

    out = {
        "symbol": symbol,
        "timeframe": timeframe,
        "n": n,
        "window_days": int(window_days),
        "regime": str(regime) if regime is not None else None,
        "ratio": ratio,
        "avg_pnl": _r2(agg["p"]),
        "avg_bars": _r2(agg["b"]),
        "max_up_avg": _r2(agg["u"]),
        "max_dn_avg": _r2(agg["d"]),
        "samples": sample_rows,
    }

    if regime is None:
        reg_rows = conn.execute(
            f"SELECT regime, COUNT(*) AS c, AVG(pnl_pct) AS p FROM snapshots "
            f"WHERE {' AND '.join(where)} AND regime IS NOT NULL "
            f"GROUP BY regime", params).fetchall()
        by_regime = {}
        for r in reg_rows:
            rc = int(r["c"])
            by_regime[r["regime"]] = {
                "n": rc,
                "share": _r2(rc / n),
                "up": _r2(_regime_counts(conn, where, params, r["regime"], 1) / rc),
                "down": _r2(_regime_counts(conn, where, params, r["regime"], -1) / rc),
                "flat": _r2(_regime_counts(conn, where, params, r["regime"], 0) / rc),
                "avg_pnl": _r2(r["p"]),
            }
        if by_regime:
            out["by_regime"] = by_regime
    return out


def _regime_counts(conn, where, params, regime, outcome):
    row = conn.execute(
        f"SELECT COUNT(*) AS c FROM snapshots WHERE {' AND '.join(where)} "
        f"AND regime=? AND outcome=?", params + [str(regime), outcome]
    ).fetchone()
    return int(row["c"])


def get_similar_summary(conn, index_or_symbol, snapshot_or_tf, symbol=None,
                        tf=None, bar_ts=None, K=None, min_distance=None,
                        regime_match=None, window_days=None, min_samples=None,
                        stats=None, max_rows=3, include_neighbors=False):
    """Сводка «похожих» случаев для снимка (Этап 2: FAISS + k-NN).

    Настоящий k-NN по нормализованным 44-фич векторам:
      1. vec = snapshot_to_vector(current_snapshot)
      2. vec_norm = apply_normalization(vec, rolling_stats на bar_ts)
      3. distances, ids = index.search(vec_norm, K * 3)   # запас на фильтры
      4. КРИТИЧНЫЙ фильтр времени: ts_соседа + horizon < bar_ts
         (outcome соседа вычисляется по будущим барам за horizon баров;
         если последний бар его окна >= bar_ts — outcome «из будущего»,
         такого соседа выбрасываем — это анти-look-ahead)
      5. Фильтр distance >= min_distance (отсекаем почти-дубликаты)
      6. Фильтр regime (если regime_match=True)
      7. Фильтр времени: ts > bar_ts - window_days (окно истории)
      8. Топ-K после фильтров
      9. Агрегация: winrate_up/down/flat + avg_pnl_pct + median_bars +
         avg_distance + regime_match_rate + recent_examples
      10. Confidence = min(1, n/30) * (1 - avg_dist/10) * regime_match_rate
      n < min_samples (default config.CBR_BLEND_MIN_SAMPLES=10) ->
      {"n": n, "insufficient_data": True}.

    bar_ts — ОБЯЗАТЕЛЬНЫЙ Unix-ts бара-запроса (этап 2). Если отсутствует —
    ValueError: default «сейчас» воспроизводит look-ahead для исторических
    запросов. include_neighbors=True добавляет в ответ neighbors_ts (список
    ts соседей) — диагностика (аудит look-ahead), в проде не используется.

    Pоддерживаются ДВЕ сигнатуры:
      Этап 2 (index):  (conn, index, current_snapshot, symbol, tf,
                        bar_ts, ...)
      Этап 1 (stub):   (conn, symbol, timeframe, current_snapshot,
                        window_days=.., min_samples=.., max_rows=..)
    Режим выбирается по типу второго аргумента: str,str = Этап 1.
    """
    # ---- Этап 1 (заглушка без индекса): делегирует в get_stats_for_llm.
    if isinstance(index_or_symbol, str) and isinstance(snapshot_or_tf, str):
        _symbol = index_or_symbol
        _tf = snapshot_or_tf
        _snap = symbol
        if not isinstance(_snap, dict):
            raise TypeError("current_snapshot must be a dict")
        features = snapshot_to_vector(_snap)
        regime = classify_regime(features)
        wd = int(config.CBR_QUERY_WINDOW_DAYS if window_days is None
                 else window_days)
        return get_stats_for_llm(conn, _symbol, _tf, regime=regime,
                                 window_days=wd,
                                 min_samples=int(min_samples or 10),
                                 max_rows=int(max_rows or 3))

    # ---- Этап 2: настоящий k-NN по индексу.
    index = index_or_symbol
    current_snapshot = snapshot_or_tf
    if not isinstance(current_snapshot, dict):
        raise TypeError("current_snapshot must be a dict")
    if bar_ts is None:
        raise ValueError(
            "get_similar_summary (k-NN) requires bar_ts: neighbours must "
            "have their outcome known strictly before this bar timestamp; "
            "bar_ts=now would reintroduce look-ahead")
    bar_ts = int(bar_ts)
    K = int(config.CBR_K_DEFAULT if K is None else K)
    min_distance = float(config.CBR_MIN_DISTANCE if min_distance is None
                         else min_distance)
    regime_match = bool(config.CBR_REGIME_MATCH if regime_match is None
                        else regime_match)
    wd = int(config.CBR_QUERY_WINDOW_DAYS if window_days is None
             else window_days)
    min_ok = int(config.CBR_BLEND_MIN_SAMPLES if min_samples is None
                 else min_samples)
    symbol = str(symbol or "").upper()
    tf = str(tf or "").strip()
    return _similar_summary_knn(conn, index, current_snapshot, symbol, tf,
                                bar_ts=bar_ts, K=K, min_distance=min_distance,
                                regime_match=regime_match, window_days=wd,
                                min_ok=min_ok, stats=stats,
                                include_neighbors=bool(include_neighbors))


# ------------------------------------------------------- кэш метаданных k-NN
# Метаданные строк индекса (позиции по-баровые, ts ASC): те же строки, что в
# индексе (outcome IS NOT NULL). Кэш по (id(conn), symbol, tf) — бэктест
# делает тысячи запросов к одной БД.
_KNN_META = {}


def _load_knn_meta(conn, symbol, tf):
    """Массивы метаданных по позициям индекса (ts/regime/outcome/pnl/bars/ses)."""
    key = (id(conn), symbol, tf)
    cached = _KNN_META.get(key)
    if cached is not None:
        return cached
    rows = conn.execute(
        "SELECT features, ts, regime, outcome, pnl_pct, bars_to_outcome, "
        "session FROM snapshots WHERE symbol=? AND timeframe=? "
        "AND outcome IS NOT NULL ORDER BY ts ASC",
        (symbol, tf)).fetchall()
    out = {"ts": [], "regime": [], "outcome": [], "pnl_pct": [],
           "bars": [], "session": []}
    for r in rows:
        if cbr_index._decode_features(r["features"]) is None:
            continue  # повреждённый вектор → его нет в индексе
        out["ts"].append(int(r["ts"]))
        out["regime"].append(r["regime"])
        out["outcome"].append(int(r["outcome"]))
        out["pnl_pct"].append(r["pnl_pct"])
        out["bars"].append(r["bars_to_outcome"])
        out["session"].append(r["session"])
    meta = {k: np.asarray(v) for k, v in out.items()}
    _KNN_META[key] = meta
    return meta


def _tf_seconds(tf):
    """Секунды бара таймфрейма (канонизация `1h`/`1H` -> `1H`)."""
    tf = str(tf or "").strip()
    for canon, sec in config.TF_SECONDS.items():
        if canon.lower() == tf.lower():
            return int(sec)
    return 3600


def _similar_summary_knn(conn, index, snapshot, symbol, tf, bar_ts, K=50,
                         min_distance=2.0, regime_match=True, window_days=180,
                         min_ok=10, stats=None, include_neighbors=False):
    """Реализация k-NN: поиск + фильтры + агрегация (см. get_similar_summary).

    bar_ts — обязательный Unix-ts бара-запроса. В соседи попадают ТОЛЬКО бары,
    чей outcome заведомо известен на bar_ts: ts_соседа + horizon < bar_ts
    (outcome соседа считается по будущим барам за horizon; последний бар окна
    соседа обязан быть строго раньше bar_ts, иначе outcome «протекает»).
    """
    vec = snapshot_to_vector(snapshot)
    regime = classify_regime(vec)
    q_ts = int(bar_ts)

    if stats is None:
        stats = cbr_norm.load_stats(str(config.CBR_NORMALIZE_STATS_PATH))
    if not isinstance(stats, dict) or "mean" not in stats:
        return {"n": 0, "insufficient_data": True}

    vec_norm = cbr_norm.apply_normalization(vec, cbr_norm.stats_at(stats, q_ts))

    # Поиск с запасом на фильтры (ТЗ: K * 3).
    distances, positions = cbr_index.search_knn(index, vec_norm, max(1, K * 3))

    meta = _load_knn_meta(conn, symbol, tf)
    if len(meta["ts"]) == 0:
        return {"n": 0, "insufficient_data": True}
    meta_ts = meta["ts"]

    horizon_sec = int(config.CBR_BACKFILL_HORIZON) * _tf_seconds(tf)
    lo_ts = q_ts - int(window_days) * 86400
    kept_dist, kept_pos = [], []
    for pos, dist in zip(positions, distances):
        if pos < 0 or pos >= len(meta_ts):
            continue
        if not np.isfinite(dist) or dist < min_distance:
            continue  # почти-дубликат (близнец) или невалидная дистанция
        pos = int(pos)
        n_ts = int(meta_ts[pos])
        if n_ts + horizon_sec >= q_ts:
            continue  # LOOK-AHEAD: outcome соседа неизвестен на бар запроса
        if n_ts < lo_ts:
            continue  # старше окна истории
        kept_dist.append(float(dist))
        kept_pos.append(pos)

    kept_dist = np.asarray(kept_dist, dtype="float64")
    kept_pos = np.asarray(kept_pos, dtype="int64")

    # regime_match_rate — доля соседей в том же режиме ДО regime-фильтра.
    if kept_pos.size == 0:
        return {"n": 0, "insufficient_data": True}
    neigh_regimes = np.asarray([meta["regime"][p] for p in kept_pos],
                               dtype=object)
    regime_match_rate = float(np.mean([
        1.0 if r == regime else 0.0 for r in neigh_regimes]))
    if regime_match:
        mask = np.asarray([r == regime for r in neigh_regimes])
        kept_dist = kept_dist[mask]
        kept_pos = kept_pos[mask]
        if kept_pos.size == 0:
            return {"n": 0, "insufficient_data": True}

    # Топ-K после фильтров (порядок = близость).
    cap = min(len(kept_pos), K)
    top_d = kept_dist[:cap]
    top_p = kept_pos[:cap]
    n = int(cap)

    if n < min_ok:
        return {"n": n, "insufficient_data": True}

    outcomes = np.asarray([int(meta["outcome"][p]) for p in top_p])
    pnl = np.asarray(
        [float(v) if v is not None else np.nan for v in
         [meta["pnl_pct"][p] for p in top_p]])
    bars = np.asarray(
        [float(v) if v is not None else np.nan for v in
         [meta["bars"][p] for p in top_p]])

    avg_dist = float(np.mean(top_d))
    avg_pnl = float(np.nanmean(pnl)) if np.isfinite(pnl).any() else 0.0
    median_bars = float(np.nanmedian(bars)) if np.isfinite(bars).any() \
        else float("nan")
    confidence = min(1.0, n / 30.0) * \
        max(0.0, 1.0 - min(avg_dist, 10.0) / 10.0) * regime_match_rate

    order = np.argsort(-meta_ts[top_p])  # самые свежие первыми
    recent = []
    for j in order[:3]:
        p = int(top_p[j])
        recent.append({
            "ts": int(meta_ts[p]),
            "outcome": int(meta["outcome"][p]),
            "pnl_pct": _r2(meta["pnl_pct"][p]),
            "bars_to_outcome": int(meta["bars"][p])
            if meta["bars"][p] is not None else None,
            "regime": meta["regime"][p],
            "distance": round(float(top_d[j]), 2),
        })

    out = {
        "symbol": symbol,
        "timeframe": tf,
        "n": n,
        "n_filled": n,
        "winrate_up": _r2(float(np.mean(outcomes == 1))),
        "winrate_down": _r2(float(np.mean(outcomes == -1))),
        "winrate_flat": _r2(float(np.mean(outcomes == 0))),
        "avg_pnl_pct": _r2(avg_pnl),
        "median_bars_to_outcome": round(median_bars)
        if np.isfinite(median_bars) else None,
        "confidence": round(confidence, 4),
        "avg_distance": round(avg_dist, 2),
        "regime_match_rate": round(regime_match_rate, 4),
        "regime": regime,
        "min_distance": min_distance,
        "K": K,
        "window_days": int(window_days),
        "recent_examples": recent,
    }
    if include_neighbors:
        out["neighbors_ts"] = [int(meta_ts[p]) for p in top_p]
    return out


def format_for_prompt(stats, max_chars=250):
    """Компактная строка CBR-статистики для промпта (<= max_chars).

    Одна строка: символ, ТФ, n, раскладка up/down/flat %, средний pnl,
    средний срок исхода, экскурсии, режим (была выбрана), confidence.
    Жёсткий truncate по max_chars гарантирует лимит даже на аномальном входе.
    Понимает обе схемы: get_stats_for_llm (ratio/avg_pnl/...) и
    get_similar_summary (winrate_up/winrate_down/...).
    """
    if not isinstance(stats, dict):
        return ""
    if stats.get("insufficient_data"):
        line = (f"CBR {stats.get('symbol')} {stats.get('timeframe')} "
                f"insufficient n={stats.get('n')}")
        return line[:max_chars]
    if "winrate_up" in stats:  # новая схема Этапа 2
        wd = int(stats.get("window_days") or config.CBR_QUERY_WINDOW_DAYS)
        k = int(stats.get("K") or config.CBR_K_DEFAULT)
        line = (
            f"CBR history (last {wd}d, K={k} similar): "
            f"up={_pct(stats.get('winrate_up'))} "
            f"down={_pct(stats.get('winrate_down'))} "
            f"flat={_pct(stats.get('winrate_flat'))} "
            f"n={stats.get('n')} conf={stats.get('confidence', 0.0):.2f}"
        )
        return line[:max_chars]
    ratio = stats.get("ratio") or {}
    by_regime = stats.get("by_regime") or {}
    regime_tag = stats.get("regime")
    if regime_tag is None and by_regime:
        # Режим не выбран — указываем доминирующий по доле.
        regime_tag = max(by_regime,
                         key=lambda k: by_regime[k].get("share", 0.0))
    parts = [
        (f"CBR {stats.get('symbol')} {stats.get('timeframe')} "
        f"n={stats.get('n')}"),
        (f"up={_pct(ratio.get('up'))} dn={_pct(ratio.get('down'))} "
        f"flat={_pct(ratio.get('flat'))}"),
        (f"pnl_avg={stats.get('avg_pnl')}% "
        f"bars_avg={stats.get('avg_bars')}"),
        (f"max_up={stats.get('max_up_avg')}% "
        f"max_dn={stats.get('max_dn_avg')}%"),
    ]
    if regime_tag:
        parts.append(f"reg={regime_tag}")
    line = " | ".join(p for p in parts if p)
    return line[:max_chars]


def _pct(ratio):
    """Доля 0-1 -> строка процента (None -> пусто)."""
    if ratio is None:
        return "-"
    return f"{round(float(ratio) * 100.0, 1)}%"


def stats_to_json(stats):
    """Компактный JSON-вид статистики (для scripts/cbr_token_check.py)."""
    return json.dumps(stats, ensure_ascii=False, separators=(",", ":"))