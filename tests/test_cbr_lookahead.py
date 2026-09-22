# -*- coding: utf-8 -*-
"""Тесты анти-look-ahead для CBR k-NN (Этап 2): tests/test_cbr_lookahead.py.

После фикса get_similar_summary соседями выбираются ТОЛЬКО бары, чей
outcome известен на момент бара запроса: ts_соседа + horizon < bar_ts.
Без этого k-NN «протекает» из будущего и метрики бэктеста становятся
невозможными.

Покрытие:
  - future_neighbors_flagged  — детект подсаженных будущих соседей;
  - outcome_known_at_query    — на синтетике ни один сосед не лежит в
    будущем и не имеет неизвестного исхода;
  - requires_bar_ts           — без bar_ts get_similar_summary падает;
  - insufficient_no_past      — ранний бар без прошлых соседей -> n=0;
  - verdict_logic             — verdict аудита чист на чистой паре;
  - trade_coverage_logic      — coverage < 20% у селективного сигнала и
    ≥ 20% у неселективного.
"""

import numpy as np
import pytest

from app_pkg.cbr import query_api
from scripts.cbr_lookahead_audit import (
    COVERAGE_WARN,
    audit_verdict,
    check_neighbor_pairs,
    direction_from_summary,
    trade_coverage,
)

HORIZON_SEC = 20 * 3600  # config.CBR_BACKFILL_HORIZON x 1H


# ------------------------------------------------------- фикстура (синтетика)


@pytest.fixture(scope="function")
def env():
    import os
    import tempfile

    from app_pkg import config
    from app_pkg.cbr import index as cbr_index
    from app_pkg.cbr import normalize as cbr_norm
    from app_pkg.cbr import schema, store

    rng = np.random.RandomState(42)
    fnames = store.FEATURE_NAMES

    def row_features(i):
        row = {name: float(rng.normal() * (1.0 + (i % 5) * 0.05))
               for name in fnames}
        row.update({"clock.hour_utc": float(i % 24),
                    "clock.dow": float(i % 7),
                    "clock.market_open": 1.0,
                    "clock.london_ny_overlap": 0.0,
                    "clock.session_code": float(i % 5)})
        return row

    tmp = tempfile.mkdtemp(prefix="cbr_la_")
    conn = schema.init_db(os.path.join(tmp, "cbr.db"))
    ids = []
    base = 1_700_000_000
    n = 300
    for i in range(n):
        ts = base + i * 3600
        snap = {"symbol": "BTCUSDT", "timeframe": "1H", "ts": ts,
                "source": "test", **row_features(i)}
        rid = store.store_snapshot(conn, snap, entry_price=100.0)
        if rid is not None:
            ids.append((rid, ts))
    conn.commit()
    label_n = int(len(ids) * 0.9)
    for j, (rid, _ts) in enumerate(ids):
        if j >= label_n:
            continue
        y = 1 if j % 4 == 0 else -1 if j % 4 == 1 else 0
        conn.execute(
            "UPDATE snapshots SET outcome=?, pnl_pct=?, bars_to_outcome=? "
            "WHERE id=?",
            (y, round(float(np.random.RandomState(j).randn()), 4),
             int(12 + j % 8), rid))
    conn.commit()

    X, ts, regimes, y = cbr_index.load_feature_matrix(conn, "BTCUSDT", "1H")
    stats = cbr_norm.compute_rolling_stats(X, ts=ts)
    Xn = cbr_norm.normalize_matrix(X, ts, stats)
    index = cbr_index.build_faiss_index(Xn)
    return {"conn": conn, "index": index, "stats": stats, "X": X, "ts": ts,
            "FNAMES": fnames, "base": base}


def _snap(env, i):
    vec = env["X"][i]
    snap = dict(zip(env["FNAMES"], vec.tolist()))
    snap["symbol"] = "BTCUSDT"
    snap["timeframe"] = "1H"
    snap["ts"] = int(env["ts"][i])
    return snap


# ------------------------------------------------------- проверки будущего


def test_future_neighbors_flagged():
    """check_neighbor_pairs детектит будущих соседей и неизвестные исходы."""
    q = 1_700_000_000
    # n1 в будущем, n2 с известным исходом, n3 с ещё не состоявшимся
    pairs = [(q, [q + 3600, q - 10 * 3600, q - 30 * 3600])]
    future, not_known, deltas = check_neighbor_pairs(pairs, HORIZON_SEC)
    assert future == 1            # q+3600 > q
    assert not_known == 2         # q+3600 и q-10*3600 (10h < 20h горизонт)
    assert (np.asarray(deltas) < 0).sum() == 1


def test_outcome_known_at_query(env):
    """На синтетике ни один сосед не из будущего и не с неизвестным
    исходом (по равноотстоящим барам 1H с horizon=20 это bar_ts - n_ts,
    кратное 3600, но не 1)."""
    q = int(env["ts"][250])
    res = query_api.get_similar_summary(
        env["conn"], env["index"], _snap(env, 250), "BTCUSDT", "1H",
        bar_ts=q, K=30, min_distance=0.0, regime_match=False, window_days=180,
        stats=env["stats"], include_neighbors=True)
    assert not res.get("insufficient_data")
    nbrs = res["neighbors_ts"]
    q = int(q)
    for n in nbrs:
        assert n < q                        # не из будущего
        assert n + HORIZON_SEC < q          # outcome уже известен


def test_requires_bar_ts(env):
    """k-NN ветка без bar_ts — запрещена намеренно (нет неявного now)."""
    with pytest.raises(ValueError):
        query_api.get_similar_summary(
            env["conn"], env["index"], _snap(env, 100), "BTCUSDT", "1H",
            K=30, min_distance=0.0, regime_match=False, window_days=180,
            stats=env["stats"])


def test_insufficient_no_past(env):
    """Самый ранний бар: прошлых соседей с известным исходом нет -> n=0."""
    i = 20
    q = int(env["ts"][i])
    res = query_api.get_similar_summary(
        env["conn"], env["index"], _snap(env, i), "BTCUSDT", "1H",
        bar_ts=q, K=30, min_distance=0.0, regime_match=False, window_days=180,
        stats=env["stats"])
    assert res == {"n": 0, "insufficient_data": True}


# ------------------------------------------------------- логика аудита


def test_verdict_logic():
    """Вердикт чист только при отсутствии утечек и min-delta > horizon."""
    assert audit_verdict(0, 0, 0, min_delta_days=1.0,
                         horizon_days=HORIZON_SEC / 86400.0).startswith("NO")
    assert audit_verdict(1, 0, 0, 1.0, 0.83).startswith("LOOK-AHEAD")
    assert audit_verdict(0, 1, 0, 1.0, 0.83).startswith("LOOK-AHEAD")
    assert audit_verdict(0, 0, 1, 1.0, 0.83).startswith("LOOK-AHEAD")
    assert audit_verdict(0, 0, 0, 0.5, 0.83).startswith("LOOK-AHEAD")


def test_trade_coverage_logic():
    """Селективный сигнал (F почти всегда) -> coverage < 20%, неселективный
    (L/S почти всегда) -> >= 20%."""
    selective = ["F"] * 900 + ["L"] * 25 + ["S"] * 25
    nonselective = ["L"] * 400 + ["S"] * 400 + ["F"] * 200
    sel = trade_coverage(selective)
    nsel = trade_coverage(nonselective)
    assert sel is not None and nsel is not None
    assert sel < COVERAGE_WARN
    assert nsel >= COVERAGE_WARN
    assert trade_coverage([]) is None
    # направление: спред > 0.15 -> сделка; иначе F
    assert direction_from_summary({"winrate_up": 0.72, "winrate_down": 0.40}) == "L"
    assert direction_from_summary({"winrate_up": 0.30, "winrate_down": 0.62}) == "S"
    assert direction_from_summary({"winrate_up": 0.51, "winrate_down": 0.49}) == "F"
    assert direction_from_summary({"n": 0, "insufficient_data": True}) == "F"