"""Тесты CBR-базы Харона (Этап 1+): schema/store/backfill/query_api + хуки.

Покрытие:
  - schema: создание таблицы/индексов, WAL, idempotent re-init, migrate;
- store: три формата снапшота -> вектор 44, порядок == charon_features,
      классификатор режима, хэш-дедуп по (symbol, tf, ts), session/verdict/
      levels/atr_pct;
  - backfill: triple-barrier (up/down/flat), SL-приоритет, таймаут,
    идемпотентность, пропуск недостающих баров, ValueError на вырожденном
    atr_pct, pnl_pct по направлению вердикта;
  - query_api: фиксированная схема, недостаток данных, окно по MAX(ts),
    regime-фильтр, by_regime ТОЛЬКО при regime=None, sample-лимит 3,
    get_similar_summary (та же схема), format_for_prompt < 300 символов;
  - хуки: _flush_pending (source=backtest), levels_for_slice (live/backtest,
    уровни delta/prob), compact_snapshot live (source=live), гейт выключен.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app_pkg import config
from app_pkg.cbr import backfill, query_api, schema, store

BASE = 1_700_000_000  # эпоха фикстур (UTC)
SESSION_NAMES = ["asia", "london", "london_ny", "ny", "off_hours", "weekend"]
_MODE_BARS = {
    "up": {"high": 104.0, "low": 99.5, "close": 104.0},
    "down": {"high": 100.5, "low": 96.0, "close": 96.0},
    "flat": {"high": 100.5, "low": 99.5, "close": 100.0},
}


def _compact(i, ts=None, mode=None, atr=2.0, close=100.0, ses=None,
             sig=None, source="test", symbol="BTCUSDT", tf="1H"):
    """Компактный снапшот (схема v3), детерминированный по i.

    Режим привязывается к i%4 (trend_up/reversal/flat) независимо от
    исхода (i%3 up/down/flat) — by_regime не коррелирует с outcome.
    Вектор уникален по i (rsi, час, сессия).
    """
    mode = mode or ["up", "down", "flat"][i % 3]
    if sig is None:
        sig = {"up": "L", "down": "S", "flat": "F"}[mode]
    ses = ses or SESSION_NAMES[i % 6]
    r = i % 4
    if r == 0:
        hurst, adx, er = 0.62, 30.0, 1.02   # trend_up
    elif r == 1:
        hurst, adx, er = 0.40, 20.0, 0.98   # reversal
    else:
        hurst, adx, er = 0.50, 20.0, 0.98   # flat
    h = (8 + i) % 24
    dow = i % 7
    return {
        "symbol": symbol, "timeframe": tf,
        "ts": ts if ts is not None else BASE + i * 3 * 86400,
        "source": source, "verdict": sig,
        "t": {"rsi": 30.0 + (i % 5) * 5.0, "atr": atr, "bb": 0.5, "sma": 0.4,
              "ap": 0.6, "dh": -0.2, "dl": 0.2, "close": close},
        "v": {"rv": 1.0, "obv": 0.0, "vz": 0.1, "vp": 0.5, "vwd": 0.0},
        "tr": {"adx": adx, "pdi": 22.0, "mdi": 20.0, "ds": 2.0, "er": er,
               "rs": 0.1},
        "mo": {"macd": 0.05, "ms": 0.04, "mh": 0.01, "mhs": 0.0, "rsi_s": 0.5},
        "vl": {"bw": 0.05, "bwp": 0.5, "hv": 30.0, "atc": 1.0, "atr_r": 1.0},
        "rg": {"h": hurst, "ac1": 0.05, "er": 0.3},
        "div": {"rsi": 0, "macd": 0, "obv": 0},
        "cnd": {"br": 0.5, "uw": 0.25, "lw": 0.25, "eng": 0, "pin": 0},
        "c": {"h": h, "dow": dow, "ses": ses,
              "ovl": 1 if (13 <= h <= 16) and dow < 5 else 0, "open": 1},
    }


def _raw_from_compact(c):
    """Компакт -> raw-формат (полные имена блоков, clock.session — имя)."""
    map_full = {"t": "technicals", "v": "volume", "tr": "trend",
                "mo": "momentum", "vl": "volatility", "rg": "regime",
                "div": "divergence", "cnd": "candle"}
    map_fields = {
        "t": {"rsi": "rsi", "atr": "atr", "bb": "bb_pct_b", "sma": "sma20_diff_pct",
              "ap": "atr_percentile", "dh": "dist_to_high_pct",
              "dl": "dist_to_low_pct", "close": "close"},
        "v": {"rv": "rel_vol", "obv": "obv_slope", "vz": "vol_zscore",
              "vp": "vol_percentile", "vwd": "vwap_dev"},
        "tr": {"adx": "adx", "pdi": "plus_di", "mdi": "minus_di",
               "ds": "di_spread", "er": "ema20_50_ratio", "rs": "reg_slope_20"},
        "mo": {"macd": "macd", "ms": "macd_signal", "mh": "macd_hist",
               "mhs": "macd_hist_slope", "rsi_s": "rsi_slope"},
        "vl": {"bw": "bb_width", "bwp": "bb_width_pct", "hv": "hv20",
               "atc": "atr_change_pct", "atr_r": "atr_ratio_short_long"},
        "rg": {"h": "hurst", "ac1": "autocorr_lag1", "er": "efficiency_ratio"},
        "div": {"rsi": "rsi_price", "macd": "macd_price", "obv": "obv_price"},
        "cnd": {"br": "body_ratio", "uw": "upper_wick_ratio",
                "lw": "lower_wick_ratio", "eng": "engulfing", "pin": "pinbar"},
    }
    out = {"symbol": c["symbol"], "timeframe": c["timeframe"], "ts": c["ts"],
           "source": c["source"]}
    for blk, src in map_full.items():
        out[src] = dict(c.get(blk) or {})
    for key in list(out.get("candle", {}))[::-1]:
        pass
    # нормализуем ключи raw-блоков до полных имён
    raw = {}
    for blk, src in map_full.items():
        fmap = map_fields[blk]
        raw[src] = {fmap[k]: v for k, v in (c.get(blk) or {}).items()
                    if k in fmap}
    out.update(raw)
    clk = dict(c.get("c") or {})
    if isinstance(clk.get("ses"), str):
        clk["session"] = clk.pop("ses")
    out["clock"] = {"hour_utc": clk.get("h"), "dow": clk.get("dow"),
                    "session": clk.get("session"),
                    "london_ny_overlap": clk.get("ovl"),
                    "market_open": clk.get("open")}
    return out


def seed_snapshots(conn, n=50):
    """n снапшотов (i%3 up/down/flat) + бары их 20-барных окон. (conn, bars)."""
    bars = {}
    for i in range(n):
        mode = ["up", "down", "flat"][i % 3]
        ts = BASE + i * 3 * 86400
        store.store_snapshot(conn, _compact(i, ts, mode), entry_price=100.0)
        mb = _MODE_BARS[mode]
        for j in range(1, 21):
            bars[ts + j * 3600] = {"ts": ts + j * 3600, "open": 100.0,
                                   "high": mb["high"], "low": mb["low"],
                                   "close": mb["close"]}
    return conn, [bars[k] for k in sorted(bars)]


def run_backfill(conn, bars, symbol="BTCUSDT", tf="1H", **kwargs):
    def price_fn(sym, timef, frm, to):
        return [b for b in bars if b["ts"] >= frm
                and (to is None or b["ts"] <= to)]
    return backfill.backfill_outcomes(conn, symbol, tf, price_fn, **kwargs)


# ------------------------------------------------------------- фикстуры
@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "cbr.db")


@pytest.fixture
def conn(db_path):
    c = schema.init_db(db_path)
    yield c
    c.close()


@pytest.fixture
def seeded_db(conn):
    return seed_snapshots(conn, 50)


@pytest.fixture
def filled_db(seeded_db):
    c, bars = seeded_db
    run_backfill(c, bars, horizon=20, atr_k=1.5)
    return c


# ============================================================== schema
def test_schema_creates_table_and_indexes(conn):
    names = {r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','index')")}
    assert "snapshots" in names
    assert {"idx_snapshots_hash", "idx_snapshots_key",
            "idx_snapshots_pending", "idx_snapshots_outcome"} <= names


def test_schema_wal_and_busy_timeout(conn):
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000


def test_schema_row_factory(conn):
    row = conn.execute("SELECT COUNT(*) AS c FROM snapshots").fetchone()
    assert row["c"] == 0  # доступ по имени — sqlite3.Row


def test_schema_reinit_idempotent(db_path):
    c1 = schema.init_db(db_path)
    c1.close()
    c2 = schema.init_db(db_path)
    n = c2.execute("SELECT COUNT(*) c FROM sqlite_master "
                   "WHERE type='table' AND name='snapshots'").fetchone()["c"]
    c2.close()
    assert n == 1


def test_schema_connect_and_migrate(db_path):
    c = schema.connect(db_path)
    assert c.execute("SELECT 1").fetchone()[0] == 1
    assert schema.migrate(c) is c
    c.close()


# ============================================================== store
def test_vector_44_float32():
    v = store.snapshot_to_vector(_compact(0))
    assert v.shape == (44,)
    assert v.dtype == np.float32


def test_vector_compact_vs_raw_same_values():
    c = _compact(7)
    vc = store.snapshot_to_vector(c)
    vr = store.snapshot_to_vector(_raw_from_compact(c))
    assert np.array_equal(vc, vr)


def test_vector_session_code_mapping():
    c = _compact(5)
    c["c"]["ses"] = "off_hours"
    assert store.snapshot_to_vector(c)[43] == 4.0
    r = _raw_from_compact(c)
    r["clock"]["session"] = "weekend"
    assert store.snapshot_to_vector(r)[43] == 5.0
    c2 = dict(c)
    c2["c"]["ses"] = 2
    assert store.snapshot_to_vector(c2)[43] == 2.0


def test_vector_missing_fields_zero():
    v = store.snapshot_to_vector({"t": {"close": 100.0}})
    assert v.sum() == 0.0  # всё кроме close — 0; close не входит в 44
    assert v[0] == 0.0


def test_vector_raises_on_non_dict():
    with pytest.raises(ValueError):
        store.snapshot_to_vector([1, 2, 3])


def test_classify_regime():
    def v(hurst, adx, er):
        vec = np.zeros(44, dtype="float32")
        vec[store._IDX["regime.hurst"]] = hurst
        vec[store._IDX["trend.adx"]] = adx
        vec[store._IDX["trend.ema20_50_ratio"]] = er
        return vec

    assert store.classify_regime(v(0.62, 30.0, 1.02)) == "trend_up"
    assert store.classify_regime(v(0.62, 30.0, 0.95)) == "trend_down"
    assert store.classify_regime(v(0.40, 20.0, 1.0)) == "reversal"
    assert store.classify_regime(v(0.50, 20.0, 1.0)) == "flat"


def test_feature_order_matches_labels():
    """FEATURE_NAMES == колонки charon_features; вектор == строка признаков."""
    from app_pkg.ml import labels
    rng = np.random.default_rng(7)
    rows = 320
    ts = (pd.Timestamp("2024-01-01", tz="UTC")
          + pd.to_timedelta(np.arange(rows), unit="h"))
    close = 100.0 + np.cumsum(rng.normal(0, 1.0, rows))
    spread = np.abs(rng.normal(0, 0.5, rows))
    df = pd.DataFrame({
        "timestamp": ts,
        "open": np.roll(close, 1),
        "high": close + spread,
        "low": close - spread,
        "close": close,
        "volume": np.abs(rng.normal(100, 30, rows)) + 5.0,
    })
    df.loc[0, "open"] = df.loc[0, "close"]
    feat = labels.charon_features(df, symbol="BTCUSDT")
    assert list(feat.columns) == store.FEATURE_NAMES
    i = 200
    row = {c: float(v) for c, v in feat.iloc[i].items()}
    vec = store.snapshot_to_vector(row).astype("float64")
    exp = feat.iloc[i].to_numpy(dtype="float64")
    assert np.allclose(vec, exp, atol=1e-4)


def test_store_insert_and_fields(conn):
    rid = store.store_snapshot(conn, _compact(0))
    assert rid is not None
    row = conn.execute("SELECT * FROM snapshots WHERE id=?", (rid,)).fetchone()
    assert dict(row)["symbol"] == "BTCUSDT"
    assert dict(row)["timeframe"] == "1H"
    assert dict(row)["ts"] == BASE
    assert dict(row)["source"] == "test"
    assert dict(row)["atr_pct"] == pytest.approx(0.02)
    assert dict(row)["entry_price"] == pytest.approx(100.0)
    assert len(json.loads(row["features"])) == 44
    assert len(row["feature_hash"]) == 32


def test_store_dedup_by_hash(conn):
    s1 = _compact(0)
    rid1 = store.store_snapshot(conn, s1)
    rid2 = store.store_snapshot(conn, s1)  # тот же вектор (тот же i/ts)
    assert rid1 is not None
    assert rid2 is None
    assert conn.execute("SELECT COUNT(*) c FROM snapshots").fetchone()["c"] == 1


def test_dedup_by_timestamp_not_vector(conn):
    """Два снапшота с ОДИНАКОВЫМ вектором, но разным ts — оба записаны."""
    s1 = _compact(0, ts=BASE)
    s2 = dict(s1)
    s2["ts"] = BASE + 86400
    rid1 = store.store_snapshot(conn, s1)
    rid2 = store.store_snapshot(conn, s2)
    assert rid1 is not None and rid2 is not None
    assert rid1 != rid2
    assert conn.execute("SELECT COUNT(*) c FROM snapshots").fetchone()["c"] == 2


def test_dedup_same_bar_twice(conn):
    """Тот же (symbol, tf, ts, vec) — вторая запись возвращает None."""
    s = _compact(0, ts=BASE)
    rid1 = store.store_snapshot(conn, s)
    rid2 = store.store_snapshot(conn, dict(s))
    assert rid1 is not None
    assert rid2 is None
    assert conn.execute("SELECT COUNT(*) c FROM snapshots").fetchone()["c"] == 1


def test_hash_includes_symbol_tf_ts(conn):
    """feature_hash меняется при изменении любого из (symbol, tf, ts)."""
    base = _compact(0, ts=BASE)
    rid = store.store_snapshot(conn, base)
    h_base = conn.execute("SELECT feature_hash FROM snapshots WHERE id=?",
                          (rid,)).fetchone()["feature_hash"]
    for mut in ({"symbol": "ETHUSDT"}, {"timeframe": "4H"},
                {"ts": BASE + 86400}):
        s = dict(base)
        s.update(mut)
        _rid = store.store_snapshot(conn, s)
        assert _rid is not None  # если hash совпал — запись потеряна
        h = conn.execute("SELECT feature_hash FROM snapshots WHERE id=?",
                         (_rid,)).fetchone()["feature_hash"]
        assert h != h_base


def test_store_requires_key_fields(conn):
    with pytest.raises(ValueError):
        store.store_snapshot(conn, {"symbol": "BTCUSDT", "ts": 1})
    with pytest.raises(ValueError):
        store.store_snapshot(conn, {"symbol": "BTCUSDT", "timeframe": "1H"})
    with pytest.raises(ValueError):
        store.store_snapshot(conn, {"symbol": "BTCUSDT", "timeframe": "1H",
                                    "ts": None})


def test_store_entry_fallback_to_close(conn):
    s = _compact(1)
    s.pop("t")  # close брать неоткуда — только явная цена
    rid = store.store_snapshot(conn, s, entry_price=50.0)
    row = conn.execute("SELECT entry_price, atr_pct FROM snapshots WHERE id=?",
                       (rid,)).fetchone()
    # вектор без t.atr => atr=0 => atr_pct None, entry сохранён
    assert row["entry_price"] == pytest.approx(50.0)
    assert row["atr_pct"] is None


def test_store_atr_pct_none_on_degenerate(conn):
    s = _compact(2)
    s["t"]["atr"] = 0.0
    rid = store.store_snapshot(conn, s, entry_price=100.0)
    row = conn.execute("SELECT atr_pct FROM snapshots WHERE id=?", (rid,)).fetchone()
    assert row["atr_pct"] is None


def test_store_verdict_normalization(conn):
    s = _compact(3)
    s["verdict"] = "FLAT"
    rid = store.store_snapshot(conn, s)
    assert dict(conn.execute("SELECT verdict FROM snapshots WHERE id=?",
                             (rid,)).fetchone())["verdict"] == "F"
    s2 = _compact(4)
    s2["verdict"] = "NULL"
    rid2 = store.store_snapshot(conn, s2)
    assert conn.execute("SELECT verdict FROM snapshots WHERE id=?",
                        (rid2,)).fetchone()["verdict"] is None


def test_store_levels_json_roundtrip(conn):
    s = _compact(5)
    s["levels"] = [{"delta": 4.0, "prob": 0.6}, {"delta": -3.0, "prob": 0.4}]
    rid = store.store_snapshot(conn, s)
    row = conn.execute("SELECT levels_json FROM snapshots WHERE id=?",
                       (rid,)).fetchone()
    assert json.loads(row["levels_json"]) == [{"delta": 4.0, "prob": 0.6},
                                              {"delta": -3.0, "prob": 0.4}]


def test_store_regime_and_session(conn):
    rid = store.store_snapshot(conn, _compact(0))  # i%4==0 trend_up, ses asia
    row = conn.execute("SELECT regime, session FROM snapshots WHERE id=?",
                       (rid,)).fetchone()
    assert row["regime"] == "trend_up"
    assert row["session"] == "asia"
    rid2 = store.store_snapshot(conn, _compact(5))  # i%4==1 reversal, ses weekend
    row2 = conn.execute("SELECT regime, session FROM snapshots WHERE id=?",
                        (rid2,)).fetchone()
    assert row2["regime"] == "reversal"
    assert row2["session"] == "weekend"


def test_get_cbr_conn_respects_config_path(monkeypatch, tmp_path):
    from app_pkg.cbr import store as st
    monkeypatch.setattr(config, "CBR_ENABLED", True)
    p1 = str(tmp_path / "a.db")
    monkeypatch.setattr(config, "CBR_DB_PATH", p1)
    c1 = st.get_cbr_conn()
    p2 = str(tmp_path / "b.db")
    monkeypatch.setattr(config, "CBR_DB_PATH", p2)
    c2 = st.get_cbr_conn()
    assert c1 is not c2
    assert Path(p2).exists()


# ============================================================== backfill
def test_backfill_outcome_balance_and_metrics(filled_db):
    mix = {int(r["outcome"]): int(r["c"]) for r in filled_db.execute(
        "SELECT outcome, COUNT(*) c FROM snapshots GROUP BY outcome")}
    assert mix == {1: 17, -1: 17, 0: 16}
    pending = filled_db.execute(
        "SELECT COUNT(*) c FROM snapshots WHERE outcome IS NULL").fetchone()["c"]
    assert pending == 0


def test_backfill_hit_semantics_up(filled_db):
    row = filled_db.execute(
        "SELECT * FROM snapshots WHERE outcome=1 LIMIT 1").fetchone()
    assert row["bars_to_outcome"] == 1
    assert row["exit_price"] == pytest.approx(103.0)
    assert row["max_up"] == pytest.approx(4.0)
    assert row["max_dn"] == pytest.approx(-0.5)
    assert row["pnl_pct"] == pytest.approx(3.0)  # sig=L


def test_backfill_hit_semantics_down(filled_db):
    row = filled_db.execute(
        "SELECT * FROM snapshots WHERE outcome=-1 LIMIT 1").fetchone()
    assert row["bars_to_outcome"] == 1
    assert row["exit_price"] == pytest.approx(97.0)
    assert row["pnl_pct"] == pytest.approx(3.0)  # sig=S: падение -> плюс
    assert row["max_dn"] == pytest.approx(-4.0)


def test_backfill_timeout_flat(filled_db):
    row = filled_db.execute(
        "SELECT * FROM snapshots WHERE outcome=0 LIMIT 1").fetchone()
    assert row["bars_to_outcome"] == 20
    assert row["exit_price"] == pytest.approx(100.0)
    assert row["pnl_pct"] == pytest.approx(0.0)  # sig=F


def test_backfill_idempotent(filled_db):
    counts = filled_db.execute("SELECT outcome, COUNT(*) c FROM snapshots "
                               "GROUP BY outcome").fetchall()
    run_backfill(filled_db, seed_snapshots(filled_db)[1], horizon=20, atr_k=1.5)
    after = filled_db.execute("SELECT outcome, COUNT(*) c FROM snapshots "
                              "GROUP BY outcome").fetchall()
    assert sorted((r["outcome"], r["c"]) for r in after) == \
        sorted((r["outcome"], r["c"]) for r in counts)


def test_backfill_sl_priority_on_same_bar(conn):
    s = _compact(0)
    s["ts"] = BASE
    store.store_snapshot(conn, s, entry_price=100.0)  # atr=2 => up=103 dn=97
    bars = [{"ts": BASE + 3600 * (j + 1), "open": 100.0,
             "high": 104.0, "low": 96.0, "close": 100.0} for j in range(20)]
    assert run_backfill(conn, bars, horizon=20, atr_k=1.5) == 1
    row = conn.execute("SELECT outcome, exit_price FROM snapshots").fetchone()
    assert row["outcome"] == -1
    assert row["exit_price"] == pytest.approx(97.0)


def test_backfill_skips_when_not_enough_future(conn):
    s = _compact(1)
    s["ts"] = BASE
    store.store_snapshot(conn, s, entry_price=100.0)
    bars = [{"ts": BASE + 3600 * (j + 1), "open": 100.0, "high": 100.5,
             "low": 99.5, "close": 100.0} for j in range(5)]  # только 5<20
    assert run_backfill(conn, bars, horizon=20, atr_k=1.5) == 0
    assert conn.execute("SELECT outcome FROM snapshots").fetchone()["outcome"] \
        is None


def test_backfill_valueerror_on_degenerate_atr(conn):
    s = _compact(2)
    s["ts"] = BASE
    store.store_snapshot(conn, s, entry_price=100.0)  # atr_pct=0.02
    conn.execute("UPDATE snapshots SET atr_pct=0.0")  # вырождаем вручную
    conn.commit()
    bars = [{"ts": BASE + 3600 * (j + 1), "open": 100.0, "high": 101.0,
             "low": 99.0, "close": 100.0} for j in range(20)]
    with pytest.raises(ValueError):
        run_backfill(conn, bars, horizon=20, atr_k=1.5)


def test_backfill_pnl_sign_by_verdict(conn):
    for i in (0, 1):  # up(L) и down(S), оба с близким профилем
        s = _compact(i)
        s["ts"] = BASE + i * 86400
        store.store_snapshot(conn, s, entry_price=100.0)
    bars = []
    for i in (0, 1):  # первый бар окна бьёт ВЕРХ (103) для обоих
        for j in range(1, 21):
            bars.append({"ts": BASE + i * 86400 + j * 3600, "open": 100.0,
                         "high": 104.0, "low": 99.5, "close": 104.0})
    assert run_backfill(conn, bars, horizon=20, atr_k=1.5) == 2
    by_sig = {r["verdict"]: r["pnl_pct"] for r in conn.execute(
        "SELECT verdict, pnl_pct FROM snapshots")}
    # L: целимся вверх = +3%; S: движение вверх = убыток -3%
    assert by_sig["L"] == pytest.approx(3.0)
    assert by_sig["S"] == pytest.approx(-3.0)


# ============================================================== query_api
def test_stats_fixed_schema(filled_db):
    st = query_api.get_stats_for_llm(filled_db, "BTCUSDT", "1H",
                                     window_days=365)
    assert set(st.keys()) == set(query_api._STATS_KEYS)
    assert st["symbol"] == "BTCUSDT"
    assert st["regime"] is None
    assert st["n"] == 50
    assert sorted(st["ratio"]) == ["down", "flat", "up"]
    assert st["ratio"]["up"] + st["ratio"]["down"] + st["ratio"]["flat"] \
        == pytest.approx(1.0, abs=0.01)


def test_stats_insufficient_empty(conn):
    st = query_api.get_stats_for_llm(conn, "BTCUSDT", "1H")
    assert st == {"n": 0, "insufficient_data": True}


def test_stats_insufficient_min_samples(filled_db):
    st = query_api.get_stats_for_llm(filled_db, "BTCUSDT", "1H",
                                     window_days=9990, min_samples=999)
    assert st == {"n": 50, "insufficient_data": True}


def test_stats_regime_filter(filled_db):
    st = query_api.get_stats_for_llm(filled_db, "BTCUSDT", "1H",
                                     regime="trend_up", window_days=365)
    assert "by_regime" not in st
    assert st["regime"] == "trend_up"
    assert all(s["regime"] == "trend_up" for s in st["samples"])


def test_stats_by_regime_only_when_no_filter(filled_db):
    st = query_api.get_stats_for_llm(filled_db, "BTCUSDT", "1H",
                                     window_days=365)
    assert "by_regime" in st
    keys = set(st["by_regime"])
    assert keys == {"trend_up", "reversal", "flat"}
    total_share = sum(v["share"] for v in st["by_regime"].values())
    assert total_share == pytest.approx(1.0, abs=0.01)


def test_stats_window_relative_to_max(conn):
    # 5 «старых» строк за пределами окна (max-90d) и 10 свежих
    BASE + 100 * 86400
    for i in range(10):
        store.store_snapshot(conn, _compact(i, ts=BASE + i * 3600))
    for i in range(5):
        store.store_snapshot(conn, _compact(100 + i, ts=BASE - 200 * 86400))
    conn.execute("UPDATE snapshots SET outcome=1, bars_to_outcome=3, "
                 "pnl_pct=1.0, max_up=2.0, max_dn=-1.0")
    conn.commit()
    st = query_api.get_stats_for_llm(conn, "BTCUSDT", "1H",
                                     window_days=90, min_samples=1)
    assert st["n"] == 10  # старые вне окна MAX(ts)-90d исключены
    assert st["ratio"]["up"] == 1.0
    assert "insufficient_data" not in st


def test_stats_samples_limited_and_recent(filled_db):
    st = query_api.get_stats_for_llm(filled_db, "BTCUSDT", "1H",
                                     window_days=365, max_rows=3)
    assert len(st["samples"]) == 3
    max_ts = filled_db.execute("SELECT MAX(ts) m FROM snapshots").fetchone()["m"]
    assert st["samples"][0]["ts"] == max_ts
    assert st["samples"][0]["pnl_pct"] in (3.0, -3.0, 0.0)


def test_stats_percent_two_decimals(filled_db):
    st = query_api.get_stats_for_llm(filled_db, "BTCUSDT", "1H",
                                     window_days=365)
    for r in st["ratio"].values():
        assert r == round(r, 2)
    for reg in st["by_regime"].values():
        assert reg["share"] == round(reg["share"], 2)
    assert st["avg_pnl"] == round(st["avg_pnl"], 2)


def test_similar_summary_same_schema(filled_db):
    c = _compact(1)
    c["rg"]["h"] = 0.40  # reversal
    st = query_api.get_similar_summary(filled_db, "BTCUSDT", "1H", c,
                                        window_days=365)
    assert set(st.keys()) == set(query_api._STATS_KEYS) - {"by_regime"}
    assert st["regime"] == "reversal"
    # тот же результат, что явный фильтр (схема не добавляет ключей)
    direct = query_api.get_stats_for_llm(filled_db, "BTCUSDT", "1H",
                                         regime="reversal", window_days=365)
    assert st["n"] == direct["n"]


def test_similar_summary_insufficient_empty(conn):
    st = query_api.get_similar_summary(conn, "BTCUSDT", "1H", _compact(0))
    assert st == {"n": 0, "insufficient_data": True}


def test_format_prompt_compact_and_short(filled_db):
    st = query_api.get_stats_for_llm(filled_db, "BTCUSDT", "1H",
                                     window_days=365)
    line = query_api.format_for_prompt(st)
    assert "CBR BTCUSDT 1H" in line
    assert "n=50" in line
    assert len(line) < 300
    assert line.count("\n") == 0


def test_format_prompt_truncates(filled_db):
    st = query_api.get_stats_for_llm(filled_db, "BTCUSDT", "1H",
                                     window_days=365)
    line = query_api.format_for_prompt(st, max_chars=20)
    assert len(line) <= 20


def test_format_prompt_insufficient():
    line = query_api.format_for_prompt({"n": 3, "insufficient_data": True,
                                        "symbol": "BTCUSDT",
                                        "timeframe": "1H"})
    assert "insufficient" in line
    assert len(line) < 100


# ============================================================== token-оценка
def test_token_estimate_naive(seeded_db):
    """Честная naive-оценка (SELECT LIMIT 50) на seeded_db: не 0, не миллион."""
    conn, _ = seeded_db
    from scripts.cbr_token_check import estimate_llm_naive_tokens
    est = estimate_llm_naive_tokens(conn, "BTCUSDT", "1H", window_days=3650)
    assert 0 < est < 1_000_000


def test_token_estimate_api(seeded_db):
    """API-оценка (get_stats_for_llm + format_for_prompt) < 500 токенов."""
    conn, _ = seeded_db
    from scripts.cbr_token_check import estimate_cbr_api_tokens
    est = estimate_cbr_api_tokens(conn, "BTCUSDT", "1H")
    assert 0 < est < 500


def test_grid_balance_score():
    """balance_score: 1 при ровно 35/35/30; отклонение -> падение."""
    from scripts.cbr_grid import balance_score
    assert balance_score(0.35, 0.35, 0.30) == 1.0
    assert balance_score(0.46, 0.46, 0.08) < 0.85


# ============================================================== хуки
def _enable_cbr(monkeypatch, tmp_path, enabled=True):
    monkeypatch.setattr(config, "CBR_ENABLED", enabled)
    monkeypatch.setattr(config, "CBR_DB_PATH", str(tmp_path / "cbr.db"))


def test_hook_flush_pending_stores_verdict(monkeypatch, tmp_path):
    from app_pkg.ai import ai_backtest as aibt
    _enable_cbr(monkeypatch, tmp_path, True)
    verdict = {"sig": "L", "pu": 0.7, "pd": 0.3, "pf": 0.0, "conf": 0.7,
               "tg": []}
    snap = {"t": {"rsi": 55.0, "atr": 2.0, "close": 100.0},
            "c": {"h": 14, "dow": 2, "ses": "london", "ovl": 1, "open": 1}}
    monkeypatch.setattr(aibt, "filter_verdict",
                        lambda v, s, generated_at=None: v)
    monkeypatch.setattr(aibt, "parse_verdict", lambda raw: verdict)
    monkeypatch.setattr(aibt, "parse_verdict_batch",
                        lambda raw, n: [verdict] * n)
    monkeypatch.setattr(aibt, "_llm_verdict", lambda *a, **k: "{}")
    results, cache = {}, {}
    totals = {"llm_calls": 0, "tokens_prompt": 0, "tokens_completion": 0,
              "tokens_total": 0, "cache_hits": 0}
    aibt._flush_pending([(0, BASE, snap, "k")], results, cache, totals,
                        "model", "system", symbol="BTCUSDT", timeframe="1H")
    c = schema.init_db(tmp_path / "cbr.db")
    row = c.execute("SELECT verdict, source, ts, entry_price FROM snapshots"
                    ).fetchone()
    assert row["verdict"] == "L"
    assert row["source"] == "backtest"
    assert row["ts"] == BASE
    assert row["entry_price"] == pytest.approx(100.0)
    assert results[0]["sig"] == "L"


def test_hook_flush_pending_batch(monkeypatch, tmp_path):
    from app_pkg.ai import ai_backtest as aibt
    _enable_cbr(monkeypatch, tmp_path, True)
    verdict = {"sig": "S", "pu": 0.2, "pd": 0.8, "pf": 0.0, "conf": 0.8,
               "tg": []}
    snaps = [{"t": {"rsi": 40.0 + i, "atr": 2.0, "close": 100.0},
              "c": {"h": i, "dow": 1, "ses": "asia", "ovl": 0, "open": 1}}
             for i in range(2)]
    monkeypatch.setattr(aibt, "filter_verdict",
                        lambda v, s, generated_at=None: v)
    monkeypatch.setattr(aibt, "parse_verdict_batch",
                        lambda raw, n: [verdict] * n)
    monkeypatch.setattr(aibt, "parse_verdict", lambda raw: verdict)
    monkeypatch.setattr(aibt, "_llm_verdict", lambda *a, **k: "{}")
    results, cache = {}, {}
    totals = {"llm_calls": 0, "tokens_prompt": 0, "tokens_completion": 0,
              "tokens_total": 0, "cache_hits": 0}
    aibt._flush_pending([(0, BASE, snaps[0], "k0"),
                         (1, BASE + 1, snaps[1], "k1")],
                        results, cache, totals, "m", "sys",
                        symbol="BTCUSDT", timeframe="1H")
    c = schema.init_db(tmp_path / "cbr.db")
    rows = c.execute("SELECT verdict, source FROM snapshots").fetchall()
    assert len(rows) == 2
    assert all(r["verdict"] == "S" for r in rows)
    assert all(r["source"] == "backtest" for r in rows)


def test_hook_levels_for_slice(monkeypatch, tmp_path):
    from app_pkg.ai import ai_backtest as aibt
    _enable_cbr(monkeypatch, tmp_path, True)
    snap = {"t": {"rsi": 50.0, "atr": 2.0, "close": 100.0},
            "c": {"h": 15, "dow": 3, "ses": "london_ny", "ovl": 1, "open": 1}}
    monkeypatch.setattr(aibt, "compact_snapshot",
                        lambda s, tf, upto_sec=None: snap)
    monkeypatch.setattr(aibt, "_llm_request",
                        lambda *a, **k: json.dumps(
                            {"targets": [
                                {"side": "UP", "price": 104.0,
                                 "probability": 0.6},
                                {"side": "DOWN", "price": 96.0,
                                 "probability": 0.4}]}))
    monkeypatch.setattr(aibt, "filter_levels", lambda lv, snap: lv)
    monkeypatch.setattr(aibt, "_structure_levels", lambda *a, **k: [])
    levels, price, err = aibt.levels_for_slice("BTCUSDT", "1H",
                                               upto_sec=None,
                                               current_price=100.0)
    assert err is None
    assert len(levels) == 2
    c = schema.init_db(tmp_path / "cbr.db")
    row = c.execute("SELECT source, levels_json, entry_price, ts "
                    "FROM snapshots").fetchone()
    assert row["source"] == "live"
    assert row["entry_price"] == pytest.approx(100.0)
    lv = json.loads(row["levels_json"])
    # delta = (price-close)/close*100
    assert lv[0]["delta"] == pytest.approx(4.0)
    assert lv[0]["prob"] == pytest.approx(0.6)
    assert lv[1]["delta"] == pytest.approx(-4.0)
    assert row["ts"] == int(price) or row["ts"] > 0


def test_hook_market_snapshot_live(monkeypatch, tmp_path):
    from app_pkg.data import market_snapshot as ms
    _enable_cbr(monkeypatch, tmp_path, True)
    raw = {
        "technicals": {"rsi": 50.0, "atr": 2.0, "bb_pct_b": 0.5,
                       "sma20_diff_pct": 0.4, "atr_percentile": 0.6,
                       "dist_to_high_pct": -0.2, "dist_to_low_pct": 0.2,
                       "close": 100.0, "ok": True},
        "clock": {"hour_utc": 14, "dow": 2, "session": "london",
                  "london_ny_overlap": 1, "market_open": 1, "ok": True},
    }
    df = pd.DataFrame({
        "timestamp": [pd.Timestamp(BASE, unit="s", tz="UTC")],
        "open": [99.0], "high": [101.0], "low": [98.0], "close": [100.0],
        "volume": [100.0],
    })
    monkeypatch.setattr(ms, "get_raw_market_data", lambda *a, **k: raw)
    monkeypatch.setattr(ms, "_mtf_compact", lambda *a, **k: {})
    monkeypatch.setattr(ms, "_slice_df", lambda *a, **k: df)
    out = ms.compact_snapshot("BTCUSDT", "1H")
    assert out.get("t") is not None
    c = schema.init_db(tmp_path / "cbr.db")
    row = c.execute("SELECT source, ts, entry_price FROM snapshots").fetchone()
    assert row["source"] == "live"
    assert row["ts"] == BASE
    assert row["entry_price"] == pytest.approx(100.0)


def test_hook_disabled_writes_nothing(monkeypatch, tmp_path):
    from app_pkg.ai import ai_backtest as aibt
    _enable_cbr(monkeypatch, tmp_path, False)  # CBR_ENABLED=False
    snap = {"t": {"rsi": 50.0, "atr": 2.0, "close": 100.0},
            "c": {"h": 14, "dow": 2, "ses": "london", "ovl": 1, "open": 1}}
    verdict = {"sig": "L", "pu": 0.7, "pd": 0.3, "pf": 0.0, "conf": 0.7,
               "tg": []}
    monkeypatch.setattr(aibt, "filter_verdict",
                        lambda v, s, generated_at=None: v)
    monkeypatch.setattr(aibt, "parse_verdict", lambda raw: verdict)
    monkeypatch.setattr(aibt, "_llm_verdict", lambda *a, **k: "{}")
    results, cache, totals = {}, {}, {"llm_calls": 0, "tokens_prompt": 0,
                                       "tokens_completion": 0,
                                       "tokens_total": 0, "cache_hits": 0}
    aibt._flush_pending([(0, BASE, snap, "k")], results, cache, totals,
                        "m", "sys", symbol="BTCUSDT", timeframe="1H")
    db_file = tmp_path / "cbr.db"
    assert not db_file.exists() or \
        schema.init_db(db_file).execute(
            "SELECT COUNT(*) c FROM snapshots").fetchone()["c"] == 0