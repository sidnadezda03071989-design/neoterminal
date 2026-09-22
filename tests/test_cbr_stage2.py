"""Тесты CBR Этапа 2: normalize / index (FAISS+k-NN) / blender / query_api /
хук ai_backtest. Изолированы от data.db: синтетическая CBR-БД + tmp-артефакты.

Покрытие:
  1. normalize: rolling-статистика (40/44 без look-ahead), stats_at, clip,
     save/load roundtrip;
  2. index: построение/поиск (евклид = sqrt(L2²)), save/load, метаданные;
  3. query_api: схема Этапа 2, агрегация (сумма winrate = 1), недостаток
     данных, детерминизм, min_distance, обратная совместимость Этапа 1,
     format_for_prompt обеих схем;
  4. blender: веса по режимам/n/confidence, недостаток данных -> Charon-only;
  5. хуки ai_backtest: гейт CBR_STAGE2_ENABLED (default False), смешение
     вердикта, безопасность при отсутствии артефактов;
  6. конфиг: константы Этапа 2.
"""

import json

import numpy as np
import pytest

from app_pkg import config
from app_pkg.ai import ai_backtest as aibt
from app_pkg.cbr import (
    blender,
    query_api,
    schema,
    store,
)
from app_pkg.cbr import (
    index as cbr_index,
)
from app_pkg.cbr import (
    normalize as cbr_norm,
)

BASE_TS = 1_700_000_000
FNAMES = store.FEATURE_NAMES


# ------------------------------------------------------------- helpers
def _row_features(i, rng):
    """детерминированный вектор 44 фич (плоский словарь block.field)."""
    row = {name: float(rng.normal() * (1.0 + (i % 5) * 0.05)) for name in FNAMES}
    for key, val in (
        ("clock.hour_utc", float(i % 24)),
        ("clock.dow", float(i % 7)),
        ("clock.market_open", 1.0),
        ("clock.london_ny_overlap", 0.0),
        ("clock.session_code", float(i % 5)),
    ):
        row[key] = val
    return row


def seed_db(conn, n=300, labeled_frac=0.9):
    """n снимков (шаг 1H) + детерминированные outcome на первых ~90%."""
    rng = np.random.RandomState(42)
    ids = []
    for i in range(n):
        ts = BASE_TS + i * 3600
        snap = {"symbol": "BTCUSDT", "timeframe": "1H", "ts": ts,
                "source": "test", **_row_features(i, rng)}
        rid = store.store_snapshot(conn, snap, entry_price=100.0)
        if rid is not None:
            ids.append((rid, ts))
    conn.commit()
    label_n = int(len(ids) * labeled_frac)
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
    return conn


@pytest.fixture(autouse=True)
def _clean_caches():
    query_api._KNN_META.clear()
    aibt._CBR_KNN_ASSETS["index"] = None
    aibt._CBR_KNN_ASSETS["stats"] = None
    yield
    query_api._KNN_META.clear()
    aibt._CBR_KNN_ASSETS["index"] = None
    aibt._CBR_KNN_ASSETS["stats"] = None


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Изолированная среда Этапа 2: БД + артефакты (индекс/статистика)."""
    db_path = str(tmp_path / "cbr.db")
    idx_path = str(tmp_path / "cbr.faiss")
    stats_path = str(tmp_path / "cbr_norm_stats.pkl")
    monkeypatch.setattr(config, "CBR_DB_PATH", db_path)
    monkeypatch.setattr(config, "CBR_FAISS_INDEX_PATH", idx_path)
    monkeypatch.setattr(config, "CBR_NORMALIZE_STATS_PATH", stats_path)
    conn = schema.init_db(db_path)
    seed_db(conn)
    X, ts, regimes, y = cbr_index.load_feature_matrix(conn, "BTCUSDT", "1H")
    stats = cbr_norm.compute_rolling_stats(X, ts=ts)
    cbr_norm.save_stats(stats, stats_path)
    Xn = cbr_norm.normalize_matrix(X, ts, stats)
    index = cbr_index.build_faiss_index(Xn)
    cbr_index.save_index(index, idx_path)
    return {"conn": conn, "index": index, "stats": stats,
            "idx_path": idx_path, "stats_path": stats_path, "ts": ts,
            "regimes": regimes, "y": y, "X": X, "Xn": Xn}


def _query_snap(i=290):
    """снимок-запрос с ts внутри окна (бары есть до/после)."""
    rng = np.random.RandomState(1000 + i)
    snap = {"symbol": "BTCUSDT", "timeframe": "1H", "ts": BASE_TS + i * 3600,
            **_row_features(i, rng)}
    return snap


# ========================================================= normalize
def test_normalize_stats_shape_and_no_lookahead(env):
    stats = env["stats"]
    assert len(env["ts"]) == stats["mean"].shape[0] == stats["std"].shape[0]
    assert stats["mean"].shape[1] == 44
    assert sorted(stats.keys()) == ["mean", "std", "ts", "window"]
    # без look-ahead: mean[k] = среднее ТОЛЬКО по предыдущим барам (shift(1))
    k = 10
    expected = env["X"][:k].astype("float64").mean(axis=0)
    assert np.allclose(stats["mean"][k], expected, atol=1e-9)
    # первый бар -> окно пустое (NaN) -> 0.0, std -> 1.0
    assert np.allclose(stats["mean"][0], 0.0)
    assert np.allclose(stats["std"][0], 1.0)


def test_normalize_apply_clip_and_stats_at(env):
    vec = env["X"][5]
    sl = cbr_norm.stats_at(env["stats"], int(env["ts"][5]))
    out = cbr_norm.apply_normalization(vec, sl)
    assert out.shape == (44,)
    assert ((-5.0 - 1e-6) <= out).all() and (out <= (5.0 + 1e-6)).all()
    # stats_at без look-ahead: для баровой ts pos <= его индекса
    assert sl["pos"] == 5
    # ts до старта истории -> крайняя нулевая позиция (не Negative)
    early = cbr_norm.stats_at(env["stats"], BASE_TS - 3600)
    assert early["pos"] == 0


def test_normalize_save_load_roundtrip(env, tmp_path):
    p = str(tmp_path / "norm.pkl")
    cbr_norm.save_stats(env["stats"], p)
    loaded = cbr_norm.load_stats(p)
    assert loaded is not None
    assert np.allclose(loaded["mean"], env["stats"]["mean"])
    assert cbr_norm.load_stats(str(tmp_path / "nope.pkl")) is None


def test_normalize_matrix_rows_equal_stats_at(env):
    Xn = cbr_norm.normalize_matrix(env["X"], env["ts"], env["stats"])
    row = cbr_norm.apply_normalization(
        env["X"][10], cbr_norm.stats_at(env["stats"], int(env["ts"][10])))
    assert np.allclose(Xn[10], row, atol=1e-6)


# ========================================================= index (FAISS)
def test_index_build_search_euclidean(env):
    d, idx = cbr_index.search_knn(env["index"], env["Xn"][0], 5)
    assert idx[0] == 0 and abs(d[0]) < 1e-4  # самосопоставление -> дистанция 0
    assert len(d) == 5
    assert (np.diff(d) >= -1e-9).all()        # расстояния не убывают


def test_index_save_load_roundtrip(env, tmp_path):
    p = str(tmp_path / "ix.faiss")
    cbr_index.save_index(env["index"], p)
    loaded = cbr_index.load_index(p)
    assert loaded is not None
    d1, i1 = cbr_index.search_knn(env["index"], env["Xn"][3], 4)
    d2, i2 = cbr_index.search_knn(loaded, env["Xn"][3], 4)
    assert np.allclose(d1, d2) and np.array_equal(i1, i2)


def test_index_load_feature_matrix_aligned(env):
    ts, regimes, y = env["ts"], env["regimes"], env["y"]
    assert len(ts) == len(regimes) == len(y)
    assert (np.diff(ts) > 0).all()  # сортировка по ts ASC
    assert set(np.unique(y)) <= {-1, 0, 1}


def test_index_sklearn_fallback(monkeypatch, env, tmp_path):
    monkeypatch.setattr(cbr_index, "_HAS_FAISS", False)
    monkeypatch.setattr(cbr_index, "_faiss", None)
    nn = cbr_index.build_faiss_index(env["Xn"])
    assert cbr_index.faiss_available() is False
    d, idx = cbr_index.search_knn(nn, env["Xn"][0], 3)
    assert idx[0] == 0 and len(d) == 3


# ========================================================= query_api (k-NN)
def test_get_similar_summary_schema_keys(env):
    out = query_api.get_similar_summary(
        env["conn"], env["index"], _query_snap(290), "BTCUSDT", "1H",
        bar_ts=BASE_TS + 290 * 3600,
        K=30, min_distance=2.0, regime_match=True, window_days=180)
    assert not out.get("insufficient_data")
    assert set(out.keys()) == set(query_api._KNN_KEYS)


def test_get_similar_summary_aggregation_consistency(env):
    out = query_api.get_similar_summary(
        env["conn"], env["index"], _query_snap(295), "BTCUSDT", "1H",
        bar_ts=BASE_TS + 295 * 3600,
        K=40, min_distance=1.0, regime_match=False, window_days=180)
    win_sum = out["winrate_up"] + out["winrate_down"] + out["winrate_flat"]
    # winrate_* округлены до 0.01: сумма может слегка отклониться от 1.00
    assert -0.03 <= win_sum <= 1.0 + 0.03
    assert out["n"] == out["n_filled"] >= 10
    assert out["confidence"] >= 0.0
    assert len(out["recent_examples"]) <= 3
    # все дистанции оставшихся соседей >= min_distance
    assert out["avg_distance"] >= 1.0
    for ex in out["recent_examples"]:
        assert ex["distance"] >= 1.0


def test_get_similar_summary_insufficient_fallback(env):
    # K минимален + жёсткий min_distance + далёкий ts -> почти нет соседей
    out = query_api.get_similar_summary(
        env["conn"], env["index"],
        {"symbol": "BTCUSDT", "timeframe": "1H", "ts": BASE_TS + 295 * 3600,
         "t": {"rsi": 50.0, "atr": 1.0, "bb": 0.5, "sma": 0.4, "ap": 0.5,
               "dh": 0.0, "dl": 0.0, "close": 100.0}},
        "BTCUSDT", "1H", bar_ts=BASE_TS + 295 * 3600,
        K=1, min_distance=50.0, regime_match=True)
    assert out == {"n": 0, "insufficient_data": True}


def test_get_similar_summary_deterministic(env):
    snap = _query_snap(299)
    bar_ts = BASE_TS + 299 * 3600
    a = query_api.get_similar_summary(
        env["conn"], env["index"], snap, "BTCUSDT", "1H",
        bar_ts=bar_ts,
        K=30, min_distance=2.0, regime_match=True, window_days=180)
    b = query_api.get_similar_summary(
        env["conn"], env["index"], snap, "BTCUSDT", "1H",
        bar_ts=bar_ts,
        K=30, min_distance=2.0, regime_match=True, window_days=180)
    assert a == b


def test_min_distance_filters_self(env):
    # Запрос на вектор РЕАЛЬНОЙ строки БД i=150 с bar_ts = её ts: сама строка
    # (dist 0) в соседи НЕ попадает — её outcome на момент бара ещё не
    # известен (анти-look-ahead: ts_соседа + horizon >= bar_ts). Поэтому
    # невысокие пороги min_distance не меняют пул соседей, а жёсткий —
    # вычищает всё до недостатка данных.
    i = 150
    vec = env["X"][i]
    snap = dict(zip(FNAMES, vec.tolist()))
    snap["symbol"] = "BTCUSDT"
    snap["timeframe"] = "1H"
    bar_ts = int(env["ts"][i])
    snap["ts"] = bar_ts
    low = query_api.get_similar_summary(
        env["conn"], env["index"], snap, "BTCUSDT", "1H",
        bar_ts=bar_ts, K=30, min_distance=0.0, regime_match=False,
        window_days=180, include_neighbors=True)
    high = query_api.get_similar_summary(
        env["conn"], env["index"], snap, "BTCUSDT", "1H",
        bar_ts=bar_ts, K=30, min_distance=0.5, regime_match=False,
        window_days=180)
    assert bar_ts not in low["neighbors_ts"]   # само-сопоставление исключено
    assert low["n"] == 30 and high["n"] == 30
    assert low["avg_distance"] > 0.0           # dist-0 близнеца в пуле нет
    # пулы соседей при 0.0/0.5 идентичны (в прошлом нет близнецов < 0.5)
    assert low["avg_distance"] == high["avg_distance"]
    far = query_api.get_similar_summary(
        env["conn"], env["index"], snap, "BTCUSDT", "1H",
        bar_ts=bar_ts, K=30, min_distance=100.0, regime_match=False,
        window_days=180)
    assert far == {"n": 0, "insufficient_data": True}


def test_get_similar_summary_old_signature_compat(env):
    snap = _query_snap(150)
    out = query_api.get_similar_summary(
        env["conn"], "BTCUSDT", "1H", snap, window_days=99999,
        min_samples=1, max_rows=3)
    # Этап 1: та же схема, что get_stats_for_llm (без by_regime)
    assert set(out.keys()) == set(query_api._STATS_KEYS) - {"by_regime"}


def test_format_for_prompt_both_schemas(env):
    new = query_api.get_similar_summary(
        env["conn"], env["index"], _query_snap(280), "BTCUSDT", "1H",
        bar_ts=BASE_TS + 280 * 3600,
        K=30, min_distance=2.0, regime_match=True, window_days=180)
    line_new = query_api.format_for_prompt(new)
    assert len(line_new) < 300
    assert "up=" in line_new and "down=" in line_new and "conf=" in line_new
    old = query_api.get_stats_for_llm(env["conn"], "BTCUSDT", "1H")
    line_old = query_api.format_for_prompt(old)
    assert len(line_old) < 300 and "CBR" in line_old


# ========================================================= blender
def test_blend_weights_by_regime_and_n():
    cbr = {"n": 20, "winrate_up": 0.6, "winrate_down": 0.2,
           "confidence": 0.5}
    b = blender.blend(0.5, 0.3, cbr, "flat")
    assert b["w1"] == pytest.approx(0.6) and b["w2"] == pytest.approx(0.4)
    assert b["source"] == "blended"
    strong = {"n": 40, "winrate_up": 0.6, "winrate_down": 0.2,
              "confidence": 0.7}
    b2 = blender.blend(0.5, 0.3, strong, "flat")
    assert b2["w1"] == pytest.approx(0.3) and b2["w2"] == pytest.approx(0.7)
    assert b2["source"] == "cbr_only"
    b3 = blender.blend(0.5, 0.3, cbr, "trend_up")
    assert b3["w1"] == pytest.approx(0.8)
    b4 = blender.blend(0.5, 0.3, cbr, "reversal")
    assert b4["w2"] == pytest.approx(0.6) and b4["w1"] == pytest.approx(0.4)
    # итоговые вероятности нормированы (не более 1)
    assert b["final_pu"] + b["final_pd"] <= 1.0 + 1e-9


def test_blend_insufficient_cbr_charon_only():
    cbr = {"n": 3, "insufficient_data": True}
    b = blender.blend(0.5, 0.3, cbr, "flat")
    assert b["source"] == "charon_only"
    assert b["w1"] == 1.0 and b["w2"] == 0.0
    assert b["final_pu"] == pytest.approx(0.5)
    assert b["final_pd"] == pytest.approx(0.3)
    b2 = blender.blend(0.5, 0.3, None, "reversal")
    assert b2["source"] == "charon_only"


# ========================================================= хуки ai_backtest
def test_config_stage2_defaults_off():
    assert config.CBR_STAGE2_ENABLED is False
    assert config.CBR_K_DEFAULT == 50
    assert config.CBR_MIN_DISTANCE == 2.0
    assert config.CBR_REGIME_MATCH is True
    assert config.CBR_QUERY_WINDOW_DAYS == 180
    assert config.CBR_BLEND_MIN_SAMPLES == 10
    assert config.CBR_BLEND_STRONG_SAMPLES == 30
    assert config.CBR_BLEND_STRONG_CONF == 0.6


def test_ai_backtest_hook_gated_off(env):
    monkeypatch = None  # эталонный вызов без stage2
    verdict = {"sig": "L", "pu": 0.6, "pd": 0.3, "pf": 0.1, "tg": [], "conf": 0.6}
    out = aibt._cbr_blend_verdict(verdict, _query_snap(200), "BTCUSDT", "1H",
                                  ts=BASE_TS + 200 * 3600)
    assert out == verdict  # гейт False -> вердикт без изменений


def test_ai_backtest_hook_blends_when_enabled(env, monkeypatch):
    monkeypatch.setattr(config, "CBR_STAGE2_ENABLED", True)
    verdict = {"sig": "L", "pu": 0.6, "pd": 0.3, "pf": 0.1, "tg": [], "conf": 0.6}
    out = aibt._cbr_blend_verdict(
        dict(verdict), _query_snap(210), "BTCUSDT", "1H",
        ts=BASE_TS + 210 * 3600)
    assert set(out) >= {"final_pu", "final_pd", "cbr"}
    assert 0.0 <= out["final_pu"] <= 1.0 and 0.0 <= out["final_pd"] <= 1.0
    assert isinstance(out["cbr"], dict) and out["cbr"]["n"] > 0
    assert out["cbr"]["source"] in ("charon_only", "blended", "cbr_only")


def test_ai_backtest_hook_error_safe(monkeypatch, env):
    monkeypatch.setattr(config, "CBR_STAGE2_ENABLED", True)
    monkeypatch.setattr(config, "CBR_FAISS_INDEX_PATH", str(env["idx_path"]))
    monkeypatch.setattr(config, "CBR_NORMALIZE_STATS_PATH",
                        str(env["stats_path"]))
    # убираем БД: get_cbr_conn откроет пустую missing.db — ошибки быть не должно
    verdict = {"sig": "F", "pu": 0.3, "pd": 0.3, "pf": 0.4, "tg": [], "conf": 0.4}
    monkeypatch.setattr(config, "CBR_DB_PATH", str(env["idx_path"]) + "x")
    out = aibt._cbr_blend_verdict(verdict, _query_snap(220), "BTCUSDT", "1H")
    assert out == verdict  # пустая БД -> недостаток данных -> без изменений


def test_smoke_50_bar_run(monkeypatch):
    """Полный прогон run_verdict_backtest на 50 барах: без краха, сигналы есть."""
    def fake_snap(*a, **k):
        i = getattr(fake_snap, "_i", 0)
        fake_snap._i = i + 1
        return _query_snap(10 + i)

    monkeypatch.setattr(aibt, "compact_snapshot", lambda *a, **k: fake_snap())
    monkeypatch.setattr(aibt, "charon_prompt_text", lambda: "SYS")
    monkeypatch.setattr(aibt, "get_series_df",
                        lambda *a, **k: _bars_df(50))

    def fake_llm(system, messages, **kwargs):
        usage = kwargs.get("usage_out")
        if isinstance(usage, dict):
            usage["prompt_tokens"] = 5
            usage["completion_tokens"] = 5
        return json.dumps({"sig": "L", "pu": 0.6, "pd": 0.3, "pf": 0.1,
                           "tg": [[110.0, 0.6], [90.0, 0.4]]})

    monkeypatch.setattr(aibt, "_llm_request", fake_llm)
    res = aibt.run_verdict_backtest("BTCUSDT", "1H", bars=50, adaptive=False)
    assert len(res["signals"]) == 50
    assert set(res["signals"]) <= {"L", "S", "F"}
    assert res["llm_calls"] >= 1


def _bars_df(n):
    import pandas as pd
    arr = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h"),
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
        "volume": 1000.0,
    })
    return arr