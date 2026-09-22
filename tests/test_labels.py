# -*- coding: utf-8 -*-
"""Triple-barrier разметка (app_pkg/ml/labels.py).

Без сети, синтетические ряды. Проверяем:
  - восходящий тренд -> +1, нисходящий -> -1, флэт -> 0 (таймаут);
  - обе линии на одном баре -> -1 (приоритет SL, конвенция движка);
  - хвост короче horizon -> NaN (метка невычислима), вырожденный
    барьер (atr_pct<=0) -> 0;
  - meta: n_computable, n_dropped, class_balance;
  - build_labeled_dataset: признаки + метки, dropna, запись CSV,
    round-trip катится, пустой df не падает.
"""

import numpy as np
import pandas as pd
import pytest

from app_pkg.data import market_snapshot
from app_pkg.ml.labels import (build_labeled_dataset, charon_features,
                               default_features, triple_barrier_labels)

N = 300
HORIZON = 20


def _df(close, high_low=0.5):
    """OHLCV со строками по возрастанию (без timestamp — clock не применим).

    volume НЕ константный (иначе у Charon-блока volume.vol_zscore вечный
    None и dropna при сборке датасета выбросит все строки).
    """
    n = len(close)
    vol = 1000.0 + 400.0 * np.sin(np.arange(n) / 5.0) + np.arange(n) * 2.0
    return pd.DataFrame({
        "open": np.asarray(close) - high_low / 2,
        "high": np.asarray(close) + high_low,
        "low": np.asarray(close) - high_low,
        "close": np.asarray(close, dtype=float),
        "volume": vol,
    })


def _df_with_ts(close, step=3600, end_ts=1_700_000_000):
    """_df + колонка timestamp (для блока clock в Charon-фичах)."""
    df = _df(close)
    ts = [end_ts - i * step for i in range(len(close))][::-1]
    df["timestamp"] = pd.to_datetime(ts, unit="s", utc=True)
    return df


def _int_labels(labels):
    """Учебные метки как int (NaN остаётся NaN)."""
    return np.where(np.isnan(labels.astype(float)), np.nan,
                    labels.astype(float)).astype(float)


# -------------------------------------------------------------- направление
def test_upward_trend_labels_plus_one():
    df = _df(100.0 + np.arange(N))
    labels, meta = triple_barrier_labels(df, k=2.0, horizon=HORIZON)
    valid = labels[:meta["n_computable"]]
    assert all(v == 1 for v in valid)
    assert meta["class_balance"] == {-1: 0, 0: 0, 1: meta["n_computable"]}
    assert meta["n_computable"] == N - HORIZON
    assert meta["n_dropped"] == HORIZON


def test_downward_trend_labels_minus_one():
    df = _df(300.0 - np.arange(N))
    labels, meta = triple_barrier_labels(df, k=2.0, horizon=HORIZON)
    valid = labels[:meta["n_computable"]]
    assert all(v == -1 for v in valid)


def test_flat_series_timeout_zero():
    close = np.full(N, 100.0)
    df = _df(close)
    labels, meta = triple_barrier_labels(df, k=2.0, horizon=HORIZON)
    valid = labels[:meta["n_computable"]]
    assert all(v == 0 for v in valid)


# -------------------------------------------------- обе линии на одном баре
def test_same_bar_both_touches_gives_sl():
    """Низкие/высокие пробиты на ОДНОМ баре -> -1 (приоритет нижнего)."""
    close = np.full(60, 100.0)
    df = _df(close)
    # atr≈1 (high-low=1) -> upper≈102, lower≈98
    df.loc[30, "high"] = 103.0   # >= upper
    df.loc[30, "low"] = 97.0     # <= lower
    labels, _ = triple_barrier_labels(df, k=2.0, horizon=HORIZON)
    # бар 30 попадает в окна t=10..29 (t+1..t+20)
    assert labels[10] == -1 and labels[15] == -1 and labels[29] == -1
    assert labels[5] == 0        # окно 6..25 бар 30 не содержит
    assert labels[28] == -1


def test_same_bar_up_only_gives_win():
    """Если пробит только верхний барьер -> +1 (контроль к -1 выше)."""
    close = np.full(60, 100.0)
    df = _df(close)
    df.loc[30, "high"] = 103.0
    df.loc[30, "low"] = 100.5   # по-прежнему > lower
    labels, _ = triple_barrier_labels(df, k=2.0, horizon=HORIZON)
    assert labels[10] == 1 and labels[29] == 1


# --------------------------------------------------------- крайние случаи
def test_tail_nan_and_degenerate():
    close = np.full(N, 100.0)
    df = _df(close)
    labels, meta = triple_barrier_labels(df, k=2.0, horizon=HORIZON)
    tail = labels[meta["n_computable"]:]
    assert all(np.isnan(float(v)) for v in tail)
    # баров меньше horizon+1 -> все метки NaN, без исключения
    df_small = _df(np.full(10, 100.0))
    lbl, m = triple_barrier_labels(df_small, k=2.0, horizon=HORIZON)
    assert len(lbl) == 10 and m["n_computable"] == 0
    assert all(np.isnan(float(v)) for v in lbl)


def test_degenerate_zero_atr():
    """high==low==close -> atr_pct=0, барьер вырожден -> метка 0."""
    df = pd.DataFrame({
        "open": [100.0] * 40, "high": [100.0] * 40, "low": [100.0] * 40,
        "close": [100.0] * 40, "volume": [1.0] * 40,
    })
    labels, meta = triple_barrier_labels(df, k=2.0, horizon=HORIZON)
    valid = labels[:meta["n_computable"]]
    assert all(v == 0 for v in valid)


# ------------------------------------------------------------- meta / края
def test_meta_geometry():
    df = _df(100.0 + np.arange(N))
    labels, meta = triple_barrier_labels(df, k=2.0, horizon=HORIZON)
    assert meta["k"] == 2.0 and meta["horizon"] == HORIZON
    assert meta["n_rows"] == N
    assert meta["n_computable"] + meta["n_dropped"] == N
    assert len(labels) == N


def test_empty_df():
    empty = pd.DataFrame({
        "open": [], "high": [], "low": [], "close": [], "volume": []})
    labels, meta = triple_barrier_labels(empty)
    assert len(labels) == 0
    X, y, meta2 = build_labeled_dataset(empty)
    assert len(X) == 0 and len(y) == 0
    assert meta2["n_computable"] == 0


def test_bad_k_or_horizon_returns_nan_labels():
    df = _df(100.0 + np.arange(N))
    labels, _ = triple_barrier_labels(df, k=0.0, horizon=HORIZON)
    assert all(np.isnan(float(v)) for v in labels)
    labels2, _ = triple_barrier_labels(df, k=2.0, horizon=0)
    assert all(np.isnan(float(v)) for v in labels2)


# ------------------------------------------------------- build_dataset
def test_build_dataset_consistency():
    """По умолчанию X собирается из Charon-фич (market_snapshot), не OHLCV."""
    df = _df(100.0 + np.arange(N))
    X, y, meta = build_labeled_dataset(df, k=2.0, horizon=HORIZON)
    assert len(X) == len(y) == meta["n_labeled"]
    assert meta["n_computable"] == N - HORIZON
    assert meta["n_skipped"] == meta["n_computable"] - len(y)
    assert set(np.unique(y)) == {1}   # тренд вверх, все +1
    assert "label" not in X.columns
    # реальные фичи Харона: плоские 'block.field', а не 8 OHLCV-индикаторов
    assert len(X.columns) > 25
    assert {"technicals.rsi", "trend.adx", "momentum.macd_hist",
            "regime.hurst", "divergence.rsi_price", "candle.pinbar",
            "volume.rel_vol", "volatility.hv20"} <= set(X.columns)
    # прогрев Charon-блоков (~100 баров) отброшен dropna
    assert 100 < len(y) < N - HORIZON


def test_build_dataset_ohlcv_baseline_explicit():
    """OHLCV-baseline остаётся доступным явно (сравнительный «пол»)."""
    df = _df(100.0 + np.arange(N))
    X, y, meta = build_labeled_dataset(
        df, k=2.0, horizon=HORIZON, features_fn=default_features)
    assert len(X.columns) <= 8
    assert set(np.unique(y)) == {1}


def test_build_dataset_csv_roundtrip(tmp_path):
    df = _df(100.0 + np.arange(N))
    X, y, meta = build_labeled_dataset(
        df, k=2.0, horizon=HORIZON, out_dir=str(tmp_path),
        csv_name="labels.csv")
    path = tmp_path / "labels.csv"
    assert path.exists()
    reloaded = pd.read_csv(path)
    assert len(reloaded) == len(X)
    assert "label" in reloaded.columns
    assert reloaded["label"].isna().sum() == 0
    for col in X.columns:
        assert np.allclose(reloaded[col].to_numpy(), X[col].to_numpy(),
                           equal_nan=True)


def test_custom_features_callable():
    df = _df(100.0 + np.arange(N))
    feat = lambda d: pd.DataFrame({"trend": np.asarray(d["close"])})
    X, y, meta = build_labeled_dataset(
        df, k=2.0, horizon=HORIZON, features_fn=feat)
    assert list(X.columns) == ["trend"]


# ------------------------------------------------------------ Charon-фичи
def test_charon_features_flat_numeric_vector():
    df = _df_with_ts(100.0 + np.arange(N))
    f = charon_features(df)
    assert len(f) == N
    # плоские колонки 'block.field' из market_snapshot + clock
    assert "technicals.rsi" in f.columns
    assert "clock.hour_utc" in f.columns
    assert len(f.columns) == 44          # technicals.close удалён (было 45)
    assert "technicals.close" not in f.columns
    assert (f.dtypes == "float64").all()        # только числа
    assert not ({"ok", "applicable"} & set(f.columns))
    assert f.iloc[250].notna().all()            # после прогрева всё конечно


def test_charon_features_matches_market_snapshot_window():
    """Признак бара t == снимок market_snapshot по окну, кончающемуся t.

    Проверяем по барам 250 и последнему для ВСЕХ локальных блоков (44 поля)
    — причинный расчёт на полном ряду даёт те же числа, что окно снимка.
    """
    df = _df_with_ts(100.0 + np.arange(N))
    f = charon_features(df)
    blocks = [
        ("technicals", market_snapshot._technicals_from_df),
        ("volume", market_snapshot._volume_block),
        ("trend", market_snapshot._trend_block),
        ("momentum", market_snapshot._momentum_block),
        ("volatility", market_snapshot._volatility_block),
        ("regime", market_snapshot._regime_block),
        ("divergence", market_snapshot._divergence_block),
        ("candle", market_snapshot._candle_block),
    ]
    for t in (150, N - 1):
        window = df.iloc[:t + 1].tail(300)
        for prefix, fn in blocks:
            res = fn(window)
            if not res.get("ok"):
                continue
            for k, v in res.items():
                if k in ("ok", "applicable") or v is None:
                    continue
                col = prefix + "." + k
                if col not in f.columns:     # поля, намеренно не включённые в вектор
                    continue
                assert abs(float(f[col].iloc[t]) - float(v)) < 1e-12, \
                    f"{col}@t={t}: {f[col].iloc[t]} != {v}"


def test_charon_features_warmup_nan():
    df = _df_with_ts(100.0 + np.arange(N))
    f = charon_features(df)
    assert np.isnan(f.iloc[0]["technicals.rsi"])  # блок ok:false -> NaN
    assert np.isnan(f.iloc[0]["trend.adx"])
    assert not np.isnan(f.iloc[250]["trend.adx"])


def test_charon_features_no_timestamp_no_clock():
    df = _df(100.0 + np.arange(N))              # без timestamp
    f = charon_features(df)
    assert not any(c.startswith("clock.") for c in f.columns)
    assert "technicals.rsi" in f.columns


def test_charon_features_empty_df():
    empty = pd.DataFrame({"close": []})
    f = charon_features(empty)
    assert len(f) == 0


# ----------------------------------------------------- default_features
def test_default_features_columns_and_finite():
    df = _df(100.0 + np.arange(300))
    f = default_features(df)
    assert list(f.columns) == ["rsi", "atr_pct", "bb_pct_b", "ema_diff_pct",
                               "vol_zscore", "roc_5", "dist_hi_20",
                               "dist_lo_20"]
    row = f.iloc[250]
    assert row.notna().all()          # после прогрева всё конечно
    assert np.isfinite(row.to_numpy()).all()