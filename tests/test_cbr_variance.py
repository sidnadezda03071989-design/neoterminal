"""Тесты variance-check CBR-БД перед Этапом 2 (FAISS + k-NN).

Покрытие:
  - low_variance: константные фичи детектятся (std/|mean| < 0.1);
  - pairwise_ratio: все точки одинаковые -> ratio ≈ 0;
  - pca: точки на одной прямой -> n_80 = 1;
  - knn_up_std: случайные метки -> up_std ≈ binomial (√(p(1-p)/K), p=1/3);
  - run_check: синтетика с явной структурой (3 кластера + корреляция
    исхода с режимом) -> вердикт GO.
"""

import numpy as np
import pytest

from scripts.cbr_variance_check import (
    SEED,
    metric_knn_up_std,
    metric_low_variance,
    metric_pairwise_ratio,
    metric_pca,
    run_check,
    zscore,
)


def test_low_variance_detects_constants():
    """3 константные фичи -> count == 3, остальные с шумом не считаются."""
    rng = np.random.default_rng(SEED)
    X = rng.normal(0.0, 1.0, (200, 44))
    X[:, 5] = 5.0
    X[:, 17] = 0.0
    X[:, 40] = -2.0
    m = metric_low_variance(X)
    assert m["value"] == 3
    assert m["total"] == 44
    assert m["status"] != "FAIL"


def test_pairwise_ratio_identical_points():
    """Все точки одинаковые -> все расстояния 0 -> ratio ≈ 0 (FAIL-ветка)."""
    X = np.ones((50, 44), dtype="float64") * 3.0
    m = metric_pairwise_ratio(X)
    assert m["status"] == "FAIL"
    assert m["value"] == pytest.approx(0.0, abs=1e-6)


def test_pca_on_rank1_data():
    """Точки на одной прямой (X[i, :] = i) -> 1 компонента = 100%."""
    n = 80
    X = np.tile(np.arange(n, dtype="float64")[:, None], (1, 44))
    assert np.linalg.matrix_rank(zscore(X)) == 1
    m = metric_pca(X)
    assert m["value"] == 1
    assert m["status"] == "OK"


def test_knn_up_std_on_random_labels():
    """Случайные метки {-1,0,1}, up=1 -> std ≈ sqrt(p(1-p)/K), p=1/3.

    Ожидание (binomial): sqrt((1/3)(2/3)/50) ≈ 0.067 — разумный диапазон.
    """
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(2000, 44))
    y = rng.choice(np.asarray([-1, 0, 1]), size=2000)
    m = metric_knn_up_std(X, y, k=50, n_queries=200, seed=SEED)
    assert 0.04 < m["value"] < 0.12
    assert m["status"] == "OK"


def test_verdict_go_on_balanced_data():
    """Непрерывное облако точек с градиентом исхода -> вердикт GO.

    X — низкоранговый многообразие (k=8 латентных факторов -> PCA быстро
    добирает 80%), outcome задаётся логистическим градиентом по непрерывному
    сигналу (rsi), а режим коррелирует с тем же сигналом (hurst/adx/er
    привязаны к нему) -> все 5 метрик проходят (pairwise может быть WARN —
    это не блокирует GO).
    """
    rng = np.random.default_rng(SEED)
    idx = {name: i for i, name in enumerate(_names())}
    n, k_factors, z_std, noise, gain = 900, 8, 1.5, 0.3, 2.5
    Z = rng.normal(0.0, z_std, (n, k_factors))
    W = rng.normal(0.0, 1.0, (44, k_factors))
    X = Z @ W.T + rng.normal(0.0, noise, (n, 44))
    s = X[:, idx["technicals.rsi"]]
    s = (s - s.mean()) / s.std()
    p = 1.0 / (1.0 + np.exp(-gain * s))
    y = np.where(rng.random(n) < p, 1,
                 np.where(rng.random(n) < 0.5, -1, 0))
    X[:, idx["regime.hurst"]] = 0.50 + 0.15 * s
    X[:, idx["trend.adx"]] = 24.0 + 8.0 * s
    X[:, idx["trend.ema20_50_ratio"]] = 1.0 + 0.02 * np.sign(s)
    from app_pkg.cbr import store
    regimes = [store.classify_regime(v) for v in X.astype("float64")]
    res = run_check(X.astype("float32"), y, regimes)
    assert res["verdict"] == "GO", [m["key"] for m in res["metrics"]
                                    if m["status"] == "FAIL"]
    for m in res["metrics"]:
        assert m["status"] != "FAIL"
    regime_m = next(m for m in res["metrics"]
                    if m["key"] == "regime_winrate")
    assert regime_m["value"] > 0.4  # up-rate между режимами различается


def _names():
    """Имена 44 фич в порядке вектора (store.FEATURE_NAMES)."""
    from app_pkg.cbr import store
    return store.FEATURE_NAMES