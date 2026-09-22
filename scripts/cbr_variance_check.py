"""Variance-check CBR-базы перед Этапом 2 (FAISS + k-NN).

Отвечает на вопрос: различаются ли снапшоты между собой в 44D пространстве
фич? Если все точки ~одинаковы -> k-NN вернёт константу -> FAISS бесполезен.

Считает 5 метрик по размеченным снимкам (WHERE outcome IS NOT NULL):
  1. low_variance   — число фич с std/(|mean|+eps) < 0.1 (константы);
  2. pairwise_ratio — D.min()/median(D) на z-scored выборке, D с inf-диагональю;
  3. pca            — сколько компонент нужно для 80% кумулятивной дисперсии;
  4. knn_up_std     — std доли (+1)-соседей по 200 случайным запросам, K=50.
                      up кодируется бинарно (1 vs не-1): при случайных метках
                      std ≈ sqrt(p(1-p)/50)|_{p=1/3} ≈ 0.067 (биномиальное);
  5. regime_winrate — разброс up-rate между режимами (trend_up/trend_down/
                      flat/reversal), spread = max - min.

Вердикт GO, если ни одна метрика не FAIL; иначе NO-GO -> feature selection:
топ-15 из config/charon_features_v1_perm.json (если файл есть) или топ по
std/|mean|, повторный variance-check на этих 15 фичах. Если теперь OK ->
конфиг config/cbr_selected_features.json сохраняется для
store.snapshot_to_vector на Этапе 2.

Запуск из корня проекта:
    python scripts/cbr_variance_check.py --db data/cbr.db --symbol BTCUSDT --tf 1h

Артефакты:
    reports/cbr_variance_<SYMBOL>_<tf>.json  — все метрики + вердикт
    reports/cbr_variance_<SYMBOL>_<tf>.png   — гистограмма попарных расстояний
                                               + PCA cumulative variance curve
    при NO-GO:
      config/cbr_selected_features.json      — топ-15 фич (канон для Этапа 2)
      reports/cbr_top15_features.json        — копия списка топ-15

seed=42 — воспроизводимость выборок (случайные запросы k-NN, подвыборка 1000).
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_BASE_DIR = Path(__file__).resolve().parent.parent
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors

from app_pkg import config
from app_pkg.cbr import schema, store

SEED = 42
EPS = 1e-9
TOP_N = 15

# Пороги статусов (см. ТЗ; "WARN" не блокирует GO).
LOW_VAR_WARN, LOW_VAR_FAIL = 5, 15          # счёт фич-констант
PAIR_OK, PAIR_WARN = 0.30, 0.10             # D.min()/median(D)
PCA_WARN, PCA_FAIL = 15, 25                 # компонент для 80% дисперсии
KNN_K, KNN_QUERIES = 50, 200
KNN_OK, KNN_WARN = 0.05, 0.02               # std доли (+1) у соседей
REGIME_MIN_SPREAD = 0.05                    # разброс up-rate по режимам

PERM_JSON = config.CONFIG_DIR / "charon_features_v1_perm.json"
SELECTED_JSON = config.CONFIG_DIR / "cbr_selected_features.json"
REPORTS_DIR = _BASE_DIR / "reports"

_STATUS_ICON = {"OK": "✅ OK", "WARN": "⚠️ WARN", "FAIL": "🔴 FAIL"}

_FAIL_REASONS = {
    "low_variance": "половина фич константна (std/|mean| < 0.1)",
    "pairwise_ratio": "все точки на одном расстоянии (min/median < 0.1)",
    "pca": "слишком много компонент для 80% дисперсии (проклятие размерности)",
    "knn_up_std": "k-NN всегда возвращает одно и то же (up_std < 0.02)",
    "regime_winrate": "winrate одинаков во всех режимах — фичи не различают режимы",
}


# ------------------------------------------------------------ загрузка данных
def canonical_tf(tf):
    """Приводит таймфрейм к канонической форме config.TF_SECONDS
    (1h/1H->1H, 5m->5m), иначе возвращает как есть."""
    tf = str(tf or "").strip()
    for canon in config.TF_SECONDS:
        if canon.lower() == tf.lower():
            return canon
    return tf


def _decode_features(raw):
    """features-поле -> np.ndarray float32 (44,) или None при повреждении.

    Реальный формат поля — JSON-массив float32 (schema.py), но на всякий
    случай понимаем и raw bytes numpy .tobytes() (ТЗ-вариант).
    """
    if raw is None:
        return None
    if isinstance(raw, bytes):
        try:
            return np.frombuffer(raw, dtype="<f4")
        except ValueError:
            return None
    if isinstance(raw, str) and raw.startswith("["):
        try:
            arr = json.loads(raw)
        except ValueError:
            return None
        return np.asarray(arr, dtype="<f4")
    try:
        return np.asarray(raw, dtype="<f4")
    except (ValueError, TypeError):
        return None


def load_features(conn, symbol, timeframe, dim=44):
    """(X, y, regimes) размеченных снимков: X (N, 44) float32, y int, regimes.

    Строки с повреждённым или не-44-мерным вектором молча пропускаются.
    """
    rows = conn.execute(
        "SELECT features, outcome, regime FROM snapshots "
        "WHERE symbol=? AND timeframe=? AND outcome IS NOT NULL",
        (str(symbol).upper(), str(timeframe).strip())).fetchall()
    if not rows:
        raise SystemExit(f"нет размеченных снимков для "
                         f"{symbol} {timeframe} (outcome IS NOT NULL)")
    mats, ys, regimes = [], [], []
    for r in rows:
        vec = _decode_features(r["features"])
        if vec is None or vec.size != dim:
            continue
        mats.append(vec)
        ys.append(int(r["outcome"]))
        regimes.append(r["regime"])
    if not mats:
        raise SystemExit(f"все снимки {symbol} {timeframe} повреждены "
                         f"(нет валидных features)")
    X = np.stack(mats).astype(np.float32)
    y = np.asarray(ys, dtype=np.int64)
    return X, y, np.asarray(regimes, dtype=object)


# ------------------------------------------------------------ нормализация
def zscore(X, eps=EPS):
    """Z-score по всей БД: (X - mean) / std, константные фичи -> 0 (не NaN)."""
    X = np.asarray(X, dtype="float64")
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < eps, 1.0, std)
    return (X - mean) / std


# ---------------------------------------------------------------- метрики
def metric_low_variance(X, names=None, eps=EPS):
    """Число фич со std/(|mean|+eps) < 0.1 («константы»)."""
    X = np.asarray(X, dtype="float64")
    std = X.std(axis=0)
    mean = np.abs(X.mean(axis=0))
    ratios = std / (mean + eps)
    flag = ratios < 0.1
    count = int(flag.sum())
    total = int(X.shape[1])
    if count > LOW_VAR_FAIL:
        status = "FAIL"
    elif count > LOW_VAR_WARN:
        status = "WARN"
    else:
        status = "OK"
    out = {"key": "low_variance", "value": count, "total": total,
           "status": status}
    if names is not None:
        out["flagged"] = [str(n) for n, ok in zip(names, flag) if ok]
    return out


def metric_pairwise_ratio(X, n=1000, seed=SEED):
    """ratio = D.min()/median(D) на случайной z-scored подвыборке."""
    X = np.asarray(X, dtype="float64")
    n_pts = min(int(n), len(X))
    base = {"key": "pairwise_ratio", "status": "FAIL", "n_samples": n_pts,
            "min_dist": 0.0, "median_dist": 0.0, "mean_dist": 0.0,
            "max_dist": 0.0}
    if n_pts < 2:
        base["value"] = 0.0
        return base
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), n_pts, replace=False)
    sample = zscore(X[idx])
    D = euclidean_distances(sample)
    np.fill_diagonal(D, np.inf)
    Df = D[np.isfinite(D)]
    d_min = float(Df.min())
    d_med = float(np.median(Df))
    d_mean = float(Df.mean())
    d_max = float(Df.max())
    ratio = 0.0 if d_med <= 0.0 else d_min / d_med
    if ratio >= PAIR_OK:
        status = "OK"
    elif ratio >= PAIR_WARN:
        status = "WARN"
    else:
        status = "FAIL"
    return {"key": "pairwise_ratio", "value": ratio, "status": status,
            "n_samples": n_pts, "min_dist": d_min, "median_dist": d_med,
            "mean_dist": d_mean, "max_dist": d_max, "_distances": Df}


def metric_pca(X):
    """n_80 — сколько компонент закрывают 80% кумулятивной дисперсии."""
    X = np.asarray(X, dtype="float64")
    pca = PCA()
    pca.fit(zscore(X))
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    mask = cumvar >= 0.80
    n_80 = int(np.argmax(mask) + 1) if mask.any() else len(cumvar)
    if n_80 > PCA_FAIL:
        status = "FAIL"
    elif n_80 > PCA_WARN:
        status = "WARN"
    else:
        status = "OK"
    return {"key": "pca", "value": n_80, "status": status,
            "n_components": len(cumvar), "_cumvar": cumvar}


def metric_knn_up_std(X, y, k=KNN_K, n_queries=KNN_QUERIES, seed=SEED):
    """std доли (+1)-соседей по случайным запросам (K ближайших).

    up — бинарное кодирование (1 vs не-1): per_query_up = доля up у соседей
    (winrate). На случайных метках p=1/3 std ≈ sqrt(p(1-p)/K) ≈ 0.067.
    """
    X = np.asarray(X, dtype="float64")
    up = (np.asarray(y) == 1).astype("float64")
    kk = max(1, min(int(k), len(X)))
    q = min(int(n_queries), len(X))
    nn = NearestNeighbors(n_neighbors=kk).fit(zscore(X))
    rng = np.random.default_rng(seed)
    queries = rng.choice(len(X), q, replace=False)
    _, indices = nn.kneighbors(np.asarray(X[queries], dtype="float64"))
    per_query_up = np.asarray([up[idx].mean() for idx in indices])
    up_std = float(per_query_up.std())
    if up_std >= KNN_OK:
        status = "OK"
    elif up_std >= KNN_WARN:
        status = "WARN"
    else:
        status = "FAIL"
    return {"key": "knn_up_std", "value": up_std, "status": status,
            "k": kk, "n_queries": q, "per_query_up_mean": float(per_query_up.mean())}


def metric_regime_winrate(regimes, y):
    """Разброс up-rate между режимами: spread = max - min."""
    y = np.asarray(y)
    sums, counts = {}, {}
    for rg, label in zip(regimes, y):
        if rg is None:
            continue
        key = str(rg)
        sums[key] = sums.get(key, 0.0) + (1.0 if int(label) == 1 else 0.0)
        counts[key] = counts.get(key, 0) + 1
    by_regime = {rg: {"n": counts[rg], "up_rate": sums[rg] / counts[rg]}
                 for rg in sorted(counts)}
    if not by_regime:
        return {"key": "regime_winrate", "value": 0.0, "status": "FAIL",
                "by_regime": {}, "n_regimes": 0}
    rates = np.asarray([v["up_rate"] for v in by_regime.values()])
    spread = float(rates.max() - rates.min())
    status = "OK" if spread >= REGIME_MIN_SPREAD else "FAIL"
    return {"key": "regime_winrate", "value": spread, "status": status,
            "by_regime": by_regime, "n_regimes": len(by_regime)}


# ------------------------------------------------------------------- чек
def run_check(X, y, regimes=None):
    """5 метрик на (X, y, regimes) + вердикт GO/NO-GO + данные для графиков.

    X — (N, 44) float32; y — outcome (+1/-1/0); regimes — список режимов или
    None (тогда метрика 5 вернёт FAIL с пустым by_regime). Ключи метрик,
    начинающиеся с '_', не попадают в JSON-отчёт (используются для графиков).
    """
    metrics = [
        metric_low_variance(X),
        metric_pairwise_ratio(X),
        metric_pca(X),
        metric_knn_up_std(X, y),
        metric_regime_winrate(regimes, y),
    ]
    fails = [m["key"] for m in metrics if m["status"] == "FAIL"]
    y_arr = np.asarray(y)
    counts = {int(v): int((y_arr == v).sum()) for v in (-1, 0, 1)}
    return {
        "metrics": metrics,
        "n": len(y_arr),
        "feature_dim": int(np.asarray(X).shape[1]),
        "outcome_balance": counts,
        "fails": fails,
        "verdict": "NO-GO" if fails else "GO",
    }


def sanitize_metric(m):
    """Копия метрики без служебных '_'-/ndarray-ключей (для JSON)."""
    out = {}
    for k, v in m.items():
        if k.startswith("_") or isinstance(v, np.ndarray):
            continue
        if isinstance(v, np.floating):
            v = float(v)
        elif isinstance(v, np.integer):
            v = int(v)
        elif isinstance(v, dict):
            v = sanitize_metric(v)
        out[k] = v
    return out


# -------------------------------------------------- feature selection (NO-GO)
def load_perm_features(path=None):
    """Пермутационный ranking из config/charon_features_v1_perm.json.

    Файл — список имён фич (убывание важности); None, если файла нет/битый.
    """
    path = Path(path or PERM_JSON)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    if isinstance(data, list):
        return [str(x).strip() for x in data if str(x).strip()]
    return None


def rank_by_cvariance(X, names):
    """Топ фич по std/(|mean|+eps) (фолбэк без perm-файла)."""
    X = np.asarray(X, dtype="float64")
    std = X.std(axis=0)
    mean = np.abs(X.mean(axis=0))
    ratios = std / (mean + EPS)
    order = np.argsort(-ratios)
    return [str(names[int(i)]) for i in order]


def select_top_features(X, names, top_n=TOP_N, perm_path=None):
    """(indices, selected) топ-top_n фич: perm-importance ?? variance-фолбэк.

    Из perm-списка берётся первая top_n фича, присутствующая среди names;
    если файла нет или не хватило имён — дозаполняем variance-ранкингом.
    """
    names = list(names)
    avail = set(names)
    selected = [f for f in (load_perm_features(perm_path) or []) if f in avail]
    selected = selected[:top_n]
    if len(selected) < top_n:
        for f in rank_by_cvariance(X, names):
            if f not in selected:
                selected.append(f)
            if len(selected) >= top_n:
                break
    selected = selected[:top_n]
    idx = [names.index(f) for f in selected]
    return idx, selected


# ------------------------------------------------------------------- вывод
def format_status(m):
    return _STATUS_ICON.get(m["status"], m["status"])


def format_report(res, symbol, tf_label):
    """Блок-бокс отчёта (см. ТЗ) как многострочную строку."""
    metrics = {m["key"]: m for m in res["metrics"]}
    lines = [
        f"CBR VARIANCE CHECK — {symbol} {tf_label}",
        f"N samples:                    {res['n']}",
        f"Feature dim:                  {res['feature_dim']}",
        "──",
        f"1. Low-variance features:  {metrics['low_variance']['value']:>4} "
        f"/ {metrics['low_variance']['total']}  "
        f"{format_status(metrics['low_variance'])}",
        f"2. Pairwise dist ratio:    "
        f"{metrics['pairwise_ratio']['value']:>8.4f}  "
        f"{format_status(metrics['pairwise_ratio'])}",
        f"3. PCA components (80%):   "
        f"{metrics['pca']['value']:>8}  {format_status(metrics['pca'])}",
        f"4. k-NN up_std (K={metrics['knn_up_std']['k']}): "
        f"{metrics['knn_up_std']['value']:>8.4f}  "
        f"{format_status(metrics['knn_up_std'])}",
        f"5. Regime winrate spread:  "
        f"{metrics['regime_winrate']['value']:>8.4f}  "
        f"{format_status(metrics['regime_winrate'])}",
    ]
    if res["verdict"] == "GO":
        lines.append("ВЕРДИКТ: ✅ GO — Этап 2 (FAISS) имеет смысл")
    else:
        lines.append("ВЕРДИКТ: 🔴 NO-GO — сначала feature selection")
        reasons = [f"  Причина: {_FAIL_REASONS.get(k, k)}"
                   for k in res["fails"]]
        lines.extend(reasons)
        lines.append("  Рекомендация: использовать топ-15 из permutation "
                     "importance, повторить variance check")
    return "\n".join(lines)


def _box(lines):
    """Рисует бокс ╔═╗ вокруг строк (ширина по самой длинной)."""
    width = max(len(l) for l in lines) + 4
    inner = width - 4
    out = ["╔" + "═" * width + "╗"]
    for l in lines:
        out.append("║ " + l[:inner].ljust(inner) + " ║")
    out.append("╚" + "═" * width + "╝")
    return "\n".join(out)


def save_plot(png_path, metrics):
    """2 графика: гистограмма попарных расстояний + PCA cumulative variance."""
    d = metrics["pairwise_ratio"].get("_distances")
    c = metrics["pca"].get("_cumvar")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
    if d is not None and d.size:
        axes[0].hist(d, bins=60, color="#4a90d9")
        axes[0].set_title(f"Pairwise distances (z-scored, n={d.size})")
        axes[0].set_xlabel("Euclidean distance")
        axes[0].set_ylabel("count")
        axes[0].axvline(d.min(), color="red", ls="--", label="min")
        axes[0].axvline(np.median(d), color="green", ls="--", label="median")
        axes[0].legend()
    if c is not None and len(c):
        axes[1].plot(np.arange(1, len(c) + 1), c, "o-", ms=3)
        axes[1].axhline(0.80, color="red", ls="--")
        axes[1].set_title("PCA cumulative variance")
        axes[1].set_xlabel("n components")
        axes[1].set_ylabel("cumulative explained variance")
    fig.tight_layout()
    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=110)
    plt.close(fig)
    return png_path


# ------------------------------------------------------------------- main
def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description="Variance-check CBR-БД перед Этапом 2 (FAISS + k-NN).")
    parser.add_argument("--db", default=None,
                        help="путь к CBR-БД (default: config.CBR_DB_PATH)")
    parser.add_argument("--symbol", default="BTCUSDT", help="символ")
    parser.add_argument("--tf", default="1h", help="таймфрейм")
    args = parser.parse_args(argv)

    db_path = args.db or config.CBR_DB_PATH
    symbol = str(args.symbol).upper()
    tf_query = canonical_tf(args.tf)
    tf_label = str(args.tf).strip() or tf_query

    conn = schema.init_db(db_path)
    try:
        X, y, regimes = load_features(conn, symbol, tf_query)
    finally:
        conn.close()

    res = run_check(X, y, regimes)
    report = {
        "symbol": symbol,
        "timeframe": tf_label,
        "timeframe_canonical": tf_query,
        "n": res["n"],
        "feature_dim": res["feature_dim"],
        "outcome_balance": res["outcome_balance"],
        "seed": SEED,
        "verdict": res["verdict"],
        "fail_reasons": [_FAIL_REASONS.get(k, k) for k in res["fails"]],
        "metrics": {m["key"]: sanitize_metric(m) for m in res["metrics"]},
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    slug = f"{symbol}_{tf_label}"
    json_path = REPORTS_DIR / f"cbr_variance_{slug}.json"
    png_path = REPORTS_DIR / f"cbr_variance_{slug}.png"

    print(_box(format_report(res, symbol, tf_label).splitlines()))
    print(f"\n-> {json_path}")
    print(f"-> {png_path}")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    save_plot(png_path, {m["key"]: m for m in res["metrics"]})

    if res["verdict"] == "NO-GO":
        idx, selected = select_top_features(X, store.FEATURE_NAMES)
        sub = run_check(X[:, idx], y, regimes)
        print("\n=== Feature selection (NO-GO) ===")
        src = "permutation importance" if load_perm_features() else "variance"
        print(f"Источник топ-15: {src}")
        print("Топ-15: " + ", ".join(selected))
        print(_box(format_report(sub, symbol, tf_label).splitlines()))
        top15_path = REPORTS_DIR / "cbr_top15_features.json"
        top15_path.write_text(
            json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"-> {top15_path}")
        if sub["verdict"] == "GO":
            SELECTED_JSON.parent.mkdir(parents=True, exist_ok=True)
            SELECTED_JSON.write_text(
                json.dumps(selected, ensure_ascii=False, indent=2),
                encoding="utf-8")
            print(f"-> {SELECTED_JSON} (топ-15 фич для Этапа 2)")
        else:
            print(f"Топ-15 всё ещё FAIL — {SELECTED_JSON} НЕ сохранён; "
                  "нужна ручная чистка фич.")


if __name__ == "__main__":
    main()