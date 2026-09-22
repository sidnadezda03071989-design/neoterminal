# -*- coding: utf-8 -*-
"""Feature importance + walk-forward оценка LightGBM на Charon-фичах.

Собирает X, y из walk-forward срезов (как в сканере, _wf_window_slices) и
учит 3-классовый LightGBM (triple-barrier {-1,0,1}, метки-хвост drop):
    train slice  -> build_labeled_dataset(slice, features_fn=charon_features)
    purge/embargo = horizon + буфер (по умолчанию 25 баров: 20 + 5)
    метки с NaN не заполняются нулём — строки выпадают (класс «неизвестно»).

Метрики агрегируются ПО ФОЛДАМ (pooled accuracy / f1_macro / confusion) и
сравниваются с тремя baselines:
    random       = 33.3%;
    majority     = частота самого частого класса в train;
    OHLCV-only   = те же фолды на default_features («пол», ~50%).
Печатается баланс классов (ориентир +1/-1/0 ~ 35/35/30; иначе крутить k/horizon).
SHAP: TreeExplainer per fold, аггрегация по фолдам, summary_plot (bar + beeswarm)
сохраняются PNG. Top-K фич по gain -> JSON (чистый список имён, для конфига).
Permutation importance (top-K -> JSON + WARNING при расхождении с gain).
--shuffle-labels: перетасовка y_train как sanity-контроль наличия сигнала.
--grid: перебор k x horizon на walk-forward срезах (max f1_macro -> CSV).

Запуск из корня:
    python scripts/charon_feature_importance.py --symbol BTCUSDT --timeframe 1h --days 700
    python scripts/charon_feature_importance.py --csv my_ohlcv.csv --json-out config/charon_features_v1.json --shap

Данные: нет. локального OHLCV-кэша нет — по умолчанию идёт сетевой fetch
(get_replay_df, crypto->Binance), результат можно --cache в CSV для
детерминированных офлайн-повторов. --csv берёт готовый файл (колонки
timestamp,open,high,low,close,volume; timestamp tz-aware или naive UTC).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_BASE_DIR = Path(__file__).resolve().parent.parent
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

from app_pkg import config                      # noqa: E402  (после bootstrap)
from app_pkg.ai import scanner as scanner_mod   # noqa: E402
from app_pkg.ml.labels import (                 # noqa: E402
    build_labeled_dataset, triple_barrier_labels,
    charon_features, default_features,
    DEFAULT_K, DEFAULT_HORIZON,
)

_LABELS = [-1, 0, 1]

_LGB_PARAMS = dict(
    n_estimators=500, learning_rate=0.03, num_leaves=31,
    min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
    class_weight="balanced", random_state=42, verbosity=-1,
)


def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--timeframe", "--tf", dest="timeframe", default="1h")
    p.add_argument("--days", type=int, default=700)
    p.add_argument("--csv", help="готовый OHLCV CSV (вместо fetch)")
    p.add_argument("--cache", help="CSV-кэш: если есть — читать, иначе fetch+save")
    p.add_argument("--k", type=float, default=DEFAULT_K)
    p.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    p.add_argument("--embargo", type=int, default=None,
                   help="purge-бары между train/test (default=horizon+5)")
    p.add_argument("--test-frac", type=float, default=config.SCAN_WF_TEST_FRACTION)
    p.add_argument("--train-per-test", type=int, default=config.SCAN_WF_TRAIN_PER_TEST)
    p.add_argument("--folds", type=int, default=config.SCAN_WF_FOLDS)
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--json-out", default="config/charon_features_v1.json")
    p.add_argument("--perm-json", default="config/charon_features_v1_perm.json")
    p.add_argument("--shap", action="store_true",
                   help="считать SHAP и сохранять summary plot'ы")
    p.add_argument("--figs-dir", default="scripts/charon_shap")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shuffle-labels", action="store_true",
                   help="перетасовать y_train (sanity: сигнала нет, если "
                        "shuffled ≈ real)")
    p.add_argument("--grid", action="store_true",
                   help="перебор k x horizon на walk-forward срезах -> "
                        "reports/grid_k_horizon.csv (max f1_macro)")
    p.add_argument("--grid-csv", default="reports/grid_k_horizon.csv")
    return p.parse_args(argv)


def load_df(args):
    """df с колонками timestamp,open,high,low,close,volume (asc, tz-aware)."""
    if args.csv:
        path = Path(args.csv)
        if not path.exists():
            raise SystemExit(f"CSV не найден: {args.csv}")
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df.sort_values("timestamp").reset_index(drop=True)
    if args.cache and Path(args.cache).exists():
        df = pd.read_csv(args.cache)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df.sort_values("timestamp").reset_index(drop=True)
    from app_pkg.data.fetch import get_replay_df      # noqa: E402
    limit = config.HISTORY_LIMITS.get(args.timeframe, 20000)
    from_sec = int(time.time()) - args.days * 86400
    print(f"[fetch] {args.symbol}/{args.timeframe} ~{args.days}d, limit={limit} ...")
    df = get_replay_df(args.symbol, args.timeframe, from_sec, None, limit=limit)
    if df is None or df.empty:
        raise SystemExit(f"нет данных {args.symbol}/{args.timeframe} (сеть?) — "
                         f"передайте --csv или --cache")
    if len(df) > limit:
        df = df.iloc[-limit:].reset_index(drop=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    if args.cache:
        Path(args.cache).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.cache, index=False)
        print(f"[cache] сохранено в {args.cache}")
    return df


def build_folds(df, args):
    """Срезы walk-forward как в сканере, purge = embargo (без утечки меток)."""
    embargo = args.embargo if args.embargo is not None else args.horizon + 5
    slices = scanner_mod._wf_window_slices(
        len(df), args.test_frac, args.train_per_test, args.folds, embargo)
    if not slices:
        raise SystemExit(
            f"walk-forward не влезает в {len(df)} баров "
            f"(train+{embargo}+test) — больше данных / меньше train-per-test")
    return slices, embargo


def dataset_for(df, sl, args, feature_fn):
    """Датасет среза: drop NaN-меток и NaN-признаков (прогрев)."""
    sub = df.iloc[sl[0]:sl[1]].reset_index(drop=True)
    X, y, meta = build_labeled_dataset(sub, k=args.k, horizon=args.horizon,
                                       features_fn=feature_fn)
    return X, y.astype(int), meta


def fit_fold(X_train, y_train):
    import lightgbm as lgb
    model = lgb.LGBMClassifier(**_LGB_PARAMS)
    model.fit(X_train, y_train)
    return model


def majority_acc(y_train, y_test):
    top = np.argmax(np.bincount(np.asarray(y_train) + 1))
    label = top - 1
    return float((np.asarray(y_test) == label).mean())


def pooled_metrics(y_all, p_all):
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    return (accuracy_score(y_all, p_all),
            f1_score(y_all, p_all, average="macro", labels=_LABELS),
            confusion_matrix(y_all, p_all, labels=_LABELS))


def run_model(df, slices, args, feature_fn, name, shuffle=None, with_perm=False):
    y_all, p_all = [], []
    gains, perm_means, shap_all, x_all = [], [], [], []
    maj_acc, maj_weight = 0.0, 0
    report, fold_stats = [], []
    for i, (sl_tr, sl_te) in enumerate(slices):
        Xtr, ytr, m_tr = dataset_for(df, sl_tr, args, feature_fn)
        Xte, yte, _ = dataset_for(df, sl_te, args, feature_fn)
        if shuffle is not None:      # sanity: метки train перемешаны, test нет
            ytr = pd.Series(ytr).sample(frac=1.0, random_state=shuffle).to_numpy()
        if len(Xtr) < 50 or len(Xte) < 1:
            raise SystemExit(f"{name}: фолд {i} слишком мал "
                             f"({len(Xtr)} train / {len(Xte)} test)")
        model = fit_fold(Xtr, ytr)
        pred = model.predict(Xte)
        y_all.extend(yte)
        p_all.extend(pred)
        maj_acc += majority_acc(ytr, yte) * len(yte)
        maj_weight += len(yte)
        gain = np.asarray(model.booster_.feature_importance(
            importance_type="gain"), dtype="float64")
        gain = gain / (gain.sum() if gain.sum() else 1.0)
        gains.append(gain)
        if with_perm:
            from sklearn.inspection import permutation_importance
            # loky-воркеры (n_jobs>1) на Windows+detached падают
            # (TerminatedWorkerError); на 44 фичах sequential ≈ 1с/фолд.
            perm = permutation_importance(model, Xte, yte, n_repeats=10,
                                          random_state=args.seed, n_jobs=1)
            perm_means.append(perm.importances_mean)
        if args.shap:
            import shap
            vals = np.asarray(shap.TreeExplainer(model)(Xte).values,
                              dtype="float64")
            if vals.ndim == 3:
                # shap 0.52: (n, classes, features) или (n, features, classes)
                if vals.shape[2] == Xte.shape[1]:
                    vals = vals.mean(axis=1)   # (n, features)
                elif vals.shape[1] == Xte.shape[1]:
                    vals = vals.mean(axis=2)   # (n, features)
                else:
                    vals = vals.mean(axis=1)
            assert vals.ndim == 2 and vals.shape[1] == Xte.shape[1], vals.shape
            shap_all.append(vals)
            x_all.append(Xte)
        report.append(
            f"{name:6s} fold{i}: train={len(ytr)} test={len(yte)} "
            f"test_acc={sk_acc(yte, pred):.3f} "
            f"maj={majority_acc(ytr, yte):.3f} "
            f"bal={dict(m_tr.get('class_balance', {}))}")
        fold_stats.append({"fold": i + 1, "acc": sk_acc(yte, pred),
                           "f1": sk_f1(yte, pred),
                           "n_train": len(ytr), "n_test": len(yte)})
    acc, f1, cm = pooled_metrics(np.asarray(y_all), np.asarray(p_all))
    features = list(Xtr.columns)
    return {"name": name, "acc": acc, "f1": f1, "cm": cm,
            "n_test": len(y_all), "features": features,
            "maj_acc": maj_acc / maj_weight,
            "gain": np.mean(gains, axis=0),
            "perm": (np.mean(perm_means, axis=0) if perm_means else None),
            "fold_stats": fold_stats,
            "shap_all": (np.concatenate(shap_all, axis=0) if shap_all else None),
            "x_all": (pd.concat(x_all, axis=0) if x_all else None),
            "report": report}


def sk_acc(y_true, y_pred):
    from sklearn.metrics import accuracy_score
    return accuracy_score(y_true, y_pred)


def sk_f1(y_true, y_pred):
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, average="macro", labels=_LABELS)


def save_shap(model_res, args):
    """SHAP summary: bar (gain-like) + beeswarm (направление влияния)."""
    shap_all = model_res["shap_all"]
    x_all = model_res["x_all"]
    if shap_all is None or x_all is None:
        print("[shap] не считан (нужен --shap)")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap
    out = Path(args.figs_dir)
    out.mkdir(parents=True, exist_ok=True)
    plt.figure()
    shap.summary_plot(shap_all, x_all, plot_type="bar",
                      max_display=min(25, shap_all.shape[1]))
    plt.savefig(out / "shap_bar.png", bbox_inches="tight")
    plt.close("all")
    plt.figure()
    shap.summary_plot(shap_all, x_all,
                      max_display=min(25, shap_all.shape[1]))
    plt.savefig(out / "shap_beeswarm.png", bbox_inches="tight")
    plt.close("all")
    print(f"[shap] -> {out / 'shap_bar.png'}, {out / 'shap_beeswarm.png'}")


_GRID_KS = [1.0, 1.5, 2.0, 2.5]
_GRID_HORIZONS = [10, 15, 20, 30]


def relabel_slice(X, df_slice, k, h):
    """Метки k/horizon поверх уже посчитанных фич (drop NaN как в датасете)."""
    labels, _ = triple_barrier_labels(df_slice, k=k, horizon=h)
    rows = X.copy()
    rows["label"] = labels
    rows = rows.dropna(subset=["label"]).dropna()
    return rows.drop(columns=["label"]), rows["label"].to_numpy().astype(int)


def grid_run(df, slices, args):
    """Перебор k x horizon на walk-forward срезах (фичи считаем один раз)."""
    from itertools import product
    from sklearn.metrics import accuracy_score, f1_score

    fold_subs = [
        (df.iloc[sl_tr[0]:sl_tr[1]].reset_index(drop=True),
         df.iloc[sl_te[0]:sl_te[1]].reset_index(drop=True))
        for sl_tr, sl_te in slices]
    fold_X = [(charon_features(tr), charon_features(te))
              for tr, te in fold_subs]

    rows = []
    for k, h in product(_GRID_KS, _GRID_HORIZONS):
        y_all, p_all, n_train = [], [], 0
        cb = {-1: 0, 0: 0, 1: 0}
        for (Xtr, Xte), (sub_tr, sub_te) in zip(fold_X, fold_subs):
            Xtr2, ytr = relabel_slice(Xtr, sub_tr, k, h)
            Xte2, yte = relabel_slice(Xte, sub_te, k, h)
            if len(Xtr2) < 50 or len(Xte2) < 1:
                continue
            model = fit_fold(Xtr2, ytr)
            pred = model.predict(Xte2)
            y_all.extend(yte)
            p_all.extend(pred)
            n_train += len(ytr)
            for lbl, cnt in zip(*np.unique(ytr, return_counts=True)):
                cb[int(lbl)] += int(cnt)
        if not y_all:
            continue
        rows.append({"k": k, "horizon": h,
                     "acc": accuracy_score(np.asarray(y_all),
                                           np.asarray(p_all)),
                     "f1_macro": f1_score(np.asarray(y_all),
                                          np.asarray(p_all),
                                          average="macro", labels=_LABELS),
                     "n_train": n_train, "n_test": len(y_all),
                     "bal_-1": cb[-1], "bal_0": cb[0], "bal_+1": cb[1]})
    if not rows:
        raise SystemExit("grid: ни одна пара k/horizon не дала валидных фолдов")
    grid_df = pd.DataFrame(rows)
    out = Path(args.grid_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    grid_df.to_csv(out, index=False)
    print(f"-> {out}")
    piv = grid_df.pivot_table(index="k", columns="horizon", values="f1_macro")
    print("\n=== grid f1_macro (k x horizon) ===")
    print(piv.to_string(float_format="%.3f"))
    best = grid_df.loc[grid_df["f1_macro"].idxmax()]
    print(f"[best] k={best['k']} horizon={int(best['horizon'])} "
          f"f1={best['f1_macro']:.3f} acc={best['acc']:.3f}")


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = _parse_args(argv)
    t0 = time.time()
    df = load_df(args)
    slices, embargo = build_folds(df, args)
    n_train_bars = sum(sl[1] - sl[0] for sl, _ in slices)
    n_test_bars = sum(sl[1] - sl[0] for _, sl in slices)
    print(f"[data] {len(df)} баров, embargo={embargo}, "
          f"фолдов={len(slices)} (train≈{n_train_bars}, test≈{n_test_bars})")

    if args.grid:
        grid_run(df, slices, args)
        print(f"\n[ok] {time.time() - t0:.1f}s")
        return

    charon = run_model(df, slices, args, charon_features, "charon",
                       with_perm=True)
    table_models = [charon]
    if args.shuffle_labels:
        charon_shuf = run_model(df, slices, args, charon_features,
                                "charon(shuffled)", shuffle=args.seed)
        table_models.append(charon_shuf)
    ohlcv = run_model(df, slices, args, default_features, "ohlcv")

    print("\n=== класс-баланс (train, первый фолд) ===")
    for line in charon["report"]:
        print(" ", line)
    print(f"[hint] цель ≈ +1/-1/0 35/35/30; при сильном перекосе крутить "
          f"--k/--horizon")

    print("\n=== per-fold (walk-forward) ===")
    for res in (charon, ohlcv):
        for fs in res["fold_stats"]:
            print("  %-18s Fold %d: acc=%.3f f1=%.3f n_train=%d n_test=%d"
                  % (res["name"], fs["fold"], fs["acc"], fs["f1"],
                     fs["n_train"], fs["n_test"]))
        accs = [fs["acc"] for fs in res["fold_stats"]]
        f1s = [fs["f1"] for fs in res["fold_stats"]]
        print("  %-18s Pooled:   acc=%.3f f1=%.3f"
              % (res["name"], res["acc"], res["f1"]))
        print("  %-18s mean±std: acc=%.3f±%.3f f1=%.3f±%.3f"
              % (res["name"], float(np.mean(accs)), float(np.std(accs)),
                 float(np.mean(f1s)), float(np.std(f1s))))

    print("\n=== метрики по walk-forward (pooled) ===")
    print("  %-18s %-6s %-6s %s" % ("model", "acc", "f1_macro", "confusion"))
    for res in table_models:
        print("  %-18s %-6s %-6s %s" % (res["name"], f"{res['acc']:.3f}",
                                         f"{res['f1']:.3f}",
                                         res["cm"].tolist()))
        if res["name"] == "charon":
            print("  %-18s %-6s %-6s  (baseline majority на test)"
                  % ("baseline maj", f"{res['maj_acc']:.3f}", ""))
    print("  %-18s %-6s %-6s  (baseline majority на test)"
          % ("baseline maj ohlcv", f"{ohlcv['maj_acc']:.3f}", ""))
    print("  %-18s %-6s %-6s" % ("baseline random", "0.333", "0.333"))
    gain_vs_floor = charon["acc"] - max(ohlcv["acc"], 1 / 3.0)
    print(f"[вывод] charon acc={charon['acc']:.3f} vs floor(ohlcv)="
          f"{ohlcv['acc']:.3f}: delta={gain_vs_floor:+.3f}")
    if args.shuffle_labels:
        print(f"[вывод] charon real vs shuffled: "
              f"acc {charon['acc'] - charon_shuf['acc']:+.3f}, "
              f"f1 {charon['f1'] - charon_shuf['f1']:+.3f}")

    top_k = int(args.top_k)
    assert len(charon["features"]) == len(charon["gain"])
    ranked = sorted(zip(charon["features"], charon["gain"]),
                    key=lambda x: -x[1])
    print(f"\n=== top-{top_k} фич по gain ===")
    for i, (f, g) in enumerate(ranked[:top_k], 1):
        print("  %2d %-28s %.4f" % (i, f, g))

    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump([f for f, _ in ranked[:top_k]], fh, indent=2)
    print(f"-> {args.json_out} ({len(ranked[:top_k])} фич)")

    perm = charon.get("perm")
    if perm is not None:
        assert len(charon["features"]) == len(perm)
        ranked_perm = sorted(zip(charon["features"], perm),
                             key=lambda x: -x[1])
        print(f"\n=== top-{top_k} фич по permutation ===")
        for i, (f, g) in enumerate(ranked_perm[:top_k], 1):
            print("  %2d %-28s %.4f" % (i, f, g))
        intersect = len({f for f, _ in ranked[:10]}
                        & {f for f, _ in ranked_perm[:10]})
        print(f"overlap top-10 (gain ∩ permutation) = {intersect}/10")
        if intersect < 5:
            print("WARNING: gain and permutation disagree — "
                  "gain likely biased by cardinality")
        perm_out = Path(args.perm_json)
        perm_out.parent.mkdir(parents=True, exist_ok=True)
        with open(perm_out, "w", encoding="utf-8") as fh:
            json.dump([f for f, _ in ranked_perm[:top_k]], fh, indent=2)
        print(f"-> {args.perm_json} ({len(ranked_perm[:top_k])} фич)")

    if args.shap:
        save_shap(charon, args)

    print(f"\n[ok] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()