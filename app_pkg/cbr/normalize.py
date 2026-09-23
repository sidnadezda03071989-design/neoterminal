"""Нормализация 44-фич снимков для k-NN (FAISS, Этап 2).

Ошибка глобального z-score по всей БД — look-ahead: mean/std считаются по
ВСЕЙ истории, включая будущие бары, и «протекают» в прошлые запросы. Поэтому
нормализация ROLLING: для бара t статистика берётся по окну из ПРЕДЫДУЩИХ
`window` баров (shift(1)) — значение в баре t не зависит от t и дальше.

Форматы:
  compute_rolling_stats(df, window=500) -> dict
      df — DataFrame с индексом ts (int, Unix-сек) и колонками-фичами (44,
      порядок значения не имеет — нормализация поколоночная). Возвращает
      {"mean": (N,44), "std": (N,44), "ts": (N,), "window": w}.
  apply_normalization(features, stats) -> np.ndarray float32 (44,), clip [-5,5]
      stats — либо полный dict из compute_rolling_stats (тогда берётся
      статистика ПОСЛЕДНЕГО бара — live-запрос «сейчас»), либо его срез
      {"mean": (44,), "std": (44,)} (см. stats_at).
  stats_at(stats, ts) -> срез статистики ровно на исторический бар ts
      (ближайший доступный <= ts): для offline-бэктестов без look-ahead.

  save_stats/load_stats — pickle-персистентность (config.CBR_NORMALIZE_STATS_PATH).
"""

import logging
import pickle

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Защита от выбросов после z-score (ТЗ: clip к [-5, 5]).
CLIP_MIN, CLIP_MAX = -5.0, 5.0
DEFAULT_WINDOW = 500
# std ниже порога считается вырожденной константой -> заменяется на 1.0
# (после z-score фича становится 0; деление на 0 не даёт Inf/NaN).
STD_EPS = 1e-9


def compute_rolling_stats(df, window=DEFAULT_WINDOW, ts=None):
    """Rolling mean/std по каждой фиче (без look-ahead: shift(1)).

    df — DataFrame с индексом ts (int Unix-сек, отсортирован по возрастанию)
    и 44 колонками-фичами ИЛИ np.ndarray (N,44). Для ndarray ts поставляет
    второй аргумент (иначе статистику нельзя привязать к реальным барам и
    stats_at/normalize_matrix будут считать всю матрицу «последним баром» —
    look-ahead). Для каждого бара статистика считается по окну ПРЕДЫДУЩИХ
    `window` баров (pandas rolling + shift(1)); первый бар получает окно из
    min_periods=1 (свой же бар исключён shift'ом — он уже NaN).

    Возвращает {"mean": (N,44) float64, "std": (N,44) float64,
                "ts": (N,) int64, "window": int}.
    """
    if df is None or len(df) == 0:
        return {"mean": np.zeros((0, 0), dtype="float64"),
                "std": np.ones((0, 0), dtype="float64"),
                "ts": np.zeros(0, dtype="int64"), "window": int(window)}
    if isinstance(df, np.ndarray):
        arr = np.asarray(df, dtype="float64")
        idx = np.asarray(ts, dtype="int64") if ts is not None \
            else np.arange(len(arr), dtype="int64")
        if len(idx) != len(arr):
            raise ValueError("ts must have the same length as df")
        df = pd.DataFrame(arr, index=idx)
    else:
        df = pd.DataFrame(df).reset_index(drop=True)
    win = max(1, int(window))

    mean = df.rolling(win, min_periods=1).mean().shift(1)
    std = df.rolling(win, min_periods=1).std().shift(1)

    mean_arr = np.asarray(mean, dtype="float64")
    std_arr = np.asarray(std, dtype="float64")
    std_arr = np.where(np.isfinite(std_arr) & (std_arr > STD_EPS),
                       std_arr, 1.0)
    mean_arr = np.where(np.isfinite(mean_arr), mean_arr, 0.0)

    ts = np.asarray(df.index, dtype="int64") if not isinstance(df.index, int) \
        else np.zeros(len(df), dtype="int64")
    return {"mean": mean_arr, "std": std_arr, "ts": ts,
            "window": int(win)}


def apply_normalization(features, stats):
    """Z-score вектора фич по rolling-статистике. Clip к [-5, 5].

    features — (44,) числовой вектор. stats — dict из compute_rolling_stats
    (тогда для live берётся статистика последнего бара) или его срез с
    ("mean": (44,), "std": (44,)) от stats_at. Возвращает np.ndarray float32 (44,).
    """
    x = np.asarray(features, dtype="float64").reshape(-1)
    mean = stats.get("mean", 0.0)
    std = stats.get("std", 1.0)
    mean = np.asarray(mean, dtype="float64")
    std = np.asarray(std, dtype="float64")
    if mean.ndim == 2:
        mean = mean[-1]
    if std.ndim == 2:
        std = std[-1]
    std = np.where(np.isfinite(std) & (std > STD_EPS), std, 1.0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    out = (x - mean) / std
    return np.clip(out, CLIP_MIN, CLIP_MAX).astype(np.float32)


def normalize_matrix(features, ts, stats):
    """Матричная нормализация для offline-билдинга: (N, 44) -> (N, 44) float32.

    Каждая строка нормализуется статистикой окна ПЕРЕД её баром (stats_at,
    без look-ahead): mean/std берётся по позиции ts в stats["ts"]. Это ровно
    то же, что stats_at + apply_normalization построчно, но векторизованно.
    """
    x = np.asarray(features, dtype="float64")
    if x.ndim != 2:
        raise ValueError(f"features must be (N, F), got {x.shape}")
    ts_arr = np.asarray(stats.get("ts"), dtype="int64")
    mean = np.asarray(stats.get("mean", 0.0), dtype="float64")
    std = np.asarray(stats.get("std", 1.0), dtype="float64")
    if mean.ndim != 2:
        mean = np.tile(mean, (len(ts_arr), 1))
    if std.ndim != 2:
        std = np.tile(std, (len(ts_arr), 1))
    pos = np.searchsorted(ts_arr, np.asarray(ts, dtype="int64"),
                          side="right") - 1
    pos = np.clip(pos, 0, len(ts_arr) - 1)
    m = mean[pos]
    s = std[pos]
    s = np.where(np.isfinite(s) & (s > STD_EPS), s, 1.0)
    m = np.where(np.isfinite(m), m, 0.0)
    out = (x - m) / s
    return np.clip(out, CLIP_MIN, CLIP_MAX).astype(np.float32)


def stats_at(stats, ts):
    """Срез статистики ровно на бар ts (без look-ahead: ближайший <= ts).

    Для offline-бэктеста запрос в бар ts нормализуется статистикой, которую
    можно было знать на баре ts (окно из баров ПЕРЕД ним). Для ts за пределами
    истории берётся крайний доступный бар (0..len-1).
    Возвращает {"mean": (44,), "std": (44,), "ts": int, "pos": int}.
    """
    ts_arr = stats.get("ts")
    if ts_arr is None or len(ts_arr) == 0:
        return {"mean": stats.get("mean", 0.0), "std": stats.get("std", 1.0),
                "ts": None, "pos": -1}
    pos = int(np.searchsorted(np.asarray(ts_arr, dtype="int64"),
                              int(ts), side="right")) - 1
    pos = max(0, min(pos, len(ts_arr) - 1))
    return {"mean": np.asarray(stats["mean"])[pos],
            "std": np.asarray(stats["std"])[pos],
            "ts": int(ts_arr[pos]), "pos": pos}


def save_stats(stats, path):
    """Сериализация dict статистики в pickle. Создаёт каталог при необходимости."""
    path = str(path)
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(stats, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_stats(path):
    """Чтение dict статистики из pickle. None при отсутствии/повреждении."""
    path = str(path)
    try:
        with open(path, "rb") as fh:
            stats = pickle.load(fh)
        return stats if isinstance(stats, dict) else None
    except (OSError, EOFError, pickle.UnpicklingError, ValueError, AttributeError):
        log.warning("cbr normalize_stats load failed: %s", path)
        return None