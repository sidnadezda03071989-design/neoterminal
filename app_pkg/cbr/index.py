# -*- coding: utf-8 -*-
"""FAISS-индекс CBR (Этап 2): HNSW32 / Flat, L2-метрика, сохранение/загрузка.

Индекс строится по НОРМАЛИЗОВАННЫМ 44-фич векторам (rolling z-score,
app_pkg.cbr.normalize) размеченных снимков (outcome IS NOT NULL) одного
(symbol, timeframe), отсортированных по ts ASC — позиция строки в матрице
= позиция в метаданных (regime/outcome/ts), которые читаются из той же БД.

Backend: faiss-cpu (IndexHNSWFlat M=32) — для 16701×44 HNSW быстрее Flat при
том же recall. Если faiss недоступен — фолбэк на sklearn.neighbors
(NearestNeighbors, медленнее, но ок для 17k). save/load round-trip идентичны.

Функции:
  build_faiss_index(X_normalized, index_type='HNSW') -> index
  save_index(index, path) / load_index(path)
  search_knn(index, query, k) -> (dist, idx): евклидовы расстояния (не L2²),
      idx = -1 при нехватке соседей.
  load_feature_matrix(conn, symbol, timeframe) -> (X, ts, regimes, y):
      сырые фичи БД (порядок ts ASC), для нормализации и индекса.
"""

import logging
import os

import numpy as np

log = logging.getLogger(__name__)

try:  # faiss-cpu опционален (фолбэк — sklearn).
    import faiss as _faiss
    _HAS_FAISS = True
except Exception:  # noqa: BLE001
    _faiss = None
    _HAS_FAISS = False

if not _HAS_FAISS:
    try:
        from sklearn.neighbors import NearestNeighbors
    except Exception:  # noqa: BLE001
        NearestNeighbors = None


# Параметры HNSW32 (M=32 — число связей на узел графа).
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 64
# Потолок соседей sklearn-фолбэка (покрывает K*3 запас ТЗ при K<=200).
SKLEARN_BACKEND_MAX_K = 1000
DIM = 44


def faiss_available():
    """True, если доступен настоящий FAISS backend."""
    return bool(_HAS_FAISS)


def _is_faiss_index(index):
    return _HAS_FAISS and _faiss is not None and isinstance(
        index, (_faiss.IndexHNSWFlat, _faiss.IndexFlatL2))


def build_faiss_index(X_normalized, index_type="HNSW"):
    """Строит индекс по нормализованным фичам (N, 44) float32.

    index_type: 'HNSW' (default, HNSW32) или 'FLAT'. На N=16701 HNSW быстрее
    Flat с почти тем же recall; Flat — точный перебор (эталон). Если faiss
    не установлен — возвращает sklearn NearestNeighbors (квадратичный поиск).
    """
    X = np.ascontiguousarray(np.asarray(X_normalized, dtype="float32"))
    if X.ndim != 2 or X.shape[1] != DIM:
        raise ValueError(f"X must be (N, {DIM}), got {X.shape}")
    kind = str(index_type or "HNSW").strip().upper()

    if _HAS_FAISS and _faiss is not None:
        if kind == "FLAT":
            index = _faiss.IndexFlatL2(DIM)
        else:
            index = _faiss.IndexHNSWFlat(DIM, HNSW_M)
            index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
            index.hnsw.efSearch = HNSW_EF_SEARCH
        index.add(X)
        return index

    try:
        from sklearn.neighbors import NearestNeighbors
    except Exception as exc:  # noqa: BLE001
        raise ImportError(f"neither faiss nor sklearn.neighbors available: {exc}")
    k = max(1, min(len(X), SKLEARN_BACKEND_MAX_K + 1))
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(X)
    return nn


def save_index(index, path):
    """Сохраняет индекс на диск. faiss — бинарный фaiss-формат, sklearn — pickle."""
    import os
    from pathlib import Path
    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if _is_faiss_index(index):
        _faiss.write_index(index, path)
    else:
        import pickle
        with open(path, "wb") as fh:
            pickle.dump(index, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_index(path):
    """Загружает индекс (faiss-файл или pickle). None при отсутствии/ошибке.

    После загрузки HNSW фиксирует efSearch (параметр не персистится в файл).
    """
    path = str(path)
    if not os.path.exists(path):
        return None
    if _HAS_FAISS and _faiss is not None:
        try:
            index = _faiss.read_index(path)
            try:
                index.hnsw.efSearch = HNSW_EF_SEARCH
            except AttributeError:
                pass
            return index
        except Exception as exc:  # noqa: BLE001 — файл мог быть pickle
            log.debug("faiss read_index failed (%s): %s", path, exc)
    import pickle
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except (OSError, EOFError, pickle.UnpicklingError, AttributeError, ValueError):
        log.warning("cbr index load failed: %s", path)
        return None


def search_knn(index, query, k):
    """Поиск k ближайших для одного нормализованного вектора.

    Возвращает (dist, idx): dist — ЕВКЛИДОВЫ расстояния (фaiss отдаёт L2² —
    конвертируем sqrt; sklearn уже евклид), idx — позиции соседей в матрице
    индекса, -1 если соседей меньше k. Вход query — (44,) или (1,44).
    """
    q = np.asarray(np.asarray(query, dtype="float32").reshape(1, -1),
                   dtype="float32")
    kk = max(1, int(k))
    if _is_faiss_index(index):
        dist, idx = index.search(np.ascontiguousarray(q), kk)
        dist = np.sqrt(np.maximum(dist, 0.0)).astype("float64")
    else:
        n_total = getattr(index, "n_samples_fit_", len(index._fit_X))
        kk = min(kk, n_total)
        dist, idx = index.kneighbors(q, n_neighbors=kk)
        dist = np.asarray(dist, dtype="float64")
        idx = np.asarray(idx, dtype="int64")
        if len(idx[0]) < k:  # добиваем -1 до запрошенной длины
            pad = np.full((1, k - len(idx[0])), -1, dtype="int64")
            idx = np.concatenate([idx, pad], axis=1)
            dist = np.concatenate(
                [dist, np.full((1, k - len(dist[0])), np.inf)], axis=1)
    return dist[0], idx[0]


def load_feature_matrix(conn, symbol, timeframe, dim=DIM):
    """(X, ts, regimes, y) размеченных снимков (symbol, timeframe) по ts ASC.

    Матрица фич — порядок БД `ORDER BY ts ASC` (позиции = метаданным).
    Строки с повреждённым вектором молча пропускаются.
    """
    rows = conn.execute(
        "SELECT features, ts, regime, outcome FROM snapshots "
        "WHERE symbol=? AND timeframe=? AND outcome IS NOT NULL "
        "ORDER BY ts ASC",
        (str(symbol).upper(), str(timeframe).strip())).fetchall()
    mats, ts, regimes, ys = [], [], [], []
    for r in rows:
        vec = _decode_features(r["features"], dim)
        if vec is None:
            continue
        mats.append(vec)
        ts.append(int(r["ts"]))
        regimes.append(r["regime"])
        ys.append(int(r["outcome"]))
    if not mats:
        raise ValueError(f"no labeled snapshots for {symbol} {timeframe}")
    X = np.stack(mats).astype(np.float32)
    return (X, np.asarray(ts, dtype="int64"),
            np.asarray(regimes, dtype=object), np.asarray(ys, dtype="int64"))


def _decode_features(raw, dim=DIM):
    """features-поле -> (dim,) float32 или None при повреждении."""
    import json
    if raw is None:
        return None
    if isinstance(raw, bytes):
        try:
            v = np.frombuffer(raw, dtype="<f4")
        except ValueError:
            return None
    elif isinstance(raw, str) and raw.startswith("["):
        try:
            v = np.asarray(json.loads(raw), dtype="<f4")
        except ValueError:
            return None
    else:
        try:
            v = np.asarray(raw, dtype="<f4")
        except (ValueError, TypeError):
            return None
    return v if v.size == dim else None