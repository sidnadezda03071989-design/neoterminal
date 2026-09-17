"""Общие утилиты NeoTerminal: время, очистка чисел, атомарная запись JSON."""

import json
import math
import os
import tempfile
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd


def now_sec() -> float:
    """Текущее время в Unix-секундах (float)."""
    return time.time()


def now_iso() -> str:
    """Текущее время UTC в ISO-формате YYYY-MM-DDTHH:MM:SSZ."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _clean(v):
    """Приводит значение к float; NaN/Inf/None/нечисло -> None."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _atomic_json_write(path, data) -> None:
    """Атомарная запись JSON: tempfile + os.replace."""
    path = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def epoch_secs(series) -> np.ndarray:
    """Переводит серию дат/таймстампов в int64 Unix-секунды через datetime64[s]."""
    arr = np.asarray(pd.to_datetime(series))
    return arr.astype("datetime64[s]").astype("int64")