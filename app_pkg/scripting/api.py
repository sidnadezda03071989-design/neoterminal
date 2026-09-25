"""Read-only, JSON-friendly market API exposed to user scripts.

Real pandas/numpy objects are intentionally not returned to the script.  They
are wrapped in small facades whose public methods are sufficient for OHLCV
indicators and strategies, while private attributes, imports, file access and
dynamic callables remain outside the script's reach.
"""

from __future__ import annotations

import math as _math
import numbers
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as _np
import pandas as _pd

_MAX_SAFE_ITEMS = 2_000
_MAX_SAFE_DEPTH = 12
_MAX_OUTPUT_NODES = 10_000
_SAFE_FIELDS = ("timestamp", "open", "high", "low", "close", "volume")
_SAFE_PARAM_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_RESERVED_PARAM_NAMES = {
    "data", "df", "snapshot", "levels", "params", "symbol", "timeframe",
    "bar_time", "math", "np", "pd", "numpy", "pandas", "ta", "stats",
    "result", "print",
}


def _unwrap(value: Any) -> Any:
    if isinstance(value, SafeSeries):
        return value._series
    if isinstance(value, SafeFrame):
        return value._frame
    if isinstance(value, SafeArray):
        return value._array
    if isinstance(value, SafeMapping):
        return {key: _unwrap(item) for key, item in value._data.items()}
    if isinstance(value, SafeSequence):
        return [_unwrap(item) for item in value._items]
    return value


def _values(value: Any) -> list[Any]:
    value = _unwrap(value)
    if isinstance(value, SafeSequence):
        return list(value._items)
    if isinstance(value, (list, tuple, set, range)):
        return list(value)
    if isinstance(value, (str, bytes)):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def _scalar(value: Any) -> Any:
    value = _unwrap(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    return value


def _safe_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not _math.isfinite(number):
        return None
    return number


def _epoch_seconds(series: _pd.Series) -> _pd.Series:
    """Convert datetime-like or numeric epoch columns to Unix seconds."""
    if _pd.api.types.is_numeric_dtype(series):
        values = _pd.to_numeric(series, errors="coerce")
        finite = values.abs().dropna()
        scale = 1.0
        if len(finite):
            magnitude = float(finite.max())
            if magnitude > 1e17:
                scale = 1e9
            elif magnitude > 1e14:
                scale = 1e6
            elif magnitude > 1e11:
                scale = 1e3
        return values / scale
    converted = _pd.to_datetime(series, utc=True, errors="coerce")
    divisor = {"s": 1, "ms": 1_000, "us": 1_000_000, "ns": 1_000_000_000}.get(
        getattr(converted.dtype, "unit", "ns"), 1_000_000_000
    )
    return converted.astype("int64") / divisor


class SafeSequence:
    """Read-only sequence facade used for levels and small numeric results."""

    def __init__(self, values: Sequence[Any] | None = None) -> None:
        self._items = [
            _safe_public_value(value, 1)
            for value in list(values or [])[:_MAX_SAFE_ITEMS]
        ]

    def __getitem__(self, key: Any) -> Any:
        value = self._items[key]
        if isinstance(value, (SafeMapping, SafeSequence)):
            return value
        return _public_value(value)

    def __iter__(self):
        return iter([
            value if isinstance(value, (SafeMapping, SafeSequence)) else _public_value(value)
            for value in self._items
        ])

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, value: Any) -> bool:
        return value in self._items

    def index(self, value: Any) -> int:
        return self._items.index(value)

    def count(self, value: Any) -> int:
        return self._items.count(value)


class SafeMapping:
    """Read-only mapping facade for snapshots and parameter dictionaries."""

    def __init__(self, values: Mapping[str, Any] | None = None) -> None:
        self._data: dict[str, Any] = {}
        for key, value in list((values or {}).items())[:_MAX_SAFE_ITEMS]:
            if isinstance(key, str):
                self._data[key] = _safe_public_value(value, 1)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        if key in self._data:
            return self._data[key]
        return _safe_public_value(default, 1)

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def keys(self):
        return SafeSequence(self._data.keys())

    def values(self):
        return SafeSequence(self._data.values())

    def items(self):
        return SafeSequence([(key, value) for key, value in self._data.items()])


class SafeILoc:
    def __init__(self, owner: SafeSeries | SafeFrame) -> None:
        self._owner = owner

    def __getitem__(self, key: Any) -> Any:
        owner = self._owner
        if isinstance(owner, SafeSeries):
            value = owner._series.iloc[key]
            if isinstance(key, slice):
                return SafeSeries(value)
            return _scalar(value)
        if isinstance(key, slice):
            return SafeFrame(owner._frame.iloc[key])
        row = owner._frame.iloc[key]
        if isinstance(row, _pd.Series):
            return SafeMapping({str(col): row.get(col) for col in owner._frame.columns})
        return SafeFrame(row.to_frame().T)


class SafeRolling:
    def __init__(self, series: SafeSeries, window: int, min_periods: int | None = None) -> None:
        self._rolling = series._series.rolling(
            window=max(1, int(window)),
            min_periods=min_periods,
        )

    def mean(self) -> SafeSeries:
        return SafeSeries(self._rolling.mean())

    def std(self, ddof: int = 1) -> SafeSeries:
        return SafeSeries(self._rolling.std(ddof=ddof))

    def min(self) -> SafeSeries:
        return SafeSeries(self._rolling.min())

    def max(self) -> SafeSeries:
        return SafeSeries(self._rolling.max())

    def sum(self) -> SafeSeries:
        return SafeSeries(self._rolling.sum())


class SafeEwm:
    def __init__(self, series: SafeSeries, span: int | None = None,
                 alpha: float | None = None, adjust: bool = False) -> None:
        kwargs: dict[str, Any] = {"adjust": bool(adjust)}
        if span is not None:
            kwargs["span"] = max(1, int(span))
        if alpha is not None:
            kwargs["alpha"] = float(alpha)
        self._ewm = series._series.ewm(**kwargs)

    def mean(self) -> SafeSeries:
        return SafeSeries(self._ewm.mean())

    def std(self, ddof: int = 1) -> SafeSeries:
        return SafeSeries(self._ewm.std(ddof=ddof))


class SafeSeries:
    """Small pandas-Series-like read-only facade."""

    def __init__(self, values: Any, name: str | None = None) -> None:
        if isinstance(values, SafeSeries):
            series = values._series.copy()
        else:
            series = _pd.Series(_unwrap(values), name=name)
        if not _pd.api.types.is_numeric_dtype(series):
            series = _pd.to_numeric(series, errors="coerce")
        self._series = series
        self._name = name or (str(series.name) if series.name is not None else "")

    @property
    def iloc(self) -> SafeILoc:
        return SafeILoc(self)

    @property
    def name(self) -> str:
        return self._name

    def __len__(self) -> int:
        return int(len(self._series))

    def __iter__(self):
        for value in self._series.tolist():
            yield _scalar(value)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, slice):
            return SafeSeries(self._series.iloc[key])
        if isinstance(key, (list, tuple, _np.ndarray)):
            return SafeSeries(self._series.iloc[key])
        return _scalar(self._series.iloc[key])

    def __float__(self) -> float:
        if len(self._series) != 1:
            raise TypeError("A numeric conversion requires exactly one value")
        return float(self._series.iloc[0])

    def __repr__(self) -> str:
        return f"<SafeSeries name={self._name!r} length={len(self)}>"

    def _binary(self, other: Any, op: str) -> SafeSeries:
        rhs = _unwrap(other)
        return SafeSeries(getattr(self._series, op)(rhs))

    def __add__(self, other: Any) -> SafeSeries: return self._binary(other, "__add__")
    def __radd__(self, other: Any) -> SafeSeries: return SafeSeries(_unwrap(other) + self._series)
    def __sub__(self, other: Any) -> SafeSeries: return self._binary(other, "__sub__")
    def __rsub__(self, other: Any) -> SafeSeries: return SafeSeries(_unwrap(other) - self._series)
    def __mul__(self, other: Any) -> SafeSeries: return self._binary(other, "__mul__")
    def __rmul__(self, other: Any) -> SafeSeries: return SafeSeries(_unwrap(other) * self._series)
    def __truediv__(self, other: Any) -> SafeSeries: return self._binary(other, "__truediv__")
    def __rtruediv__(self, other: Any) -> SafeSeries: return SafeSeries(_unwrap(other) / self._series)
    def __neg__(self) -> SafeSeries: return SafeSeries(-self._series)
    def __abs__(self) -> SafeSeries: return SafeSeries(self._series.abs())
    def __gt__(self, other: Any) -> SafeSeries: return self._binary(other, "__gt__")
    def __ge__(self, other: Any) -> SafeSeries: return self._binary(other, "__ge__")
    def __lt__(self, other: Any) -> SafeSeries: return self._binary(other, "__lt__")
    def __le__(self, other: Any) -> SafeSeries: return self._binary(other, "__le__")
    def __eq__(self, other: Any) -> SafeSeries: return self._binary(other, "__eq__")

    def rolling(self, window: int, min_periods: int | None = None) -> SafeRolling:
        return SafeRolling(self, window, min_periods)

    def ewm(self, span: int | None = None, alpha: float | None = None,
            adjust: bool = False) -> SafeEwm:
        return SafeEwm(self, span=span, alpha=alpha, adjust=adjust)

    def mean(self) -> float | None: return _safe_number(self._series.mean())
    def std(self, ddof: int = 1) -> float | None: return _safe_number(self._series.std(ddof=ddof))
    def min(self) -> float | None: return _safe_number(self._series.min())
    def max(self) -> float | None: return _safe_number(self._series.max())
    def sum(self) -> float | None: return _safe_number(self._series.sum())
    def median(self) -> float | None: return _safe_number(self._series.median())
    def last(self) -> float | None: return _safe_number(self._series.iloc[-1]) if len(self) else None
    def count(self) -> int: return int(self._series.count())
    def stddev(self) -> float | None: return self.std()

    def diff(self, periods: int = 1) -> SafeSeries: return SafeSeries(self._series.diff(periods=periods))
    def pct_change(self, periods: int = 1) -> SafeSeries: return SafeSeries(self._series.pct_change(periods=periods))
    def shift(self, periods: int = 1) -> SafeSeries: return SafeSeries(self._series.shift(periods=periods))
    def dropna(self) -> SafeSeries: return SafeSeries(self._series.dropna())
    def fillna(self, value: Any = 0.0) -> SafeSeries: return SafeSeries(self._series.fillna(_unwrap(value)))
    def abs(self) -> SafeSeries: return SafeSeries(self._series.abs())
    def exp(self) -> SafeSeries: return SafeSeries(self._series.exp())
    def log(self) -> SafeSeries: return SafeSeries(self._series.map(_math.log))
    def cumsum(self) -> SafeSeries: return SafeSeries(self._series.cumsum())
    def round(self, decimals: int = 0) -> SafeSeries: return SafeSeries(self._series.round(decimals))
    def quantile(self, q: float) -> float | None: return _safe_number(self._series.quantile(q))
    def to_list(self) -> list[Any]: return [_scalar(value) for value in self._series.tolist()]
    def tolist(self) -> list[Any]: return self.to_list()

    def __bool__(self) -> bool:
        raise TypeError("A SafeSeries cannot be used as a scalar truth value")


class SafeFrame:
    """Read-only DataFrame facade for OHLCV data."""

    def __init__(self, frame: Any = None) -> None:
        if isinstance(frame, SafeFrame):
            frame = frame._frame
        elif isinstance(frame, Mapping):
            frame = _pd.DataFrame({
                str(key): _unwrap(value) for key, value in frame.items()
            })
        if not isinstance(frame, _pd.DataFrame):
            frame = _pd.DataFrame(columns=_SAFE_FIELDS)
        columns = [column for column in _SAFE_FIELDS if column in frame.columns]
        if not columns:
            columns = []
        prepared_window = bool(getattr(frame, "attrs", {}).get("_nt_prepared_window"))
        selected = frame.loc[:, columns].tail(300)
        self._frame = selected if prepared_window else selected.copy()
        if "timestamp" in self._frame.columns and not _pd.api.types.is_numeric_dtype(
            self._frame["timestamp"]
        ):
            self._frame = self._frame.copy()
            self._frame["timestamp"] = _epoch_seconds(self._frame["timestamp"])
        if not prepared_window:
            self._frame.reset_index(drop=True, inplace=True)

    @property
    def columns(self) -> SafeSequence:
        return SafeSequence([str(column) for column in self._frame.columns])

    @property
    def empty(self) -> bool:
        return bool(self._frame.empty)

    @property
    def shape(self) -> tuple[int, int]:
        return (int(len(self._frame)), int(len(self._frame.columns)))

    @property
    def iloc(self) -> SafeILoc:
        return SafeILoc(self)

    def __len__(self) -> int:
        return int(len(self._frame))

    def __iter__(self):
        return iter([str(column) for column in self._frame.columns])

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            if key not in self._frame.columns:
                raise KeyError(key)
            return SafeSeries(self._frame[key], name=key)
        if isinstance(key, list):
            return SafeFrame(self._frame[key])
        raise TypeError("DataFrame key must be a column name or column list")

    def __repr__(self) -> str:
        return f"<SafeFrame rows={len(self)} columns={len(self.columns)}>"

    def head(self, n: int = 5) -> SafeFrame: return SafeFrame(self._frame.head(n))
    def tail(self, n: int = 5) -> SafeFrame: return SafeFrame(self._frame.tail(n))
    def copy(self) -> SafeFrame: return SafeFrame(self._frame.copy())
    def reset_index(self, drop: bool = False) -> SafeFrame: return SafeFrame(self._frame.reset_index(drop=drop))
    def rolling(self, window: int, min_periods: int | None = None) -> SafeFrame:
        # A frame-level rolling facade is intentionally minimal; column-level
        # rolling is the documented/supported indicator path.
        return SafeFrame(self._frame.rolling(window, min_periods=min_periods).mean())


class SafeRow(SafeMapping):
    """Named row mapping returned by ``df.iloc[row]``."""


class SafeArray:
    """Read-only numeric ndarray facade."""

    def __init__(self, values: Any) -> None:
        array = _np.asarray(_unwrap(values))
        if array.dtype == object:
            array = _np.asarray([_safe_number(value) or 0.0 for value in array.tolist()])
        self._array = array

    def __len__(self) -> int: return int(len(self._array))
    def __iter__(self):
        for value in self._array.tolist(): yield _scalar(value)
    def __getitem__(self, key: Any) -> Any:
        value = self._array[key]
        return SafeArray(value) if isinstance(key, slice) else _scalar(value)
    def __repr__(self) -> str: return f"<SafeArray shape={self._array.shape}>"
    def tolist(self) -> list[Any]: return [_scalar(value) for value in self._array.tolist()]
    def mean(self) -> float | None: return _safe_number(_np.nanmean(self._array))
    def std(self) -> float | None: return _safe_number(_np.nanstd(self._array))
    def min(self) -> float | None: return _safe_number(_np.nanmin(self._array))
    def max(self) -> float | None: return _safe_number(_np.nanmax(self._array))
    def sum(self) -> float | None: return _safe_number(_np.nansum(self._array))
    def __float__(self) -> float:
        if self._array.size != 1: raise TypeError("A numeric conversion requires one value")
        return float(self._array.reshape(-1)[0])
    def __add__(self, other: Any) -> SafeArray: return SafeArray(self._array + _unwrap(other))
    def __sub__(self, other: Any) -> SafeArray: return SafeArray(self._array - _unwrap(other))
    def __mul__(self, other: Any) -> SafeArray: return SafeArray(self._array * _unwrap(other))
    def __truediv__(self, other: Any) -> SafeArray: return SafeArray(self._array / _unwrap(other))
    def __neg__(self) -> SafeArray: return SafeArray(-self._array)
    def __abs__(self) -> SafeArray: return SafeArray(_np.abs(self._array))


class SafeFunction:
    """Callable facade whose function object belongs to this API module."""

    def __init__(self, function: Any) -> None:
        self._function = function

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._function(*args, **kwargs)


class _MathFacade:
    _NAMES = {
        "acos", "asin", "atan", "atan2", "ceil", "cos", "cosh", "degrees",
        "exp", "fabs", "floor", "fmod", "hypot", "isfinite", "isnan",
        "log", "log10", "pow", "radians", "sin", "sinh", "sqrt", "tan",
        "tanh", "trunc", "pi", "e", "tau",
    }

    def __getattr__(self, name: str) -> Any:
        if name not in self._NAMES or not hasattr(_math, name):
            raise AttributeError(name)
        value = getattr(_math, name)
        return SafeFunction(value) if callable(value) else value


class _NumpyFacade:
    _NAMES = {
        "array", "asarray", "arange", "linspace", "where", "maximum",
        "minimum", "sqrt", "abs", "log", "exp", "sin", "cos", "tan",
        "isfinite", "isnan", "nan_to_num", "clip", "mean", "median", "std",
        "var", "percentile", "quantile", "sum", "amin", "amax",
    }

    def __getattr__(self, name: str) -> Any:
        if name not in self._NAMES or not hasattr(_np, name):
            raise AttributeError(name)

        def call(*args: Any, **kwargs: Any) -> Any:
            raw_args = [_unwrap(value) for value in args]
            raw_kwargs = {key: _unwrap(value) for key, value in kwargs.items()}
            value = getattr(_np, name)(*raw_args, **raw_kwargs)
            if name in {"where", "maximum", "minimum", "sqrt", "abs", "log", "exp", "sin", "cos", "tan", "isfinite", "isnan", "nan_to_num", "clip", "array", "asarray", "arange", "linspace"}:
                return SafeArray(value)
            return _scalar(value)

        return SafeFunction(call)


class _PandasFacade:
    def __getattr__(self, name: str) -> Any:
        if name in {"Series", "DataFrame"}:
            if name == "Series":
                return SafeFunction(lambda value, *args, **kwargs: SafeSeries(value, kwargs.get("name")))
            return SafeFunction(lambda value=None, *args, **kwargs: SafeFrame(value))
        if name in {"isna", "notna"}:
            return SafeFunction(lambda value: SafeArray(getattr(_pd, name)(_unwrap(value))))
        if name == "to_numeric":
            return SafeFunction(lambda value: SafeSeries(_pd.to_numeric(_unwrap(value), errors="coerce")))
        raise AttributeError(name)


class _TAFacade:
    @staticmethod
    def _rsi(series: Any, period: int = 14) -> SafeSeries:
        values = _unwrap(series)
        delta = _pd.Series(values).astype("float64").diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1 / max(1, int(period)), adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / max(1, int(period)), adjust=False).mean()
        with _np.errstate(divide="ignore", invalid="ignore"):
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        rsi = rsi.where(avg_loss != 0, 100).where(~((avg_gain == 0) & (avg_loss != 0)), 0)
        return SafeSeries(rsi.fillna(50))

    def __getattr__(self, name: str) -> Any:
        if name in {"rsi", "RSI"}:
            return SafeFunction(self._rsi)
        if name in {"sma", "SMA"}:
            return SafeFunction(lambda series, period: SafeSeries(_unwrap(series).rolling(int(period)).mean()))
        if name in {"ema", "EMA"}:
            return SafeFunction(lambda series, period: SafeSeries(_unwrap(series).ewm(span=int(period), adjust=False).mean()))
        if name in {"atr", "ATR"}:
            def atr(high: Any, low: Any, close: Any, period: int = 14) -> SafeSeries:
                h, l, c = map(_unwrap, (high, low, close))
                prev = c.shift(1)
                tr = _pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
                return SafeSeries(tr.ewm(alpha=1 / max(1, int(period)), adjust=False).mean())
            return SafeFunction(atr)
        raise AttributeError(name)


class _StatsFacade:
    def __getattr__(self, name: str) -> Any:
        if name in {"zscore", "zscore_score"}:
            def zscore(values: Any, axis: int | None = None) -> SafeArray:
                array = _np.asarray(_unwrap(values), dtype=float)
                mean = _np.nanmean(array)
                std = _np.nanstd(array)
                return SafeArray((array - mean) / (std if std > 1e-12 else 1.0))
            return SafeFunction(zscore)
        if name == "pearsonr":
            def pearsonr(x: Any, y: Any) -> tuple[float | None, float | None]:
                result = _np.corrcoef(_unwrap(x), _unwrap(y))
                return (_safe_number(result[0, 1]), None)
            return SafeFunction(pearsonr)
        raise AttributeError(name)


_SAFE_MODULES = {
    "math": _MathFacade(),
    "numpy": _NumpyFacade(),
    "pandas": _PandasFacade(),
    "ta": _TAFacade(),
    "stats": _StatsFacade(),
    "scipy.stats": _StatsFacade(),
}


def get_safe_module(name: str) -> Any:
    """Return an allow-listed facade; never import the requested module."""
    return _SAFE_MODULES.get(name, _SAFE_MODULES["numpy"])


def make_safe_frame(frame: Any) -> SafeFrame:
    return SafeFrame(frame)


def make_safe_mapping(values: Any) -> SafeMapping:
    return SafeMapping(values if isinstance(values, Mapping) else {})


def make_safe_levels(values: Any) -> SafeSequence:
    if isinstance(values, SafeSequence):
        return values
    if not isinstance(values, (list, tuple)):
        return SafeSequence([])
    return SafeSequence(values)


def make_safe_params(values: Any) -> SafeMapping:
    if not isinstance(values, Mapping):
        return SafeMapping({})
    clean: dict[str, Any] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not _SAFE_PARAM_RE.match(key):
            continue
        if key in _RESERVED_PARAM_NAMES or key in {"__builtins__", "__import__"}:
            continue
        if isinstance(value, bool) or isinstance(value, (int, float, str)):
            clean[key] = value
    return SafeMapping(clean)


def _plain_value(value: Any, depth: int = 0) -> Any:
    if depth > _MAX_SAFE_DEPTH:
        return None
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if _math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item, depth + 1)
                for key, item in list(value.items())[:_MAX_SAFE_ITEMS]}
    if isinstance(value, (list, tuple, set)):
        return [_plain_value(item, depth + 1) for item in list(value)[:_MAX_SAFE_ITEMS]]
    return None


def normalise_context(
    context: Mapping[str, Any], max_rows: int = 300,
) -> dict[str, Any]:
    """Copy only the public context fields into a worker payload."""
    if not isinstance(context, Mapping):
        raise ValueError("Script context must be a mapping")
    raw_df = context.get("df")
    if not isinstance(raw_df, _pd.DataFrame):
        raw_df = _pd.DataFrame(columns=_SAFE_FIELDS)
    raw_df = raw_df.loc[:, [field for field in _SAFE_FIELDS if field in raw_df.columns]]
    raw_df = raw_df.tail(max(1, min(int(max_rows), 2_000))).copy()
    params = context.get("params")
    if not isinstance(params, Mapping):
        params = {}
    clean_params: dict[str, Any] = {}
    for key, value in params.items():
        if not isinstance(key, str):
            continue
        if isinstance(value, (list, tuple)):
            # A saved training grid is not a valid scalar runtime value.  Use
            # the first candidate for Run; ReplayTrainer expands the full grid
            # only on the training endpoint.
            value = value[0] if value else None
        if isinstance(value, (bool, int, float, str)):
            clean_params[key] = value
    return {
        "df": raw_df,
        "snapshot": _plain_value(context.get("snapshot") or {}),
        "levels": _plain_value(context.get("levels") or []),
        "params": clean_params,
        "symbol": str(context.get("symbol") or "")[:32],
        "timeframe": str(context.get("timeframe") or "")[:16],
        "bar_time": context.get("bar_time"),
    }


def _public_value(value: Any) -> Any:
    if isinstance(value, SafeMapping):
        return {key: _public_value(item) for key, item in value._data.items()}
    if isinstance(value, SafeSequence):
        return [_public_value(item) for item in value._items]
    if isinstance(value, SafeSeries):
        return value.to_list()
    if isinstance(value, SafeArray):
        return value.tolist()
    if isinstance(value, float) and not _math.isfinite(value):
        return None
    return value


def _safe_public_value(value: Any, depth: int = 0) -> Any:
    if depth > _MAX_SAFE_DEPTH:
        return None
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if _math.isfinite(value) else None
    if isinstance(value, Mapping):
        return SafeMapping(value)
    if isinstance(value, (list, tuple)):
        return SafeSequence(value)
    return None


def make_json_safe(value: Any) -> Any:
    """Bounded conversion used for the worker's response."""
    nodes = 0

    def convert(item: Any, depth: int = 0) -> Any:
        nonlocal nodes
        nodes += 1
        if nodes > _MAX_OUTPUT_NODES or depth > _MAX_SAFE_DEPTH:
            return None
        if item is None or isinstance(item, (bool, int, str)):
            return item[:2_000] if isinstance(item, str) else item
        if isinstance(item, float):
            return float(item) if _math.isfinite(float(item)) else None
        if isinstance(item, numbers.Integral):
            return int(item)
        if isinstance(item, numbers.Real):
            return _safe_number(item)
        if isinstance(item, SafeMapping):
            return {str(key): convert(val, depth + 1) for key, val in item._data.items()}
        if isinstance(item, SafeSequence):
            return [convert(val, depth + 1) for val in item._items]
        if isinstance(item, SafeSeries):
            return [convert(val, depth + 1) for val in item.to_list()]
        if isinstance(item, SafeArray):
            return [convert(val, depth + 1) for val in item.tolist()]
        if isinstance(item, Mapping):
            return {str(key): convert(val, depth + 1) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(val, depth + 1) for val in item]
        try:
            if isinstance(item, numbers.Real):
                return _safe_number(item)
        except (TypeError, ValueError):
            pass
        return None

    return convert(value)
