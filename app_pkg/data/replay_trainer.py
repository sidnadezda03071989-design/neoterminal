"""Leakage-safe replay execution and walk-forward parameter selection."""

from __future__ import annotations

import itertools
import logging
import math
import time
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from app_pkg import config, utils
from app_pkg.ai.volume_profile import calculate_vp
from app_pkg.data.fetch import get_replay_df
from app_pkg.scripting.engine import ScriptEngine

log = logging.getLogger(__name__)


TF_SECONDS = dict(config.TF_SECONDS)


class ReplayTrainer:
    """Execute a script at each closed historical bar.

    The script sees a frame ending at the current bar only.  Future prices are
    used after execution solely to score the already-produced signal, never to
    construct the context passed to the script.
    """

    LOOKBACK = 300
    MAX_BARS = 2_000
    MAX_COMBINATIONS = 32
    MAX_WALL_SECONDS = 30.0
    SCRIPT_TIMEOUT_SEC = 0.75

    def train(
        self,
        script_code: str,
        symbol: str,
        timeframe: str,
        start_sec: int,
        end_sec: int,
        params: dict,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict:
        """Run a bounded walk-forward optimisation and return JSON metrics."""
        if symbol not in config.SYMBOLS:
            raise ValueError("Invalid symbol")
        if timeframe not in config.TF_SECONDS:
            raise ValueError("Invalid timeframe")
        if int(end_sec) <= int(start_sec):
            raise ValueError("Invalid replay range")
        engine = ScriptEngine()
        started = time.monotonic()
        combinations = self._expand_params(params)
        total = len(combinations)
        try:
            frame = get_replay_df(
                symbol,
                timeframe,
                from_sec=int(start_sec),
                to_sec=int(end_sec),
                limit=self.MAX_BARS,
            )
            frame = self._prepare_frame(frame, int(start_sec), int(end_sec))
            bars = len(frame)
            timestamps = utils.epoch_secs(frame["timestamp"]).tolist()
            if bars < 2:
                return {
                    "metrics": self._empty_metrics(),
                    "best_params": combinations[0] if combinations else {},
                    "bars": bars,
                    "grid": combinations,
                    "signals": [],
                    "errors": 0,
                }

            cutoff = max(1, int(bars * 0.70))
            all_results: list[dict[str, Any]] = []
            combo_reports: list[dict[str, Any]] = []
            errors = 0
            snapshots: list[dict[str, Any]] = []
            for index in range(bars):
                window = frame.iloc[max(0, index - self.LOOKBACK + 1):index + 1]
                bar_time = int(timestamps[index])
                snapshots.append(self._replay_snapshot(
                    window, symbol, timeframe, bar_time,
                ))

            for combo_index, combo in enumerate(combinations, start=1):
                remaining = self.MAX_WALL_SECONDS - (time.monotonic() - started)
                if remaining <= 0:
                    break
                try:
                    items = engine.execute_series(
                        script_code, frame, timestamps, snapshots, combo,
                        symbol=symbol, timeframe=timeframe,
                        lookback=self.LOOKBACK,
                        timeout_sec=min(remaining, 25.0),
                    )
                except Exception as exc:
                    errors += bars
                    log.warning("script batch failed for %s: %s", combo, type(exc).__name__)
                    continue

                records: list[dict[str, Any]] = []
                for index, item in enumerate(items):
                    bar_time = int(timestamps[index])
                    if not item.get("ok"):
                        errors += 1
                        log.warning("script failed at %s: %s", bar_time, item.get("kind", "error"))
                        continue
                    output = item.get("result")
                    if not isinstance(output, dict):
                        output = {}
                    signal = self._signal_value(output.get("signal"))
                    confidence = self._confidence(output.get("confidence"))
                    close = utils._clean(frame.iloc[index]["close"])
                    records.append({
                        "time": bar_time,
                        "signal": signal,
                        "confidence": confidence,
                        "params": dict(combo),
                        "price": close,
                        "index": index,
                    })
                train_metrics = self._calculate_metrics(records, cutoff, "train")
                test_metrics = self._calculate_metrics(records, cutoff, "test")
                score = self._score(train_metrics)
                combo_reports.append({
                    "params": dict(combo), "train": train_metrics,
                    "test": test_metrics, "score": score,
                })
                all_results.extend(records)
                if progress:
                    progress({
                        "done": combo_index, "total": total,
                        "current": f"{symbol}:{timeframe}:{combo}",
                        "score": score,
                    })

            if combo_reports:
                best_report = max(
                    combo_reports,
                    key=lambda item: (item["score"], -item["params"].__len__()),
                )
                best_params = dict(best_report["params"])
                best_records = [
                    record for record in all_results
                    if record["params"] == best_params
                ]
                metrics = {
                    "train": best_report["train"],
                    "test": best_report["test"],
                    "full": self._calculate_metrics(best_records, cutoff, "full"),
                    "walk_forward": True,
                    "cutoff_index": cutoff,
                }
                signals = [
                    {"time": record["time"], "signal": record["signal"],
                     "confidence": record["confidence"], "price": record["price"]}
                    for record in best_records
                ]
            else:
                best_params = {}
                metrics = self._empty_metrics()
                signals = []

            return {
                "metrics": metrics,
                "best_params": best_params,
                "bars": bars,
                "grid": combinations,
                "signals": signals,
                "errors": errors,
                "elapsed_sec": round(time.monotonic() - started, 3),
            }
        finally:
            engine.close()

    @staticmethod
    def _expand_params(params: Any) -> list[dict[str, Any]]:
        if not isinstance(params, Mapping) or not params:
            return [{}]
        keys = sorted(str(key) for key in params)
        values: list[list[Any]] = []
        for key in keys:
            raw = params[key]
            if isinstance(raw, (list, tuple)):
                choices = list(raw)
            else:
                choices = [raw]
            if not choices or len(choices) > 20:
                raise ValueError(f"Invalid parameter grid for '{key}'")
            values.append(choices[:20])
        total = math.prod(len(items) for items in values)
        if total > ReplayTrainer.MAX_COMBINATIONS:
            raise ValueError(
                f"Too many parameter combinations: {total} > "
                f"{ReplayTrainer.MAX_COMBINATIONS}"
            )
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    @staticmethod
    def _prepare_frame(frame: Any, start_sec: int, end_sec: int) -> pd.DataFrame:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        result = frame.copy()
        if "timestamp" not in result.columns:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        raw_timestamp = result["timestamp"]
        if pd.api.types.is_numeric_dtype(raw_timestamp):
            numeric = pd.to_numeric(raw_timestamp, errors="coerce")
            finite = numeric.abs().dropna()
            magnitude = float(finite.max()) if len(finite) else 0.0
            unit = "s" if magnitude < 1e11 else "ms" if magnitude < 1e14 else "us" if magnitude < 1e17 else "ns"
            result["timestamp"] = pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
        else:
            result["timestamp"] = pd.to_datetime(raw_timestamp, utc=True, errors="coerce")
        result = result.dropna(subset=["timestamp"])
        if result.empty:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        stamps = utils.epoch_secs(result["timestamp"])
        mask = (stamps >= start_sec) & (stamps <= end_sec)
        result = result.loc[mask]
        result = result.sort_values("timestamp").drop_duplicates("timestamp")
        return result.head(ReplayTrainer.MAX_BARS).reset_index(drop=True)

    @staticmethod
    def _replay_snapshot(
        frame: pd.DataFrame, symbol: str, timeframe: str, bar_time: int
    ) -> dict[str, Any]:
        """Build a local, zero-network snapshot from the current window."""
        from app_pkg.indicators import _atr, rsi_wilder

        close = pd.to_numeric(frame["close"], errors="coerce")
        high = pd.to_numeric(frame["high"], errors="coerce")
        low = pd.to_numeric(frame["low"], errors="coerce")
        vp = calculate_vp(frame)
        last_close = utils._clean(close.iloc[-1]) if len(close) else None
        rsi = utils._clean(rsi_wilder(close, 14).iloc[-1]) if len(close) >= 15 else None
        atr = utils._clean(_atr(high, low, close, 14).iloc[-1]) if len(close) >= 15 else None

        tg: list[list[float]] = []
        for key, probability in (("poc", 0.45), ("vah", 0.275), ("val", 0.275)):
            price = utils._clean(vp.get(key)) if isinstance(vp, Mapping) else None
            if price is not None:
                tg.append([float(price), probability])

        return {
            "t": {
                "timestamp": int(bar_time),
                "close": last_close,
                "rsi": rsi,
                "atr": atr,
            },
            "vp": vp,
            # No live derivatives are inserted here.  A historical funding
            # series is not available from the OHLCV source, and inserting the
            # current value would be look-ahead leakage.
            "d": {},
            "tg": tg,
            "replay": True,
        }

    @staticmethod
    def _signal_value(signal: Any) -> int:
        if isinstance(signal, bool):
            return 0
        if isinstance(signal, (int, float)) and math.isfinite(float(signal)):
            return 1 if float(signal) > 0 else -1 if float(signal) < 0 else 0
        text = str(signal or "").strip().upper()
        if text in {"BUY", "LONG", "L", "UP", "BULL", "1"}:
            return 1
        if text in {"SELL", "SHORT", "S", "DOWN", "BEAR", "-1"}:
            return -1
        return 0

    @staticmethod
    def _confidence(value: Any) -> float:
        number = utils._clean(value)
        if number is None:
            return 0.0
        return min(1.0, max(0.0, float(number)))

    @classmethod
    def _calculate_metrics(
        cls, records: list[dict[str, Any]], cutoff: int | None = None,
        segment: str = "full",
    ) -> dict[str, Any]:
        if cutoff is None:
            cutoff = len(records)
        if segment == "train":
            selected = [record for record in records if record["index"] < cutoff]
        elif segment == "test":
            selected = [record for record in records if record["index"] >= cutoff]
        else:
            selected = list(records)

        returns: list[float] = []
        wins = 0
        losses = 0
        for previous, current in zip(selected, selected[1:]):
            direction = cls._signal_value(previous.get("signal"))
            old_price = utils._clean(previous.get("price"))
            new_price = utils._clean(current.get("price"))
            if direction == 0 or old_price is None or new_price is None or old_price <= 0:
                returns.append(0.0)
                continue
            ret = direction * (new_price / old_price - 1.0) - 0.0005
            returns.append(float(ret))
            if ret > 0:
                wins += 1
            elif ret < 0:
                losses += 1

        if not returns:
            return {
                "trades": 0, "win_rate": 0.0, "total_return": 0.0,
                "sharpe": 0.0, "max_drawdown": 0.0, "profit_factor": 0.0,
            }
        equity = 1.0
        peak = 1.0
        max_dd = 0.0
        for ret in returns:
            equity *= 1.0 + ret
            peak = max(peak, equity)
            max_dd = max(max_dd, (peak - equity) / peak)
        mean = float(np.mean(returns))
        std = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
        sharpe = mean / std * math.sqrt(252.0) if std > 1e-12 else 0.0
        gross_win = sum(ret for ret in returns if ret > 0)
        gross_loss = abs(sum(ret for ret in returns if ret < 0))
        return {
            "trades": wins + losses,
            "win_rate": round(wins / (wins + losses), 4) if wins + losses else 0.0,
            "total_return": round(equity - 1.0, 6),
            "sharpe": round(sharpe, 4),
            "max_drawdown": round(max_dd, 6),
            "profit_factor": round(gross_win / gross_loss, 4) if gross_loss > 1e-12 else 0.0,
        }

    @classmethod
    def _optimize_params(cls, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Return the best parameter set from already-produced replay rows."""
        groups: dict[tuple[tuple[str, Any], ...], list[dict[str, Any]]] = {}
        for record in results:
            params = record.get("params") or {}
            key = tuple(sorted((str(name), repr(value)) for name, value in params.items()))
            groups.setdefault(key, []).append(record)
        candidates = []
        for rows in groups.values():
            metrics = cls._calculate_metrics(rows)
            candidates.append((cls._score(metrics), dict(rows[0].get("params") or {}), metrics))
        if not candidates:
            return {}
        return max(candidates, key=lambda item: item[0])[1]

    @staticmethod
    def _score(metrics: Mapping[str, Any]) -> float:
        trades = float(metrics.get("trades") or 0)
        if trades < 2:
            return -999.0
        return round(float(metrics.get("sharpe") or 0.0), 6)

    @staticmethod
    def _empty_metrics() -> dict[str, Any]:
        return {
            "train": {"trades": 0, "win_rate": 0.0, "total_return": 0.0,
                      "sharpe": 0.0, "max_drawdown": 0.0, "profit_factor": 0.0},
            "test": {"trades": 0, "win_rate": 0.0, "total_return": 0.0,
                     "sharpe": 0.0, "max_drawdown": 0.0, "profit_factor": 0.0},
            "full": {"trades": 0, "win_rate": 0.0, "total_return": 0.0,
                     "sharpe": 0.0, "max_drawdown": 0.0, "profit_factor": 0.0},
            "walk_forward": True,
        }
