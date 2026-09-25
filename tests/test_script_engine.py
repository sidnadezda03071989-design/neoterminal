"""Security and functional regression tests for the Python scripting engine."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from app_pkg.data import replay_trainer as replay_trainer_module
from app_pkg.data.replay_trainer import ReplayTrainer
from app_pkg.routes.scripts import STARTER_TEMPLATES
from app_pkg.scripting import engine as engine_module
from app_pkg.scripting.engine import (
    ScriptEngine,
    ScriptExecutionError,
    ScriptTimeoutError,
    SecurityError,
)


def _frame(size: int = 300) -> pd.DataFrame:
    """Small deterministic OHLCV frame with enough history for the templates."""
    index = np.arange(size, dtype=float)
    close = 100.0 + index * 0.03 + np.sin(index / 4.0) * 1.8
    return pd.DataFrame({
        "timestamp": pd.date_range(
            "2024-01-01", periods=size, freq="h", tz="UTC",
        ),
        "open": close - 0.1,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": 1_000.0 + index,
    })


def _context(
    size: int = 300,
    *,
    params: dict | None = None,
    snapshot: dict | None = None,
) -> dict:
    return {
        "df": _frame(size),
        "snapshot": snapshot or {
            "t": {"close": 100.0, "atr": 1.0},
            "vp": {"poc": 100.0},
            "d": {"funding_zscore": 2.0},
        },
        "levels": [],
        "params": params or {},
        "symbol": "BTCUSDT",
        "timeframe": "1H",
    }


def _template(template_id: str) -> str:
    return next(item["code"] for item in STARTER_TEMPLATES
                if item["id"] == template_id)


class TestScriptEngineSecurity:
    def test_blocks_os_import(self):
        engine = ScriptEngine()
        try:
            with pytest.raises(SecurityError, match="os"):
                engine.execute("import os\nresult = {}", _context(2))
        finally:
            engine.close()

    def test_blocks_subprocess(self):
        engine = ScriptEngine()
        try:
            with pytest.raises(SecurityError, match="subprocess"):
                engine.execute(
                    "import subprocess\nresult = {}", _context(2),
                )
        finally:
            engine.close()

    def test_blocks_eval_exec(self):
        engine = ScriptEngine()
        try:
            for code in ("result = eval('1 + 1')",
                         "result = exec('x = 1')"):
                with pytest.raises(SecurityError):
                    engine.execute(code, _context(2))
        finally:
            engine.close()

    def test_blocks_file_access(self):
        engine = ScriptEngine()
        try:
            with pytest.raises(SecurityError, match="open"):
                engine.execute(
                    "result = open('secret.txt').read()", _context(2),
                )
        finally:
            engine.close()

    def test_timeout_on_infinite_loop(self):
        engine = ScriptEngine()
        started = time.monotonic()
        try:
            with pytest.raises(ScriptTimeoutError):
                engine.execute(
                    "while True:\n    pass\nresult = {}",
                    _context(2),
                    timeout_sec=0.2,
                )
            # Worker startup is not charged to the script deadline; the loop
            # itself must nevertheless be stopped promptly.
            assert time.monotonic() - started < 5.0
        finally:
            engine.close()

    def test_memory_limit_enforced(self, monkeypatch):
        """Exercise the parent RSS guard without allocating 512 MiB in CI."""
        monkeypatch.setattr(
            engine_module,
            "_worker_memory_bytes",
            lambda _pid: engine_module._CHILD_MEMORY_BYTES + 1,
        )
        engine = ScriptEngine()
        try:
            with pytest.raises(ScriptExecutionError, match="memory limit"):
                engine.execute(
                    "while True:\n    pass\nresult = {}",
                    _context(2),
                    timeout_sec=1.0,
                )
        finally:
            engine.close()


class TestScriptEngineFunctional:
    def test_rsi_vp_template_runs(self):
        engine = ScriptEngine()
        try:
            result = engine.execute(
                _template("rsi_vp"),
                _context(300, params={"rsi_period": 14, "atr_distance": 0.5}),
                timeout_sec=3,
            )
        finally:
            engine.close()

        assert result["signal"] in {"LONG", "SHORT", "NEUTRAL"}
        assert 0.0 <= result["confidence"] <= 1.0
        assert result["rsi"] is not None
        assert len(result["preview"]) == 120

    def test_scanner_template_returns_results(self):
        engine = ScriptEngine()
        frame = _frame(300)
        last_close = float(frame["close"].iloc[-1])
        snapshot = {
            "t": {"close": last_close, "atr": 1.0},
            "vp": {"poc": last_close},
            "d": {"funding_zscore": 2.0},
        }
        try:
            result = engine.execute(
                _template("vp_funding_scanner"),
                _context(300, params={"atr_distance": 0.5,
                                       "funding_zscore": 1.0},
                         snapshot=snapshot),
                timeout_sec=3,
            )
        finally:
            engine.close()

        assert result["signal"] == "MATCH"
        assert result["confidence"] > 0.0
        assert result["funding_zscore"] == pytest.approx(2.0)
        assert result["poc"] == pytest.approx(last_close)
        assert len(result["preview"]) == 120

    def test_trainable_ma_returns_metrics(self, monkeypatch):
        size = 120
        frame = _frame(size)
        start_sec = 1_704_067_200  # 2024-01-01T00:00:00Z
        end_sec = start_sec + (size - 1) * 3_600
        monkeypatch.setattr(
            replay_trainer_module,
            "get_replay_df",
            lambda *args, **kwargs: frame.copy(),
        )

        report = ReplayTrainer().train(
            _template("ma_crossover"),
            "BTCUSDT",
            "1H",
            start_sec,
            end_sec,
            {"fast": [3, 5], "slow": [8]},
        )

        assert report["bars"] == size
        assert report["errors"] == 0
        assert report["grid"] == [
            {"fast": 3, "slow": 8},
            {"fast": 5, "slow": 8},
        ]
        assert report["best_params"] in report["grid"]
        assert report["metrics"]["walk_forward"] is True
        assert set(report["metrics"]) >= {"train", "test", "full"}
        assert "sharpe" in report["metrics"]["full"]
        assert len(report["signals"]) > 0
