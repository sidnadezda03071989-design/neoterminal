"""REST API and editor page for secure user scripts."""

from __future__ import annotations

import json
import logging
import math
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from flask import Blueprint, jsonify, render_template, request

from app_pkg import config, utils
from app_pkg.data.fetch import get_series_df
from app_pkg.data.market_snapshot import compact_snapshot
from app_pkg.data.replay_trainer import ReplayTrainer
from app_pkg.scripting.engine import (
    ScriptEngine,
    ScriptExecutionError,
    ScriptTimeoutError,
    SecurityError,
)
from app_pkg.ws import _ws_push

log = logging.getLogger(__name__)
bp = Blueprint("scripts", __name__)

_ENGINE = ScriptEngine()
_SCRIPT_DIR = config.DATA_DIR / "scripts"
_SCRIPT_ID_RE = re.compile(r"^[a-f0-9]{32}$")
_MAX_SCRIPT_NAME = 100
_MAX_DESCRIPTION = 500
_MAX_TRAIN_RANGE_SEC = 190 * 86400
_MAX_ACTIVE_TRAININGS = 2
_TRAIN_RUNS_MAX = 30
_TRAIN_LOCK = threading.RLock()
_TRAIN_RUNS: dict[str, dict[str, Any]] = {}


# The templates intentionally use only the read-only public API.  They are
# returned to the editor and can also be saved as ordinary user scripts.
STARTER_TEMPLATES = [
    {
        "id": "rsi_vp",
        "name": "RSI + Volume Profile",
        "description": "Long/short confluence: RSI extremes near POC.",
        "default_params": {"rsi_period": 14, "atr_distance": 0.5},
        "code": '''# RSI + Volume Profile confluence
rsi_period = int(params.get("rsi_period", 14))
atr_distance = float(params.get("atr_distance", 0.5))
close = data["close"]
rsi_line = ta.rsi(close, rsi_period)
rsi_now = rsi_line.iloc[-1]
price = close.iloc[-1]
vp = snapshot.get("vp", {})
poc = vp.get("poc")
atr = snapshot.get("t", {}).get("atr")
signal = "NEUTRAL"
confidence = 0.0
if poc is not None and atr is not None and atr > 0:
    distance = abs(price - poc) / atr
    if rsi_now < 32 and distance <= atr_distance:
        signal = "LONG"
        confidence = min(0.95, 0.55 + (32 - rsi_now) / 100)
    elif rsi_now > 68 and distance <= atr_distance:
        signal = "SHORT"
        confidence = min(0.95, 0.55 + (rsi_now - 68) / 100)
preview = []
start = max(0, len(data) - 120)
for i in range(start, len(data)):
    preview.append({"time": data["timestamp"][i], "value": close[i], "signal": signal if i == len(data) - 1 else 0})
result = {"signal": signal, "confidence": confidence, "rsi": rsi_now,
          "poc": poc, "atr": atr, "preview": preview}
''',
    },
    {
        "id": "vp_funding_scanner",
        "name": "VP + Funding Scanner",
        "description": "Current asset matches price near POC and funding_zscore > 1.",
        "default_params": {"atr_distance": 0.5, "funding_zscore": 1.0},
        "code": '''# Scanner condition for the current asset
atr_distance = float(params.get("atr_distance", 0.5))
funding_limit = float(params.get("funding_zscore", 1.0))
price = data["close"].iloc[-1]
vp = snapshot.get("vp", {})
poc = vp.get("poc")
atr = snapshot.get("t", {}).get("atr")
funding_zscore = snapshot.get("d", {}).get("funding_zscore")
near_poc = poc is not None and atr is not None and atr > 0 and abs(price - poc) <= atr_distance * atr
funding_ok = funding_zscore is not None and funding_zscore > funding_limit
signal = "MATCH" if near_poc and funding_ok else "REJECT"
confidence = 0.9 if signal == "MATCH" else 0.0
preview = []
for i in range(max(0, len(data) - 120), len(data)):
    preview.append({"time": data["timestamp"][i], "value": data["close"][i], "signal": signal if i == len(data) - 1 else 0})
result = {"signal": signal, "confidence": confidence, "near_poc": near_poc,
          "funding_ok": funding_ok, "poc": poc, "atr": atr,
          "funding_zscore": funding_zscore, "preview": preview}
''',
    },
    {
        "id": "ma_crossover",
        "name": "MA Crossover (Trainable)",
        "description": "Fast/slow moving-average crossover with replay optimisation.",
        "default_params": {"fast": [5, 10], "slow": [21, 50]},
        "code": '''# Trainable moving-average crossover
fast = int(params.get("fast", 8))
slow = int(params.get("slow", 34))
close = data["close"]
fast_line = close.rolling(fast).mean()
slow_line = close.rolling(slow).mean()
fast_now = fast_line.iloc[-1]
slow_now = slow_line.iloc[-1]
fast_prev = fast_line.iloc[-2] if len(close) > 1 else None
slow_prev = slow_line.iloc[-2] if len(close) > 1 else None
signal = "FLAT"
if fast_now is not None and slow_now is not None and fast_prev is not None and slow_prev is not None:
    if fast_prev <= slow_prev and fast_now > slow_now:
        signal = "LONG"
    elif fast_prev >= slow_prev and fast_now < slow_now:
        signal = "SHORT"
confidence = 0.7 if signal != "FLAT" else 0.0
preview = []
for i in range(max(0, len(data) - 120), len(data)):
    preview.append({"time": data["timestamp"][i], "value": close[i], "signal": 1 if signal == "LONG" else -1 if signal == "SHORT" else 0})
result = {"signal": signal, "confidence": confidence, "fast": fast,
          "slow": slow, "fast_value": fast_now, "slow_value": slow_now,
          "preview": preview}
''',
    },
]


def _script_path(script_id: str) -> Path | None:
    if not _SCRIPT_ID_RE.fullmatch(script_id or ""):
        return None
    return _SCRIPT_DIR / f"{script_id}.json"


def _read_script(script_id: str) -> dict[str, Any] | None:
    path = _script_path(script_id)
    if path is None or not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError, TypeError) as exc:
        log.warning("cannot read script %s: %s", script_id, type(exc).__name__)
        return None


def _write_script(value: dict[str, Any]) -> None:
    script_id = value["id"]
    path = _script_path(script_id)
    if path is None:
        raise ValueError("invalid script id")
    _SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    utils._atomic_json_write(path, value)


def _clean_text(value: Any, limit: int, fallback: str = "") -> str:
    if not isinstance(value, str):
        return fallback
    return value.strip()[:limit]


def _clean_params(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    clean: dict[str, Any] = {}
    for raw_key, raw_value in list(value.items())[:20]:
        key = str(raw_key)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", key):
            continue
        if isinstance(raw_value, (list, tuple)):
            values = list(raw_value)[:20]
            if all(isinstance(item, (int, float)) and not isinstance(item, bool)
                   and math.isfinite(float(item)) for item in values):
                clean[key] = values
        elif isinstance(raw_value, (int, float, str, bool)):
            if not isinstance(raw_value, float) or math.isfinite(raw_value):
                clean[key] = raw_value
    return clean


def _levels_from_snapshot(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    levels = snapshot.get("tg")
    if isinstance(levels, list):
        return [item for item in levels if isinstance(item, (list, tuple, dict))][:100]
    vp = snapshot.get("vp")
    if not isinstance(vp, dict):
        return []
    out: list[dict[str, Any]] = []
    for key, probability in (("poc", 0.45), ("vah", 0.275), ("val", 0.275)):
        value = vp.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            out.append({"type": key.upper(), "price": float(value),
                        "probability": probability})
    return out


def _public_script(value: dict[str, Any], include_code: bool = False) -> dict[str, Any]:
    result = {
        "id": value.get("id"),
        "name": value.get("name") or "Untitled",
        "description": value.get("description") or "",
        "created_at": value.get("created_at"),
        "updated_at": value.get("updated_at"),
    }
    if include_code:
        result["code"] = value.get("code") or ""
        result["params"] = value.get("params") or {}
    return result


@bp.before_request
def _limit_script_request_body():
    length = request.content_length
    if length is not None and length > 256 * 1024:
        return jsonify({"error": "Request body is too large"}), 413



@bp.route("/script-editor")
def script_editor():
    return render_template("script_editor.html")


@bp.route("/api/scripts", methods=["GET"])
def list_scripts():
    items: list[dict[str, Any]] = []
    try:
        paths = sorted(_SCRIPT_DIR.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    except OSError:
        paths = []
    for path in paths:
        value = _read_script(path.stem)
        if value:
            items.append(_public_script(value))
    return jsonify({"scripts": items, "templates": STARTER_TEMPLATES})


@bp.route("/api/scripts/validate", methods=["POST"])
def validate_script():
    body = request.get_json(silent=True) or {}
    code = body.get("code") if isinstance(body, dict) else None
    ok, error = _ENGINE.validate(code)
    return jsonify({"valid": ok, "error": error})


@bp.route("/api/scripts", methods=["POST"])
def save_script():
    body = request.get_json(silent=True) or {}
    if not isinstance(body, dict):
        return jsonify({"error": "JSON object required"}), 400
    code = body.get("code")
    ok, error = _ENGINE.validate(code)
    if not ok:
        return jsonify({"error": error}), 400
    now = utils.now_iso()
    script_id = str(body.get("id") or uuid.uuid4().hex)
    if not _SCRIPT_ID_RE.fullmatch(script_id):
        return jsonify({"error": "Invalid script id"}), 400
    old = _read_script(script_id) or {}
    value = {
        "id": script_id,
        "name": _clean_text(body.get("name"), _MAX_SCRIPT_NAME, "Untitled") or "Untitled",
        "description": _clean_text(body.get("description"), _MAX_DESCRIPTION),
        "code": code,
        "params": _clean_params(body.get("params") or old.get("params") or {}),
        "created_at": old.get("created_at") or now,
        "updated_at": now,
    }
    try:
        _write_script(value)
    except (OSError, ValueError) as exc:
        log.exception("script save failed")
        return jsonify({"error": "Could not save script"}), 500
    return jsonify({"script": _public_script(value, include_code=True)}), 201


@bp.route("/api/scripts/<script_id>", methods=["GET"])
def get_script(script_id: str):
    value = _read_script(script_id)
    if value is None:
        return jsonify({"error": "Script not found"}), 404
    return jsonify({"script": _public_script(value, include_code=True)})


@bp.route("/api/scripts/<script_id>", methods=["PUT"])
def update_script(script_id: str):
    old = _read_script(script_id)
    if old is None:
        return jsonify({"error": "Script not found"}), 404
    body = request.get_json(silent=True) or {}
    if not isinstance(body, dict):
        return jsonify({"error": "JSON object required"}), 400
    code = body.get("code", old.get("code"))
    ok, error = _ENGINE.validate(code)
    if not ok:
        return jsonify({"error": error}), 400
    now = utils.now_iso()
    value = {
        **old,
        "name": _clean_text(body.get("name", old.get("name")), _MAX_SCRIPT_NAME, "Untitled") or "Untitled",
        "description": _clean_text(body.get("description", old.get("description")), _MAX_DESCRIPTION),
        "code": code,
        "params": _clean_params(body.get("params", old.get("params") or {})),
        "updated_at": now,
    }
    try:
        _write_script(value)
    except (OSError, ValueError):
        log.exception("script update failed")
        return jsonify({"error": "Could not update script"}), 500
    return jsonify({"script": _public_script(value, include_code=True)})


@bp.route("/api/scripts/<script_id>", methods=["DELETE"])
def delete_script(script_id: str):
    path = _script_path(script_id)
    if path is None or not path.is_file():
        return jsonify({"error": "Script not found"}), 404
    try:
        path.unlink()
    except OSError:
        log.exception("script delete failed")
        return jsonify({"error": "Could not delete script"}), 500
    return jsonify({"deleted": True, "id": script_id})


@bp.route("/api/scripts/<script_id>/run", methods=["POST"])
def run_script(script_id: str):
    value = _read_script(script_id)
    if value is None:
        return jsonify({"error": "Script not found"}), 404
    body = request.get_json(silent=True) or {}
    symbol = str(body.get("symbol") or "BTCUSDT").upper()
    timeframe = str(body.get("timeframe") or config.DEFAULT_TIMEFRAME)
    if symbol not in config.SYMBOLS or timeframe not in config.TF_SECONDS:
        return jsonify({"error": "Invalid symbol or timeframe"}), 400
    try:
        snapshot = compact_snapshot(symbol, timeframe, rules_only=True)
        frame = get_series_df(symbol, timeframe, limit=300, history_limit=300)
        result = _ENGINE.execute(
            value.get("code") or "",
            {"df": frame, "snapshot": snapshot,
             "levels": _levels_from_snapshot(snapshot),
             "params": value.get("params") or {}, "symbol": symbol,
             "timeframe": timeframe},
            timeout_sec=5,
        )
        return jsonify({"result": result, "logs": _ENGINE.last_logs})
    except SecurityError as exc:
        return jsonify({"error": str(exc)}), 400
    except ScriptTimeoutError as exc:
        return jsonify({"error": str(exc)}), 408
    except ScriptExecutionError as exc:
        return jsonify({"error": str(exc)}), 422
    except Exception:
        log.exception("script run failed")
        return jsonify({"error": "Script execution failed"}), 500


def _prune_train_runs() -> None:
    if len(_TRAIN_RUNS) <= _TRAIN_RUNS_MAX:
        return
    oldest = sorted(_TRAIN_RUNS, key=lambda key: _TRAIN_RUNS[key].get("created_at", 0))
    for run_id in oldest[:len(_TRAIN_RUNS) - _TRAIN_RUNS_MAX]:
        _TRAIN_RUNS.pop(run_id, None)


def _train_worker(run_id: str, code: str, symbol: str, timeframe: str,
                  start_sec: int, end_sec: int, params: dict[str, Any]) -> None:
    def progress(data: dict[str, Any]) -> None:
        with _TRAIN_LOCK:
            if run_id in _TRAIN_RUNS:
                _TRAIN_RUNS[run_id]["progress"] = data
        _ws_push("script_train_progress", {"run_id": run_id, **data})

    with _TRAIN_LOCK:
        if run_id in _TRAIN_RUNS:
            _TRAIN_RUNS[run_id]["status"] = "running"
    try:
        result = ReplayTrainer().train(
            code, symbol, timeframe, start_sec, end_sec, params, progress=progress
        )
        with _TRAIN_LOCK:
            _TRAIN_RUNS[run_id].update({"status": "done", "result": result,
                                        "finished_at": time.time()})
        _ws_push("script_train_progress", {"run_id": run_id, "done": True,
                                           "status": "done"})
    except Exception as exc:  # keep API responsive and avoid leaking traceback
        log.exception("script training failed")
        with _TRAIN_LOCK:
            _TRAIN_RUNS[run_id].update({"status": "error", "error": type(exc).__name__,
                                        "finished_at": time.time()})
        _ws_push("script_train_progress", {"run_id": run_id, "done": True,
                                           "status": "error"})


@bp.route("/api/scripts/<script_id>/train", methods=["POST"])
def train_script(script_id: str):
    value = _read_script(script_id)
    if value is None:
        return jsonify({"error": "Script not found"}), 404
    body = request.get_json(silent=True) or {}
    if not isinstance(body, dict):
        return jsonify({"error": "JSON object required"}), 400
    symbol = str(body.get("symbol") or "BTCUSDT").upper()
    timeframe = str(body.get("timeframe") or config.DEFAULT_TIMEFRAME)
    try:
        start_sec = int(body["start_sec"])
        end_sec = int(body["end_sec"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "start_sec and end_sec are required integers"}), 400
    if symbol not in config.SYMBOLS or timeframe not in config.TF_SECONDS:
        return jsonify({"error": "Invalid symbol or timeframe"}), 400
    if start_sec <= 0 or end_sec <= start_sec or end_sec - start_sec > _MAX_TRAIN_RANGE_SEC:
        return jsonify({"error": "Invalid replay range"}), 400
    ok, error = _ENGINE.validate(value.get("code") or "")
    if not ok:
        return jsonify({"error": error}), 400
    try:
        ReplayTrainer._expand_params(body.get("params") or value.get("params") or {})
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    with _TRAIN_LOCK:
        active = sum(1 for item in _TRAIN_RUNS.values()
                     if item.get("status") in {"queued", "running"})
        if active >= _MAX_ACTIVE_TRAININGS:
            return jsonify({"error": "Too many active training jobs"}), 429
        run_id = uuid.uuid4().hex
        _TRAIN_RUNS[run_id] = {
            "run_id": run_id, "script_id": script_id, "status": "queued",
            "created_at": time.time(), "progress": {"done": 0, "total": 0},
        }
        _prune_train_runs()
    thread = threading.Thread(
        target=_train_worker,
        args=(run_id, value.get("code") or "", symbol, timeframe, start_sec,
              end_sec, _clean_params(body.get("params") or value.get("params") or {})),
        name=f"script-train-{script_id[:8]}", daemon=True,
    )
    thread.start()
    return jsonify({"run_id": run_id, "status": "queued", "script_id": script_id}), 202


@bp.route("/api/scripts/train/<run_id>", methods=["GET"])
def train_status(run_id: str):
    with _TRAIN_LOCK:
        value = _TRAIN_RUNS.get(run_id)
        if value is None:
            return jsonify({"error": "Training run not found"}), 404
        return jsonify(dict(value))
