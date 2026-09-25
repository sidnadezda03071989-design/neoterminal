"""Secure execution boundary for user supplied indicator/strategy scripts.

The public :class:`ScriptEngine` API deliberately accepts only market context.
It validates source with an AST policy, then sends the source and a *prepared*
context to a disposable worker process.  RestrictedPython is useful for
restricting Python syntax, but its own documentation explicitly says that it
is not a complete sandbox; process isolation and the allow-listed facade in
``app_pkg.scripting.api`` are therefore part of the security boundary.
"""

from __future__ import annotations

import ast
import multiprocessing as _mp
import re
import threading
import time
import warnings
from collections.abc import Mapping
from typing import Any

from RestrictedPython import compile_restricted
from RestrictedPython.Guards import (
    full_write_guard,
    guarded_iter_unpack_sequence,
    guarded_unpack_sequence,
    safer_getattr,
)
from RestrictedPython.transformer import RestrictingNodeTransformer

# A child process is deliberately conservative.  These limits are additional
# protection around the language guard, not a replacement for it.
_CHILD_MEMORY_BYTES = 512 * 1024 * 1024
_BATCH_MEMORY_BYTES = 1024 * 1024 * 1024
_MAX_OUTPUT_NODES = 10_000
_MAX_SOURCE_BYTES = 64 * 1024
_MAX_AST_NODES = 2_500


class SecurityError(RuntimeError):
    """Raised when source or a requested operation violates the sandbox policy."""


class ScriptExecutionError(RuntimeError):
    """Raised for an error raised by an otherwise validated user script."""


class ScriptTimeoutError(ScriptExecutionError):
    """Raised when the worker exceeds the requested wall-clock deadline."""


class _PrintCollector:
    """Small object consumed by RestrictedPython's transformed ``print``."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def _call_print(self, *values: Any, **_kwargs: Any) -> None:
        if len(self.lines) >= 100:
            return
        text = " ".join(str(value) for value in values)
        self.lines.append(text[:2_000])

    def __call__(self, *values: Any, **_kwargs: Any) -> "_PrintCollector":
        # RestrictedPython's print transform may leave the collector itself as
        # the callable in some AST shapes.  Returning ``self`` makes the
        # generated ``_print._call_print`` path safe and keeps output local.
        if values and values[0] is _guarded_getattr:
            return self
        self._call_print(*values, **_kwargs)
        return self

class _RestrictedScriptPolicy(RestrictingNodeTransformer):
    """RestrictedPython policy plus an explicit import/call allow-list.

    ``ast.Call`` is not discarded: every call is rewritten to
    ``_guarded_call_``.  This is important because ordinary calls are not
    guarded by RestrictedPython by default.
    """

    def _import_call(self, module: str, node: ast.AST) -> ast.Call:
        call = ast.Call(
            func=ast.Name(id="_nt_import", ctx=ast.Load()),
            args=[ast.Constant(value=module)],
            keywords=[],
        )
        return ast.fix_missing_locations(ast.copy_location(call, node))

    def _import_error(self, node: ast.AST, message: str) -> None:
        self.error(node, message)

    def visit_Import(self, node: ast.Import) -> Any:
        assignments: list[ast.stmt] = []
        for alias in node.names:
            if alias.name not in ScriptEngine.ALLOWED_MODULES:
                self._import_error(node, f"Import '{alias.name}' is not allowed")
                continue
            bound_name = alias.asname or alias.name.split(".")[0]
            if bound_name.startswith("_") or bound_name in {"eval", "exec", "open"}:
                self._import_error(node, f"Import alias '{bound_name}' is not allowed")
                continue
            assignments.append(
                ast.fix_missing_locations(ast.copy_location(
                    ast.Assign(
                        targets=[ast.Name(id=bound_name, ctx=ast.Store())],
                        value=self._import_call(alias.name, node),
                        type_comment=None,
                    ),
                    node,
                ))
            )
        return assignments or ast.copy_location(ast.Pass(), node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        if node.level or node.module not in ScriptEngine.ALLOWED_MODULES:
            self._import_error(node, f"Import from '{node.module}' is not allowed")
            return ast.copy_location(ast.Pass(), node)
        assignments: list[ast.stmt] = []
        for alias in node.names:
            if alias.name == "*" or alias.name.startswith("_"):
                self._import_error(node, "Wildcard/private imports are not allowed")
                continue
            bound_name = alias.asname or alias.name
            if bound_name.startswith("_") or bound_name in {"eval", "exec", "open"}:
                self._import_error(node, f"Import alias '{bound_name}' is not allowed")
                continue
            value = ast.fix_missing_locations(ast.copy_location(ast.Call(
                func=ast.Name(id="_getattr_", ctx=ast.Load()),
                args=[
                    self._import_call(node.module or "", node),
                    ast.Constant(value=alias.name),
                ],
                keywords=[],
            ), node))
            assignments.append(
                ast.fix_missing_locations(ast.copy_location(
                    ast.Assign(
                        targets=[ast.Name(id=bound_name, ctx=ast.Store())],
                        value=value,
                        type_comment=None,
                    ),
                    node,
                ))
            )
        return assignments or ast.copy_location(ast.Pass(), node)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        # Visit children first so RestrictedPython's attribute/print rewrites
        # happen before the call is placed behind our guard.
        node = self.node_contents_visit(node)
        guarded = ast.Call(
            func=ast.Name(id="_guarded_call_", ctx=ast.Load()),
            args=[node.func, *node.args],
            keywords=node.keywords,
        )
        return ast.fix_missing_locations(ast.copy_location(guarded, node))

    def visit_Global(self, node: ast.Global) -> Any:
        self.error(node, "global declarations are not allowed")
        return node

    def visit_Nonlocal(self, node: ast.Nonlocal) -> Any:
        self.error(node, "nonlocal declarations are not allowed")
        return node


class ScriptEngine:
    """Validate and execute scripts in a restricted worker process.

    Imports are mapped to small in-process facades; no Python import statement
    is evaluated by the worker.  Consequently ``import os`` is rejected by
    validation and there is no ``__import__`` in script globals.
    """

    ALLOWED_MODULES = {"math", "numpy", "pandas", "ta", "scipy.stats"}
    # Kept as a public compatibility constant.  Calls are not rejected
    # wholesale: the policy rewrites them to ``_guarded_call_`` and the runtime
    # guard checks the callable.  Imports are handled separately above.
    BANNED_NODES = {ast.Import, ast.ImportFrom, ast.Call}
    BANNED_NAMES = {
        "__import__", "anext", "aiter", "breakpoint", "classmethod", "compile",
        "delattr", "dir", "eval", "exec", "exit", "getattr", "globals", "help",
        "input", "locals", "memoryview", "object", "open", "property", "quit",
        "setattr", "staticmethod", "super", "type", "vars", "hasattr",
    }
    BANNED_ATTRIBUTES = {
        "__builtins__", "__class__", "__closure__", "__code__", "__dict__",
        "__globals__", "__mro__", "__subclasses__", "f_back", "f_builtins",
        "f_code", "f_globals", "f_locals", "func_globals", "gi_code",
        "gi_frame", "im_class", "im_func", "im_self", "tb_frame",
        "ag_frame", "cr_frame", "mro",
    }

    _SAFE_BUILTIN_NAMES = {
        "abs", "all", "any", "bool", "chr", "dict", "divmod", "enumerate",
        "float", "int", "len", "list", "max", "min", "ord", "pow", "range",
        "reversed", "round", "set", "sorted", "str", "sum", "tuple", "zip",
    }

    def __init__(self) -> None:
        self._ctx = _mp.get_context("spawn")
        self._lock = threading.RLock()
        self._process: Any = None
        self._connection: Any = None
        self._validation_cache: dict[str, tuple[bool, str]] = {}
        self.last_logs: list[str] = []

    @classmethod
    def _validate_import_node(cls, node: ast.Import | ast.ImportFrom) -> str | None:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in cls.ALLOWED_MODULES:
                    return f"Import '{alias.name}' is not allowed"
            return None
        if node.level or node.module not in cls.ALLOWED_MODULES:
            return f"Import from '{node.module}' is not allowed"
        for alias in node.names:
            if alias.name == "*" or alias.name.startswith("_"):
                return "Wildcard/private imports are not allowed"
        return None


    def validate(self, code: str) -> tuple[bool, str]:
        """Perform AST and RestrictedPython validation without executing code."""
        if not isinstance(code, str):
            return False, "Code must be a string"
        cached = self._validation_cache.get(code)
        if cached is not None:
            return cached
        if len(self._validation_cache) >= 256:
            self._validation_cache.clear()
        if not code.strip():
            return False, "Code is empty"
        if len(code.encode("utf-8")) > _MAX_SOURCE_BYTES:
            result = (False, f"Code is too large (max {_MAX_SOURCE_BYTES} bytes)")
            self._validation_cache[code] = result
            return result
        try:
            tree = ast.parse(code, filename="<user_script>", mode="exec")
        except (SyntaxError, ValueError, TypeError, RecursionError, MemoryError) as exc:
            result = (False, f"Syntax error: {exc}")
            self._validation_cache[code] = result
            return result

        if sum(1 for _ in ast.walk(tree)) > _MAX_AST_NODES:
            result = (False, f"AST is too complex (max {_MAX_AST_NODES} nodes)")
            self._validation_cache[code] = result
            return result
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                error = self._validate_import_node(node)
                if error:
                    result = (False, error)
                    self._validation_cache[code] = result
                    return result
            elif isinstance(node, ast.Name) and node.id in self.BANNED_NAMES:
                result = (False, f"Name '{node.id}' is banned")
                self._validation_cache[code] = result
                return result
            elif isinstance(node, ast.Attribute):
                if node.attr.startswith("__") or node.attr in self.BANNED_ATTRIBUTES:
                    result = (False, f"Attribute '{node.attr}' is banned")
                    self._validation_cache[code] = result
                    return result
            elif isinstance(node, (ast.Global, ast.Nonlocal, ast.Await,
                                   ast.AsyncFor, ast.AsyncFunctionDef,
                                   ast.AsyncWith, ast.Yield, ast.YieldFrom)):
                result = (False, f"'{type(node).__name__}' is not allowed")
                self._validation_cache[code] = result
                return result
            elif isinstance(node, ast.Constant) and isinstance(node.value, bytes):
                result = (False, "bytes literals are not allowed")
                self._validation_cache[code] = result
                return result

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                compile_restricted(
                    code,
                    filename="<user_script>",
                    mode="exec",
                    policy=_RestrictedScriptPolicy,
                )
        except (SyntaxError, TypeError, ValueError, RecursionError, MemoryError) as exc:
            result = (False, f"RestrictedPython: {exc}")
            self._validation_cache[code] = result
            return result
        result = (True, "")
        self._validation_cache[code] = result
        return result

    def _ensure_worker(self) -> None:
        if self._process is not None and self._process.is_alive() and self._connection is not None:
            return
        self._stop_worker()
        parent_conn, child_conn = self._ctx.Pipe(duplex=True)
        process = self._ctx.Process(
            target=_worker_main,
            args=(child_conn,),
            name="nt-script-worker",
            daemon=True,
        )
        try:
            process.start()
        except OSError as exc:
            parent_conn.close()
            child_conn.close()
            raise ScriptExecutionError("Script worker failed to start") from exc
        child_conn.close()
        self._process = process
        self._connection = parent_conn
        ready_deadline = time.monotonic() + 10.0
        while time.monotonic() < ready_deadline:
            if self._connection.poll(0.05):
                packet = self._connection.recv()
                if packet.get("type") == "ready":
                    return
                if packet.get("type") == "startup_error":
                    self._stop_worker()
                    raise ScriptExecutionError("Script worker failed to start")
                self._stop_worker()
                raise ScriptExecutionError("Invalid worker handshake")
            if not process.is_alive():
                self._stop_worker()
                raise ScriptExecutionError("Script worker stopped during startup")
        self._stop_worker()
        raise ScriptExecutionError("Script worker startup timed out")

    def _stop_worker(self) -> None:
        process, connection = self._process, self._connection
        self._process = None
        self._connection = None
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass
        if process is not None and process.is_alive():
            _terminate_process_tree(process)
        if process is not None:
            try:
                process.join(timeout=0.5)
            except (OSError, RuntimeError):
                pass

    def execute(self, code: str, context: dict, timeout_sec: int = 5) -> dict:
        """Execute ``code`` and return its JSON-compatible ``result`` mapping.

        The parent waits for a response from a separate process.  A timeout
        terminates the whole worker tree; it is not implemented with a thread
        that could leave an infinite loop running in Flask.
        """
        ok, error = self.validate(code)
        if not ok:
            raise SecurityError(error)
        try:
            timeout = max(0.1, min(float(timeout_sec), 5.0))
        except (TypeError, ValueError) as exc:
            raise SecurityError("timeout_sec must be numeric") from exc

        from app_pkg.scripting.api import normalise_context

        prepared = normalise_context(context or {})
        with self._lock:
            try:
                self._ensure_worker()
            except ScriptExecutionError:
                # Windows can transiently refuse a spawn immediately after a
                # previous worker was reaped. One fresh retry is safe because
                # no user code ran in the failed worker.
                time.sleep(0.1)
                self._ensure_worker()
            assert self._connection is not None
            try:
                self._connection.send({"type": "execute", "code": code, "context": prepared})
            except (BrokenPipeError, EOFError, OSError) as exc:
                self._stop_worker()
                raise ScriptExecutionError("Script worker is unavailable") from exc

            # Worker startup is deliberately outside the script deadline.  A
            # timeout must bound user code, not the one-time Windows spawn cost.
            started = time.monotonic()
            deadline = started + timeout
            packet: dict[str, Any] | None = None
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._stop_worker()
                    raise ScriptTimeoutError(f"Script exceeded {timeout:.2f}s timeout")
                try:
                    ready = self._connection.poll(min(0.05, remaining))
                except (EOFError, OSError):
                    ready = False
                if ready:
                    try:
                        packet = self._connection.recv()
                    except (EOFError, OSError) as exc:
                        self._stop_worker()
                        raise ScriptExecutionError("Script worker stopped") from exc
                    break
                if self._process is None or not self._process.is_alive():
                    self._stop_worker()
                    raise ScriptExecutionError("Script worker stopped unexpectedly")
                if _worker_memory_bytes(self._process.pid) > _CHILD_MEMORY_BYTES:
                    self._stop_worker()
                    raise ScriptExecutionError("Script exceeded the memory limit")

        if not packet:
            raise ScriptExecutionError("Script returned no response")
        if not packet.get("ok"):
            message = str(packet.get("error") or "Script execution failed")
            if packet.get("kind") == "security":
                raise SecurityError(message)
            raise ScriptExecutionError(message)
        self.last_logs = list(packet.get("logs") or [])
        result = packet.get("result")
        return result if isinstance(result, dict) else {"value": result}
    def execute_series(
        self,
        code: str,
        frame: Any,
        bar_times: list[int],
        snapshots: list[dict[str, Any]],
        params: dict[str, Any],
        *,
        symbol: str = "",
        timeframe: str = "",
        lookback: int = 300,
        timeout_sec: float = 25.0,
    ) -> list[dict[str, Any]]:
        """Execute a replay series while exposing only the current window."""
        ok, error = self.validate(code)
        if not ok:
            raise SecurityError(error)
        from app_pkg.scripting.api import _epoch_seconds, normalise_context

        prepared = normalise_context({"df": frame, "params": params}, max_rows=2_000)
        prepared_frame = prepared["df"].copy()
        if "timestamp" in prepared_frame.columns:
            prepared_frame["timestamp"] = _epoch_seconds(prepared_frame["timestamp"])
        prepared_frame.attrs["_nt_prepared_window"] = True
        clean_params = normalise_context(
            {"df": prepared_frame.head(0), "params": params}, max_rows=1,
        )["params"]
        count = len(prepared_frame)
        if count < 1 or len(bar_times) != count or len(snapshots) != count:
            raise SecurityError("Invalid replay batch")
        if count > 2_000:
            raise SecurityError("Replay batch is too large")
        request = {
            "type": "execute_series", "code": code, "frame": prepared_frame,
            "bar_times": [int(value) for value in bar_times],
            "snapshots": list(snapshots), "params": clean_params,
            "symbol": str(symbol)[:32], "timeframe": str(timeframe)[:16],
            "lookback": max(1, min(int(lookback), 300)),
        }
        timeout = max(0.5, min(float(timeout_sec), 30.0))
        with self._lock:
            try:
                self._ensure_worker()
            except ScriptExecutionError:
                time.sleep(0.1)
                self._ensure_worker()
            assert self._connection is not None
            try:
                self._connection.send(request)
            except (BrokenPipeError, EOFError, OSError) as exc:
                self._stop_worker()
                raise ScriptExecutionError("Script worker is unavailable") from exc
            deadline = time.monotonic() + timeout
            packet: dict[str, Any] | None = None
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._stop_worker()
                    raise ScriptTimeoutError(f"Replay batch exceeded {timeout:.2f}s timeout")
                if self._connection.poll(min(0.05, remaining)):
                    packet = self._connection.recv()
                    break
                if self._process is None or not self._process.is_alive():
                    self._stop_worker()
                    raise ScriptExecutionError("Script worker stopped unexpectedly")
                if _worker_memory_bytes(self._process.pid) > _BATCH_MEMORY_BYTES:
                    self._stop_worker()
                    raise ScriptExecutionError("Replay exceeded the memory limit")
        if not packet:
            raise ScriptExecutionError("Script returned no response")
        if not packet.get("ok"):
            message = str(packet.get("error") or "Replay execution failed")
            if packet.get("kind") == "security":
                raise SecurityError(message)
            raise ScriptExecutionError(message)
        items = packet.get("items")
        if not isinstance(items, list) or len(items) != count:
            raise ScriptExecutionError("Invalid replay batch response")
        logs: list[str] = []
        for item in items:
            if isinstance(item, Mapping):
                logs.extend(str(line) for line in item.get("logs") or [])
        self.last_logs = logs[:100]
        return [dict(item) if isinstance(item, Mapping) else {} for item in items]



    def close(self) -> None:
        with self._lock:
            self._stop_worker()

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown path
        try:
            self.close()
        except Exception:
            pass


def _worker_memory_bytes(pid: int | None) -> int:
    if not pid:
        return 0
    try:
        import psutil

        return int(psutil.Process(pid).memory_info().rss)
    except Exception:
        return 0


def _terminate_process_tree(process: Any) -> None:
    """Terminate the worker and descendants without leaving a child behind."""
    try:
        import psutil

        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.terminate()
            except (psutil.Error, OSError):
                pass
        try:
            parent.terminate()
        except (psutil.Error, OSError):
            pass
        gone, alive = psutil.wait_procs(children + [parent], timeout=0.25)
        for item in alive:
            try:
                item.kill()
            except (psutil.Error, OSError):
                pass
    except Exception:
        try:
            process.terminate()
        except (OSError, AttributeError):
            pass
    try:
        process.join(timeout=0.5)
    except (OSError, RuntimeError):
        pass


def _set_child_limits() -> None:
    """Best-effort OS limits; Windows lacks the POSIX resource interface."""
    try:
        import resource

        # BLAS runtimes may reserve a large virtual address space; enforce
        # resident memory in the parent instead of imposing RLIMIT_AS here.
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    except (ImportError, OSError, ValueError):
        pass


def _safe_builtins() -> dict[str, Any]:
    import builtins

    return {
        name: getattr(builtins, name)
        for name in ScriptEngine._SAFE_BUILTIN_NAMES
        if hasattr(builtins, name)
    } | {"True": True, "False": False, "None": None}


def _make_globals(context: dict[str, Any], print_collector: _PrintCollector) -> dict[str, Any]:
    from app_pkg.scripting.api import (
        get_safe_module,
        make_safe_frame,
        make_safe_levels,
        make_safe_mapping,
        make_safe_params,
    )

    safe_globals: dict[str, Any] = {
        "__builtins__": _safe_builtins(),
        "__name__": "<user_script>",
        "_getattr_": _guarded_getattr,
        "_getitem_": _guarded_getitem,
        "_getiter_": _guarded_getiter,
        "_write_": full_write_guard,
        "_unpack_sequence_": guarded_unpack_sequence,
        "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
        "_nt_import": get_safe_module,
        "_print_": print_collector,
    }
    # Install the closure after the dictionary exists: the guard needs the
    # identity of this exact globals mapping to admit user-defined functions.
    safe_globals["_guarded_call_"] = _make_guarded_call(safe_globals)
    safe_frame = make_safe_frame(context.get("df"))
    safe_params = make_safe_params(context.get("params") or {})
    safe_globals.update({
        "data": safe_frame,
        "df": safe_frame,
        "snapshot": make_safe_mapping(context.get("snapshot") or {}),
        "levels": make_safe_levels(context.get("levels") or []),
        "params": safe_params,
        "symbol": str(context.get("symbol") or ""),
        "timeframe": str(context.get("timeframe") or ""),
        "bar_time": context.get("bar_time"),
    })
    for key, value in safe_params.items():
        safe_globals[key] = value
    for module_name in ("math", "numpy", "pandas", "ta", "stats"):
        safe_globals[module_name] = get_safe_module(module_name)
    safe_globals["np"] = get_safe_module("numpy")
    safe_globals["pd"] = get_safe_module("pandas")
    return safe_globals


def _guarded_getattr(obj: Any, name: str, default: Any = None) -> Any:
    # RestrictedPython's generated call for print may pass ``None`` as the
    # default of ``_getattr_``.  The collector itself is callable, so accepting
    # the object here is safe and keeps print output in the worker.
    if isinstance(obj, _PrintCollector):
        if name == "_call_print":
            return obj._call_print
        if name == "__call__":
            return obj
    return safer_getattr(obj, name, default)


def _guarded_getitem(obj: Any, key: Any) -> Any:
    # All objects reachable from the public context implement a deliberately
    # small read-only protocol.  Primitive containers are safe to index too.
    return obj[key]


def _guarded_getiter(obj: Any):
    return iter(obj)


def _make_guarded_call(safe_globals: dict[str, Any]):
    import types

    builtin_values = tuple(_safe_builtins().values())

    def guarded_call(func: Any, *args: Any, **kwargs: Any) -> Any:
        owner = getattr(func, "__self__", None)
        safe_builtin_method = (
            type(owner) in {list, dict, set, str, tuple}
            and getattr(func, "__name__", "") in {
                "append", "extend", "insert", "pop", "remove", "clear", "reverse",
                "sort", "get", "keys", "values", "items", "update", "setdefault",
                "copy", "count", "index", "join", "upper", "lower", "strip",
                "replace", "split", "startswith", "endswith", "find", "add",
                "discard", "union", "intersection", "difference",
            }
        )
        allowed = (
            safe_builtin_method
            or func in builtin_values
            or (isinstance(func, types.FunctionType)
                and getattr(func, "__globals__", None) is safe_globals)
            or getattr(func, "__module__", "") in {
                "app_pkg.scripting.api", "app_pkg.scripting.engine",
            }
            or getattr(type(func), "__module__", "") in {
                "app_pkg.scripting.api", "app_pkg.scripting.engine",
            }
            or getattr(getattr(func, "__self__", None), "__class__", type(None)).__module__
            in {"app_pkg.scripting.api", "app_pkg.scripting.engine"}
        )
        if not allowed:
            raise SecurityError("Callable is not allowed by the script sandbox")
        return func(*args, **kwargs)

    return guarded_call


def _execute_bytecode(
    bytecode: Any, context: dict[str, Any], collector: _PrintCollector,
) -> dict[str, Any]:
    """Execute one prepared request and release cyclic references eagerly."""
    from app_pkg.scripting.api import make_json_safe

    safe_globals = _make_globals(context, collector)
    local_vars: dict[str, Any] = {}
    try:
        exec(bytecode, safe_globals, local_vars)  # noqa: S102 - isolated worker only
        return {
            "ok": True,
            "result": make_json_safe(local_vars.get("result", {})),
            "logs": collector.lines,
        }
    finally:
        local_vars.clear()
        safe_globals.clear()


def _execute_series_batch(bytecode: Any, packet: Mapping[str, Any]) -> dict[str, Any]:
    """Run a pre-validated code object against successive causal windows."""
    frame = packet.get("frame")
    bar_times = packet.get("bar_times")
    snapshots = packet.get("snapshots")
    params = packet.get("params") or {}
    lookback = max(1, min(int(packet.get("lookback") or 300), 300))
    if frame is None or not isinstance(bar_times, list) or not isinstance(snapshots, list):
        raise SecurityError("Invalid replay batch")
    if len(frame) != len(bar_times) or len(frame) != len(snapshots):
        raise SecurityError("Invalid replay batch length")
    items: list[dict[str, Any]] = []
    for index in range(len(frame)):
        collector = _PrintCollector()
        snapshot = snapshots[index] if isinstance(snapshots[index], Mapping) else {}
        window = frame.iloc[max(0, index - lookback + 1):index + 1]
        window.attrs["_nt_prepared_window"] = True
        context = {
            "df": window,
            "snapshot": snapshot,
            "levels": snapshot.get("tg", []),
            "params": params,
            "symbol": packet.get("symbol") or "",
            "timeframe": packet.get("timeframe") or "",
            "bar_time": bar_times[index],
        }
        try:
            item = _execute_bytecode(bytecode, context, collector)
            full_result = item.get("result")
            if isinstance(full_result, Mapping):
                item["result"] = {
                    key: full_result[key]
                    for key in ("signal", "confidence")
                    if key in full_result
                }
            item["logs"] = []
            items.append(item)
        except SecurityError as exc:
            items.append({"ok": False, "kind": "security", "error": str(exc)})
        except BaseException as exc:
            message = f"{type(exc).__name__}: {str(exc)[:200]}"
            message = re.sub(r"(?:[A-Za-z]:[\\/]|/)[^\s'\"]+", "<path>", message)
            items.append({"ok": False, "kind": "runtime", "error": message})
    return {"ok": True, "items": items}



def _worker_main(connection: Any) -> None:
    """Entrypoint executed in the isolated process."""
    _set_child_limits()
    try:
        # Warm heavy imports before the first script deadline.  The parent waits
        # for this handshake, so pandas/numpy startup is not charged to user
        # code or to a 2-second replay bar.
        try:
            from app_pkg.scripting.api import make_json_safe
            connection.send({"type": "ready"})
        except BaseException as exc:
            connection.send({"type": "startup_error", "error": type(exc).__name__})
            return

        cached_source: str | None = None
        cached_bytecode: Any = None
        while True:
            try:
                packet = connection.recv()
            except (EOFError, OSError):
                break
            request_type = packet.get("type") if isinstance(packet, Mapping) else None
            if request_type not in {"execute", "execute_series"}:
                connection.send({"ok": False, "kind": "security", "error": "Invalid worker request"})
                continue
            try:
                code = packet.get("code")
                if not isinstance(code, str):
                    raise SecurityError("Invalid script payload")
                if cached_source == code and cached_bytecode is not None:
                    bytecode = cached_bytecode
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        bytecode = compile_restricted(
                            code, filename="<user_script>", mode="exec",
                            policy=_RestrictedScriptPolicy,
                        )
                    cached_source = code
                    cached_bytecode = bytecode
                if request_type == "execute":
                    context = packet.get("context") or {}
                    if not isinstance(context, dict):
                        raise SecurityError("Invalid script context")
                    connection.send(_execute_bytecode(bytecode, context, _PrintCollector()))
                else:
                    connection.send(_execute_series_batch(bytecode, packet))
            except SecurityError as exc:
                connection.send({"ok": False, "kind": "security", "error": str(exc)})
            except BaseException as exc:
                message = f"{type(exc).__name__}: {str(exc)[:300]}"
                message = re.sub(r"(?:[A-Za-z]:[\\/]|/)[^\s'\"]+", "<path>", message)
                connection.send({"ok": False, "kind": "runtime", "error": message})

    finally:
        try:
            connection.close()
        except OSError:
            pass
