"""Тесты внешней обёртки Cline CLI (без запуска самого cline).

Проверяем: сборку argv (`--json --yolo -P -m -t`), резолвинг исполняемого
файла, контракт ответа {success, output, error, exit_code}, обработку таймаута
с убийством дерева процессов, отсутствие cline, чтение задачи из stdin
и разбор NDJSON из `--json`.
"""

import io
import json
import subprocess
import sys
from typing import ClassVar

import pytest

from cline_bridge import (
    DEFAULT_TIMEOUT,
    TIMEOUT_MARGIN,
    _build_argv,
    _kill_tree,
    _looks_like_crash,
    _resolve_cline_command,
    main,
    parse_events,
    run_cline_task,
)


class _FakePopen:
    """Заглушка subprocess.Popen: отдаёт заранее заданный итог запуска.

    script — список прогонов {"rc","out","err","raise_timeout"}: i-й запуск
    берёт свой итог. Нужно для проверки повторов после аварийного старта.
    """

    rc = 0
    stdout = ""
    stderr = ""
    raise_timeout = False
    script: ClassVar[list] = []
    instances: ClassVar[list] = []

    def __init__(self, argv, **kwargs):
        index = len(_FakePopen.instances)
        self.argv = list(argv)
        self.kwargs = kwargs
        self.pid = 4242
        spec = _FakePopen.script[index] if index < len(_FakePopen.script) else {}
        self.rc = spec.get("rc", _FakePopen.rc)
        self.out = spec.get("out", _FakePopen.stdout)
        self.err = spec.get("err", _FakePopen.stderr)
        self.raise_timeout = spec.get("raise_timeout", _FakePopen.raise_timeout)
        self.returncode = self.rc
        self.communicate_calls = 0
        _FakePopen.instances.append(self)

    def communicate(self, timeout=None):
        self.communicate_calls += 1
        if self.raise_timeout and self.communicate_calls == 1:
            raise subprocess.TimeoutExpired(cmd=self.argv, timeout=timeout)
        return self.out, self.err

    def poll(self):
        return self.returncode

    def kill(self):
        self.returncode = -9


@pytest.fixture(autouse=True)
def _reset_fake():
    _FakePopen.rc = 0
    _FakePopen.stdout = ""
    _FakePopen.stderr = ""
    _FakePopen.raise_timeout = False
    _FakePopen.script = []
    _FakePopen.instances = []
    yield


@pytest.fixture(autouse=True)
def _fake_cline(monkeypatch):
    """Подменяем и резолвинг бинаря, и Popen — cline реально не запускается."""
    monkeypatch.setattr("cline_bridge._resolve_cline_command", lambda: ["cline"])
    monkeypatch.setattr("cline_bridge.subprocess.Popen", _FakePopen)
    monkeypatch.setattr("cline_bridge.os.getcwd", lambda: r"C:\neoterminal")


# --- сборка argv ---------------------------------------------------------

def test_build_argv_has_yolo_json_and_task_last():
    argv = _build_argv("сделай X", 600, "openai-compatible", "qwen3.8-flash",
                       None, None, True)

    assert argv[0] == "cline"
    assert "--json" in argv
    assert "--yolo" in argv
    assert argv[-1] == "сделай X"
    assert argv[argv.index("-P") + 1] == "openai-compatible"
    assert argv[argv.index("-m") + 1] == "qwen3.8-flash"


def test_build_argv_cli_timeout_is_below_subprocess_timeout():
    argv = _build_argv("t", 600, None, None, None, None, True)

    assert argv[argv.index("-t") + 1] == str(600 - TIMEOUT_MARGIN)


def test_build_argv_omits_empty_optionals():
    argv = _build_argv("t", 60, None, None, None, None, False)

    assert "--yolo" not in argv
    assert "-P" not in argv
    assert "-m" not in argv
    assert "-c" not in argv
    assert argv[-1] == "t"


def test_build_argv_passes_cwd_thinking():
    argv = _build_argv("t", 90, "p", "m", "high", r"C:\Projects", True)

    assert argv[argv.index("-c") + 1] == r"C:\Projects"
    assert argv[argv.index("--thinking") + 1] == "high"


# --- резолвинг исполняемого файла ---------------------------------------

def test_resolve_prefers_env_override(monkeypatch):
    monkeypatch.setenv("CLINE_BIN", r"C:\custom\cline.cmd")
    monkeypatch.setattr("cline_bridge.shutil.which", lambda _n: r"C:\other\cline.CMD")

    assert _resolve_cline_command() == [r"C:\custom\cline.cmd"]


def test_resolve_uses_shutil_which(monkeypatch):
    monkeypatch.delenv("CLINE_BIN", raising=False)
    monkeypatch.setattr("cline_bridge.shutil.which",
                        lambda _n: r"C:\Users\kiril\AppData\Roaming\npm\cline.CMD")

    assert _resolve_cline_command() == [r"C:\Users\kiril\AppData\Roaming\npm\cline.CMD"]


def test_resolve_falls_back_to_npm_dir(monkeypatch):
    monkeypatch.delenv("CLINE_BIN", raising=False)
    monkeypatch.setattr("cline_bridge.shutil.which", lambda _n: None)
    monkeypatch.setenv("APPDATA", r"C:\Users\kiril\AppData\Roaming")
    monkeypatch.setattr("cline_bridge.os.path.exists",
                        lambda p: p.endswith("cline.cmd"))

    assert _resolve_cline_command() == [
        r"C:\Users\kiril\AppData\Roaming\npm\cline.cmd"]


# --- контракт ответа -----------------------------------------------------

def test_run_cline_task_success_returns_four_keys():
    _FakePopen.rc = 0
    _FakePopen.stdout = '{"type":"run_result","text":"ok"}\n'
    _FakePopen.stderr = ""

    result = run_cline_task("скажи ok", timeout=120)

    assert set(result) == {"success", "output", "error", "exit_code"}
    assert result["success"] is True
    assert result["output"] == '{"type":"run_result","text":"ok"}\n'
    assert result["error"] == ""
    assert result["exit_code"] == 0


def test_run_cline_task_nonzero_exit():
    _FakePopen.rc = 2
    _FakePopen.stdout = "partial"
    _FakePopen.stderr = "boom"

    result = run_cline_task("t")

    assert result["success"] is False
    assert result["output"] == "partial"
    assert result["error"] == "boom"
    assert result["exit_code"] == 2


def test_run_cline_task_passes_cwd_and_defaults():
    run_cline_task("t", timeout=60, cwd=r"C:\Projects")

    argv = _FakePopen.instances[0].argv
    assert argv[argv.index("-c") + 1] == r"C:\Projects"
    # cwd задан явно -> не берём os.getcwd()
    assert _FakePopen.instances[0].kwargs["cwd"] == r"C:\Projects"


def test_run_cline_task_defaults_to_current_dir():
    run_cline_task("t", timeout=60)

    assert _FakePopen.instances[0].kwargs["cwd"] == r"C:\neoterminal"


def test_run_cline_task_timeout_kills_and_reports(monkeypatch):
    _FakePopen.raise_timeout = True
    killed = []
    monkeypatch.setattr("cline_bridge._kill_tree",
                        lambda proc: killed.append(proc.pid))

    result = run_cline_task("долгая задача", timeout=600)

    assert result["success"] is False
    assert result["exit_code"] == -1
    assert "Таймаут" in result["error"]
    assert killed == [4242]  # процесс реально останавливается


def test_run_cline_task_binary_missing(monkeypatch):
    def boom(*_a, **_k):
        raise FileNotFoundError("[WinError 2] Не удается найти указанный файл")

    monkeypatch.setattr("cline_bridge.subprocess.Popen", boom)

    result = run_cline_task("t")

    assert result["success"] is False
    assert result["exit_code"] == -1
    assert "не найдена" in result["error"]


_CRASH = 3221225477  # 0xC0000005: так cline изредка падает на старте в Windows


def test_looks_like_crash_detects_abnormal_empty_run():
    assert _looks_like_crash(
        {"success": False, "output": "", "error": "", "exit_code": _CRASH})
    assert not _looks_like_crash(
        {"success": True, "output": "ok", "error": "", "exit_code": 0})
    # таймаут несёт текст ошибки -> это не аварийный старт
    assert not _looks_like_crash(
        {"success": False, "output": "", "error": "Таймаут", "exit_code": -1})


def test_run_cline_task_retries_after_startup_crash():
    _FakePopen.script = [{"rc": _CRASH}, {"rc": 0, "out": "готово"}]

    result = run_cline_task("t", timeout=60, retries=1)

    assert len(_FakePopen.instances) == 2
    assert result["success"] is True
    assert result["output"] == "готово"


def test_run_cline_task_exhausts_retries_and_reports_crash():
    _FakePopen.script = [{"rc": _CRASH}, {"rc": _CRASH}]

    result = run_cline_task("t", timeout=60, retries=1)

    assert len(_FakePopen.instances) == 2
    assert result["success"] is False
    assert result["exit_code"] == _CRASH


def test_run_cline_task_no_retry_by_default():
    _FakePopen.script = [{"rc": _CRASH}, {"rc": 0, "out": "готово"}]

    result = run_cline_task("t", timeout=60)

    assert len(_FakePopen.instances) == 1  # поведение по умолчанию не меняется
    assert result["exit_code"] == _CRASH


def test_run_cline_task_does_not_retry_timeout(monkeypatch):
    monkeypatch.setattr("cline_bridge._kill_tree", lambda proc: None)
    _FakePopen.script = [{"raise_timeout": True}, {"rc": 0, "out": "поздно"}]

    result = run_cline_task("t", timeout=1, retries=3)

    assert len(_FakePopen.instances) == 1  # таймаут не повторяем
    assert "Таймаут" in result["error"]


# --- убийство дерева процессов ------------------------------------------

def test_kill_tree_uses_taskkill_on_windows(monkeypatch):
    calls = []
    monkeypatch.setattr("cline_bridge.os.name", "nt")
    monkeypatch.setattr("cline_bridge.subprocess.run",
                        lambda argv, **kw: calls.append(argv))

    class _Proc:
        pid = 777

        @staticmethod
        def poll():
            return None

    _kill_tree(_Proc())

    assert calls and calls[0][:3] == ["taskkill", "/F", "/T"]
    assert calls[0][-1] == "777"


def test_kill_tree_noop_when_process_finished(monkeypatch):
    calls = []
    monkeypatch.setattr("cline_bridge.subprocess.run",
                        lambda argv, **kw: calls.append(argv))

    class _Proc:
        pid = 777

        @staticmethod
        def poll():
            return 0

    _kill_tree(_Proc())

    assert calls == []


# --- разбор NDJSON -------------------------------------------------------

def test_parse_events_extracts_text_tools_and_usage():
    stdout = "\n".join([
        'not json at all',
        json.dumps({"type": "agent_event", "event": {
            "type": "content_end", "contentType": "tool",
            "toolName": "execute_command"}}),
        json.dumps({"type": "run_result", "finishReason": "completed",
                    "text": "готово", "usage": {"totalCost": 0}}),
    ])

    ev = parse_events(stdout)

    assert ev["text"] == "готово"
    assert ev["finish_reason"] == "completed"
    assert ev["tools"] == ["execute_command"]
    assert ev["usage"] == {"totalCost": 0}


def test_parse_events_tolerates_empty():
    assert parse_events("")["text"] == ""


# --- CLI -----------------------------------------------------------------

def test_main_reports_json_and_exit_zero(monkeypatch, capsys):
    monkeypatch.setattr("cline_bridge.run_cline_task",
                        lambda *a, **k: {"success": True, "output": "ok",
                                         "error": "", "exit_code": 0})
    monkeypatch.setattr(sys, "argv",
                        ["cline_bridge.py", "--timeout", "5", "задача"])

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["success"] is True and payload["output"] == "ok"


def test_main_survives_output_outside_console_encoding(monkeypatch):
    """Вывод агента с символами вне cp1251 не должен ронять печать JSON.

    Регрессия: реальный запуск упал с UnicodeEncodeError на '\\u2716' при
    кодировке консоли cp1251 — уже ПОСЛЕ успешного выполнения задачи.
    """
    raw_out, raw_err = io.BytesIO(), io.BytesIO()
    monkeypatch.setattr(sys, "stdout",
                        io.TextIOWrapper(raw_out, encoding="cp1251",
                                         errors="strict"))
    monkeypatch.setattr(sys, "stderr",
                        io.TextIOWrapper(raw_err, encoding="cp1251",
                                         errors="strict"))
    monkeypatch.setattr(sys, "argv", ["cline_bridge.py", "--timeout", "5", "задача"])
    monkeypatch.setattr("cline_bridge.run_cline_task",
                        lambda *a, **k: {"success": True, "output": "✖ fail",
                                         "error": "", "exit_code": 0})

    with pytest.raises(SystemExit) as exc:
        main()

    sys.stdout.flush()
    assert exc.value.code == 0
    payload = json.loads(raw_out.getvalue().decode("utf-8"))
    assert payload["success"] is True
    assert payload["output"] == "✖ fail"

def test_main_reads_task_from_stdin(monkeypatch, capsys):
    seen = {}

    def fake(task, **kwargs):
        seen["task"] = task
        return {"success": True, "output": "", "error": "", "exit_code": 0}

    monkeypatch.setattr("cline_bridge.run_cline_task", fake)
    monkeypatch.setattr(sys, "argv", ["cline_bridge.py"])
    monkeypatch.setattr(sys, "stdin", type("S", (), {
        "isatty": staticmethod(lambda: False), "read": staticmethod(lambda: "из stdin")})())

    with pytest.raises(SystemExit):
        main()

    assert seen["task"] == "из stdin"


def test_main_without_task_exits_one(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["cline_bridge.py"])
    monkeypatch.setattr(sys, "stdin", type("S", (), {
        "isatty": staticmethod(lambda: False), "read": staticmethod(lambda: "")})())

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["success"] is False
    assert payload["exit_code"] == -1


def test_default_timeout_is_six_hundred():
    assert DEFAULT_TIMEOUT == 600
