# -*- coding: utf-8 -*-
"""cline_bridge.py — внешняя обёртка для управления Cline CLI из другого процесса.

Принимает задачу (аргумент командной строки или stdin), запускает Cline
в headless-режиме (`--json --yolo`) и печатает результат как JSON:

    {"success": bool, "output": str, "error": str, "exit_code": int}

Проверено на Cline CLI 3.0.62 (Windows):
  * `-y/--yolo`    — пропуск подтверждений + submit_and_exit. Без submit_and_exit
    агент в one-shot режиме не закрывает сессию сам;
  * `--json`       — NDJSON-поток событий в stdout;
  * `-t/--timeout` — таймаут самого рана. Ставим его МЕНЬШЕ таймаута subprocess,
    чтобы агент успел завершиться штатно, а не был убит;
  * bare-имя `cline` не резолвится в subprocess на Windows (CreateProcess не
    применяет PATHEXT), поэтому путь ищется через shutil.which.

Примеры:
    python cline_bridge.py "добавь hello world в main.py"
    echo "поправь баг" | python cline_bridge.py --timeout 900
    python cline_bridge.py --model qwen3.8-max --thinking high "перепиши тесты"

Порядок в модуле: header, imports, log, constants, def, main.
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from typing import Optional

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 600
DEFAULT_PROVIDER = "openai-compatible"
DEFAULT_MODEL = "qwen3.8-flash"
# CLI-таймаут должен срабатывать раньше внешнего: иначе вместо NDJSON-финала
# получим принудительное убийство и пустой вывод.
TIMEOUT_MARGIN = 15


def _resolve_cline_command() -> list:
    """Найти cline и вернуть argv-префикс для subprocess.run/Popen."""
    override = os.environ.get("CLINE_BIN", "").strip()
    if override:
        return [override]

    found = shutil.which("cline")
    if found:
        return [found]

    npm_dir = os.path.join(os.environ.get("APPDATA", ""), "npm")
    for name in ("cline.cmd", "cline.ps1", "cline"):
        cand = os.path.join(npm_dir, name)
        if os.path.exists(cand):
            return [cand]

    # последний шанс: отдать как есть, ошибку поймает run_cline_task
    return ["cline"]


def _build_argv(task: str, timeout: int, provider: Optional[str],
                model: Optional[str], thinking: Optional[str],
                cwd: Optional[str], yolo: bool) -> list:
    """Собрать argv для cline CLI (без запуска)."""
    argv = _resolve_cline_command()
    argv.append("--json")
    if yolo:
        argv.append("--yolo")
    if provider:
        argv += ["-P", provider]
    if model:
        argv += ["-m", model]
    if thinking:
        argv += ["--thinking", thinking]
    if cwd:
        argv += ["-c", cwd]
    argv += ["-t", str(max(1, timeout - TIMEOUT_MARGIN))]
    argv.append(task)
    return argv


def _kill_tree(proc: "subprocess.Popen") -> None:
    """Убить процесс вместе с потомками.

    cline — это .cmd-обёртка, запускающая вложенный бинарь, поэтому обычный
    proc.kill() оставил бы агента работать в фоне и держать сессию.
    """
    if proc.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, timeout=30)
            return
        except Exception as exc:  # noqa: BLE001
            log.warning("taskkill не сработал для pid=%s: %s", proc.pid, exc)
    try:
        proc.kill()
    except Exception as exc:  # noqa: BLE001
        log.warning("kill не сработал для pid=%s: %s", proc.pid, exc)


def _looks_like_crash(result: dict) -> bool:
    """Процесс умер на старте: аварийный код и полностью пустой вывод.

    На Windows у cline изредка наблюдается 0xC0000005 (3221225477) сразу после
    запуска — агент при этом ничего не успевает сделать, поэтому повтор безопасен.
    Таймаут сюда не попадает: у него есть текст ошибки.
    """
    return (result["exit_code"] != 0
            and not result["output"].strip()
            and not result["error"].strip())


def run_cline_task(
    task: str,
    timeout: int = DEFAULT_TIMEOUT,
    cwd: Optional[str] = None,
    provider: Optional[str] = DEFAULT_PROVIDER,
    model: Optional[str] = DEFAULT_MODEL,
    thinking: Optional[str] = None,
    yolo: bool = True,
    retries: int = 0,
) -> dict:
    """Запустить Cline CLI с задачей и вернуть результат.

    Parameters
    ----------
    task : str
        Текст задачи для Cline.
    timeout : int
        Таймаут выполнения в секундах (по умолчанию 600).
    cwd : str, optional
        Рабочая директория для агента (по умолчанию текущая).
    provider : str, optional
        Id провайдера (`-P`). None — использовать сохранённый в Cline.
    model : str, optional
        Id модели (`-m`). None — использовать сохранённую в Cline.
    thinking : str, optional
        Уровень рассуждений: none|low|medium|high|xhigh.
    yolo : bool
        Пропускать подтверждения инструментов (YOLO-режим, флаг `-y`).
    retries : int
        Сколько раз повторить запуск, если процесс упал на старте (см.
        _looks_like_crash). Таймауты не повторяются. По умолчанию 0.

    Returns
    -------
    dict
        {"success": bool, "output": str, "error": str, "exit_code": int}
    """
    attempt = 0
    while True:
        result = _launch(task, timeout, cwd, provider, model, thinking, yolo)
        if attempt >= retries or not _looks_like_crash(result):
            return result
        attempt += 1
        log.warning("cline упал на старте (exit_code=%s): повтор %d/%d",
                    result["exit_code"], attempt, retries)


def _launch(
    task: str,
    timeout: int,
    cwd: Optional[str],
    provider: Optional[str],
    model: Optional[str],
    thinking: Optional[str],
    yolo: bool,
) -> dict:
    """Один запуск cline; контракт ответа — см. run_cline_task."""
    argv = _build_argv(task, timeout, provider, model, thinking, cwd, yolo)
    log.debug("cline argv: %s", argv[1:])

    try:
        proc = subprocess.Popen(
            argv,
            cwd=cwd or os.getcwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        return {
            "success": False,
            "output": "",
            "error": "Команда 'cline' не найдена. Установите Cline CLI: "
                     "npm install -g cline (или задайте CLINE_BIN)",
            "exit_code": -1,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "success": False,
            "output": "",
            "error": f"Не удалось запустить cline: {exc}",
            "exit_code": -1,
        }

    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        try:
            out, err = proc.communicate(timeout=30)
        except Exception:  # noqa: BLE001
            out, err = "", ""
        return {
            "success": False,
            "output": out or "",
            "error": f"Таймаут выполнения: {timeout} секунд (процесс остановлен)",
            "exit_code": -1,
        }
    except Exception as exc:  # noqa: BLE001
        _kill_tree(proc)
        return {
            "success": False,
            "output": "",
            "error": f"Непредвиденная ошибка: {exc}",
            "exit_code": -1,
        }

    return {
        "success": proc.returncode == 0,
        "output": out or "",
        "error": err or "",
        "exit_code": proc.returncode,
    }


def parse_events(stdout: str) -> dict:
    """Разобрать NDJSON из `--json` и вернуть финальный результат агента.

    Возвращает {"text": str, "tools": [str], "finish_reason": str|None,
    "usage": dict}. Нужен потребителям, которым мало сырого stdout.
    """
    text, finish, usage, tools = "", None, {}, []
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            evt = json.loads(line)
        except ValueError:
            continue
        if evt.get("type") == "run_result":
            text = evt.get("text") or text
            finish = evt.get("finishReason")
            usage = evt.get("usage") or {}
        event = evt.get("event") or {}
        if isinstance(event, dict) and event.get("contentType") == "tool":
            name = event.get("toolName")
            if name:
                tools.append(name)
    return {"text": text, "tools": tools, "finish_reason": finish,
            "usage": usage}


def _ensure_utf8_stdio() -> None:
    """Перевести stdout/stderr на UTF-8, чтобы печать JSON не падала.

    Вывод агента содержит произвольные символы (эмодзи, ✖, кириллицу), а
    консоль Windows по умолчанию отдаёт cp1251/cp866 — тогда print() роняет
    скрипт с UnicodeEncodeError уже ПОСЛЕ успешного выполнения задачи.
    Потоки, которые уже UTF-8 (pip-каптуры тестов, перенаправления), не трогаем.
    """
    for stream in (sys.stdout, sys.stderr):
        encoding = getattr(stream, "encoding", None) or ""
        if "utf-8" in encoding.lower() or "utf8" in encoding.lower():
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):  # не-текстовый/закрытый
            pass


def main() -> None:
    """Прочитать задачу из argv/stdin, запустить Cline, напечатать JSON."""
    _ensure_utf8_stdio()
    parser = argparse.ArgumentParser(
        description="Запуск задачи в Cline CLI (headless, YOLO) с JSON-ответом.",
    )
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                        help=f"таймаут, секунды (по умолчанию {DEFAULT_TIMEOUT})")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER,
                        help=f"id провайдера (по умолчанию {DEFAULT_PROVIDER}); "
                             "пустая строка — сохранённый в Cline")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"id модели (по умолчанию {DEFAULT_MODEL}); "
                             "пустая строка — сохранённая в Cline")
    parser.add_argument("--cwd", default=None,
                        help="рабочая директория агента")
    parser.add_argument("--thinking", default=None,
                        choices=["none", "low", "medium", "high", "xhigh"],
                        help="уровень рассуждений модели")
    parser.add_argument("--no-yolo", dest="yolo", action="store_false",
                        help="требовать подтверждения инструментов")
    parser.add_argument("--retries", type=int, default=0,
                        help="повторов при аварийном старте cline "
                             "(0xC0000005); таймауты не повторяются")
    parser.add_argument("--parse", action="store_true",
                        help="добавить в вывод разобранный NDJSON (поле events)")
    parser.add_argument("task", nargs=argparse.REMAINDER,
                        help="текст задачи; если пусто — читается из stdin")
    args = parser.parse_args()

    task = " ".join(args.task).strip()
    if not task and not sys.stdin.isatty():
        task = sys.stdin.read().strip()

    if not task:
        print(json.dumps({
            "success": False,
            "output": "",
            "error": "Не указан текст задачи. Передайте через аргумент или stdin.",
            "exit_code": -1,
        }, ensure_ascii=False, indent=2))
        sys.exit(1)

    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    result = run_cline_task(
        task,
        timeout=args.timeout,
        cwd=args.cwd,
        provider=args.provider or None,
        model=args.model or None,
        thinking=args.thinking,
        yolo=args.yolo,
        retries=max(0, args.retries),
    )
    if args.parse:
        result = dict(result)
        result["events"] = parse_events(result.get("output", ""))

    print(json.dumps(result, ensure_ascii=False, indent=2))
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()