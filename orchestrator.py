"""orchestrator.py — FastAPI-сервер чата с оркестратором на Qwen.

Qwen (OpenAI-совместимый API Dashscope) получает 4 инструмента и сам решает,
когда их вызывать; сервер исполняет tool_calls и возвращает результаты обратно
в модель, пока она не даст финальный ответ (максимум MAX_ITERATIONS кругов).

Инструменты:
    run_cline_task(task, cwd)  — запуск Cline через python cline_bridge.py --parse
    read_file(path)            — чтение файла
    list_dir(path)             — список файлов/папок
    run_command(command, cwd)  — команда PowerShell

Запуск:
    python orchestrator.py            # uvicorn на 127.0.0.1:8000

Зависимости: fastapi, uvicorn, requests (pydantic ставится с fastapi).
.env: QWEN_API_KEY, QWEN_BASE_URL, QWEN_MODEL (см. .env.example).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
BRIDGE_PATH = BASE_DIR / "cline_bridge.py"
CHAT_HTML_PATH = BASE_DIR / "chat.html"

SYSTEM_PROMPT = (
    "Ты оркестратор. У тебя есть инструмент run_cline_task, который запускает "
    "Cline через python cline_bridge.py. Разбей задачу пользователя на "
    "подзадачи, для каждой вызывай run_cline_task, проверяй результат через "
    "read_file/list_dir/run_command, отчитывайся."
)

MAX_ITERATIONS = 20
# cline_bridge сам держит свой CLI-таймаут меньше внешнего (с запасом 15 с),
# поэтому внешнему subprocess даём чуть больше, чем дефолтные 600 с бриджа.
CLINE_TASK_TIMEOUT = 660
COMMAND_TIMEOUT = 180
FILE_READ_LIMIT = 200_000  # символов, чтобы не заваливать контекст модели
HTTP_TIMEOUT = 300


def _load_env(path: Path) -> dict:
    """Прочитать .env без внешних зависимостей: KEY=VALUE построчно."""
    env = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


_ENV = _load_env(ENV_PATH)
API_KEY = os.environ.get("QWEN_API_KEY") or _ENV.get("QWEN_API_KEY", "")
BASE_URL = (
    os.environ.get("QWEN_BASE_URL")
    or _ENV.get("QWEN_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
).rstrip("/")
MODEL = os.environ.get("QWEN_MODEL") or _ENV.get("QWEN_MODEL", "qwen3.8-flash")

app = FastAPI(title="NeoTerminal Orchestrator")

# История диалога: одна сессия в памяти, без system-сообщения (он подставляется
# в начало messages при каждом запросе).
history: list = []


class ChatIn(BaseModel):
    text: str


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_cline_task",
            "description": (
                "Запустить задачу в Cline (автономный агент, YOLO) через "
                "python cline_bridge.py. Возвращает JSON "
                "{success, output, error, exit_code, events}. "
                "Выполняется долго (до 10 минут) — используй для реальной "
                "работы с кодом в проекте."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Текст задачи для Cline.",
                    },
                    "cwd": {
                        "type": "string",
                        "description": (
                            "Рабочая директория проекта, например C:\\\\neoterminal."
                        ),
                    },
                },
                "required": ["task"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Прочитать текстовый файл и вернуть его содержимое.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Абсолютный путь к файлу."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "Получить список файлов и папок в директории.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Путь к директории."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Выполнить команду PowerShell и вернуть stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Команда PowerShell."},
                    "cwd": {
                        "type": "string",
                        "description": "Рабочая директория команды (опционально).",
                    },
                },
                "required": ["command"],
            },
        },
    },
]


def _run_cline_task(task: str, cwd: str | None = None) -> str:
    """python cline_bridge.py --parse "<task>" (флаги строго ДО задачи —
    argparse бриджа собран на REMAINDER и всё после задачи считает её текстом)."""
    if not BRIDGE_PATH.exists():
        return json.dumps({"success": False, "error": f"Не найден {BRIDGE_PATH}"},
                          ensure_ascii=False)
    argv = [sys.executable, str(BRIDGE_PATH), "--parse", "--retries", "2", task]
    try:
        proc = subprocess.run(
            argv,
            cwd=cwd or str(BASE_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=CLINE_TASK_TIMEOUT,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return json.dumps(
            {"success": False, "error": f"Таймаут {CLINE_TASK_TIMEOUT} с"},
            ensure_ascii=False,
        )
    try:
        payload = json.loads(proc.stdout)
    except ValueError:
        payload = {"success": proc.returncode == 0, "output": proc.stdout,
                   "error": proc.stderr, "exit_code": proc.returncode}
    return json.dumps(payload, ensure_ascii=False)


def _read_file(path: str) -> str:
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)
    truncated = len(text) > FILE_READ_LIMIT
    return json.dumps(
        {"path": path, "content": text[:FILE_READ_LIMIT],
         "truncated": truncated},
        ensure_ascii=False,
    )


def _list_dir(path: str) -> str:
    try:
        entries = sorted(Path(path).iterdir(),
                         key=lambda p: (p.is_file(), p.name.lower()))
    except OSError as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)
    items = [
        {"name": e.name, "type": "dir" if e.is_dir() else "file",
         "size": e.stat().st_size if e.is_file() else None}
        for e in entries[:500]
    ]
    return json.dumps({"path": path, "entries": items}, ensure_ascii=False)


def _run_command(command: str, cwd: str | None = None) -> str:
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
            cwd=cwd or str(BASE_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=COMMAND_TIMEOUT,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return json.dumps(
            {"error": f"Таймаут {COMMAND_TIMEOUT} с"}, ensure_ascii=False)
    return json.dumps(
        {"stdout": proc.stdout, "stderr": proc.stderr,
         "exit_code": proc.returncode},
        ensure_ascii=False,
    )


TOOL_IMPL = {
    "run_cline_task": _run_cline_task,
    "read_file": _read_file,
    "list_dir": _list_dir,
    "run_command": _run_command,
}


def _call_qwen(messages: list) -> dict:
    """Один запрос к Qwen (OpenAI-совместимый /chat/completions)."""
    response = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}",
                 "Content-Type": "application/json"},
        json={
            "model": MODEL,
            "messages": messages,
            "tools": TOOLS,
            "temperature": 0.3,
            # qwen3-модели Dashscope в non-streaming требуют выключить thinking
            "enable_thinking": False,
        },
        timeout=HTTP_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]


@app.get("/")
def index() -> FileResponse:
    """Отдать chat.html; если его ещё нет — понятная 404."""
    if not CHAT_HTML_PATH.exists():
        raise HTTPException(status_code=404,
                            detail=f"{CHAT_HTML_PATH.name} не найден")
    return FileResponse(CHAT_HTML_PATH)


@app.post("/chat")
def chat(body: ChatIn) -> dict:
    """Принять текст, прогнать цикл оркестратора, вернуть reply + log."""
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Поле 'text' пустое")

    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                *history,
                {"role": "user", "content": text}]
    log: list = []
    reply = ""

    for _ in range(MAX_ITERATIONS):
        assistant = _call_qwen(messages)
        messages.append(assistant)

        tool_calls = assistant.get("tool_calls") or []
        if not tool_calls:
            reply = assistant.get("content") or ""
            break

        for call in tool_calls:
            name = call["function"]["name"]
            log.append(name)
            try:
                args = json.loads(call["function"].get("arguments") or "{}")
            except ValueError:
                args = {}
            impl = TOOL_IMPL.get(name)
            try:
                result = impl(**args) if impl else json.dumps(
                    {"error": f"Неизвестный инструмент: {name}"},
                    ensure_ascii=False)
            except Exception as exc:  # noqa: BLE001 — ошибка инструмента не
                # должна ронять цикл: модель получит её текстом
                result = json.dumps({"error": f"{type(exc).__name__}: {exc}"},
                                    ensure_ascii=False)
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "content": result})
    else:
        # 20 итераций исчерпаны: модель всё ещё вызывает инструменты
        reply = "Остановлено: достигнут лимит итераций (20)."

    history.extend(messages[1:])  # в историю system не пишем
    return {"reply": reply, "log": log}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
