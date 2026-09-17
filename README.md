# neoterminal — внешнее управление Cline

## cline_bridge.py

Обёртка для запуска задач в Cline CLI (headless, YOLO) из внешних приложений.

```bash
# задачу аргументом
python cline_bridge.py "добавь hello world в main.py"

# задачу через stdin
echo "поправь баг" | python cline_bridge.py --timeout 900

# с повторами при аварийном старте cline (0xC0000005 на Windows)
python cline_bridge.py --retries 2 --timeout 600 "задача"

# разобранный NDJSON (текст ответа, инструменты, usage)
python cline_bridge.py --parse "задача"
```

Выход — JSON: `{"success": bool, "output": str, "error": str, "exit_code": int}`.
Exit code процесса: 0 при успехе, 1 при ошибке.

### Параметры CLI

| Флаг | По умолчанию | Описание |
|---|---|---|
| `--timeout` | 600 | таймаут задачи, сек (CLI-таймаут = минус 15 c запасом) |
| `--provider` | `openai-compatible` | id провайдера (`-P`); пустая строка — сохранённый |
| `--model` | `qwen3.8-flash` | id модели (`-m`); пустая строка — сохранённая |
| `--cwd` | текущая | рабочая директория агента (`-c`) |
| `--thinking` | — | `none\|low\|medium\|high\|xhigh` |
| `--no-yolo` | выкл. | требовать подтверждения инструментов |
| `--retries` | 0 | повторы при аварийном старте (0xC0000005); таймауты не повторяются |
| `--parse` | выкл. | добавить поле `events` с разбором NDJSON |

### Библиотечное использование

```python
from cline_bridge import run_cline_task, parse_events

result = run_cline_task("задача", timeout=600, retries=2)
events = parse_events(result["output"])  # {"text", "tools", "finish_reason", "usage"}
```

### Примечания

* Windows: bare `cline` не резолвится из `subprocess` (CreateProcess не применяет
  PATHEXT) — путь ищется через `shutil.which`, можно переопределить переменной
  `CLINE_BIN`.
* stdout/stderr бриджа принудительно переводятся на UTF-8 (только если текущая
  кодировка не UTF-8), иначе `print(json.dumps(...))` падает с
  `UnicodeEncodeError` на символах вне cp1251 — уже после успешной задачи.
* Провайдер Qwen (OpenAI-compatible, Dashscope) настроен на уровне Cline CLI
  (`cline config`); бридж передаёт `-P openai-compatible -m qwen3.8-flash`
  на каждый запуск.

## MCP-серверы

Конфигурация: `C:\Users\kiril\.cline\data\settings\cline_mcp_settings.json`

* `filesystem` — `@modelcontextprotocol/server-filesystem`, доступ: `C:\Projects`, `C:\neoterminal`
* `shell` — `mcp-shell-server`, выполнение команд PowerShell/cmd

## Тесты

```bash
cd C:\neoterminal
python -m pytest tests -q
```
