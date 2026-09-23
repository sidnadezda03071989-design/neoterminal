"""NeoTerminal — проверка API-ключа QwenCloud без запуска приложения.

Ключ в файле не хранится: берётся из --key или из окружения
(QWEN_API_KEY_1 / DASHSCOPE_API_KEY). Документация QwenCloud:

    base_url = https://dashscope-intl.aliyuncs.com/compatible-mode/v1
    POST {base_url}/chat/completions, заголовок Authorization: Bearer <key>

Формат ключа pay-as-you-go — sk-ws-... . Ключ Token Plan (sk-sp-...) привязан
к другому хосту (token-plan.ap-southeast-1.maas.aliyuncs.com) и с этим
base_url не работает.

Примеры:
    python check_qwen_key.py --key sk-ws-xxxxx
    python check_qwen_key.py --model qwen3.8-max

Порядок в модуле: header, imports, log, def.
"""

import argparse
import logging
import os
import sys

import requests

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.8-flash"
TOKEN_PLAN_HOST = "token-plan.ap-southeast-1.maas.aliyuncs.com"


def _mask(key):
    """Ключ для печати: первые 8 и последние 4 символа + длина."""
    if not key:
        return "(пусто)"
    if len(key) <= 16:
        return key[:4] + "***"
    return f"{key[:8]}...{key[-4:]} (длина {len(key)})"


def _probe_models(session, base_url, key, timeout):
    """GET /models — быстрая проверка ключа. Возвращает (ok, сообщение)."""
    try:
        resp = session.get(
            f"{base_url}/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        log.warning("QWEN: сбой запроса /models: %s", exc)
        return False, f"сеть: {exc}"
    if resp.status_code != 200:
        return False, f"HTTP {resp.status_code}: {resp.text[:300]}"
    data = resp.json() or {}
    ids = [m.get("id") for m in (data.get("data") or []) if m.get("id")]
    tail = f", например {', '.join(ids[:3])}" if ids else ""
    return True, f"HTTP 200, моделей: {len(ids)}{tail}"


def _probe_chat(session, base_url, key, model, timeout):
    """POST /chat/completions — боевой вызов. Возвращает (ok, сообщение, ответ)."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Ответь одним словом: ok"}],
        "max_tokens": 16,
        "temperature": 0,
    }
    try:
        resp = session.post(
            f"{base_url}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        log.warning("QWEN: сбой запроса /chat/completions: %s", exc)
        return False, f"сеть: {exc}", None
    if resp.status_code != 200:
        return False, f"HTTP {resp.status_code}: {resp.text[:400]}", None
    data = resp.json() or {}
    choices = data.get("choices") or []
    answer = ""
    if choices:
        answer = (choices[0].get("message") or {}).get("content") or ""
    usage = data.get("usage") or {}
    info = (f"HTTP 200, model={data.get('model') or model}, "
            f"tokens={usage.get('total_tokens', '?')}")
    return True, info, answer.strip()


def main(argv=None):
    """Проверяет ключ: GET /models + POST /chat/completions. 0 — рабочий."""
    parser = argparse.ArgumentParser(
        description="Проверка API-ключа QwenCloud (Qwen/DashScope)."
    )
    parser.add_argument("--key", default="",
                        help="API-ключ (по умолчанию — из окружения)")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                        help=f"base_url OpenAI-совместимого API (по умолчанию {DEFAULT_BASE_URL})")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"ID модели для вызова (по умолчанию {DEFAULT_MODEL})")
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="таймаут запроса, секунды")
    args = parser.parse_args(argv)

    key = (args.key
           or os.getenv("QWEN_API_KEY_1", "")
           or os.getenv("DASHSCOPE_API_KEY", "")).strip()
    if not key:
        print("Ключ не задан: укажите --key либо QWEN_API_KEY_1 / DASHSCOPE_API_KEY")
        return 2

    base_url = args.base_url.rstrip("/")
    prefix = key.split("-")[1] + "-" if key.count("-") >= 1 else "?"
    print(f"Ключ:     {_mask(key)}")
    print(f"Тип:      {prefix}*")
    print(f"base_url: {base_url}")
    print(f"model:    {args.model}")

    session = requests.Session()
    _ok_models, models_msg = _probe_models(session, base_url, key, args.timeout)
    print(f"[1/2] GET  /models           -> {models_msg}")
    ok_chat, chat_msg, answer = _probe_chat(
        session, base_url, key, args.model, args.timeout
    )
    print(f"[2/2] POST /chat/completions -> {chat_msg}")
    if answer:
        print(f"      ответ модели: {answer}")

    if ok_chat:
        print("ИТОГ: ключ рабочий.")
        return 0

    print("ИТОГ: ключ НЕ прошёл проверку.")
    if key.startswith("sk-sp-"):
        print(f"      это ключ Token Plan: нужен хост {TOKEN_PLAN_HOST}")
    elif key.startswith("sk-"):
        print("      проверьте, что ключ не удалён и у workspace есть доступ к модели")
    return 1


if __name__ == "__main__":
    sys.exit(main())
