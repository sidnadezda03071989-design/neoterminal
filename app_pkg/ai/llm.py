# -*- coding: utf-8 -*-
"""Клиент LLM (OpenAI-совместимый API) для всего ИИ-стека.

Fallback-цепочка из трёх провайдеров: Qwen (DashScope, QWEN_*) -> Groq
(GROQ_*) -> DeepSeek через aitunnel.ru (DEEPSEEK_*). Порядок задаётся
config.LLM_PROVIDER_ORDER (env LLM_PROVIDER_ORDER, по умолчанию
"qwen,groq,deepseek"). Провайдер без ключа пропускается; при HTTP
401/403/404 или сетевом сбое запрос уходит к следующему.

- _LLM_SESSION — глобальная requests.Session (keep-alive; старое имя
      _QWEN_SESSION оставлено алиасом);
- _extract_json — достаёт JSON из текста модели;
- _provider_config — креды провайдера по имени (qwen/groq/deepseek);
- _call_with_fallback — единая точка запроса: перебор цепочки,
      попытка 1 у каждого провайдера — response_format={"type":"json_object"};
      HTTP 400 (модель не поддерживает json-режим) -> попытка 2 без
      response_format, но с явным «Отвечай строго валидным JSON» в system;
      HTTP 400 в текстовом режиме — терминально (raise_for_status);
      HTTP 429 -> RuntimeError("rate limit") — в fallback НЕ уходим;
      сетевая ошибка (ConnectionError/Timeout/ConnectionResetError) ->
      "provider X failed (network: ...), fallback to Y";
      все провайдеры упали -> None (текст) / RuntimeError (vision);
      system-дубликаты во входных messages удаляются (свой system
      добавляется здесь ровно один);
- _llm_request / _llm_vision_request — обёртки над _call_with_fallback
      (старые имена _qwen_request / _qwen_vision_request оставлены
      алиасами). Явные креда (api_key/base_url/model/timeout) включают
      legacy-режим «один провайдер» — цепочка не применяется;
- _llm_vision_request — multimodal (text + image_url base64); для vision
      Qwen-vl идёт первым независимо от LLM_PROVIDER_ORDER (qwen-vl-max —
      единственная заведомо рабочая vision-модель); поддержка images
      проверяется по GET /models (architecture.input_modalities
      содержит "image"), иначе провайдер пропускается, а если vision
      недоступен нигде — RuntimeError("vision unavailable").
"""

import hashlib
import json
import logging
import math
import re

import requests
from requests.adapters import HTTPAdapter

from app_pkg import config
from app_pkg.cache import get_cached_llm_response, set_cached_llm_response
from app_pkg.config import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    DEEPSEEK_VISION_MODEL, DEEPSEEK_TIMEOUT, DEEPSEEK_MAX_TOKENS,
    GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL, GROQ_VISION_MODEL,
    GROQ_TIMEOUT, GROQ_MAX_TOKENS,
    QWEN_API_KEY_1, QWEN_BASE_URL_1, QWEN_MODEL_1, QWEN_VL_MODEL,
    QWEN_TIMEOUT, QWEN_MAX_TOKENS,
    LLM_PROVIDER_ORDER,
)

log = logging.getLogger(__name__)

_LLM_SESSION = requests.Session()
_LLM_SESSION.mount(
    "https://", HTTPAdapter(pool_connections=4, pool_maxsize=8, max_retries=0)
)
_LLM_SESSION.mount(
    "http://", HTTPAdapter(pool_connections=4, pool_maxsize=8, max_retries=0)
)
# Legacy-имя сессии (старые тесты/импорты).
_QWEN_SESSION = _LLM_SESSION


def _extract_json(text):
    """Извлекает первый сбалансированный JSON-объект {...} из текста.

    Убирает markdown-обёртку ```json ... ```. Возвращает распарсенный
    объект или None.
    """
    if not text:
        return None
    text = text.strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            pass
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        try:
            return json.loads(m.group(1))
        except (json.JSONDecodeError, ValueError):
            pass
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except (json.JSONDecodeError, ValueError):
                        break
    return None


def _fmt_ai_num(x):
    """Компактная строка числа через repr(float(x)); 'null', если не число."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return "null"
    if math.isnan(f) or math.isinf(f):
        return "null"
    return repr(f)


def _llm_headers(api_key):
    """Заголовки Groq: только Authorization (HTTP-Referer не нужен)."""
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


# Legacy-имя (старые импорты/тесты).
_openrouter_headers = _llm_headers


def _strip_system_duplicates(messages):
    """Удаляет system-сообщения из messages (свой system добавит запрос).

    Защита от двойного system: LLM API ломается на двух system-
    сообщениях. Если дубликаты были — warning в лог.
    """
    clean = []
    removed = 0
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "system":
            removed += 1
            continue
        clean.append(m)
    if removed:
        log.warning(
            "removed %d system message(s) from messages "
            "(system is prepended by _llm_request)", removed)
    return clean


def _post_chat(url, headers, payload, timeout, model):
    """POST /chat/completions; 429 -> RuntimeError('rate limit')."""
    resp = _QWEN_SESSION.post(url, json=payload, headers=headers,
                              timeout=timeout)
    if resp.status_code == 429:
        raise RuntimeError("rate limit")
    if resp.status_code >= 400:
        # Тело ошибки критично для диагностики (модель, квоты, json-режим).
        log.warning("LLM %s HTTP %d: %s", model, resp.status_code,
                    resp.text[:500])
    return resp


def _build_payload(model, system, messages, max_tokens):
    """Payload /chat/completions (system уже очищен/дополнен вызывающим)."""
    return {
        "model": model,
        "messages": [{"role": "system", "content": system}] + list(messages),
        "temperature": 0.3,
        "max_tokens": max_tokens,
    }


def _attempt_chat(base_url, api_key, model, system, messages, max_tokens,
                  timeout):
    """Одна попытка у провайдера: json-режим, при HTTP 400 — повтор без него.

    Решение о fallback на другого провайдера принимает вызывающий код,
    поэтому raise_for_status() здесь не вызывается.
    """
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = _llm_headers(api_key)
    payload = _build_payload(model, system, messages, max_tokens)
    resp = _post_chat(url, headers,
                      dict(payload, response_format={"type": "json_object"}),
                      timeout, model)
    if resp.status_code == 400:
        # Модель/роут не поддерживает json-режим — повторяем без
        # response_format, но с явной инструкцией в system.
        log.warning("LLM %s rejected response_format (HTTP 400) "
                    "— retry без него", model)
        payload["messages"][0]["content"] = (
            f"{system}\nОтвечай строго валидным JSON.")
        resp = _post_chat(url, headers, payload, timeout, model)
    return resp


# Сетевые сбои (DNS/TLS/timeout/reset), в отличие от HTTP-статусов,
# не дают response — их ловим отдельно от fallback по 401/403/404.
_NETWORK_ERRORS = (requests.RequestException, ConnectionResetError)

# HTTP-статусы, при которых провайдер считается недоступным и запрос
# уходит к следующему в цепочке (LLM_PROVIDER_ORDER).
_CHAIN_FALLBACK_STATUSES = (401, 403, 404)


def _provider_config(name):
    """Креды провайдера по имени; None — неизвестное имя.

    qwen -> Aliyun DashScope (qwen-plus / qwen-vl-max),
    groq -> Groq, deepseek -> aitunnel.ru.
    vision-модель лежит в отдельном поле vl_model.
    """
    name = (name or "").strip().lower()
    if name == "qwen":
        return {
            "name": "qwen",
            "base_url": QWEN_BASE_URL_1,
            "api_key": QWEN_API_KEY_1,
            "model": QWEN_MODEL_1,
            "vl_model": QWEN_VL_MODEL,
            "timeout": QWEN_TIMEOUT,
            "max_tokens": QWEN_MAX_TOKENS,
        }
    if name == "groq":
        return {
            "name": "groq",
            "base_url": GROQ_BASE_URL,
            "api_key": GROQ_API_KEY,
            "model": GROQ_MODEL,
            "vl_model": GROQ_VISION_MODEL,
            "timeout": GROQ_TIMEOUT,
            "max_tokens": GROQ_MAX_TOKENS,
        }
    if name == "deepseek":
        return {
            "name": "deepseek",
            "base_url": DEEPSEEK_BASE_URL,
            "api_key": DEEPSEEK_API_KEY,
            "model": DEEPSEEK_MODEL,
            "vl_model": DEEPSEEK_VISION_MODEL,
            "timeout": DEEPSEEK_TIMEOUT,
            "max_tokens": DEEPSEEK_MAX_TOKENS,
        }
    return None


def _chain_providers(vision=False, api_key=None, base_url=None, model=None,
                     timeout=None):
    """Список конфигов провайдеров, которые пробуем по очереди.

    1) Явные креда (api_key/base_url/model/timeout) -> один провайдер
       «explicit»: legacy-вызовы (agents._run_agent и т.п.) по цепочке
       не ходят, но сетевая защита сохраняется.
    2) Иначе — имена из config.LLM_PROVIDER_ORDER. Для vision Qwen идёт
       ПЕРВЫМ независимо от порядка: qwen-vl-max — единственная заведомо
       рабочая vision-модель (Groq-vision может быть снят с производства,
       DeepSeek-vision — экспериментальная).
    """
    if api_key or base_url or model or timeout:
        return [{
            "name": "explicit",
            "base_url": base_url or DEEPSEEK_BASE_URL,
            "api_key": api_key or DEEPSEEK_API_KEY,
            "model": model or DEEPSEEK_MODEL,
            "vl_model": model or DEEPSEEK_VISION_MODEL,
            "timeout": timeout or DEEPSEEK_TIMEOUT,
            "max_tokens": DEEPSEEK_MAX_TOKENS,
        }]
    order = list(LLM_PROVIDER_ORDER) or ["qwen", "groq", "deepseek"]
    if vision:
        order = ["qwen"] + [p for p in order if p != "qwen"]
    chain = []
    for name in order:
        cfg = _provider_config(name)
        if cfg is None:
            log.warning("provider %s skipped (unknown name)", name)
            continue
        chain.append(cfg)
    return chain


def _completion_content(resp):
    """content (или reasoning_content) первого choice; None, если пусто."""
    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        return None
    message = choices[0].get("message") or {}
    return message.get("content") or message.get("reasoning_content")


def _purpose_max_tokens(purpose):
    """max_tokens по purpose (токен-диета): analysis -> 800, chat -> 400.

    None/неизвестное значение -> None (используется max_tokens провайдера).
    """
    if purpose == "analysis":
        return config.LLM_MAX_TOKENS_ANALYSIS
    if purpose == "chat":
        return config.LLM_MAX_TOKENS_CHAT
    return None


def _call_with_fallback(system, messages=None, vision=False, user_text=None,
                        image_base64=None, api_key=None, base_url=None,
                        model=None, timeout=None, purpose=None):
    """Единая точка запроса к LLM: перебор провайдеров fallback-цепочки.

    vision=False -> текстовый POST {base}/chat/completions (_attempt_chat);
    vision=True  -> multimodal (_attempt_vision: system + text + image_url).

    purpose ("analysis"|"chat") задаёт max_tokens ответа независимо от
    провайдера (токен-диета); None -> max_tokens провайдера.

    Правила:
      * нет ключа / нет base_url или модели -> провайдер пропускается
        ("provider X skipped (no key)") и берётся следующий;
      * HTTP 401/403/404, сетевая ошибка (requests.RequestException /
        ConnectionResetError), а для vision ещё и HTTP 400 (модель без
        images) -> "provider X failed (...), fallback to Y" и следующий;
      * HTTP 400 в текстовом режиме — терминально (raise_for_status):
        _attempt_chat уже перепробовал запрос без response_format;
      * HTTP 429 -> RuntimeError("rate limit") из _post_chat (терминально,
        в fallback НЕ уходим);
      * все провайдеры упали -> None (текст) / RuntimeError (vision).
    """
    if not vision:
        messages = _strip_system_duplicates(messages or [])
        # Страховка для json-режима: подстрока "json" в нижнем регистре.
        _all_text = system + "".join(
            str(m.get("content") or "") for m in messages
            if isinstance(m, dict))
        if "json" not in _all_text:
            system = f"{system}\njson"
            log.warning("auto-added json keyword for API compatibility")

    chain = _chain_providers(vision=vision, api_key=api_key,
                             base_url=base_url, model=model, timeout=timeout)
    for idx, cfg in enumerate(chain):
        name = cfg["name"]
        nxt = chain[idx + 1]["name"] if idx + 1 < len(chain) else "nothing"
        prompt_model = cfg["vl_model"] if vision else cfg["model"]
        if not cfg["api_key"]:
            log.warning("provider %s skipped (no key)", name)
            continue
        if not cfg["base_url"] or not prompt_model:
            log.warning("provider %s skipped (no base_url/model: %s, %s)",
                        name, cfg["base_url"], prompt_model)
            continue
        if vision and not _vision_supported(cfg["base_url"], cfg["api_key"],
                                            prompt_model):
            log.warning("provider %s failed (model %s без images), "
                        "fallback to %s", name, prompt_model, nxt)
            continue
        try:
            if vision:
                resp = _attempt_vision(cfg["base_url"], cfg["api_key"],
                                       prompt_model, cfg["max_tokens"],
                                       cfg["timeout"], system, user_text,
                                       image_base64)
            else:
                resp = _attempt_chat(cfg["base_url"], cfg["api_key"],
                                     prompt_model, system, messages,
                                     _purpose_max_tokens(purpose)
                                     or cfg["max_tokens"], cfg["timeout"])
        except _NETWORK_ERRORS as exc:
            log.warning("provider %s failed (network: %s), fallback to %s",
                        name, exc, nxt)
            continue

        fallback_needed = resp.status_code in _CHAIN_FALLBACK_STATUSES
        if vision and resp.status_code == 400:
            # 400 у vision = модель не принимает images (json-режим уже
            # перепробован в _attempt_vision) — пробуем следующего.
            fallback_needed = True
        if fallback_needed:
            log.warning("provider %s failed (HTTP %d), fallback to %s",
                        name, resp.status_code, nxt)
            continue

        resp.raise_for_status()
        return _completion_content(resp)

    if vision:
        log.warning("LLM-VL: все провайдеры цепочки недоступны "
                    "— vision unavailable")
        raise RuntimeError("vision unavailable")
    log.warning("LLM: все провайдеры цепочки недоступны — return None")
    return None


def _llm_cache_parts(system, messages):
    """(sha1-ключ, есть_ли_свеча) по system + messages + последняя свеча.

    Время последней свечи вычисляется регэкспом по содержимому messages
    (формат свечи: "[<unix_sec> ..." — компактный, или "[<unix_sec>,..."
    — legacy). Ключ меняется только когда контекст обновился: новая
    свеча -> новый ключ -> поход в сеть. Если свечей нет, кешировать
    нельзя (chat/болтовня не должна залипать) — has_candle=False.
    """
    text = system + "".join(
        str(m.get("content") or "") for m in messages if isinstance(m, dict))
    last_candle = ""
    for m in re.finditer(r"\[(\d{9,13})[ ,]", text):
        last_candle = m.group(1)
    key = hashlib.sha1((text + last_candle).encode("utf-8")).hexdigest()
    return key, bool(last_candle)


def _llm_request(system, messages, api_key=None, base_url=None,
                 model=None, timeout=None, purpose="analysis"):
    """Текстовый запрос: цепочка qwen -> groq -> deepseek.

    Порядок — config.LLM_PROVIDER_ORDER. Явные креда (api_key/base_url/
    model/timeout) включают legacy-режим «один провайдер» (agents.py).
    purpose ("analysis"|"chat") задаёт max_tokens (токен-диета: 800/400).
    Прочие ошибки: 429 -> RuntimeError("rate limit"); HTTP 400 после двух
    попыток (json-режим) -> HTTPError; иные >= 400 -> HTTPError.
    Возвращает content (или reasoning_content) модели, либо None.

    Кеш (токен-диета): analysis-запросы без явных креда кешируются на
    config.LLM_RESPONSE_CACHE_TTL сек — повторный запрос с тем же
    контекстом (та же последняя свеча) идёт без сети. Chat не кешируется
    (нужна свежесть беседы); legacy-режим (явные креда) тоже.
    """
    explicit = bool(api_key or base_url or model or timeout)
    use_cache = (purpose == "analysis" and not explicit)
    key = None
    if use_cache:
        key, has_candle = _llm_cache_parts(system, messages or [])
        if not has_candle:
            # Нет свечей в контексте — кешировать нечего (не analysis-контекст).
            use_cache = False
    if use_cache:
        cached = get_cached_llm_response(key)
        if cached is not None:
            log.info("LLM response served from cache (key=%s...)",
                     str(key)[:8])
            return cached
    result = _call_with_fallback(
        system, messages, vision=False, api_key=api_key, base_url=base_url,
        model=model, timeout=timeout, purpose=purpose)
    if use_cache and key and result is not None:
        set_cached_llm_response(key, result)
    return result


# Legacy-имя (старые импорты/тесты).
_qwen_request = _llm_request


_VISION_CAPS_CACHE = {}


def _vision_supported(base_url, api_key, model):
    """True, если модель принимает images (GET /models, input_modalities).

    Результат кешируется. Ошибка запроса списка моделей, модель не
    найдена в списке или провайдер не отдаёт architecture/input_modalities
    (например aitunnel) -> True (не блокируем: пусть решает сам запрос).
    """
    key = (base_url.rstrip("/"), model)
    if key in _VISION_CAPS_CACHE:
        return _VISION_CAPS_CACHE[key]
    try:
        resp = _LLM_SESSION.get(
            f"{base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {api_key}"}, timeout=15)
        resp.raise_for_status()
        for m in (resp.json() or {}).get("data") or []:
            if m.get("id") == model:
                arch = m.get("architecture")
                if not arch:
                    # Модальности не заявлены — не блокируем, решает запрос.
                    _VISION_CAPS_CACHE[key] = True
                    return True
                modalities = arch.get("input_modalities") or []
                ok = "image" in modalities
                _VISION_CAPS_CACHE[key] = ok
                return ok
    except Exception as exc:  # noqa: BLE001
        log.warning("vision capability check failed (%s) — пропускаем", exc)
        return True
    return True


def _attempt_vision(base_url, api_key, model, max_tokens, timeout,
                    system, user_text, image_base64):
    """Одна vision-попытка у провайдера (HTTP 400 -> повтор без json-режима)."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = _llm_headers(api_key)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64," + image_base64,
                }},
            ]},
        ],
        "temperature": 0.3,
        "max_tokens": max_tokens,
    }
    resp = _post_chat(url, headers,
                      dict(payload, response_format={"type": "json_object"}),
                      timeout, model)
    if resp.status_code == 400:
        log.warning("LLM-VL %s rejected response_format "
                    "(HTTP 400) — retry без него", model)
        resp = _post_chat(url, headers, payload, timeout, model)
    return resp


def _llm_vision_request(system, user_text, image_base64,
                        api_key=None, base_url=None, model=None):
    """Vision-запрос (multimodal): цепочка провайдеров, Qwen-vl первым.

    Формат image_url — OpenAI-совместимый:
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}.
    Для vision Qwen идёт первым независимо от LLM_PROVIDER_ORDER
    (qwen-vl-max — единственная заведомо рабочая vision-модель: у Groq
    vision-модель могла быть снята с производства, у DeepSeek —
    экспериментальная). Явные креда (api_key/base_url/model) включают
    legacy-режим «один провайдер».
    Ошибки:
        нет ключа / нет поддержки images / 400-404 / сетевой сбой у ВСЕХ
                провайдеров -> RuntimeError("vision unavailable")
        429     -> RuntimeError("rate limit")
    Возвращает content модели (строку) или None.
    """
    return _call_with_fallback(system, vision=True, user_text=user_text,
                               image_base64=image_base64, api_key=api_key,
                               base_url=base_url, model=model)


# Legacy-имя (старые импорты/тесты).
_qwen_vision_request = _llm_vision_request
