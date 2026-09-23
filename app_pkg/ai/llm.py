"""Клиент LLM (OpenAI-совместимый API) для всего ИИ-стека.

Fallback-цепочка из трёх провайдеров: DeepSeek через aitunnel.ru
(DEEPSEEK_*) -> Qwen (DashScope, QWEN_*) -> Groq (GROQ_*). Порядок задаётся
config.LLM_PROVIDER_ORDER (env LLM_PROVIDER_ORDER, по умолчанию
"deepseek,qwen,groq"). Провайдер без ключа пропускается; при HTTP
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
      HTTP 429 (_RateLimitError) — состояние КОНКРЕТНОГО провайдера:
      запрос уходит к следующему в цепочке (иначе один залимиченный Groq
      блокировал рабочий DeepSeek); если 429 у ВСЕХ доступных —
      наружу RuntimeError("rate limit");
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
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MAX_TOKENS,
    DEEPSEEK_MODEL,
    DEEPSEEK_TIMEOUT,
    DEEPSEEK_VISION_MODEL,
    GROQ_API_KEY,
    GROQ_BASE_URL,
    GROQ_MAX_TOKENS,
    GROQ_MODEL,
    GROQ_TIMEOUT,
    GROQ_VISION_MODEL,
    LLM_PROVIDER_ORDER,
    QWEN_API_KEY_1,
    QWEN_BASE_URL_1,
    QWEN_MAX_TOKENS,
    QWEN_MODEL_1,
    QWEN_TIMEOUT,
    QWEN_VL_MODEL,
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
    """Извлекает JSON-объект {...} из текста.

    Убирает markdown-обёртку ```json ... ```; при неудаче — ленивый парс
    (_extract_json_loose): типографские кавычки/дефисы, хвостовые запятые и
    перебор сбалансированных {...} (модель может вставить мусор до/после или
    сгенерить два объекта — берём первый валидный). Возвращает объект или None.
    """
    if not text:
        return None
    text = text.strip()
    if text.startswith(("{", "[")):
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
    return _extract_json_loose(text)


def _extract_json_loose(raw):
    """Ленивый парс нестрогого вывода LLM: кавычки-фигурки, хвостовые и
    задвоенные запятые (json их запрещает), затем каждый сбалансированный
    {...} по очереди — первый верный объект и есть ответ."""
    text = (raw.replace("\u201c", '"').replace("\u201d", '"')
                .replace("\u2018", "'").replace("\u2019", "'")
                .replace("\u2013", "-").replace("\u2014", "-"))
    # Запятая перед }/] в JSON всегда невалидна — безопасно вырезать;
    # отдельные "," между элементами не трогаем.
    text = re.sub(r",\s*([}\]])", r"\1", text)
    text = re.sub(r",{2,}", ",", text)
    start = 0
    while True:
        start = text.find("{", start)
        if start < 0:
            return None
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
                        # Первый кандидат битый — пробуем следующий {...}.
                        start = i + 1
                        break
        else:
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


class _RateLimitError(RuntimeError):
    """HTTP 429 у конкретного провайдера (ловится в _call_with_fallback).

    Отдельный класс, а не голый RuntimeError: так цикл не спутает его с
    «vision unavailable», который тоже RuntimeError, но поднимается после
    цикла. Наружу (если лимит у всех) уходит именно RuntimeError("rate
    limit") — на это завязаны тесты и _llm_error_hint в ai_backtest.
    """


def _log_safe(text):
    """ASCII-only строка для лога: тела ошибок провайдеров содержат символы
    вне кодировки консоли Windows (cp1251) — например китайские скобки \uff08
    в ответе Qwen. Без этого StreamHandler падает с UnicodeEncodeError,
    лог-запись глотается целиком (--- Logging error ---) и причина падения
    провайдера не видна в логе вообще.
    """
    return str(text).encode("ascii", "backslashreplace").decode("ascii")


def _post_chat(url, headers, payload, timeout, model):
    """POST /chat/completions; 429 -> _RateLimitError('rate limit').

    Исключение ловит _call_with_fallback: 429 — состояние КОНКРЕТНОГО
    провайдера, поэтому запрос уходит к следующему в цепочке (иначе один
    залимиченный Groq блокировал рабочий DeepSeek). Если залимичены ВСЕ —
    наружу уходит RuntimeError('rate limit').
    """
    resp = _QWEN_SESSION.post(url, json=payload, headers=headers,
                              timeout=timeout)
    if resp.status_code == 429:
        log.warning("LLM %s HTTP 429 (rate limit)", model)
        raise _RateLimitError("rate limit")
    if resp.status_code >= 400:
        # Тело ошибки критично для диагностики (модель, квоты, json-режим).
        log.warning("LLM %s HTTP %d: %s", model, resp.status_code,
                    _log_safe(resp.text[:500]))
    return resp


def _build_payload(model, system, messages, max_tokens, temperature=0.3,
                   reasoning_effort=None):
    """Payload /chat/completions (system уже очищен/дополнен вызывающим).

    temperature переопределяем вызывающим (AI Backtest-вердикты — 0.0 для
    детерминизма); дефолт 0.3 сохранён для остальных агентов.
    reasoning_effort ("none"|"low"|...|None) добавляется в payload ТОЛЬКО
    когда задан явно: reasoning-модели (deepseek-v4-flash) иначе тратят
    весь max_tokens на chain-of-thought и не отдают JSON.
    """
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}] + list(messages),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort
    return payload


def _attempt_chat(base_url, api_key, model, system, messages, max_tokens,
                  timeout, temperature=0.3, reasoning_effort=None):
    """Одна попытка у провайдера: json-режим, при HTTP 400 — повтор без него.

    При 400 снимаем и response_format, и необязательный reasoning_effort:
    провайдер мог отклонить любой из них, а 400 в текстовом режиме
    терминален для цепочки (см. _call_with_fallback). Решение о fallback на
    другого провайдера принимает вызывающий код, поэтому raise_for_status()
    здесь не вызывается.
    """
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = _llm_headers(api_key)
    payload = _build_payload(model, system, messages, max_tokens, temperature,
                             reasoning_effort)
    resp = _post_chat(url, headers,
                      dict(payload, response_format={"type": "json_object"}),
                      timeout, model)
    if resp.status_code == 400:
        # Модель/роут не поддерживает json-режим (или reasoning_effort) —
        # повторяем без response_format и без необязательного
        # reasoning_effort, но с явной инструкцией в system.
        log.warning("LLM %s rejected response_format (HTTP 400) "
                    "— retry без него", model)
        payload.pop("reasoning_effort", None)
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
    order = list(LLM_PROVIDER_ORDER) or ["deepseek", "qwen", "groq"]
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
    """content (или reasoning_content/reasoning) первого choice; None, если пусто.

    Часть моделей (напр. openai/gpt-oss-120b на Groq) отвечает HTTP 200 с
    ПУСТЫМ message.content, а текст кладёт в message.reasoning (не
    reasoning_content) — читаем и его. Но если ответ ОБРЕЗАН
    (finish_reason="length") и content пуст, то reasoning — это незавершённый
    chain-of-thought (весь max_tokens ушёл на размышления), а не ответ:
    возвращаем None, чтобы _call_with_fallback ушёл к следующему провайдеру.
    Иначе на панель попадал текст «We need to process the input snapshot...»
    вместо уровней. Если текста нет нигде — тоже None.
    """
    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        return None
    choice = choices[0]
    message = choice.get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        content = content.strip()
    if content:
        return content
    if choice.get("finish_reason") == "length":
        return None
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if isinstance(reasoning, str):
        reasoning = reasoning.strip()
    return reasoning or None


def _collect_usage(resp, usage_out):
    """Суммировать usage.{prompt,completion,total}_tokens ответа в usage_out.

    Токен-диета AI Backtest: точный расход считаем ТОЛЬКО по факту ответа
    API. usage_out — обычный dict-аккумулятор; None — учёт не нужен.
    Нечисловые/отсутствующие поля не ломают прогон.
    """
    if not isinstance(usage_out, dict):
        return
    try:
        usage = (resp.json() or {}).get("usage") or {}
    except Exception:  # noqa: BLE001 — тело без JSON не должно ронять запрос
        return
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        try:
            value = int(usage.get(key) or 0)
        except (TypeError, ValueError):
            continue
        if value:
            usage_out[key] = usage_out.get(key, 0) + value


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
                        model=None, timeout=None, purpose=None,
                        max_tokens=None, temperature=None, usage_out=None,
                        reasoning_effort=None):
    """Единая точка запроса к LLM: перебор провайдеров fallback-цепочки.

    vision=False -> текстовый POST {base}/chat/completions (_attempt_chat);
    vision=True  -> multimodal (_attempt_vision: system + text + image_url).

    purpose ("analysis"|"chat") задаёт max_tokens ответа независимо от
    провайдера (токен-диета); None -> max_tokens провайдера.
    max_tokens/temperature — явный перебор: AI Backtest-вердикты просят
    120 токенов при temperature=0.0; temperature=None -> 0.3 (дефолт).
    reasoning_effort ("none"|...|None) — глушит chain-of-thought у
    reasoning-моделей (None -> параметр в payload не идёт).
    usage_out (dict) — аккумулятор usage.{prompt,completion}_tokens успешного
    ответа (см. _collect_usage).

    Правила:
      * нет ключа / нет base_url или модели -> провайдер пропускается
        ("provider X skipped (no key)") и берётся следующий;
      * HTTP 401/403/404, сетевая ошибка (requests.RequestException /
        ConnectionResetError), а для vision ещё и HTTP 400 (модель без
        images) -> "provider X failed (...), fallback to Y" и следующий;
      * HTTP 400 в текстовом режиме — терминально (raise_for_status):
        _attempt_chat уже перепробовал запрос без response_format;
      * HTTP 429 (_RateLimitError из _post_chat) — состояние конкретного
        провайдера: fallback к следующему; если 429 у ВСЕХ доступных —
        RuntimeError("rate limit") наружу (текст и vision);
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
    rate_limited = []
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
                tok = (max_tokens or _purpose_max_tokens(purpose)
                       or cfg["max_tokens"])
                temp = 0.3 if temperature is None else temperature
                resp = _attempt_chat(cfg["base_url"], cfg["api_key"],
                                     prompt_model, system, messages, tok,
                                     cfg["timeout"], temp, reasoning_effort)
        except _NETWORK_ERRORS as exc:
            log.warning("provider %s failed (network: %s), fallback to %s",
                        name, exc, nxt)
            continue
        except _RateLimitError:
            # 429 — лимит/квота конкретного провайдера, а не ошибка запроса:
            # пробуем следующего (иначе один залимиченный Groq блокировал
            # бы рабочий DeepSeek).
            log.warning("provider %s rate limited (HTTP 429), fallback to %s",
                        name, nxt)
            rate_limited.append(name)
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
        content = _completion_content(resp)
        if content:
            _collect_usage(resp, usage_out)
            return content
        # HTTP 200 с пустым текстом (модель вернула только reasoning/ничего):
        # не «съедаем» ответ — пробуем следующего провайдера цепочки.
        log.warning("provider %s returned empty content, fallback to %s",
                    name, nxt)
        continue

    if rate_limited:
        # Цикл дошёл до конца без контента и хотя бы один провайдер был
        # именно 429 — наружу уходит rate limit (иначе он терялся бы за
        # безликим None / «vision unavailable»).
        log.warning("LLM: HTTP 429 у %d провайдера(ов) цепочки — "
                    "RuntimeError('rate limit')", len(rate_limited))
        raise RuntimeError("rate limit")
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
                 model=None, timeout=None, purpose="analysis",
                 max_tokens=None, temperature=None, usage_out=None,
                 reasoning_effort=None):
    """Текстовый запрос: цепочка qwen -> groq -> deepseek.

    Порядок — config.LLM_PROVIDER_ORDER. Явные креда (api_key/base_url/
    model/timeout) включают legacy-режим «один провайдер» (agents.py).
    purpose ("analysis"|"chat") задаёт max_tokens (токен-диета: 800/400).
    Прочие ошибки: 429 -> fallback к следующему, при 429 у ВСЕХ —
    RuntimeError("rate limit"); HTTP 400 после двух попыток (json-режим)
    -> HTTPError; иные >= 400 -> HTTPError.
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
        model=model, timeout=timeout, purpose=purpose, max_tokens=max_tokens,
        temperature=temperature, usage_out=usage_out,
        reasoning_effort=reasoning_effort)
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
