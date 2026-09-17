"""NeoTerminal — централизованная конфигурация.

Все константы и настройки приложения собраны в этом модуле.
Остальной код не должен хардкодить значения — только импорт настроек отсюда.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------- пути
BASE_DIR = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------- символы
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT",            # крипта
    "EURUSD", "GBPUSD", "USDCHF", "USDJPY",     # форекс
    "USDCAD", "EURJPY", "XAUUSD",               # форекс / золото
]
CRYPTO_SYMBOLS = {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
FOREX_SYMBOLS = {s for s in SYMBOLS if s not in CRYPTO_SYMBOLS}

# ---------------------------------------------------------- таймфреймы
TIMEFRAMES = ["1m", "5m", "15m", "1H", "4H", "1D"]
DEFAULT_TIMEFRAME = "5m"

BINANCE_TF = {
    "1m": "1m", "5m": "5m", "15m": "15m",
    "1H": "1h", "4H": "4h", "1D": "1d",
}
TF_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "1H": 3600, "4H": 14400, "1D": 86400}

YF_TICKER = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDCHF": "USDCHF=X",
    "USDJPY": "JPY=X",
    "USDCAD": "USDCAD=X",
    "EURJPY": "EURJPY=X",
    "XAUUSD": "GC=F",
}
YF_PERIODS = {"1m": "2d", "5m": "5d", "15m": "1mo", "1H": "1mo", "4H": "3mo", "1D": "max"}

PRICE_PRECISION = {
    "BTCUSDT": 1, "ETHUSDT": 2, "SOLUSDT": 3,
    "EURUSD": 5, "GBPUSD": 5, "USDCHF": 5,
    "USDJPY": 3, "USDCAD": 5, "EURJPY": 3, "XAUUSD": 2,
}

REPLAY_LIMIT = 20000
# Потолок свечей для бэктеста: 5m × 30 дней = ~8600 баров, 20 страниц Binance
# через прокси рвутся и копят retry (3-5 мин). 5000 баров ≈ 17 дней по 5m —
# быстрее и без обрывов; больше можно запросить селектом #bt-limit в панели.
BACKTEST_LIMIT = 5000

# ------------------------------------------------------------- каталоги
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data"))).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "data.db"
DRAWINGS_FILE = DATA_DIR / "drawings.json"  # legacy: файловое хранилище (не используется)
CHAT_FILE = DATA_DIR / "chat.json"          # legacy: файловое хранилище (не используется)

# ------------------------------------------------------------- сервер
APP_NAME = "NeoTerminal"
APP_VERSION = "0.1.0"
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "5000"))

# ------------------------------------------------------------ источники
BINANCE_API_TIMEOUT = 10.0
BINANCE_WS_URL = "wss://stream.binance.com:9443/stream?streams="
LIVE_RECONNECT_DELAY = 8.0
FOREX_LIVE_POLL_INTERVAL = 90.0

# ------------------------------------------------------------- кеши
DATA_CACHE_TTL_CRYPTO = 3600.0
DATA_CACHE_TTL_FOREX = 120.0
DATA_CACHE_TTL_FOREX_1D = 1800.0
# Максимум ключей (symbol, tf) в data_cache (LRU): 10 символов × 6 ТФ = 60.
DATA_CACHE_MAX_ITEMS = 60
CONTEXT_CACHE_TTL = 45.0
VERDICT_CACHE_TTL = 10.0
CONTEXT_TF_LIMITS = {"1m": 60, "5m": 60, "15m": 60, "1H": 80, "4H": 100, "1D": 150}
CONTEXT_WORKERS = 6

# ------------------------------------------------------------- алерты
ALERT_CHECK_INTERVAL_SECONDS = 10.0
ALERT_CHECK_TIMEFRAME = "5m"

# ------------------------------------------------------- интеграции
MT5_PATH = os.getenv("MT5_PATH", "")
# Автозапуск терминала MT5 (SDK сам стартует его по MT5_PATH), если
# initialize() без пути не удался. 0 — отключить автозапуск.
MT5_AUTOSTART = os.getenv("MT5_AUTOSTART", "1") == "1"
# Сколько секунд ждать запуска терминала при автозапуске.
MT5_START_TIMEOUT = float(os.getenv("MT5_START_TIMEOUT", "30"))
# Запас календарного окна при авто-расчёте диапазона в fetch_mt5 (задана
# только одна граница): форекс торгуется 5 дней из 7, лимит в свечах
# нужно умножать на календарный коэффициент (7/5 = 1.4 + запас).
FOREX_CALENDAR_PADDING = 1.5
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
DEFAULT_REQUEST_TIMEOUT = 120.0

# ------------------------------------------------------------ DeepSeek
# Основной LLM-провайдер: DeepSeek через aitunnel.ru (OpenAI-совместимый
# API). POST {DEEPSEEK_BASE_URL}/chat/completions, GET .../models,
# Authorization: Bearer sk-aitunnel-...
# base_url проверен вручную: https://api.aitunnel.ru/v1 -> HTTP 200
# (310 моделей, включая deepseek-v4-flash и deepseek-v4-flash-vision-exp).
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY") or ""
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL",
                              "https://api.aitunnel.ru/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
DEEPSEEK_MODEL_2 = os.getenv("DEEPSEEK_MODEL_2", "deepseek-v4-flash")
DEEPSEEK_VISION_MODEL = os.getenv("DEEPSEEK_VISION_MODEL",
                                  "deepseek-v4-flash-vision-exp")
DEEPSEEK_TIMEOUT = float(os.getenv("DEEPSEEK_TIMEOUT", "120"))
DEEPSEEK_MAX_TOKENS = int(os.getenv("DEEPSEEK_MAX_TOKENS", "2000"))

# ------------------------------------------------------- Groq (fallback)
# Резервный провайдер: включается ТОЛЬКО если DeepSeek вернул 401/403/404
# (см. app_pkg/ai/llm.py). По умолчанию не используется.
# OpenAI-совместимый API, заголовок Authorization: Bearer gsk_...
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or ""
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_MODEL_2 = os.getenv("GROQ_MODEL_2", "llama-3.1-8b-instant")
# Актуальный список моделей: GET https://api.groq.com/openai/v1/models.
# (llama-3.2-90b-vision-preview мог быть снят с производства — тогда
# vision штатно уходит в fallback на текстовый анализ.)
GROQ_VISION_MODEL = os.getenv("GROQ_VISION_MODEL",
                              "llama-3.2-90b-vision-preview")
GROQ_TIMEOUT = float(os.getenv("GROQ_TIMEOUT", "120"))
GROQ_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "2000"))
# Режим анализа: 0/1 — один агент, 2 — два агента (Price + Indicators).
GROQ_AGENT_MODE = int(os.getenv("GROQ_AGENT_MODE",
                                os.getenv("QWEN_AGENT_MODE", "0")))
VISION_ENABLED = os.getenv("VISION_ENABLED", "1") == "1"

# ------------------------------------------------- Qwen / DashScope (Qwen)
# OpenAI-совместимый режим Aliyun DashScope. Ключ проверен вручную:
# GET {base}/models -> HTTP 200 (169 моделей). ВАЖНО: qwen-flash отдаёт
# HTTP 403 AllocationQuota.FreeTierOnly (free-tier квота исчерпана),
# поэтому рабочий дефолт — qwen-plus. Диагностика: check_qwen_key.py.
QWEN_API_KEY_1 = (os.getenv("QWEN_API_KEY_1")
                  or os.getenv("QWEN_API_KEY") or "")
QWEN_API_KEY_2 = os.getenv("QWEN_API_KEY_2") or QWEN_API_KEY_1
QWEN_BASE_URL_1 = (os.getenv("QWEN_BASE_URL_1")
                   or os.getenv("QWEN_BASE_URL")
                   or "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
QWEN_BASE_URL = QWEN_BASE_URL_1  # legacy-имя
QWEN_BASE_URL_2 = os.getenv("QWEN_BASE_URL_2") or QWEN_BASE_URL_1
QWEN_MODEL_1 = (os.getenv("QWEN_MODEL_1")
                or os.getenv("QWEN_MODEL") or "qwen-plus")
QWEN_MODEL = QWEN_MODEL_1  # legacy-имя
QWEN_MODEL_2 = os.getenv("QWEN_MODEL_2") or QWEN_MODEL_1
QWEN_VL_MODEL = os.getenv("QWEN_VL_MODEL") or "qwen-vl-max"
QWEN_TIMEOUT = float(os.getenv("QWEN_TIMEOUT", "120"))
QWEN_TIMEOUT_2 = float(os.getenv("QWEN_TIMEOUT_2", str(QWEN_TIMEOUT)))
QWEN_MAX_TOKENS = int(os.getenv("QWEN_MAX_TOKENS", "2000"))
QWEN_AGENT_MODE = GROQ_AGENT_MODE

OPENROUTER_API_KEY = GROQ_API_KEY
OPENROUTER_BASE_URL = GROQ_BASE_URL
OPENROUTER_MODEL = GROQ_MODEL
OPENROUTER_VISION_MODEL = GROQ_VISION_MODEL
OPENROUTER_AGENT_MODE = GROQ_AGENT_MODE
OPENROUTER_TIMEOUT = GROQ_TIMEOUT
OPENROUTER_MAX_TOKENS = GROQ_MAX_TOKENS

# ------------------------------------------------ fallback-цепочка провайдеров
# Порядок опроса в app_pkg/ai/llm.py: qwen -> groq -> deepseek. Первый
# провайдер с непустым ключом используется; при 401/403/404 или сетевом
# сбое запрос уходит к следующему. Переопределяется env LLM_PROVIDER_ORDER
# (имена через запятую). Неизвестные имена пропускаются с warning.
LLM_PROVIDER_ORDER = [
    p.strip() for p in os.getenv("LLM_PROVIDER_ORDER",
                                 "qwen,groq,deepseek").split(",")
    if p.strip()
]


def llm_enabled() -> bool:
    """True, если настроен хотя бы один провайдер fallback-цепочки."""
    return bool(
        (QWEN_API_KEY_1 and QWEN_BASE_URL_1 and QWEN_MODEL_1)
        or (GROQ_API_KEY and GROQ_BASE_URL and GROQ_MODEL)
        or (DEEPSEEK_API_KEY and DEEPSEEK_BASE_URL and DEEPSEEK_MODEL)
    )


def deepseek_enabled() -> bool:
    """Алиас llm_enabled() (совместимость: раньше основной был DeepSeek)."""
    return llm_enabled()


def any_llm_enabled() -> bool:
    """Алиас llm_enabled(): настроен хотя бы один из трёх провайдеров."""
    return llm_enabled()


def qwen_enabled() -> bool:
    """Legacy-алиас llm_enabled() (старые вызовы не ломаем)."""
    return llm_enabled()


def openrouter_enabled() -> bool:
    """Legacy-алиас llm_enabled() (старые вызовы не ломаем)."""
    return llm_enabled()

# ------------------------------------------------------------- данные
DEFAULT_LIMIT = 500
MAX_DATA_LIMIT = 20000
# Максимум свечей на таймфрейм: столько тянет и держит кэш get_series_df,
# обрезка до запрошенного limit делается только на выходе. Значения
# синхронизированы с HISTORY_LIMIT из static/js/config.js. Binance отдаёт
# максимум 1000 свечей за запрос, поэтому крипта тянется постранично
# (binance.fetch_binance_paged) — доступно до 20000 свечей.
HISTORY_LIMITS = {"1m": 20000, "5m": 20000, "15m": 20000, "1H": 20000, "4H": 20000, "1D": 20000}
# Дефолт limit /api/data при холодном старте: 1 страница Binance (~1с).
# Фронт затем в фоне сам запрашивает 20000 — кеш дозагружается по частям.
INITIAL_LOAD_CANDLES = 1000
# Глубина истории для /api/watchlist (передаётся в history_limit).
WATCHLIST_HISTORY = 100
# Глубина истории для мульти-ТФ трендов и ИИ-контекста.
TRENDS_HISTORY = 200
# Глубина истории для проверки алертов (нужны только последние бары).
ALERTS_HISTORY = 2
# Хвост кеша для /api/last-bar: по нему пересчитываются индикаторы последнего
# бара (стабильные SMA50/RSI14 требуют хотя бы ~150 баров истории).
LAST_BAR_HISTORY = 200
CHAT_HISTORY_LIMIT = 20
# Сколько последних сообщений помнить как контекст беседы (уходит в промпт).
CHAT_MEMORY_MESSAGES = 10
MAX_AI_DRAWABLES = 5
AI_MAX_CANDLES_BY_TF = {"1m": 50, "5m": 100, "15m": 150, "1H": 200, "4H": 120, "1D": 80}

# ------------------------------------------------------------- метрики
METRICS_QUANTILES = (0.5, 0.95)