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

# --------------------------------------- полная история ручного бэктеста
# True — гнать бэктест по ВСЕЙ доступной истории до BACKTEST_MAX_CANDLES
# (крипта ~20k, форекс ~13k свечей), а не по BACKTEST_LIMIT. Блоки сделок
# на графике должны совпадать с числом сделок в статистике (BLOCK-30).
BACKTEST_USE_FULL_HISTORY = True
# Максимум свечей для полноисторийного бэктеста = MAX_DATA_LIMIT (20k).
BACKTEST_MAX_CANDLES = 20000
# 0 = без лимита: возвращать ВСЕ сделки в trades_full/trades
# (НЕ последние 20 — иначе блоки на графике не совпадают со статистикой).
BACKTEST_MAX_TRADES_RETURNED = 0
# Dataset по умолчанию для /api/backtest/trades (кнопка «Показать на графике»):
# train | test | full. По умолчанию test — out-of-sample (последние 30%),
# цифра сделок совпадает с колонкой TEST Trades сканера (BLOCK-33).
BACKTEST_DATASET_DEFAULT = "test"

# ------------------------------------------- TP/SL бэктестов (BLOCK-42)
# ЕДИНАЯ модель выхода на весь проект: сканер (train/test окна) и
# /api/backtest/trades считают ОДИН и тот же прогон — выход ТОЛЬКО по
# касанию линии TP или SL. Поэтому число сделок в панели сканера и число
# блоков на графике совпадают 1:1, и у каждой сделки есть tp_price/sl_price
# (пунктирные IN/OUT/TP/SL).
#
# Стоп — первичен (риск трейдера): SL = entry − BACKTEST_SL_ATR×ATR.
# Тейк строится СТРОГО от стопа с фиксированным R/R = BACKTEST_RR:
#   TP = entry + BACKTEST_RR × (entry − SL),
# то есть расстояние TP↔entry ровно в BACKTEST_RR раз больше entry↔SL — для
# КАЖДОЙ сделки, всегда, независимо от переданного tp_atr (он оставлен на
# обратную совместимость вызовов, но не влияет на уровень).
# Результат сделки = линия, которой цена коснулась первой (exit_reason tp|sl);
# открытые позиции, не дожившие до уровня, в сделки НЕ попадают.
# ВНИМАНИЕ: смена этих чисел меняет winrate/trades всех прогонов —
# старые строки scan_results становятся невалидными (их нужно чистить).
BACKTEST_SL_ATR = 1.0     # размер риска (стоп) = 1×ATR(14) от entry
BACKTEST_RR = 2.0         # Take Profit = 2×ATR от entry (entry↔TP = rr × entry↔SL)
# Обратная совместимость: старые вызывающие могли передать tp_atr напрямую.
BACKTEST_TP_ATR = BACKTEST_RR * BACKTEST_SL_ATR  # 2.0

# --------------------------------------------- grid-search сканер стратегий
# Лимит комбинаций параметров на один запуск сканера. Если декартово
# произведение грида больше — берётся случайная подвыборка (seed=42,
# воспроизводимый набор при одинаковом гриде), см. app_pkg/ai/scanner.py.
# 5000 -> 50000: масштабный прогон на несколько часов — 10 символов × 6 ТФ ×
# 12 стратегий (402 комбинации на символ/ТФ) = 24 120, плюс запас на
# кастомные гриды (10 символов × 402 × запас).
SCAN_MAX_COMBINATIONS = 50000
# Число потоков ThreadPoolExecutor для параллельного прогона бэктестов.
SCAN_WORKERS = 8
# --- оценка длительности прогона (ответ POST /api/scan, подсказка в UI) ---
# Средняя длительность одного бэктеста комбинации (сек) и фетча данных на
# один символ (сек). Оценка: combos × sec_combo / SCAN_WORKERS + symbols × 15.
SCAN_SECONDS_PER_COMBINATION = 0.3
SCAN_FETCH_SECONDS_PER_SYMBOL = 15.0
# Прогон дольше 30 минут -> warning в ответе POST /api/scan (UI: жёлтый
# блок + confirm перед запуском). 3600 -> 1800: длинные прогоны стали
# реальной ситуацией, предупреждать нужно раньше.
SCAN_WARN_SECONDS = 1800
# Жёсткий потолок длительности прогона: оценка больше 4 часов -> 400
# (скан не стартует, объём надо сократить или запускать по частям).
SCAN_HARD_LIMIT_SECONDS = 14400
# Каждые N обработанных комбинаций SSE-событие scan_progress шлёт eta_seconds.
SCAN_ETA_PUSH_EVERY = 10
# Таймаут на одну пару (symbol, tf) в сканере: больше — пара пропускается,
# скан переходит к следующей (защита от зависания на «тяжёлом» символе,
# напр. XAUUSD с 20k+ свечей).
SCAN_SYMBOL_TIMEOUT_SECONDS = 600
# Доля истории на train; остаток (30%) — out-of-sample test.
SCAN_TRAIN_SPLIT = 0.7
# Глубина истории скана в днях от текущего момента (1 год вместо 2):
# 730 дней давали 20k+ свечей и ~40 сек фетча на символ.
SCAN_PERIOD_DAYS = 365
# Потолок свечей на символ в сканере: 5000 -> 20000 (полная история по
# умолчанию, = MAX_DATA_LIMIT): ~20k крипта / 13k+ форекс. Фетч берёт
# HISTORY_LIMITS[tf], а не этот потолок, при SCAN_USE_FULL_HISTORY.
SCAN_REPLAY_LIMIT = 20000
# True — сканер тянет полную историю: fetch_limit = HISTORY_LIMITS[tf]
# (максимум доступных свечей). False — откат на быстрый режим
# (fetch_limit = SCAN_REPLAY_LIMIT; верните ему 5000 для старого поведения).
SCAN_USE_FULL_HISTORY = True
# При полной истории (20k) один бэктест в ~4 раза дороже 5k, поэтому на один
# прогон разумно держать меньше комбинаций. Порог «больше лимита» в UI,
# когда чекбокс «Полная история» включён (жёсткий серверный лимит на запуск
# остаётся SCAN_MAX_COMBINATIONS).
SCAN_MAX_COMBINATIONS_FULL = 2000
# Минимум сделок хотя бы на одном из окон (train/test): комбинации с
# меньшим числом отбрасываются («повезло на 3 сделках» не проходит).
SCAN_MIN_TRADES = 30
# Сколько лучших комбинаций возвращает GET /api/scan/<run_id>.
SCAN_TOP_N = 20
# Общий грид Bollinger-стратегий (bb_reversal / bb_breakout): 4 × 3 = 12.
_BB_GRID = {"period": [15, 20, 25, 30], "std": [1.5, 2.0, 2.5]}

# Сетки параметров по стратегиям (ключи = ключи STRATEGY_MAP в
# app_pkg/ai/backtest.py). Число комбинаций стратегии — в комментарии,
# сумма по всем 12 стратегиям = 402 на символ и один ТФ.
SCAN_GRIDS = {
    "sma_cross": {
        "fast": [5, 8, 10, 12, 15, 20, 25, 30, 40, 50],
        "slow": [20, 30, 40, 50, 60, 80, 100, 150, 200]
    },
    "rsi_reversal": {
        "period": [7, 10, 14, 18, 21],
        "oversold": [20, 25, 30, 35],
        "overbought": [65, 70, 75, 80]
    },
    "macd_cross": {
        "fast": [8, 10, 12, 15],
        "slow": [20, 24, 26, 30],
        "signal": [5, 7, 9, 12]
    },
    "ema_cross": {                          # 5 × 5 = 25
        "fast": [8, 10, 12, 15, 20],
        "slow": [20, 26, 30, 40, 50]
    },
    "bb_reversal": dict(_BB_GRID),          # 4 × 3 = 12
    "bb_breakout": dict(_BB_GRID),          # 4 × 3 = 12 (тот же грид)
    "supertrend": {                         # 4 × 4 = 16
        "period": [7, 10, 14, 20],
        "multiplier": [2.0, 2.5, 3.0, 3.5]
    },
    "stoch_reversal": {                     # 3 × 2 × 3 × 3 = 54
        "k_period": [10, 14, 21],
        "d_period": [3, 5],
        "oversold": [15, 20, 25],
        "overbought": [75, 80, 85]
    },
    "stoch_cross": {                        # 3 × 2 = 6
        "k_period": [10, 14, 21],
        "d_period": [3, 5]
    },
    "cci_reversal": {                       # 3 × 3 × 3 = 27
        "period": [14, 20, 30],
        "oversold": [-150, -100, -80],      # свои зоны CCI (не 0..100)
        "overbought": [80, 100, 150]
    },
    "vwap_reversal": {                      # 4
        "vwap_threshold": [0.003, 0.005, 0.008, 0.012]  # доля отклонения
    },
    "adx_trend": {                          # 3 × 4 = 12
        "period": [10, 14, 21],
        "adx_threshold": [20, 25, 30, 35]   # порог силы тренда
    }
}
# Доступные стратегии сканера: человекочитаемая метка, список параметров и
# дефолтный грид (custom_grids из POST /api/scan мержатся поверх дефолта).
SCAN_ALLOWED_STRATEGIES = {
    "sma_cross": {
        "label": "SMA Cross",
        "params": ["fast", "slow"],
        "default_grid": SCAN_GRIDS["sma_cross"],
    },
    "rsi_reversal": {
        "label": "RSI Reversal",
        "params": ["period", "oversold", "overbought"],
        "default_grid": SCAN_GRIDS["rsi_reversal"],
    },
    "macd_cross": {
        "label": "MACD",
        "params": ["fast", "slow", "signal"],
        "default_grid": SCAN_GRIDS["macd_cross"],
    },
    "ema_cross": {
        "label": "EMA Cross",
        "params": ["fast", "slow"],
        "default_grid": SCAN_GRIDS["ema_cross"],
    },
    "bb_reversal": {
        "label": "BB Reversal",
        "params": ["period", "std"],
        "default_grid": SCAN_GRIDS["bb_reversal"],
    },
    "bb_breakout": {
        "label": "BB Breakout",
        "params": ["period", "std"],
        "default_grid": SCAN_GRIDS["bb_breakout"],
    },
    "supertrend": {
        "label": "Supertrend Follow",
        "params": ["period", "multiplier"],
        "default_grid": SCAN_GRIDS["supertrend"],
    },
    "stoch_reversal": {
        "label": "Stoch Reversal",
        "params": ["k_period", "d_period", "oversold", "overbought"],
        "default_grid": SCAN_GRIDS["stoch_reversal"],
    },
    "stoch_cross": {
        "label": "Stoch Cross",
        "params": ["k_period", "d_period"],
        "default_grid": SCAN_GRIDS["stoch_cross"],
    },
    "cci_reversal": {
        "label": "CCI Reversal",
        "params": ["period", "oversold", "overbought"],
        "default_grid": SCAN_GRIDS["cci_reversal"],
    },
    "vwap_reversal": {
        "label": "VWAP Reversal",
        "params": ["vwap_threshold"],
        "default_grid": SCAN_GRIDS["vwap_reversal"],
    },
    "adx_trend": {
        "label": "ADX Trend",
        "params": ["period", "adx_threshold"],
        "default_grid": SCAN_GRIDS["adx_trend"],
    },
}
# Разумные пределы значений параметров (валидация custom_grids). Границы
# с плавающей точкой (std / multiplier / vwap_threshold) разрешают дробные
# значения — см. app_pkg/routes/scanner.py::_validate_custom_grids.
SCAN_PARAM_LIMITS = {
    "fast": (2, 200),
    "slow": (2, 300),
    "signal": (2, 50),
    "period": (2, 50),
    # oversold/overbought общие для RSI/Stoch (5..50 / 50..95) и CCI
    # (-150..-80 / 80..150) — границы расширены под оба случая.
    "oversold": (-200, 50),
    "overbought": (50, 200),
    "std": (0.5, 4.0),              # Bollinger: ширина канала в σ
    "multiplier": (1.0, 5.0),       # Supertrend: множитель ATR
    "k_period": (5, 30),            # Stoch: период %K
    "d_period": (2, 10),            # Stoch: период %D
    "adx_threshold": (10, 50),      # ADX Trend: порог силы тренда
    "vwap_threshold": (0.001, 0.05),  # VWAP: доля отклонения от VWAP
}
# Максимум значений в одном параметре custom grid (20 -> 30: расширенные
# гриды новых стратегий, напр. stoch_reversal 3×3×3).
SCAN_MAX_CUSTOM_VALUES = 30
# Таймфреймы сканера (5m добавлен первым). Один прогон может идти по
# НЕСКОЛЬКИМ ТФ сразу (body.timeframes в POST /api/scan): объём считается
# как symbols × timeframes × Σ combos, поэтому лимит SCAN_MAX_COMBINATIONS
# расходуется быстрее — см. app_pkg/ai/scanner.py::plan_totals.
SCAN_TIMEFRAMES = ["5m", "15m", "1H", "4H", "1D"]
# ТФ по умолчанию в UI сканера (чекбоксы #scan-timeframes):
# отдаётся фронту в GET /api/scan/grids как default_timeframes.
SCAN_DEFAULT_TIMEFRAMES = ["15m", "1H"]

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
# --- Токен-диета контекста (см. app_pkg/ai/context.py) ---
# Главный ТФ: последние N свечей полностью, старше — строка-сводка.
CONTEXT_CANDLES_MAIN = 100
# Второстепенные ТФ: только последние N свечей.
CONTEXT_CANDLES_OTHER = 20
# Хвост значений каждого индикатора (вместо единственного последнего).
CONTEXT_INDICATOR_TAIL = 10
# Максимум рисунков в одном AI-ответе.
CONTEXT_DRAWINGS_MAX = 3
# ТФ, которые не тащим в секции Candles, если main_tf их «перекрывает»
# (при main=15m минутки/пятиминутки — шум). В Trends остаются все ТФ.
CONTEXT_SKIP_TF = ("1m", "5m")

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
CHAT_MEMORY_MESSAGES = 5
# Обрезка старых сообщений чата в промпте: первые 150 + "..." + последние 40.
CHAT_MESSAGE_MAX_CHARS = 200
# Максимум рисунков в одном AI-ответе (токен-диета: было 5).
MAX_AI_DRAWABLES = 3

# --------------------------------------------- лимиты токенов LLM (токен-диета)
# max_tokens ответа модели: analysis-запросы и chat отдельно.
LLM_MAX_TOKENS_ANALYSIS = 800   # было 2000
LLM_MAX_TOKENS_CHAT = 400       # было 2000
# Обрезка reason в нормализации ответа (было 500).
LLM_REASON_MAX_CHARS = 250
# TTL кеша ответов LLM (сек): повторный analysis-запрос с тем же контекстом
# (та же последняя свеча) в течение TTL идёт из кеша, без похода в сеть.
LLM_RESPONSE_CACHE_TTL = 300
AI_MAX_CANDLES_BY_TF = {"1m": 50, "5m": 100, "15m": 150, "1H": 200, "4H": 120, "1D": 80}

# ------------------------------------------------------------- метрики
METRICS_QUANTILES = (0.5, 0.95)