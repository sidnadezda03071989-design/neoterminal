"""CBR-база для Харона (Этап 1+): запись снимков + backfill + token-diet API.

Пакет хранит компактные 44-мерные векторы рыночных снимков (те же числа,
что видит нейросеть Харона: technicals/volume/trend/momentum/volatility/
regime/divergence/candle/clock) и асинхронно размечает их triple-barrier
outcome по конвенции app_pkg.ml.labels (SL-приоритет). Запись — из хуков
pipeline (ai_backtest/market_snapshot, гейт config.CBR_ENABLED) и из
оркестратора scripts/cbr_build.py; чтение — компактный API для промпта LLM
(query_api), выдающий статистику вместо сырых чисел (token-diet).
"""