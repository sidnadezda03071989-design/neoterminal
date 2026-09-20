"""Пакет источников данных: binance / yfinance / mt5 + общая обёртка fetch.

Ответственность модулей:
- binance — рыночные данные криптовалют (REST klines, с fallback на api1);
- yfinance — форекс/металлы через Yahoo Finance;
- mt5  — данные из терминала MetaTrader 5 (ленивый импорт);
- fetch — единая точка получения OHLCV с кешем;
- live  — фоновые потоки live-баров (Binance WS + поллинг форекса).
"""