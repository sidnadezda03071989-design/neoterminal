# -*- coding: utf-8 -*-
"""Тесты Market Snapshot: честные данные для ИИ (app_pkg/data/market_snapshot.py).

Проверяем:
  - sentiment — null-схема с ok:False (подменённый crowd-геттер -> {}),
    никаких 0/50-заглушек;
  - scanner_edge — profit_factor/trades/test_sharpe из db_get_best_scan_stats
    (monkeypatch), пропущенные поля -> null, без скана -> ok:False;
  - atr_percentile: растущая волатильность -> >0.7; константная -> 0.3-0.7;
  - dist_to_high_pct == 0.0 на 50-барном максимуме;
  - session_info: 14:00 UTC будни -> london_ny/overlap 1; суббота -> weekend;
    форекс в субботу market_open 0; крипта в субботу market_open 1;
  - мало баров (< 100) -> atr_percentile null без exception;
  - clock в get_raw_market_data считается по ПОСЛЕДНЕЙ свече среза (реплей),
    ts_override форсирует время для тестов;
  - весь снимок — валидный JSON из чисел/null/bool/строк/вложенных dict.
"""

import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from app_pkg.data import market_snapshot as ms

_SAT_TS = int(datetime(2026, 2, 21, 14, 0, tzinfo=timezone.utc).timestamp())
_MON_TS = int(datetime(2026, 2, 16, 14, 0, tzinfo=timezone.utc).timestamp())


def _mk_df(n=120, end_ts=_MON_TS, step=900, base=100.0,
           growing=False, vol=2.0):
    """Синтетический OHLCV-DF с таймстемпами до end_ts включительно."""
    ts = pd.to_datetime([end_ts - i * step for i in range(n)][::-1],
                        unit="s", utc=True)
    if growing:
        amp = [(i + 1) * 0.05 for i in range(n)]
        highs = [base + a for a in amp]
        lows = [base - a for a in amp]
    else:
        highs = [base + vol] * n
        lows = [base - vol] * n
    return pd.DataFrame({
        "timestamp": ts,
        "open": [base] * n,
        "high": highs,
        "low": lows,
        "close": [base] * n,
        "volume": [10.0] * n,
    })


@pytest.fixture()
def no_scan(monkeypatch):
    """Снимок герметичен: ни БД, ни сети (Binance/FRED/Finnhub не трогаем).

    scanner_edge не лезет в БД; доп. блоки (sentiment/macro/calendar/
    derivatives/news_sentiment) подменены на геттеры, возвращающие {} / None —
    блоки уходят в null-схемы / «не применимо».
    """
    monkeypatch.setattr(ms.db, "db_get_best_scan_stats", lambda s, tf: None)
    monkeypatch.setattr(ms, "get_crowd_snapshot", lambda s: {})
    monkeypatch.setattr(ms, "get_macro_snapshot", lambda s: None)
    monkeypatch.setattr(ms, "get_econ_calendar",
                        lambda c, *a, **k: None)
    monkeypatch.setattr(ms, "get_derivatives_snapshot", lambda s: {})
    monkeypatch.setattr(ms, "get_news_sentiment", lambda s, **k: None)


# ------------------------------------------------------------- sentiment
def test_sentiment_is_null_schema_with_ok_false(monkeypatch, no_scan):
    """Хвост-заглушка 0/0/50 удалена: честный «данных нет»."""
    monkeypatch.setattr(ms, "get_series_df", lambda *a, **k: pd.DataFrame())
    raw = ms.get_raw_market_data("BTCUSDT", "15m")
    assert raw["sentiment"] == {
        "ls_ratio": None, "long_pct": None, "fear_greed": None, "ok": False,
    }
    # данные пустые -> technicals и clock тоже в null-схеме, без exception
    assert raw["technicals"]["ok"] is False
    assert raw["clock"]["ok"] is False


# ---------------------------------------------------------- scanner_edge
def test_scanner_edge_has_extra_fields(monkeypatch):
    """profit_factor/trades/test_sharpe приходят из db_get_best_scan_stats."""
    fake = {
        "strategy": "rsi_reversal",
        "params": {"period": 14, "overbought": 70},
        "train": {"sharpe": 5.0},
        "test": {"sharpe": 4.0, "winrate": 0.606, "max_dd": 0.0076,
                 "profit_factor": 2.2, "trades": 80},
        "combined_sharpe": 4.5,
        "winrate": 0.606,
        "sharpe": 4.0,
    }
    monkeypatch.setattr(ms.db, "db_get_best_scan_stats", lambda s, tf: fake)
    out = ms._scanner_edge("BTCUSDT", "15m")

    assert out["profit_factor"] == 2.2
    assert out["trades"] == 80
    assert out["test_sharpe"] == 4.0
    assert out["sharpe"] == 4.0
    assert out["winrate"] == 0.606
    assert out["ok"] is True
    required = {"winrate", "sharpe", "max_dd", "profit_factor",
                "trades", "test_sharpe", "params"}
    assert required <= set(out)  # ничего из старых полей не удалено


def test_scanner_edge_missing_fields_are_null(monkeypatch):
    """Нет поля в test-ответе -> null (не 0!)."""
    monkeypatch.setattr(ms.db, "db_get_best_scan_stats",
                        lambda s, tf: {"strategy": "x", "params": {"a": "z"},
                                       "test": {}})
    out = ms._scanner_edge("EURUSD", "1H")
    assert out["profit_factor"] is None
    assert out["trades"] is None
    assert out["test_sharpe"] is None
    assert out["winrate"] is None
    assert out["params"] == {"a": None}  # нечисловые params -> null


def test_scanner_edge_ok_false_without_scan(monkeypatch):
    """Скана не было — вся схема null + ok:False."""
    monkeypatch.setattr(ms.db, "db_get_best_scan_stats", lambda s, tf: None)
    out = ms._scanner_edge("XAUUSD", "1H")
    assert out["ok"] is False
    for key in ("winrate", "sharpe", "max_dd", "profit_factor",
                "trades", "test_sharpe"):
        assert out[key] is None


# ------------------------------------------------------- atr_percentile
def test_atr_percentile_rising_volatility(monkeypatch, no_scan):
    """Растущая волатильность -> текущий ATR выше >70% истории."""
    monkeypatch.setattr(ms, "get_series_df",
                        lambda *a, **k: _mk_df(120, growing=True))
    raw = ms.get_raw_market_data("EURUSD", "15m")
    pct = raw["technicals"]["atr_percentile"]
    assert pct is not None
    assert pct > 0.7


def test_atr_percentile_constant_volatility(monkeypatch, no_scan):
    """Константная волатильность -> перцентиль в нейтральной зоне."""
    monkeypatch.setattr(ms, "get_series_df",
                        lambda *a, **k: _mk_df(120, growing=False))
    pct = ms.get_raw_market_data("EURUSD", "15m")["technicals"]["atr_percentile"]
    assert pct is not None
    assert 0.3 <= pct <= 0.7


# --------------------------------------------------------- dist_*_pct
def test_dist_to_high_zero_at_50_bar_max(monkeypatch, no_scan):
    """close == максимум за 50 баров -> dist_to_high_pct строго 0.0."""
    n = 120
    ts = pd.to_datetime([_MON_TS - i * 900 for i in range(n)][::-1],
                        unit="s", utc=True)
    close = [1.0] * n
    close[-1] = 200.0
    df = pd.DataFrame({
        "timestamp": ts,
        "open": [1.0] * n,
        "high": [200.0] * n,
        "low": [0.5] * n,
        "close": close,
        "volume": [10.0] * n,
    })
    monkeypatch.setattr(ms, "get_series_df", lambda *a, **k: df)
    tech = ms.get_raw_market_data("EURUSD", "15m")["technicals"]
    assert tech["dist_to_high_pct"] == 0.0


def test_dist_to_low_zero_at_50_bar_min(monkeypatch, no_scan):
    """close == минимум за 50 баров -> dist_to_low_pct строго 0.0."""
    n = 120
    ts = pd.to_datetime([_MON_TS - i * 900 for i in range(n)][::-1],
                        unit="s", utc=True)
    close = [100.0] * n
    close[-1] = 0.5
    df = pd.DataFrame({
        "timestamp": ts,
        "open": [1.0] * n,
        "high": [200.0] * n,
        "low": [0.5] * n,
        "close": close,
        "volume": [10.0] * n,
    })
    monkeypatch.setattr(ms, "get_series_df", lambda *a, **k: df)
    tech = ms.get_raw_market_data("EURUSD", "15m")["technicals"]
    assert tech["dist_to_low_pct"] == 0.0


def test_many_bars_all_technicals_present(monkeypatch, no_scan):
    """Достаточно баров -> technicals отдаёт 8+ полей (Уровень 3)."""
    monkeypatch.setattr(ms, "get_series_df",
                        lambda *a, **k: _mk_df(120, growing=True))
    tech = ms.get_raw_market_data("EURUSD", "15m")["technicals"]
    keys = set(tech) - {"ok"}
    assert len(keys) >= 8
    for field in ("atr_percentile", "dist_to_high_pct", "dist_to_low_pct"):
        assert field in tech


def test_atr_percentile_null_with_few_bars(monkeypatch, no_scan):
    """Меньше 100 баров -> atr_percentile null, без exception."""
    monkeypatch.setattr(ms, "get_series_df",
                        lambda *a, **k: _mk_df(60, growing=True))
    tech = ms.get_raw_market_data("EURUSD", "15m")["technicals"]
    assert tech["atr_percentile"] is None


# ---------------------------------------------------------- session_info
def test_session_info_london_ny_overlap():
    """Пн 14:00 UTC -> london_ny + overlap 1 (в будни)."""
    info = ms.session_info(_MON_TS, "EURUSD")
    assert info["hour_utc"] == 14
    assert info["dow"] == 0
    assert info["session"] == "london_ny"
    assert info["london_ny_overlap"] == 1


def test_session_info_saturday_weekend():
    """Сб 14:00 UTC -> weekend, overlap 0, форекс закрыт."""
    info = ms.session_info(_SAT_TS, "EURUSD")
    assert info["session"] == "weekend"
    assert info["london_ny_overlap"] == 0
    assert info["market_open"] == 0


def test_session_info_crypto_open_on_weekend():
    """Крипта в субботу -> market_open 1."""
    info = ms.session_info(_SAT_TS, "BTCUSDT")
    assert info["market_open"] == 1
    assert info["session"] == "weekend"


def test_session_info_off_hours():
    """23:00 UTC будни -> off_hours."""
    ts = int(datetime(2026, 2, 16, 23, 0, tzinfo=timezone.utc).timestamp())
    info = ms.session_info(ts, "USDJPY")
    assert info["session"] == "off_hours"
    assert info["london_ny_overlap"] == 0
    assert info["market_open"] == 1  # форекс в будни открыт


# ------------------------------------------------------- clock в snapshot
def test_clock_uses_last_candle_ts(monkeypatch, no_scan):
    """clock считается по последней свече среза, а не по системным часам.

    Срез заканчивается субботой 14:00 UTC — снимок должен получить weekend,
    даже если тест реально запущен в любое другое время.
    """
    monkeypatch.setattr(ms, "get_series_df",
                        lambda *a, **k: _mk_df(120, end_ts=_SAT_TS))
    raw = ms.get_raw_market_data("EURUSD", "15m")
    assert raw["clock"]["dow"] == 5
    assert raw["clock"]["session"] == "weekend"
    assert raw["clock"]["market_open"] == 0
    assert raw["clock"]["ok"] is True


def test_clock_upto_sec_uses_slice_ts(monkeypatch, no_scan):
    """Барьер реплея upto_sec: clock по ПОСЛЕДНЕЙ свече среза (пт).

    Полный ряд заканчивается субботой, но срез обрезан до пятницы 12:00 —
    значит session "london", рынок открыт.
    """
    fri = int(datetime(2026, 2, 20, 12, 0, tzinfo=timezone.utc).timestamp())
    monkeypatch.setattr(ms, "get_series_df",
                        lambda *a, **k: _mk_df(120, end_ts=_SAT_TS))
    raw = ms.get_raw_market_data("EURUSD", "15m", upto_sec=fri)
    assert raw["clock"]["session"] == "london"
    assert raw["clock"]["market_open"] == 1
    assert raw["clock"]["london_ny_overlap"] == 0


def test_clock_ts_override(monkeypatch, no_scan):
    """ts_override форсирует время блока clock для тестов."""
    monkeypatch.setattr(ms, "get_series_df",
                        lambda *a, **k: _mk_df(120, end_ts=_MON_TS))
    raw = ms.get_raw_market_data("EURUSD", "15m", ts_override=_SAT_TS)
    assert raw["clock"]["session"] == "weekend"
    assert raw["clock"]["market_open"] == 0


# ------------------------------------------------------- JSON-контракт
def test_snapshot_is_strict_json(monkeypatch, no_scan):
    """Весь снимок сериализуется в JSON и состоит из чисел/null/bool/строк/dict."""
    monkeypatch.setattr(ms, "get_series_df",
                        lambda *a, **k: _mk_df(120, growing=True))
    raw = ms.get_raw_market_data("EURUSD", "15m")

    dumped = json.dumps(raw)          # не должно упасть на NaN/Inf
    assert json.loads(dumped) == raw  # round-trip идентичен

    def check(node):
        if isinstance(node, dict):
            for value in node.values():
                check(value)
        else:
            assert node is None or isinstance(node, (int, float, bool, str)), \
                f"запрещённый тип в JSON: {type(node)} ({node!r})"

    check(raw)
    assert raw["technicals"]["ok"] is True
    assert raw["clock"]["ok"] is True