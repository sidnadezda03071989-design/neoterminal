"""Тесты ИИ-интеграции статистики сканера (app_pkg/ai/context.py и
app_pkg/ai/ai_backtest.py::_scanner_block).

Проверяем:
  - format_scanner_stats: пустая строка без данных; формулировки ТЗ для
    BTCUSDT (Sharpe 4.5 -> «Доверяй сигналам…») и USDJPY (Sharpe -4.0 ->
    «НЕ доверяй…»);
  - build_multi_tf_context включает секцию Scanner stats (=> её получают
    Харон, чат, AI Backtest);
  - _scanner_block (AI Backtest) даёт тот же формат;
  - в replay (upto_sec) секция по-прежнему отдаёт статистику сканера —
    скан-статистика не зависит от среза свечей.

Герметичность: DataDir/DB_PATH вычисляются в app_pkg.config ПРИ ИМПОРТЕ
(os.getenv("DATA_DIR")), поэтому до импорта модулей приложения DATA_DIR
переключается на временную папку. Тесты работают на изолированной SQLite и
проходят при ЛЮБОМ содержимом dev-БД (data/data.db).
"""

import os
import tempfile

import pandas as pd
import pytest

_DATA_DIR = tempfile.mkdtemp(prefix="neoterminal_test_")
os.environ["DATA_DIR"] = _DATA_DIR

from app_pkg import db
from app_pkg.ai import ai_backtest as aibt
from app_pkg.ai import context as ctx

T = 1_700_000_000
STEP = 900
_RUN_IDS = []


def _mk_df(n=60, end_ts=T, step=STEP):
    ts = [end_ts - i * step for i in range(n)][::-1]
    return pd.DataFrame({
        "timestamp": pd.to_datetime(ts, unit="s", utc=True),
        "open": [1.0] * n,
        "high": [2.0] * n,
        "low": [0.5] * n,
        "close": [1.5] * n,
        "volume": [10.0] * n,
    })


def _seed(symbol="BTCUSDT", tf="15m", train_sharpe=5.0, test_sharpe=4.0,
          combined=4.5, winrate=0.606):
    run_id = f"aistat{symbol}{tf}".replace("_", "")[:30]
    db.db_save_scan_result(
        run_id, symbol, tf, "rsi_reversal",
        {"period": 14, "oversold": 35, "overbought": 70},
        {"sharpe": train_sharpe, "winrate": 0.62, "max_dd": 0.01,
         "profit_factor": 2.5, "trades": 120},
        {"sharpe": test_sharpe, "winrate": winrate, "max_dd": 0.0076,
         "profit_factor": 2.2, "trades": 80},
        combined, 200)
    _RUN_IDS.append(run_id)


@pytest.fixture(autouse=True)
def _cleanup_db():
    """Изолированная SQLite: инициализируем схему (init_db), после теста
    чистим засеянные run_id — каждый тест стартует с пустой БД."""
    db._get_db()
    yield
    for run_id in _RUN_IDS:
        db.db_clear_scan_run(run_id)
    _RUN_IDS.clear()


# --------------------------------------------------- format_scanner_stats
def test_format_scanner_stats_empty_without_data():
    """Скана не было — секция пустая (не попадает в промпт)."""
    assert ctx.format_scanner_stats("BTCUSDT", "15m") == ""


def test_format_scanner_stats_trust_btc():
    """BTCUSDT: Sharpe 4.5 — «Доверяй сигналам rsi_reversal»."""
    _seed()
    out = ctx.format_scanner_stats("BTCUSDT", "15m")
    assert "=== Scanner stats (BTCUSDT 15m) ===" in out
    assert "Статистика сканера для BTCUSDT 15m" in out
    assert "rsi_reversal" in out
    # params сериализуются отсортированно (sorted по ключам)
    assert "overbought=70, oversold=35, period=14" in out
    assert "combined Sharpe 4.5" in out
    assert "winrate=60.6%" in out
    assert "Доверяй сигналам rsi_reversal" in out


def test_format_scanner_stats_distrust_usdjpy():
    """USDJPY: Sharpe -4.0 — «НЕ доверяй сигналам … уровень S/R»."""
    _seed(symbol="USDJPY", tf="15m", train_sharpe=-2.0, test_sharpe=-4.0,
          combined=-4.0, winrate=0.3)
    out = ctx.format_scanner_stats("USDJPY", "15m")
    assert "=== Scanner stats (USDJPY 15m) ===" in out
    assert "combined Sharpe -4.0" in out
    assert "НЕ доверяй сигналам rsi_reversal" in out
    assert "поддержки/сопротивления" in out


# ------------------------------------------------ build_multi_tf_context
def test_multi_tf_context_includes_scanner_stats(monkeypatch):
    """build_multi_tf_context: секция Scanner stats попадает в контекст."""
    df = _mk_df()
    monkeypatch.setattr(ctx, "get_series_df", lambda *a, **k: df)
    _seed()
    out = ctx.build_multi_tf_context("BTCUSDT", "15m")
    assert "=== Scanner stats (BTCUSDT 15m) ===" in out
    assert "Доверяй сигналам rsi_reversal" in out


def test_multi_tf_context_no_section_without_scan(monkeypatch):
    df = _mk_df()
    monkeypatch.setattr(ctx, "get_series_df", lambda *a, **k: df)
    out = ctx.build_multi_tf_context("BTCUSDT", "15m")
    assert "=== Scanner stats" not in out


def test_multi_tf_context_replay_still_has_stats(monkeypatch):
    """Replay (upto_sec): свечи обрезаны, статистика сканера остаётся."""
    df = _mk_df()
    monkeypatch.setattr(ctx, "get_series_df", lambda *a, **k: df)
    _seed()
    upto = T - 10 * STEP
    out = ctx.build_multi_tf_context("BTCUSDT", "15m", upto_sec=upto)
    assert "=== Scanner stats (BTCUSDT 15m) ===" in out


# ------------------------------------------------------- AI Backtest блок
def test_ai_backtest_scanner_block_trust():
    """_scanner_block: формат ТЗ + рекомендация «Доверяй сигналам…»."""
    _seed()
    out = aibt._scanner_block("BTCUSDT", "15m")
    assert "=== Scanner stats (BTCUSDT 15m) ===" in out
    assert "Статистика сканера для BTCUSDT 15m" in out
    assert "combined Sharpe 4.5" in out
    assert "Доверяй сигналам rsi_reversal" in out


def test_ai_backtest_scanner_block_distrust():
    _seed(symbol="USDJPY", tf="15m", train_sharpe=-2.0, test_sharpe=-4.0,
          combined=-4.0, winrate=0.3)
    out = aibt._scanner_block("USDJPY", "15m")
    assert "НЕ доверяй сигналам rsi_reversal" in out


def test_ai_backtest_scanner_block_empty_without_scan():
    assert aibt._scanner_block("BTCUSDT", "15m") == ""


def test_levels_context_includes_scanner_stats(monkeypatch):
    """Промпт AI Backtest целиком содержит секцию статистики сканера."""
    df = _mk_df()
    monkeypatch.setattr("app_pkg.ai.context.get_series_df",
                        lambda *a, **k: df)
    _seed()
    out = aibt.build_levels_context("BTCUSDT", "15m", None, current_price=1.5)
    assert "=== Scanner stats (BTCUSDT 15m) ===" in out
