"""Тесты GET /api/scanner/stats (карточка «Статистика стратегий» во вкладке
«Новости»).

Синтетические строки scan_results в общую SQLite (как в test_scanner.py),
после каждого теста — чистка. Без сети. Проверяем:
  - {} (200) когда скана по паре symbol+timeframe ещё не было;
  - выбор ЛУЧШЕЙ комбинации по combined_sharpe;
  - изоляцию по symbol/timeframe (другой ТФ -> {});
  - убыточную пару (combined_sharpe < 0) — для карточки USDJPY;
  - валидацию параметров (400 на мусор).
"""

import pytest

from app_pkg import create_app, db

_SYM = "BTCUSDT"
_TF = "15m"
_RUN_IDS = []


def _seed(run_id, symbol=_SYM, tf=_TF, train_sharpe=5.0, test_sharpe=4.0,
          combined=4.54, winrate=0.606, max_dd=0.0076):
    """Одна комбинация rsi_reversal в scan_results (train/test windows)."""
    db.db_save_scan_result(
        run_id, symbol, tf, "rsi_reversal",
        {"period": 14, "oversold": 35, "overbought": 70},
        {"sharpe": train_sharpe, "winrate": 0.62, "max_dd": 0.01,
         "profit_factor": 2.5, "trades": 120},
        {"sharpe": test_sharpe, "winrate": winrate, "max_dd": max_dd,
         "profit_factor": 2.2, "trades": 80},
        combined, 200)
    _RUN_IDS.append(run_id)


@pytest.fixture(autouse=True)
def _cleanup_db():
    """Чистим scan_results после каждого теста (общая SQLite)."""
    yield
    for run_id in _RUN_IDS:
        db.db_clear_scan_run(run_id)
    _RUN_IDS.clear()


@pytest.fixture()
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


# ------------------------------------------------------------------ пусто
@pytest.mark.debt
def test_scanner_stats_empty_returns_empty_dict(client):
    """Без прогона сканера — 200 и {}: карточка просто скрыта."""
    resp = client.get(
        f"/api/scanner/stats?symbol={_SYM}&timeframe={_TF}")
    assert resp.status_code == 200
    assert resp.get_json() == {}


def test_scanner_stats_requires_symbol(client):
    resp = client.get("/api/scanner/stats?timeframe=15m")
    assert resp.status_code == 400
    assert "symbol" in resp.get_json()["error"]


def test_scanner_stats_invalid_symbol(client):
    resp = client.get("/api/scanner/stats?symbol=FAKEUSDT&timeframe=15m")
    assert resp.status_code == 400


# ------------------------------------------------------------- лучший edge
def test_scanner_stats_best_by_combined_sharpe(client):
    """Из двух комбинаций отдаётся лучшая по combined_sharpe."""
    _seed("bestlo" + "0" * 24, combined=2.0, test_sharpe=2.5)
    _seed("besthi" + "0" * 24, combined=4.54, test_sharpe=4.0)
    resp = client.get(f"/api/scanner/stats?symbol={_SYM}&timeframe={_TF}")
    assert resp.status_code == 200
    d = resp.get_json()
    assert d["strategy"] == "rsi_reversal"
    assert d["strategy_label"] == "RSI Reversal 14/35/70"
    assert d["params"] == {"period": 14, "oversold": 35, "overbought": 70}
    assert d["train_sharpe"] == pytest.approx(5.0)
    assert d["test_sharpe"] == pytest.approx(4.0)
    assert d["combined_sharpe"] == pytest.approx(4.54)
    assert d["winrate"] == pytest.approx(0.606)
    assert d["max_dd"] == pytest.approx(0.0076)
    assert d["symbol"] == _SYM and d["timeframe"] == _TF


@pytest.mark.debt
def test_scanner_stats_timeframe_isolated(client):
    """Есть результат на 15m — запрос по 1H возвращает {}."""
    _seed("tfiso" + "0" * 26)
    resp = client.get(f"/api/scanner/stats?symbol={_SYM}&timeframe=1H")
    assert resp.status_code == 200
    assert resp.get_json() == {}


def test_scanner_stats_losing_pair_negative_sharpe(client):
    """USDJPY-кейс: лучшая комбинация убыточна (combined_sharpe < 0)."""
    _seed("loss" + "0" * 27, symbol="USDJPY", train_sharpe=-2.0,
          test_sharpe=-4.0, combined=-4.0, winrate=0.30, max_dd=0.12)
    resp = client.get("/api/scanner/stats?symbol=USDJPY&timeframe=15m")
    assert resp.status_code == 200
    d = resp.get_json()
    assert d["combined_sharpe"] == pytest.approx(-4.0)
    assert d["test_sharpe"] == pytest.approx(-4.0)
    assert d["verdict"] == "losing"
