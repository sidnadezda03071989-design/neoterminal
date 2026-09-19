# -*- coding: utf-8 -*-
"""Тесты grid-search сканера стратегий (app_pkg/ai/scanner.py).

Все тесты — с моком run_backtest и get_replay_df (без сети). Проверяем:
генерацию комбинаций (полная и срезка по лимиту, seed=42), отсев по
SCAN_MIN_TRADES, combined_sharpe = min(train, test), прогон run_scan
с записью в БД, роуты /api/scan (POST), /api/scan/<run_id> и export.csv.
"""

import threading

import pandas as pd
import pytest

from app_pkg import config, create_app, db
from app_pkg.ai import scanner

T = 1_700_000_000           # конец истории в моках
STEP = 900                  # 15m

_RUN_IDS = []               # run_id для чистки scan_results после тестов


def _mk_df(n=1000, end_ts=T, step=STEP):
    """Свечи шагом step, возрастающие по времени (как в get_replay_df)."""
    ts = [end_ts - i * step for i in range(n)][::-1]
    return pd.DataFrame({
        "timestamp": pd.to_datetime(ts, unit="s", utc=True),
        "open": [1.0] * n,
        "high": [2.0] * n,
        "low": [0.5] * n,
        "close": [1.5 + i for i in range(n)],
        "volume": [10.0] * n,
    })


def _fake_run_backtest(train_sharpe=1.5, test_sharpe=0.8, trades=50,
                       calls=None):
    """Мок run_backtest: train-окно (len(df) > 500 из 1000) -> train_sharpe.

    trades >= SCAN_MIN_TRADES по умолчанию, чтобы комбинации проходили отсев.
    """
    def fake(symbol, tf, from_sec, to_sec, strategy_name, params,
             initial_cash=10000, replay_limit=None, df=None, ind=None):
        if calls is not None:
            calls.append(len(df) if df is not None else -1)
        sharpe = train_sharpe if (df is not None and len(df) > 500) \
            else test_sharpe
        return {
            "total_return": 0.1,
            "sharpe_ratio": sharpe,
            "max_drawdown": 0.05,
            "total_trades": trades,
            "win_rate": 0.6,
            "trades": [{"pnl": 10.0 if i % 2 == 0 else -5.0}
                       for i in range(min(trades, 20))],
            "equity_curve": [],
        }
    return fake


def _split(df):
    split = int(len(df) * config.SCAN_TRAIN_SPLIT)
    return (df.iloc[:split].reset_index(drop=True),
            df.iloc[split:].reset_index(drop=True))


@pytest.fixture(autouse=True)
def _cleanup_db():
    """Чистим scan_results после каждого теста (общая SQLite)."""
    yield
    for run_id in _RUN_IDS:
        db.db_clear_scan_run(run_id)
    _RUN_IDS.clear()


# ---------------------------------------------------- _generate_combinations
def test_generate_combinations_full_grid():
    """sma_cross: 10 fast × 9 slow = 90 комбинаций, все — dict."""
    combos = scanner._generate_combinations(
        "sma_cross", config.SCAN_GRIDS["sma_cross"])
    assert len(combos) == 10 * 9
    assert all(isinstance(c, dict) for c in combos)
    assert {"fast", "slow"} == set(combos[0].keys())


def test_generate_combinations_respects_limit_and_seed():
    """90000 комбинаций -> срезка до SCAN_MAX_COMBINATIONS (50000), seed=42
    даёт воспроизводимую подвыборку."""
    big = {"a": list(range(300)), "b": list(range(300))}
    first = scanner._generate_combinations("x", big)
    second = scanner._generate_combinations("x", big)
    assert len(first) == config.SCAN_MAX_COMBINATIONS
    assert first == second


# ------------------------------------------------------------ _run_single
def test_run_single_returns_none_when_too_few_trades(monkeypatch):
    """total_trades < SCAN_MIN_TRADES -> комбинация отбрасывается (None)."""
    df = _mk_df(1000)
    monkeypatch.setattr(
        scanner, "run_backtest", _fake_run_backtest(trades=10))
    df_train, df_test = _split(df)
    res = scanner._run_single("BTCUSDT", "15m", "sma_cross",
                              {"fast": 5, "slow": 20}, df_train, df_test)
    assert res is None


def test_run_single_combined_sharpe_is_min(monkeypatch):
    """combined_sharpe = min(train_sharpe, test_sharpe)."""
    df = _mk_df(1000)
    calls = []
    monkeypatch.setattr(
        scanner, "run_backtest",
        _fake_run_backtest(train_sharpe=1.5, test_sharpe=0.8, calls=calls))
    df_train, df_test = _split(df)
    res = scanner._run_single("BTCUSDT", "15m", "sma_cross",
                              {"fast": 5, "slow": 20}, df_train, df_test)
    assert res["combined_sharpe"] == 0.8
    assert res["train"]["sharpe"] == 1.5
    assert res["test"]["sharpe"] == 0.8
    assert res["total_trades"] == 100  # 50 train + 50 test
    # profit_factor из мока: 10 сделок +10 и 10 сделок -5 в окне
    assert res["train"]["profit_factor"] == 2.0
    # два вызова: train (70% от 1000) и test (30%) — длины окон
    assert len(calls) == 2
    assert calls == [700, 300]


# ---------------------------------------------------------------- run_scan
def test_run_scan_writes_rows_to_db(monkeypatch):
    """2 символа × 1 стратегия × 5 параметров = 10 записей в БД."""
    monkeypatch.setattr(scanner, "get_replay_df",
                        lambda *a, **k: _mk_df(1000))
    monkeypatch.setattr(scanner, "run_backtest", _fake_run_backtest())
    monkeypatch.setitem(config.SCAN_GRIDS, "sma_cross",
                        {"fast": [5, 10, 15, 20, 25]})
    events = []
    monkeypatch.setattr(scanner, "_ws_push",
                        lambda event, data: events.append((event, data)))

    run_id = scanner.run_scan(["BTCUSDT", "ETHUSDT"], "15m", ["sma_cross"])
    _RUN_IDS.append(run_id)

    rows = db.db_get_scan_results(run_id, limit=1000)
    assert len(rows) == 10
    assert {r["symbol"] for r in rows} == {"BTCUSDT", "ETHUSDT"}
    assert all(r["strategy"] == "sma_cross" for r in rows)
    assert all(r["train"] and r["test"] for r in rows)
    # SSE-прогресс: done доходит до total (10), финальное событие current=None
    assert events
    assert all(ev == "scan_progress" for ev, _ in events)
    assert events[-1][1]["done"] == 10
    assert events[-1][1]["total"] == 10
    assert events[-1][1]["current"] is None


# ------------------------------------------------------------------ роуты
@pytest.fixture()
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_api_scan_start_returns_run_id(client, monkeypatch):
    """POST /api/scan -> {run_id, status: started}, run_scan вызван в фоне."""
    called = {}
    done_ev = threading.Event()

    def fake_run_scan(symbols, timeframe, strategies, run_id=None, grids=None):
        called.update(symbols=symbols, timeframe=timeframe,
                      strategies=strategies, run_id=run_id, grids=grids)
        done_ev.set()

    monkeypatch.setattr(scanner, "run_scan", fake_run_scan)
    resp = client.post("/api/scan", json={
        "symbols": ["BTCUSDT"], "timeframe": "15m",
        "strategies": ["sma_cross"],
    })
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "started"
    assert data["run_id"]
    assert done_ev.wait(timeout=5)  # фоновый поток отработал
    assert called["symbols"] == ["BTCUSDT"]
    assert called["timeframe"] == ["15m"]  # нормализовано в список _normalize_timeframes
    assert called["strategies"] == ["sma_cross"]
    assert called["grids"] is None
    assert called["run_id"] == data["run_id"]


def test_api_scan_validates_input(client):
    """Невалидный символ/таймфрейм/стратегия -> 400, скан не стартует."""
    assert client.post("/api/scan", json={
        "symbols": ["NOPE"], "timeframe": "15m",
        "strategies": ["sma_cross"]}).status_code == 400
    assert client.post("/api/scan", json={
        "symbols": ["BTCUSDT"], "timeframe": "2h",
        "strategies": ["sma_cross"]}).status_code == 400
    assert client.post("/api/scan", json={
        "symbols": ["BTCUSDT"], "timeframe": "15m",
        "strategies": ["nope"]}).status_code == 400
    assert client.post("/api/scan", json={
        "symbols": [], "timeframe": "15m",
        "strategies": ["sma_cross"]}).status_code == 400


def test_api_scan_results_sorted_top(client):
    """GET /api/scan/<run_id> — топ по combined_sharpe DESC."""
    run_id = "testrun" + "0" * 25
    _RUN_IDS.append(run_id)
    for i, sharpe in enumerate([1.0, 2.0, 0.5]):
        db.db_save_scan_result(
            run_id, "BTCUSDT", "15m", "sma_cross", {"fast": 5 + i},
            {"sharpe": sharpe}, {"sharpe": sharpe}, sharpe, 60)

    resp = client.get(f"/api/scan/{run_id}")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["run_id"] == run_id
    assert data["count"] == 3
    sharpes = [r["combined_sharpe"] for r in data["results"]]
    assert sharpes == sorted(sharpes, reverse=True)
    assert sharpes[0] == 2.0
    # JSON-поля распарсены обратно
    assert data["results"][0]["params"] == {"fast": 6}


def test_api_scan_csv_columns(client):
    """CSV содержит все нужные колонки и данные."""
    run_id = "testcsv" + "0" * 25
    _RUN_IDS.append(run_id)
    db.db_save_scan_result(
        run_id, "BTCUSDT", "15m", "sma_cross", {"fast": 5, "slow": 20},
        {"sharpe": 1.5, "winrate": 0.6, "max_dd": 0.05,
         "total_return": 0.1, "profit_factor": 2.0, "trades": 50},
        {"sharpe": 0.8, "winrate": 0.55, "max_dd": 0.07,
         "total_return": 0.08, "profit_factor": 1.5, "trades": 40},
        0.8, 90)

    resp = client.get(f"/api/scan/{run_id}/export.csv")
    assert resp.status_code == 200
    assert "text/csv" in resp.mimetype
    text = resp.get_data(as_text=True)
    header = text.splitlines()[0]
    for col in ["symbol", "strategy", "params", "train_sharpe",
                "test_sharpe", "combined_sharpe", "winrate", "max_dd",
                "profit_factor", "trades"]:
        assert col in header, f"нет колонки {col}"
    assert "BTCUSDT" in text
    assert "sma_cross" in text
    # test (out-of-sample) значения попали в winrate/max_dd/profit_factor
    row = text.splitlines()[1]
    assert "0.8" in row and "0.55" in row and "0.07" in row
