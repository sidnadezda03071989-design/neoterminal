"""Walk-forward (rolling-окна) и purge-сплит сканера.

Без сети: run_backtest/get_replay_df мокаются. Проверяем:
  - _wf_window_slices: размеры/непересекаемость тест-окон, эмбарго между
    train и test, сдвиг на test, потолок фолдов, отказ на коротком ряду;
  - _run_single в walk-forward режиме: прогон по фолдам, агрегация
    (combined = min по фолдам, train/test = худший фолд), ключ wf;
  - run_scan: purge-срез train-окна (purge_bars в RUN_STATS) и
    walk-forward (wf_folds) без поломки записи в БД.
"""


import pandas as pd
import pytest

from app_pkg import config, db
from app_pkg.ai import scanner

T = 1_700_000_000
STEP = 900

_RUN_IDS = []


def _mk_df(n=2000, end_ts=T, step=STEP):
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
    """Мок run_backtest (принимает rr — как реальная сигнатура)."""
    def fake(symbol, tf, from_sec, to_sec, strategy_name, params,
             initial_cash=10000, replay_limit=None, df=None, ind=None,
             tp_atr=None, sl_atr=None, rr=None):
        if calls is not None:
            calls.append(len(df) if df is not None else -1)
        sharpe = train_sharpe if (df is not None and len(df) > 1000) \
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


@pytest.fixture(autouse=True)
def _cleanup_db():
    yield
    for run_id in _RUN_IDS:
        db.db_clear_scan_run(run_id)
    _RUN_IDS.clear()


# ------------------------------------------------------ _wf_window_slices
def test_wf_slices_shapes_and_embargo():
    n = 1000
    folds = scanner._wf_window_slices(n, 0.1, 6, 3, purge=50)
    test = int(n * 0.1)        # 100
    train = test * 6           # 600
    assert folds == [
        ((0, train), (train + 50, train + 50 + test)),
        ((test, test + train), (test + train + 50, test + train + 50 + test)),
        ((2 * test, 2 * test + train),
         (2 * test + train + 50, 2 * test + train + 50 + test)),
    ]
    # непересекающиеся тест-окна, идущие подряд
    ends = [sl1[1] for _, sl1 in folds]
    assert ends == sorted(set(ends))


def test_wf_slices_no_overlap_train_test():
    """test-окно начинается строго после train-окна (с учётом эмбарго)."""
    for (ts, te), (vs, ve) in scanner._wf_window_slices(5000, 0.1, 6, 5, 50):
        assert te <= vs



def test_wf_slices_zero_purge():
    """purge=0 -> тест начинается ровно на конце train."""
    folds = scanner._wf_window_slices(1000, 0.1, 6, 1, purge=0)
    assert folds[0] == ((0, 600), (600, 700))


def test_wf_slices_max_folds_cap():
    n = 10000
    no_cap = scanner._wf_window_slices(n, 0.1, 6, 0, 0)
    capped = scanner._wf_window_slices(n, 0.1, 6, 2, 0)
    assert len(no_cap) > 2
    assert len(capped) == 2
    assert capped[0] == no_cap[0] and capped[1] == no_cap[1]


def test_wf_slices_too_small():
    """Данных меньше одного фолда (train + purge + test) -> []."""
    assert scanner._wf_window_slices(100, 0.5, 6, 3, 0) == []


# ------------------------------------------- _run_single walk-forward
def test_run_single_wf_aggregates_over_folds(monkeypatch):
    df = _mk_df(2000)
    folds = scanner._wf_window_slices(len(df), 0.1, 6, 2, purge=50)
    assert folds  # 200 баров тест, train 1200, эмбарго 50
    calls = []
    monkeypatch.setattr(
        scanner, "run_backtest",
        _fake_run_backtest(trades=50, calls=calls))
    monkeypatch.setattr(config, "SCAN_MIN_TRADES", 30)
    res = scanner._run_single("BTCUSDT", "15m", "sma_cross",
                              {"fast": 5, "slow": 20}, df, df,
                              ind_train=None, wf_slices=folds)
    assert res is not None
    assert len(calls) == 4  # 2 фолда × (train + test)
    folds_lens = [(1200, 200), (1200, 200)]
    assert calls == [a for ab in folds_lens for a in ab]
    assert res["wf"]["folds"] == 2
    assert res["combined_sharpe"] == 0.8  # min(1.5, 0.8) по фолдам
    assert res["train"]["sharpe"] == 1.5  # train худшего фолда (in-sample)
    assert res["test"]["sharpe"] == 0.8   # test худшего фолда (out-of-sample)
    assert res["total_trades"] == 200     # 4 × 50
    assert len(res["wf"]["per_fold"]) == 2
    assert res["wf"]["test_winrate"] == 0.6


def test_run_single_wf_rejects_when_fold_too_few_trades(monkeypatch):
    df = _mk_df(2000)
    folds = scanner._wf_window_slices(len(df), 0.1, 6, 2, purge=50)
    monkeypatch.setattr(
        scanner, "run_backtest",
        _fake_run_backtest(trades=10))  # < SCAN_MIN_TRADES
    monkeypatch.setattr(config, "SCAN_MIN_TRADES", 30)
    res = scanner._run_single("BTCUSDT", "15m", "sma_cross",
                              {"fast": 5, "slow": 20}, df, df,
                              ind_train=None, wf_slices=folds)
    assert res is None


def test_run_single_wf_no_slices_falls_back_to_simple(monkeypatch):
    df = _mk_df(2000)
    calls = []
    monkeypatch.setattr(
        scanner, "run_backtest",
        _fake_run_backtest(trades=50, calls=calls))
    monkeypatch.setattr(config, "SCAN_MIN_TRADES", 30)
    res = scanner._run_single("BTCUSDT", "15m", "sma_cross",
                              {"fast": 5, "slow": 20}, df, df,
                              ind_train=None, wf_slices=None)
    assert res is not None
    assert len(calls) == 2  # как раньше: train + test
    assert "wf" not in res


# ------------------------------------------------------- run_scan purge/wf
def test_run_scan_purge_splits_train(monkeypatch):
    """Purge-срез: train-окно короче, чем без него; запись в БД работает."""
    monkeypatch.setattr(scanner, "get_replay_df",
                        lambda *a, **k: _mk_df(1000))
    monkeypatch.setattr(scanner, "run_backtest", _fake_run_backtest())
    monkeypatch.setattr(scanner, "_ws_push", lambda event, data: None)
    monkeypatch.setattr(config, "SCAN_PURGE_BARS", 50)
    monkeypatch.setattr(config, "SCAN_WALK_FORWARD", False)
    monkeypatch.setattr(config, "SCAN_MIN_TRADES", 30)
    monkeypatch.setitem(config.SCAN_GRIDS, "sma_cross", {"fast": [5], "slow": [20]})

    run_id = scanner.run_scan(["BTCUSDT"], "15m", ["sma_cross"])
    _RUN_IDS.append(run_id)
    stats = scanner.RUN_STATS[run_id]
    assert stats.get("purge_bars") == 50
    assert stats.get("wf_folds") is None
    rows = db.db_get_scan_results(run_id, limit=10)
    assert len(rows) == 1 and rows[0]["train"] and rows[0]["test"]


def test_run_scan_walk_forward_writes_wf_stats(monkeypatch):
    """Walk-forward: RUN_STATS видит wf_folds, БД сохраняет окна."""
    monkeypatch.setattr(scanner, "get_replay_df",
                        lambda *a, **k: _mk_df(2000))
    monkeypatch.setattr(scanner, "run_backtest", _fake_run_backtest())
    monkeypatch.setattr(scanner, "_ws_push", lambda event, data: None)
    monkeypatch.setattr(config, "SCAN_WALK_FORWARD", True)
    monkeypatch.setattr(config, "SCAN_WF_FOLDS", 2)
    monkeypatch.setattr(config, "SCAN_WF_TEST_FRACTION", 0.1)
    monkeypatch.setattr(config, "SCAN_PURGE_BARS", 50)
    monkeypatch.setattr(config, "SCAN_MIN_TRADES", 30)
    monkeypatch.setitem(config.SCAN_GRIDS, "sma_cross",
                        {"fast": [5], "slow": [20]})

    run_id = scanner.run_scan(["BTCUSDT"], "15m", ["sma_cross"])
    _RUN_IDS.append(run_id)
    stats = scanner.RUN_STATS[run_id]
    assert stats.get("purge_bars") is None  # wf-путь не пишет purge
    assert stats.get("wf_folds") == 2
    rows = db.db_get_scan_results(run_id, limit=10)
    assert len(rows) == 1 and rows[0]["train"] and rows[0]["test"]


def test_run_scan_wf_too_small_falls_back(monkeypatch):
    """Данных меньше одного фолда -> обычный сплит, скан не падает."""
    monkeypatch.setattr(scanner, "get_replay_df",
                        lambda *a, **k: _mk_df(200))
    monkeypatch.setattr(scanner, "run_backtest", _fake_run_backtest())
    monkeypatch.setattr(scanner, "_ws_push", lambda event, data: None)
    monkeypatch.setattr(config, "SCAN_WALK_FORWARD", True)
    monkeypatch.setattr(config, "SCAN_WF_FOLDS", 3)
    monkeypatch.setattr(config, "SCAN_WF_TEST_FRACTION", 0.3)
    monkeypatch.setattr(config, "SCAN_WF_TRAIN_PER_TEST", 6)
    monkeypatch.setattr(config, "SCAN_PURGE_BARS", 50)
    monkeypatch.setattr(config, "SCAN_MIN_TRADES", 30)
    monkeypatch.setitem(config.SCAN_GRIDS, "sma_cross",
                        {"fast": [5], "slow": [20]})

    run_id = scanner.run_scan(["BTCUSDT"], "15m", ["sma_cross"])
    _RUN_IDS.append(run_id)
    stats = scanner.RUN_STATS[run_id]
    assert stats.get("wf_folds") is None
    assert stats.get("purge_bars") is not None  # откат на purge-сплит
    rows = db.db_get_scan_results(run_id, limit=10)
    assert len(rows) == 1