"""Тесты расширенного сканера: 12 стратегий, лимиты 50000 комбинаций / 30 мин
warning / 4 ч hard-limit, оценка длительности, ETA в прогрессе и подписи
новых стратегий.

Без сети: run_scan/run_backtest/get_replay_df подменяются моками, роуты
проверяются через Flask test_client. Дополняет tests/test_scanner.py
(базовый скан) и tests/test_backtest_new_strategies.py (сами стратегии).
"""

import threading
import time

import pandas as pd
import pytest

from app_pkg import config, create_app, db
from app_pkg.ai import scanner as scanner_mod
from app_pkg.routes.scanner import (
    _annotate,
    _estimate_seconds,
    _human_seconds,
    _strategy_label,
)

# 9 новых стратегий (к 3 прежним: sma_cross/rsi_reversal/macd_cross).
NEW_STRATEGIES = ["ema_cross", "bb_reversal", "bb_breakout", "supertrend",
                  "stoch_reversal", "stoch_cross", "cci_reversal",
                  "vwap_reversal", "adx_trend"]
# Ожидаемые размеры гридов (см. config.SCAN_GRIDS).
EXPECTED_GRID_SIZES = {
    "ema_cross": 25, "bb_reversal": 12, "bb_breakout": 12, "supertrend": 16,
    "stoch_reversal": 54, "stoch_cross": 6, "cci_reversal": 27,
    "vwap_reversal": 4, "adx_trend": 12,
}
# Все 12 стратегий с дефолтными гридами = 402 комбинации на символ и ТФ.
EXPECTED_TOTAL_COMBOS = 402

# (strategy_id, params, ожидаемая подпись) для 9 новых стратегий.
LABEL_CASES = [
    ("ema_cross", {"fast": 12, "slow": 26}, "EMA Cross 12/26"),
    ("bb_reversal", {"period": 20, "std": 2.0}, "BB Reversal 20/2.0"),
    ("bb_breakout", {"period": 15, "std": 2.5}, "BB Breakout 15/2.5"),
    ("supertrend", {"period": 10, "multiplier": 3.0},
     "Supertrend Follow 10-3.0"),
    ("stoch_reversal",
     {"k_period": 14, "d_period": 3, "oversold": 20, "overbought": 80},
     "Stoch Reversal 14/3/20/80"),
    ("stoch_cross", {"k_period": 14, "d_period": 3}, "Stoch Cross 14/3"),
    ("cci_reversal", {"period": 20, "oversold": -100, "overbought": 100},
     "CCI Reversal 20/-100/100"),
    ("vwap_reversal", {"vwap_threshold": 0.005}, "VWAP Reversal 0.005"),
    ("adx_trend", {"period": 14, "adx_threshold": 25}, "ADX Trend 14/25"),
]

# stoch_reversal 25 × 8 × 5 × 5 = 5000 комбинаций на символ/ТФ.
GRID_5000 = {"stoch_reversal": {
    "k_period": list(range(5, 30)),
    "d_period": list(range(2, 10)),
    "oversold": [15, 20, 25, 30, 35],
    "overbought": [75, 80, 85, 90, 95],
}}

# Все 10 символов config.SYMBOLS: GRID_5000 × 10 = 50000 — ровно лимит
# SCAN_MAX_COMBINATIONS (масштабный прогон масштаба «10 символов × 402 × …»).
ALL_SYMBOLS = list(config.SYMBOLS)

_RUN_IDS = []


def _mk_df(n=1000, end_ts=1_700_000_000, step=900):
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


def _fake_run_backtest(train_sharpe=1.5, test_sharpe=0.8, trades=50):
    """Мок run_backtest: train-окно (len(df) > 500 из 1000) -> train_sharpe."""
    def fake(symbol, tf, from_sec, to_sec, strategy_name, params,
             initial_cash=10000, replay_limit=None, df=None, ind=None,
             tp_atr=None, sl_atr=None, rr=None, dataset="full"):
        sharpe = train_sharpe if (df is not None and len(df) > 500) \
            else test_sharpe
        return {
            "total_return": 0.1, "sharpe_ratio": sharpe, "max_drawdown": 0.05,
            "total_trades": trades, "win_rate": 0.6,
            "trades": [{"pnl": 10.0}] * 5, "equity_curve": [],
        }
    return fake


def _patch_run_scan(monkeypatch, called=None):
    """Подмена скана в фоне: POST /api/scan не должен реально считать."""
    done_ev = threading.Event()

    def fake_run_scan(symbols, timeframe, strategies, run_id=None, grids=None,
                      use_full_history=None):
        if called is not None:
            called.update(symbols=symbols, timeframe=timeframe,
                          strategies=strategies, run_id=run_id, grids=grids,
                          use_full_history=use_full_history)
        done_ev.set()

    monkeypatch.setattr(scanner_mod, "run_scan", fake_run_scan)
    return done_ev


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


# ------------------------------------------------------------ /api/scan/grids
def _assert_full_history_grid_fields(data):
    """Режим истории в /api/scan/grids: по умолчанию полная + порог для UI."""
    assert data["default_use_full_history"] is config.SCAN_USE_FULL_HISTORY
    assert (data["max_combinations_full"]
            == config.SCAN_MAX_COMBINATIONS_FULL == 2000)


def test_api_scan_grids_returns_12_strategies(client):
    """/api/scan/grids: 12 стратегий, у каждой label/params/default_grid."""
    resp = client.get("/api/scan/grids")
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data["strategies"]) == 12
    assert set(data["strategies"]) == set(config.SCAN_GRIDS)
    for sid in NEW_STRATEGIES:
        meta = data["strategies"][sid]
        assert meta["label"] and meta["params"] and meta["default_grid"]
    # Константы для UI: лимиты и оценка времени (scanner.js считает так же)
    assert data["max_combinations"] == 50000
    assert data["max_custom_values"] == 30
    # Режим истории скана: по умолчанию полная (config), в быстром режиме —
    # меньший порог комбинаций для UI (max_combinations_full).
    _assert_full_history_grid_fields(data)
    assert data["param_limits"]["vwap_threshold"] == [0.001, 0.05]
    assert data["param_limits"]["adx_threshold"] == [10, 50]
    assert data["workers"] == config.SCAN_WORKERS
    assert (data["seconds_per_combination"]
            == config.SCAN_SECONDS_PER_COMBINATION)
    assert data["warn_seconds"] == 1800
    assert data["hard_limit_seconds"] == config.SCAN_HARD_LIMIT_SECONDS == 14400


@pytest.mark.parametrize("sid", NEW_STRATEGIES)
def test_scan_grid_sizes_for_new_strategies(sid):
    """Гриды 9 новых стратегий = ожидаемые размеры (25/12/16/54/6/27/4/12)."""
    grid = config.SCAN_GRIDS[sid]
    assert grid, f"{sid}: пустой грид"
    size = 1
    for values in grid.values():
        size *= len(values)
    assert size == EXPECTED_GRID_SIZES[sid]
    # Каждый параметр грида разрешён в SCAN_ALLOWED_STRATEGIES
    assert set(grid) == set(config.SCAN_ALLOWED_STRATEGIES[sid]["params"])


def test_all_strategies_have_grid_and_total_is_402():
    """12 стратегий в гридах = 12 в allowed, сумма гридов = 402 комбинации."""
    assert set(config.SCAN_GRIDS) == set(config.SCAN_ALLOWED_STRATEGIES)
    assert len(config.SCAN_ALLOWED_STRATEGIES) == 12
    total = 0
    for grid in config.SCAN_GRIDS.values():
        size = 1
        for values in grid.values():
            size *= len(values)
        total += size
    assert total == EXPECTED_TOTAL_COMBOS


# ------------------------------------------------------------- _strategy_label
@pytest.mark.parametrize(("sid", "params", "expected"), LABEL_CASES)
def test_strategy_label_new_strategies(sid, params, expected):
    """Подпись комбинации новой стратегии (см. _LABEL_FORMATS)."""
    assert _strategy_label({"strategy": sid, "params": params}) == expected


def test_strategy_label_partial_params_uses_question():
    """Неполные params — '?' вместо значения (прежнее поведение)."""
    assert _strategy_label({"strategy": "sma_cross", "params": {"fast": 5}}) \
        == "SMA Cross 5/?"


def test_strategy_label_unknown_keys_fallback():
    """Незнакомые ключи params -> fallback 'strategy_id key=val,...'."""
    out = _strategy_label({"strategy": "supertrend",
                           "params": {"period": 10, "multiplier": 3.0,
                                      "extra": 1}})
    assert out == "supertrend extra=1, multiplier=3.0, period=10"


def test_annotate_new_strategy_label_and_verdict():
    """_annotate: подпись новой стратегии + вердикт (логика _verdict общая)."""
    res = {
        "strategy": "adx_trend", "params": {"period": 14, "adx_threshold": 25},
        "train": {"sharpe": 1.2}, "test": {"sharpe": 0.9, "trades": 60},
        "combined_sharpe": 0.9,
    }
    ann = _annotate(res)
    assert ann["strategy_label"] == "ADX Trend 14/25"
    assert ann["verdict"] == "stable" and ann["quality"] == "good"
    # исходный dict не мутируется
    assert "strategy_label" not in res


# --------------------------------------------------------- оценка длительности
def test_estimate_seconds_formula():
    """combos × 0.3 / workers(8) + symbols × TF × 15 сек (см. config)."""
    assert _estimate_seconds(0, 1) == 15          # symbols × 1 ТФ × 15
    assert _estimate_seconds(5000, 10) == round(5000 * 0.3 / 8 + 10 * 15)
    assert _human_seconds(45) == "45 сек"
    assert _human_seconds(12 * 60) == "12 мин"
    assert _human_seconds(2 * 3600 + 15 * 60) == "2 ч 15 мин"
    assert _human_seconds(4 * 3600 + 15 * 60) == "4 ч 15 мин"
    assert _human_seconds(3600) == "1 ч"


def test_estimate_seconds_scales_fetch_by_timeframes():
    """6 ТФ -> fetch-компонент symbols × 6 × 15 (данные тянутся на каждый ТФ)."""
    assert _estimate_seconds(0, 10, 6) == 10 * 6 * 15 == 900
    assert _estimate_seconds(5000, 10, 6) == round(
        5000 * config.SCAN_SECONDS_PER_COMBINATION / config.SCAN_WORKERS
        + 10 * 6 * config.SCAN_FETCH_SECONDS_PER_SYMBOL)
    # Масштабный прогон 10 символов × 6 ТФ × 402 комб. = 24120: разница с
    # одним ТФ — ровно фетч за 5 дополнительных ТФ.
    one_tf = _estimate_seconds(24120, 10, 1)
    six_tf = _estimate_seconds(24120, 10, 6)
    assert six_tf - one_tf == 10 * 5 * 15
    # timeframes_count защищён от нуля/мусора: 0 -> 1 ТФ.
    assert _estimate_seconds(0, 3, 0) == _estimate_seconds(0, 3)


# ------------------------------------------------------ POST /api/scan: лимиты
def test_api_scan_accepts_timeframes_array(client, monkeypatch):
    """POST /api/scan с body.timeframes (массив) — масштаб по числу ТФ."""
    called = {}
    done_ev = _patch_run_scan(monkeypatch, called)
    resp = client.post("/api/scan", json={
        "symbols": ["BTCUSDT"], "timeframes": ["15m", "1H", "4H"],
        "strategies": ["sma_cross"],
    })
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["total_timeframes"] == 3
    assert data["total_combinations"] == 90 * 3  # sma_cross × 3 ТФ
    assert done_ev.wait(timeout=5)
    assert called["timeframe"] == ["15m", "1H", "4H"]  # передано без изменений


def test_api_scan_propagates_use_full_history(client, monkeypatch):
    """use_full_history: True/False из body -> в run_scan(+echo в ответе)."""
    for flag in (True, False):
        called = {}
        done_ev = _patch_run_scan(monkeypatch, called)
        resp = client.post("/api/scan", json={
            "symbols": ["BTCUSDT"], "timeframe": "15m",
            "strategies": ["sma_cross"], "use_full_history": flag,
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["use_full_history"] is flag
        assert done_ev.wait(timeout=5)
        assert called["use_full_history"] is flag


def test_api_scan_rejects_non_bool_use_full_history(client):
    """use_full_history обязан быть bool — иначе 400, скан не стартует."""
    for bad in ("yes", 1, 0, [], {}):
        resp = client.post("/api/scan", json={
            "symbols": ["BTCUSDT"], "timeframe": "15m",
            "strategies": ["sma_cross"], "use_full_history": bad,
        })
        assert resp.status_code == 400
        assert "use_full_history" in resp.get_json()["error"]


def test_api_scan_rejects_unknown_timeframe(client):
    """ТФ вне config.SCAN_TIMEFRAMES -> 400, скан не стартует."""
    resp = client.post("/api/scan", json={
        "symbols": ["BTCUSDT"], "timeframes": ["15m", "2h"],
        "strategies": ["sma_cross"],
    })
    assert resp.status_code == 400
    assert "Invalid timeframes" in resp.get_json()["error"]


def test_api_scan_default_timeframes_in_grids(client):
    """/api/scan/grids отдаёт default_timeframes для UI-чекбоксов."""
    resp = client.get("/api/scan/grids")
    assert resp.status_code == 200
    data = resp.get_json()
    assert list(data["default_timeframes"]) == list(config.SCAN_DEFAULT_TIMEFRAMES)
    assert set(data["default_timeframes"]) <= set(data["timeframes"])


def test_api_scan_all_12_strategies_default_grid(client, monkeypatch):
    """12 стратегий с дефолтными гридами: 402 комбинации, без warning."""
    done_ev = _patch_run_scan(monkeypatch)
    resp = client.post("/api/scan", json={
        "symbols": ["USDCHF"], "timeframe": "15m",
        "strategies": list(config.SCAN_GRIDS),
    })
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["total_combinations"] == EXPECTED_TOTAL_COMBOS
    assert data["estimated_seconds"] == _estimate_seconds(
        EXPECTED_TOTAL_COMBOS, 1)
    assert "warning" not in data
    assert done_ev.wait(timeout=5)


def test_api_scan_accepts_max_combinations_50000(client, monkeypatch):
    """50000 комбинаций (10 символов × GRID_5000) — ровно лимит: 200 OK.

    Оценка такого прогона (~34 мин) уже больше SCAN_WARN_SECONDS, поэтому
    ответ приходит с warning, но скан стартует (hard-limit 4ч не превышен).
    """
    done_ev = _patch_run_scan(monkeypatch)
    resp = client.post("/api/scan", json={
        "symbols": ALL_SYMBOLS, "timeframe": "15m",
        "strategies": ["stoch_reversal"], "custom_grids": GRID_5000,
    })
    assert resp.status_code == 200
    data = resp.get_json()
    assert config.SCAN_MAX_COMBINATIONS == 50000
    assert data["total_combinations"] == 50000
    assert data["estimated_seconds"] == _estimate_seconds(50000, 10) == 2025
    # Масштаб прогона в ответе — всегда (UI показывает его до старта).
    assert data["estimated_human"] == "34 мин"
    assert data["total_symbols"] == 10 and data["total_timeframes"] == 1
    assert data["hard_limit_seconds"] == config.SCAN_HARD_LIMIT_SECONDS
    assert "больше 30 минут" in data["warning"]
    assert done_ev.wait(timeout=5)


def test_api_scan_rejects_over_max_combinations(client):
    """50090 > 50000 -> 400 'Too many combinations' и скан не стартует.

    10 символов × (5000 stoch + 9 sma_cross) = 50090. Больше лимита не
    конструируется точно на 50001: на параметр не больше
    SCAN_MAX_CUSTOM_VALUES=30 значений, а 50001 = 3 × 7 × 2381 (простое).
    """
    grids = dict(GRID_5000)
    grids["sma_cross"] = {"fast": [5]}   # 1 × 9 (slow из дефолта) = +9
    resp = client.post("/api/scan", json={
        "symbols": ALL_SYMBOLS, "timeframe": "15m",
        "strategies": ["stoch_reversal", "sma_cross"], "custom_grids": grids,
    })
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["total_combinations"] == 50090
    assert "Too many combinations" in data["error"]
    assert "50090 > 50000" in data["error"]
    assert data["max_combinations"] == 50000


def test_custom_grids_validation_uses_raised_limit(client):
    """custom_grids: произведение гридов > 50000 -> 400 до материализации.

    Лимит тот же SCAN_MAX_COMBINATIONS (проверка в _validate_custom_grids
    дешёвая — по произведению длин, без генерации комбинаций).
    """
    resp = client.post("/api/scan", json={
        "symbols": ["BTCUSDT"], "timeframe": "15m",
        "strategies": ["stoch_reversal"],
        "custom_grids": {"stoch_reversal": {
            "k_period": list(range(5, 31)),        # 26
            "d_period": list(range(2, 11)),        # 9
            "oversold": list(range(11, 41)),       # 30
            "overbought": list(range(60, 90)),     # 30
        }},
    })
    assert resp.status_code == 400
    data = resp.get_json()
    assert "Too many combinations" in data["error"]
    assert "> 50000" in data["error"]
    assert data["max_combinations"] == 50000


def test_api_scan_no_warning_at_1700_seconds(client, monkeypatch):
    """Оценка 1700 сек (< SCAN_WARN_SECONDS=1800) -> 200 без warning."""
    _patch_run_scan(monkeypatch)
    # 50000 × 0.248 / 8 = 1550 + 10 символов × 15 = 1700 сек.
    monkeypatch.setattr(config, "SCAN_SECONDS_PER_COMBINATION", 0.248)
    resp = client.post("/api/scan", json={
        "symbols": ALL_SYMBOLS, "timeframe": "15m",
        "strategies": ["stoch_reversal"], "custom_grids": GRID_5000,
    })
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["estimated_seconds"] == 1700
    assert data["estimated_human"] == "28 мин"
    assert data["estimated_seconds"] < config.SCAN_WARN_SECONDS
    assert "warning" not in data


def test_api_scan_warns_over_30min(client, monkeypatch):
    """Оценка 3000 сек (> 30 мин, < 4 ч) -> 200 + warning 'Продолжить?'."""
    done_ev = _patch_run_scan(monkeypatch)
    # 50000 × 0.456 / 8 = 2850 + 150 = 3000 сек (50 мин).
    monkeypatch.setattr(config, "SCAN_SECONDS_PER_COMBINATION", 0.456)
    resp = client.post("/api/scan", json={
        "symbols": ALL_SYMBOLS, "timeframe": "15m",
        "strategies": ["stoch_reversal"], "custom_grids": GRID_5000,
    })
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["estimated_seconds"] == 3000 > config.SCAN_WARN_SECONDS
    assert data["estimated_seconds"] < config.SCAN_HARD_LIMIT_SECONDS
    assert data["estimated_human"] == "50 мин"
    assert "больше 30 минут" in data["warning"]
    assert "Продолжить?" in data["warning"]
    assert done_ev.wait(timeout=5)


def test_api_scan_rejects_run_longer_than_4hours(client, monkeypatch):
    """Оценка 15000 сек (> SCAN_HARD_LIMIT_SECONDS) -> 400 'Too long'."""
    _patch_run_scan(monkeypatch)
    # 50000 × 2.376 / 8 = 14850 + 150 = 15000 сек (4 ч 10 мин).
    monkeypatch.setattr(config, "SCAN_SECONDS_PER_COMBINATION", 2.376)
    resp = client.post("/api/scan", json={
        "symbols": ALL_SYMBOLS, "timeframe": "15m",
        "strategies": ["stoch_reversal"], "custom_grids": GRID_5000,
    })
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["estimated_seconds"] == 15000
    assert data["estimated_human"] == "4 ч 10 мин"
    assert data["estimated_seconds"] > config.SCAN_HARD_LIMIT_SECONDS == 14400
    assert "Too long" in data["error"]
    assert "> 4 часов" in data["error"]
    assert "Сократите объём или запустите по частям." in data["error"]
    assert data["total_combinations"] == 50000
    assert data["hard_limit_seconds"] == config.SCAN_HARD_LIMIT_SECONDS


def test_api_scan_no_warning_for_short_run(client, monkeypatch):
    """Короткий прогон (грид vwap ≈ 1 мин) — warning нет."""
    _patch_run_scan(monkeypatch)
    resp = client.post("/api/scan", json={
        "symbols": ["BTCUSDT"], "timeframe": "15m",
        "strategies": ["vwap_reversal"],
    })
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["total_combinations"] == 4          # vwap_reversal: 4 значения
    assert data["estimated_seconds"] < config.SCAN_WARN_SECONDS
    assert data["estimated_human"] == "15 сек"
    assert data["total_symbols"] == 1 and data["total_timeframes"] == 1


# --------------------------------------------------- custom_grids: валидация
def test_custom_grids_accept_float_new_params(client, monkeypatch):
    """std / multiplier / vwap_threshold принимают дробные значения."""
    _patch_run_scan(monkeypatch)
    resp = client.post("/api/scan", json={
        "symbols": ["USDCHF"], "timeframe": "15m",
        "strategies": ["bb_reversal", "supertrend", "vwap_reversal"],
        "custom_grids": {
            "bb_reversal": {"period": [20], "std": [1.5, 2.5]},
            "supertrend": {"period": [10], "multiplier": [2.5, 3.0]},
            "vwap_reversal": {"vwap_threshold": [0.003, 0.01]},
        },
    })
    assert resp.status_code == 200
    assert resp.get_json()["total_combinations"] == 2 + 2 + 2


def test_custom_grids_accept_cci_negative_zones(client, monkeypatch):
    """CCI oversold/overbought: отрицательные/большие значения разрешены."""
    _patch_run_scan(monkeypatch)
    resp = client.post("/api/scan", json={
        "symbols": ["XAUUSD"], "timeframe": "1H",
        "strategies": ["cci_reversal"],
        "custom_grids": {"cci_reversal": {
            "period": [20], "oversold": [-150, -100],
            "overbought": [100, 150]}},
    })
    assert resp.status_code == 200
    assert resp.get_json()["total_combinations"] == 1 * 2 * 2


@pytest.mark.parametrize(("grid", "fragment"), [
    ({"bb_reversal": {"std": [9.0]}}, "std"),
    ({"adx_trend": {"adx_threshold": [25.5]}}, "adx_threshold"),
    ({"stoch_cross": {"k_period": [4]}}, "k_period"),
    ({"sma_cross": {"fast": [500]}}, "fast"),
    ({"sma_cross": {"trend": [5]}}, "не разрешён"),
])
def test_custom_grids_reject_invalid_values(client, grid, fragment):
    """Вне диапазона / дробное у int-параметра / чужой параметр -> 400."""
    resp = client.post("/api/scan", json={
        "symbols": ["BTCUSDT"], "timeframe": "15m",
        "strategies": list(grid), "custom_grids": grid,
    })
    assert resp.status_code == 400
    assert fragment in resp.get_json()["error"]


def test_custom_grids_max_custom_values_30(client, monkeypatch):
    """30 значений на параметр — ок, 31 — 400 (лимит поднят 20 -> 30)."""
    _patch_run_scan(monkeypatch)
    resp = client.post("/api/scan", json={
        "symbols": ["BTCUSDT"], "timeframe": "15m",
        "strategies": ["sma_cross"],
        "custom_grids": {"sma_cross": {"fast": list(range(5, 35))}},  # 30
    })
    assert resp.status_code == 200
    assert resp.get_json()["total_combinations"] == 30 * 9

    resp2 = client.post("/api/scan", json={
        "symbols": ["BTCUSDT"], "timeframe": "15m",
        "strategies": ["sma_cross"],
        "custom_grids": {"sma_cross": {"fast": list(range(5, 36))}},  # 31
    })
    assert resp2.status_code == 400
    assert "от 1 до 30" in resp2.get_json()["error"]


# --------------------------------------------------------- ETA в прогрессе
def test_eta_seconds_math():
    """rate = done/elapsed, eta = (total - done) / rate."""
    start = time.time() - 10
    assert scanner_mod._eta_seconds(start, 0, 100) is None   # не начали
    assert scanner_mod._eta_seconds(start, 100, 100) is None  # закончили
    eta = scanner_mod._eta_seconds(start, 20, 100)            # 20 за 10 сек
    assert eta is not None and 35 <= eta <= 45


def test_eta_seconds_elapsed_factor_x4(monkeypatch):
    """Полная история (20k) ~4× дольше 5k: elapsed_factor растягивает прошедшее
    время и ровно в factor раз завышает ETA при той же скорости."""
    monkeypatch.setattr(scanner_mod.time, "time", lambda: 1000.0)
    base = scanner_mod._eta_seconds(0, 5, 10, elapsed_factor=1.0)
    full = scanner_mod._eta_seconds(0, 5, 10, elapsed_factor=4.0)
    assert base == 1000 and full == 4000


# ------------------------------------------------- fetch_limit / режим истории
def _train_rows(n):
    """Сколько рядов train доходит до backtest: сплит 70% минус purge-эмбарго.

    Копия расчёта из scanner.run_scan (split = n×SCAN_TRAIN_SPLIT,
    purge = min(SCAN_PURGE_BARS, n×0.05)) — тест падает, если формула
    разойдётся с продакшеном.
    """
    split = int(n * config.SCAN_TRAIN_SPLIT)
    purge = min(config.SCAN_PURGE_BARS, max(0, int(n * 0.05)))
    return max(1, split - purge)


def _run_scan_capture_fetch(monkeypatch, use_full_history, df_rows=2500,
                            history_limit=1500, replay_limit=999):
    """Запуск run_scan с замером limit у get_replay_df и строк df в run_backtest.

    Возвращает (calls, caps): calls — (symbol, tf, limit) каждого фетча;
    caps["df_rows"] — число свечей, дошедших до первого прогона backtest
    (после обрезки fetch_limit). _ws_push затыкаем, run_id чистим после.
    """
    calls = []
    caps = {}

    def fake_replay(symbol, tf, from_sec, to_sec, limit=None):
        calls.append((symbol, tf, limit))
        return _mk_df(df_rows)

    def fake_bt(symbol, tf, from_sec, to_sec, strategy_name, params,
                initial_cash=10000, replay_limit=None, df=None, ind=None,
                tp_atr=None, sl_atr=None, rr=None, dataset="full"):
        caps.setdefault("df_rows", len(df) if df is not None else None)
        return _fake_run_backtest()(symbol, tf, from_sec, to_sec,
                                    strategy_name, params, initial_cash,
                                    replay_limit, df, ind, tp_atr, sl_atr)

    monkeypatch.setattr(scanner_mod, "get_replay_df", fake_replay)
    monkeypatch.setattr(scanner_mod, "run_backtest", fake_bt)
    monkeypatch.setattr(scanner_mod, "_ws_push", lambda event, data: None)
    if history_limit is not None:
        monkeypatch.setitem(config.HISTORY_LIMITS, "15m", history_limit)
    if replay_limit is not None:
        monkeypatch.setattr(config, "SCAN_REPLAY_LIMIT", replay_limit)

    run_id = scanner_mod.run_scan(["BTCUSDT"], "15m", ["sma_cross"],
                                  use_full_history=use_full_history)
    _RUN_IDS.append(run_id)
    return calls, caps


def test_run_scan_full_history_fetch_limit(monkeypatch):
    """use_full_history=True: фетч берёт HISTORY_LIMITS[tf] (НЕ
    SCAN_REPLAY_LIMIT), избыточный df обрезается до этого лимита."""
    calls, caps = _run_scan_capture_fetch(monkeypatch, True)
    assert calls == [("BTCUSDT", "15m", 1500)]
    # 2500 рядов обрезаны до 1500, до backtest доходит лишь train-часть (70%)
    # минус purge-эмбарго между train и test (SCAN_PURGE_BARS).
    expect = _train_rows(1500)
    assert caps["df_rows"] == expect == 1000


def test_run_scan_fast_mode_fetch_limit(monkeypatch):
    """use_full_history=False: фетч берёт SCAN_REPLAY_LIMIT (старый 5000)."""
    calls, caps = _run_scan_capture_fetch(monkeypatch, False)
    assert calls == [("BTCUSDT", "15m", 999)]
    expect = _train_rows(999)
    assert caps["df_rows"] == expect == 650


def test_run_scan_warns_short_history(monkeypatch, caplog):
    """Меньше 1000 рядов на пару (символ/ТФ) -> warning в лог."""
    def fake_replay(symbol, tf, from_sec, to_sec, limit=None):
        return _mk_df(900)

    monkeypatch.setattr(scanner_mod, "get_replay_df", fake_replay)
    monkeypatch.setattr(scanner_mod, "run_backtest", _fake_run_backtest())
    monkeypatch.setattr(scanner_mod, "_ws_push", lambda event, data: None)
    monkeypatch.setattr(config, "SCAN_MIN_TRADES", 0)

    with caplog.at_level("WARNING", "app_pkg.ai.scanner"):
        run_id = scanner_mod.run_scan(["BTCUSDT"], "15m", ["sma_cross"])
        _RUN_IDS.append(run_id)

    assert any("too few rows" in r.message for r in caplog.records)


def test_run_scan_pushes_eta_every_n_combinations(monkeypatch):
    """scan_progress: eta_seconds раз в SCAN_ETA_PUSH_EVERY комбинаций."""
    monkeypatch.setattr(scanner_mod, "get_replay_df",
                        lambda *a, **k: _mk_df(1000))
    monkeypatch.setattr(scanner_mod, "run_backtest", _fake_run_backtest())
    monkeypatch.setitem(config.SCAN_GRIDS, "sma_cross",
                        {"fast": [5, 10, 15, 20, 25]})
    monkeypatch.setattr(config, "SCAN_MIN_TRADES", 0)
    events = []
    monkeypatch.setattr(scanner_mod, "_ws_push",
                        lambda event, data: events.append((event, data)))

    # 3 символа × 5 комбинаций = 15: ETA приходит на done=10, не на финале
    run_id = scanner_mod.run_scan(["BTCUSDT", "ETHUSDT", "USDCHF"], "15m",
                                  ["sma_cross"])
    _RUN_IDS.append(run_id)

    assert all(ev == "scan_progress" for ev, _ in events)
    eta_events = [d for _, d in events if "eta_seconds" in d]
    assert eta_events, "ни одного события с eta_seconds"
    assert all(d["done"] % config.SCAN_ETA_PUSH_EVERY == 0 for d in eta_events)
    # На быстром моке ETA может округлиться до 0 — важно, что это int >= 0.
    assert all(isinstance(d["eta_seconds"], int) and d["eta_seconds"] >= 0
               for d in eta_events)
    # финальное событие (done == total) — без ETA
    assert events[-1][1]["done"] == 15
    assert "eta_seconds" not in events[-1][1]
    # ETA дублируется в RUN_STATS (фолбэк поллинга UI)
    stats = scanner_mod.get_run_stats(run_id)
    assert stats["finished"] is True
    assert isinstance(stats["eta_seconds"], int)
