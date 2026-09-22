# -*- coding: utf-8 -*-
"""Тесты AI Backtest «Псевдо-Харон»: уровни вероятностей (без сети).

Проверяем:
  - parse_probability_levels: список/словарь от LLM, фильтр «неправильной»
    стороны цены, сортировка (UP по возрастанию, DOWN по убыванию), лимит
    уровней, вероятности в процентах (61.3 -> 0.613), невалидный ответ -> None;
  - build_levels_context: уходит тот же контекст, что у Харона, и в replay
    (upto_sec) в промпт не попадает ни одна свеча из будущего;
  - run_ai_backtest: один LLM-запрос, результат = только levels (+price), без
    сделок и метрик стратегии; ошибка LLM -> status=finished + error;
  - POST /api/ai-backtest: sync=true сразу отдаёт результат; mode=live
    игнорирует upto_sec, mode=replay его прокидывает;
  - GET /api/ai-backtest/<run_id>: уровни из БД доступны как "levels".
"""

import json

import pandas as pd
import pytest

from app_pkg import config
from app_pkg.ai import ai_backtest as aibt
from app_pkg.data import fetch
from app_pkg import create_app

BASE_TS = 1700000000
STEP = 3600
N = 100


def _df(n=N, step=STEP):
    """Растущий ряд: close_i = i+1 — будущее легко детектить."""
    times = [BASE_TS + i * step for i in range(n)]
    closes = [float(i + 1) for i in range(n)]
    return pd.DataFrame({
        "timestamp": pd.to_datetime(times, unit="s", utc=True),
        "open": closes,
        "high": closes,
        "low": closes,
        "close": closes,
        "volume": [10.0] * n,
    })


@pytest.fixture(autouse=True)
def _isolate():
    fetch.data_cache.clear()
    aibt._RUNS.clear()
    yield
    fetch.data_cache.clear()
    aibt._RUNS.clear()


@pytest.fixture()
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


# ------------------------------------------------------------------ парсинг
def test_parse_levels_list_sorted_and_filtered():
    """Список уровней: обе стороны, сортировка, отброс уровней не с той стороны."""
    raw = json.dumps({"levels": [
        {"side": "UP", "price": 130, "probability": 0.2},
        {"side": "UP", "price": 110, "probability": 0.6},
        {"side": "DOWN", "price": 90, "probability": 0.7},
        {"side": "DOWN", "price": 80, "probability": 0.4},
        # UP ниже цены — модель ошиблась: уровень отбрасываем.
        {"side": "UP", "price": 99, "probability": 0.9},
    ]})
    levels = aibt.parse_probability_levels(raw, current_price=100.0)

    assert levels is not None
    ups = [lv for lv in levels if lv["side"] == "UP"]
    downs = [lv for lv in levels if lv["side"] == "DOWN"]
    assert [lv["price"] for lv in ups] == [110, 130]      # ближний первым
    assert [lv["price"] for lv in downs] == [90, 80]      # ближний первым
    # diff считается от текущей цены, знак — как на пилюлях графика.
    assert ups[0]["diff"] == pytest.approx(10.0)
    assert downs[0]["diff"] == pytest.approx(-10.0)


def test_parse_levels_dict_form_and_percent_probability():
    """Модель вернула словарь + вероятность в процентах (61.3 -> 0.613)."""
    raw = json.dumps({
        "levels": {
            "UP": {"price": 120, "probability": 61.3},
            "DOWN": {"price": 70, "probability": 47.1},
        },
    })
    levels = aibt.parse_probability_levels(raw, current_price=100.0)

    by_side = {lv["side"]: lv for lv in levels}
    assert by_side["UP"]["probability"] == pytest.approx(0.613)
    assert by_side["DOWN"]["probability"] == pytest.approx(0.471)


def test_parse_levels_limits_and_invalid():
    """Лимит PROB_LEVELS_MAX на сторону; невалидный/пустой ответ -> None."""
    raw = json.dumps({"levels": [
        {"side": "UP", "price": 100 + i, "probability": 0.5} for i in range(1, 9)
    ]})
    levels = aibt.parse_probability_levels(raw, current_price=100.0)
    assert len(levels) == aibt.PROB_LEVELS_MAX
    assert [lv["price"] for lv in levels] == [101, 102, 103, 104, 105]

    assert aibt.parse_probability_levels("", 100.0) is None
    assert aibt.parse_probability_levels("не json", 100.0) is None
    assert aibt.parse_probability_levels(json.dumps({"foo": 1}), 100.0) is None
    # Уровни без цены/стороны — пусто -> None.
    assert aibt.parse_probability_levels(
        json.dumps({"levels": [{"side": "UP"}]}), 100.0) is None


# ------------------------------------------------------------------ контекст
def test_replay_context_has_no_future_candles(monkeypatch):
    """build_levels_context(upto_sec): ни одной свечи из будущего в промпте."""
    import re

    df = _df()
    upto = BASE_TS + 59 * STEP  # 60-я свеча, close=60.0
    monkeypatch.setattr("app_pkg.ai.context.get_series_df", lambda *a, **k: df)
    monkeypatch.setattr(aibt, "_scanner_block", lambda *a, **k: "")

    ctx = aibt.build_levels_context("BTCUSDT", "1H", upto, current_price=60.0)

    rows = re.findall(r"\[(\d+) ([\d.]*) ([\d.]*) ([\d.]*) ([\d.]*) ", ctx)
    assert rows, "в контекст не попали свечи"
    assert max(int(r[0]) for r in rows) <= upto
    assert all(float(r[4]) <= 60.0 for r in rows)
    assert "last_price=100.0" not in ctx


def test_live_context_uses_last_candle(monkeypatch):
    """build_levels_context без upto_sec (live) — свежий ряд целиком."""
    monkeypatch.setattr("app_pkg.ai.context.get_series_df", lambda *a, **k: _df())
    monkeypatch.setattr(aibt, "_scanner_block", lambda *a, **k: "")

    ctx = aibt.build_levels_context("BTCUSDT", "1H", None, current_price=100.0)
    assert "last_price=100.0" in ctx


# -------------------------------------------------------------------- движок
def test_run_ai_backtest_one_llm_call_and_levels_only(monkeypatch):
    """Один запрос к LLM; результат — только уровни, без сделок/метрик."""
    calls = []
    payload = json.dumps({"levels": [
        {"side": "UP", "price": 110, "probability": 0.6},
        {"side": "DOWN", "price": 90, "probability": 0.7},
    ]})

    def fake_llm(system, messages, **kwargs):
        calls.append((system, messages, kwargs))
        return payload

    monkeypatch.setattr(aibt, "_llm_request", fake_llm)
    monkeypatch.setattr(aibt, "_slice_price", lambda *a, **k: 100.0)
    monkeypatch.setattr(aibt, "build_levels_context", lambda *a, **k: "ctx")
    monkeypatch.setattr(aibt, "compact_snapshot",
                        lambda *a, **k: {"t": {"close": 100.0}})
    monkeypatch.setattr(aibt, "_structure_levels", lambda *a, **k: [])
    monkeypatch.setattr(aibt.db, "db_save_ai_backtest",
                        lambda *a, **k: 1)

    res = aibt.run_ai_backtest("BTCUSDT", "1H", upto_sec=None, mode="live")

    assert len(calls) == 1, "должен быть ровно один запрос к LLM"
    assert res["status"] == "finished"
    assert res["mode"] == "live"
    assert res["price"] == pytest.approx(100.0)
    assert len(res["levels"]) == 2
    assert res["error"] is None
    # Панель-помощник: ни сделок, ни метрик стратегии.
    assert "trades" not in res
    assert "metrics" not in res


def test_run_ai_backtest_llm_failure_reports_error(monkeypatch):
    """LLM недоступна: прогон завершается статусом finished + error."""
    monkeypatch.setattr(aibt, "_llm_request",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("net")))
    monkeypatch.setattr(aibt, "_slice_price", lambda *a, **k: 50.0)
    monkeypatch.setattr(aibt, "build_levels_context", lambda *a, **k: "ctx")
    monkeypatch.setattr(aibt, "compact_snapshot",
                        lambda *a, **k: {"t": {"close": 100.0}})

    res = aibt.run_ai_backtest("BTCUSDT", "1H", mode="live")

    assert res["status"] == "finished"
    assert res["levels"] == []
    assert res["error"]


@pytest.mark.parametrize("exc_text,needle", [
    ("rate limit", "rate limit"),
    ("AllocationQuota.FreeTierOnly / quota exhausted", "Квота"),
    ("Read timed out.", "Таймаут"),
    ("something unexpected", "Ошибка LLM"),
])
def test_llm_error_hint_maps_reasons(exc_text, needle):
    """Причина ошибки LLM переводится в понятный пользователю текст."""
    hint = aibt._llm_error_hint(RuntimeError(exc_text))
    assert needle.lower() in hint.lower()


def test_levels_for_slice_reports_no_data(monkeypatch):
    """Нет цены среза -> levels пусты и error про данные (не про LLM)."""
    monkeypatch.setattr(aibt, "_slice_price", lambda *a, **k: None)
    monkeypatch.setattr(aibt, "_llm_request",
                        lambda *a, **k: pytest.fail("LLM не должна вызываться"))

    levels, price, err = aibt.levels_for_slice("BTCUSDT", "1H")

    assert levels == [] and price is None
    assert "Нет данных" in err


def test_levels_for_slice_reports_empty_llm_answer(monkeypatch):
    """LLM вернула None (цепочка недоступна) -> понятная причина, не пустота."""
    monkeypatch.setattr(aibt, "_slice_price", lambda *a, **k: 100.0)
    monkeypatch.setattr(aibt, "build_levels_context", lambda *a, **k: "ctx")
    monkeypatch.setattr(aibt, "compact_snapshot",
                        lambda *a, **k: {"t": {"close": 100.0}})
    monkeypatch.setattr(aibt, "_llm_request", lambda *a, **k: None)

    levels, _price, err = aibt.levels_for_slice("BTCUSDT", "1H")

    assert levels == []
    assert "не вернула ответ" in err


def test_levels_for_slice_reports_unparsable_answer(monkeypatch):
    """LLM вернула не-JSON -> причина содержит фрагмент ответа для диагностики."""
    monkeypatch.setattr(aibt, "_slice_price", lambda *a, **k: 100.0)
    monkeypatch.setattr(aibt, "build_levels_context", lambda *a, **k: "ctx")
    monkeypatch.setattr(aibt, "compact_snapshot",
                        lambda *a, **k: {"t": {"close": 100.0}})
    monkeypatch.setattr(aibt, "_llm_request",
                        lambda *a, **k: "извините, не могу помочь")

    levels, _price, err = aibt.levels_for_slice("BTCUSDT", "1H")

    assert levels == []
    assert "без валидных уровней" in err
    assert "не могу помочь" in err


def test_run_ai_backtest_replay_passes_upto(monkeypatch):
    """mode=replay: upto_sec уходит в сырые данные (без look-ahead) и в результат."""
    seen = {}

    def fake_snap(symbol, tf, upto_sec=None):
        seen["upto"] = upto_sec
        return {"t": {"close": 42.0}}

    monkeypatch.setattr(aibt, "_llm_request",
                        lambda *a, **k: json.dumps({"targets": []}))
    monkeypatch.setattr(aibt, "_slice_price", lambda *a, **k: 42.0)
    monkeypatch.setattr(aibt, "compact_snapshot", fake_snap)

    upto = BASE_TS + 10 * STEP
    res = aibt.run_ai_backtest("BTCUSDT", "1H", upto_sec=upto, mode="replay")

    assert seen["upto"] == upto
    assert res["upto_sec"] == upto
    assert res["mode"] == "replay"


def test_levels_for_slice_hybrid_messages(monkeypatch):
    """Гибрид: system = промпт из файла, user = чистый JSON с цифрами.

    LLM получает ровно ДВА сообщения: правила (charon_prompt.txt) и
    Market Snapshot (без единого текстового описания рынка).
    """
    calls = {}

    def fake_llm(system, messages, **kwargs):
        calls["system"] = system
        calls["messages"] = messages
        return json.dumps({"prob_up": 0.63, "signal": "BUY", "targets": [
            {"side": "UP", "price": 110, "probability": 0.6},
            {"side": "DOWN", "price": 90, "probability": 0.7},
        ]})

    monkeypatch.setattr(aibt, "_llm_request", fake_llm)
    monkeypatch.setattr(aibt, "_structure_levels",
                        lambda *a, **k: [])
    monkeypatch.setattr(aibt, "_slice_price", lambda *a, **k: 100.0)
    monkeypatch.setattr(aibt, "charon_prompt_text", lambda: "RULES-FROM-FILE")
    monkeypatch.setattr(aibt, "compact_snapshot",
                        lambda *a, **k: {"t": {"rsi": 62.0, "close": 100.0}})

    levels, price, err = aibt.levels_for_slice("BTCUSDT", "1H")
    assert err is None and price == pytest.approx(100.0)

    assert calls["system"] == "RULES-FROM-FILE"
    # Два сообщения: system-промпт (передаётся отдельным аргументом) + user JSON.
    assert calls["messages"] == [{"role": "user", "content": (
        '{"current_price":100.0,"t":{"rsi":62.0,"close":100.0}}')}]

    ups = [lv for lv in levels if lv["side"] == "UP"]
    downs = [lv for lv in levels if lv["side"] == "DOWN"]
    assert [lv["price"] for lv in ups] == [110]
    assert [lv["price"] for lv in downs] == [90]


def test_parse_probability_levels_targets_format(monkeypatch):
    """Ответ в гибридном формате {prob_up, signal, targets} -> уровни для рендера."""
    raw = json.dumps({
        "prob_up": 0.63, "signal": "BUY",
        "targets": [
            {"side": "UP", "price": 120, "probability": 0.2},
            {"side": "UP", "price": 110, "probability": 0.6},
            {"side": "DOWN", "price": 90, "probability": 0.7},
        ],
    })
    levels = aibt.parse_probability_levels(raw, current_price=100.0)

    assert levels is not None
    ups = [lv for lv in levels if lv["side"] == "UP"]
    downs = [lv for lv in levels if lv["side"] == "DOWN"]
    assert [lv["price"] for lv in ups] == [110, 120]
    assert [lv["price"] for lv in downs] == [90]
    assert ups[0]["probability"] == pytest.approx(0.6)
    assert downs[0]["diff"] == pytest.approx(-10.0)


# -------------------------------------------------------- цена среза (live/replay)
def test_slice_price_cuts_by_upto_sec(monkeypatch):
    """_slice_price: replay — историческое окно до барьера, live — хвост."""
    df = _df()
    monkeypatch.setattr(aibt, "get_replay_df", lambda *a, **k: df)
    monkeypatch.setattr(aibt, "get_series_df", lambda *a, **k: df)
    upto = BASE_TS + 9 * STEP  # 10-я свеча, close=10.0

    assert aibt._slice_price("BTCUSDT", "1H", upto) == pytest.approx(10.0)
    # live: последняя свеча ряда.
    assert aibt._slice_price("BTCUSDT", "1H", None) == pytest.approx(100.0)
    # Срез раньше первой свечи — данных нет.
    assert aibt._slice_price("BTCUSDT", "1H", BASE_TS - STEP) is None


# --------------------------------------------------------------------- роуты
def test_route_sync_returns_levels(monkeypatch):
    """POST sync=true: результат сразу, без фонового потока."""
    def fake_run(symbol, timeframe, upto_sec=None, mode="live", model=None,
                 run_id=None):
        return {"run_id": "r1", "status": "finished", "symbol": symbol,
                "timeframe": timeframe, "mode": mode, "upto_sec": upto_sec,
                "price": 100.0,
                "levels": [{"side": "UP", "price": 110,
                            "probability": 0.6, "diff": 10.0}]}

    monkeypatch.setattr("app_pkg.routes.ai_backtest.run_ai_backtest", fake_run)
    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()

    resp = client.post("/api/ai-backtest", json={
        "symbol": "BTCUSDT", "timeframe": "1H", "mode": "live", "sync": True})

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["levels"][0]["price"] == 110
    assert data["mode"] == "live"


@pytest.mark.parametrize("payload,expected_upto", [
    ({"mode": "replay", "upto_sec": 1700013200}, 1700013200),
    ({"mode": "replay", "upto_sec": "1700013200.7"}, 1700013200),
    ({"mode": "replay"}, None),
    # live не режем: upto_sec игнорируется.
    ({"mode": "live", "upto_sec": 1700013200}, None),
])
def test_route_upto_sec_passthrough(client, monkeypatch, payload,
                                    expected_upto):
    """routes/ai_backtest: upto_sec = только для mode=replay."""
    captured = {}

    def fake_async(symbol, timeframe, upto_sec=None, mode="live", model=None,
                   run_id=None):
        captured.update(upto_sec=upto_sec, mode=mode)
        return "run-1"

    monkeypatch.setattr("app_pkg.routes.ai_backtest.run_ai_backtest_async",
                        fake_async)

    resp = client.post("/api/ai-backtest", json={
        "symbol": "BTCUSDT", "timeframe": "1H", **payload})

    assert resp.status_code == 200
    assert captured["upto_sec"] == expected_upto
    assert captured["mode"] == payload.get("mode", "live")


def test_route_rejects_bad_symbol_and_tf(client):
    """Валидация symbol/timeframe осталась на месте."""
    assert client.post("/api/ai-backtest", json={
        "symbol": "NOPE", "timeframe": "1H"}).status_code == 400
    assert client.post("/api/ai-backtest", json={
        "symbol": "BTCUSDT", "timeframe": "7m"}).status_code == 400


def test_route_result_from_db_exposes_levels(client, monkeypatch):
    """GET <run_id>: уровни из БД (signals_json) отдаются как levels."""
    monkeypatch.setattr("app_pkg.routes.ai_backtest.db_get_ai_backtest",
                        lambda run_id: {"run_id": run_id,
                                        "status": "finished",
                                        "levels": [{"side": "DOWN",
                                                    "price": 90,
                                                    "probability": 0.5}]})
    resp = client.get("/api/ai-backtest/abc")
    assert resp.status_code == 200
    assert resp.get_json()["levels"][0]["side"] == "DOWN"


def test_db_get_ai_backtest_aliases_levels(monkeypatch):
    """db_get_ai_backtest: signals_json -> levels (+status по умолчанию)."""
    from app_pkg import db

    class _Row(dict):
        pass

    class _Conn:
        def execute(self, *a, **k):
            class _Cur:
                def fetchone(self):
                    return {
                        "id": 1, "run_id": "r", "symbol": "BTCUSDT",
                        "timeframe": "1H",
                        "params_json": json.dumps({"mode": "replay"}),
                        "metrics_json": "{}",
                        "signals_json": json.dumps(
                            [{"side": "UP", "price": 110,
                              "probability": 0.6}]),
                        "trades_json": "[]",
                        "created_at": "2025-01-01T00:00:00Z",
                    }
            return _Cur()

    monkeypatch.setattr(db, "_get_db", lambda: _Conn())
    out = db.db_get_ai_backtest("r")

    assert out["levels"] == [{"side": "UP", "price": 110,
                              "probability": 0.6}]
    assert out["status"] == "finished"
    assert out["trades"] == []


def test_config_symbols_match_scanner_list():
    """Список символов панели совпадает с config.SYMBOLS (как у сканера)."""
    assert config.SYMBOLS[0] == "BTCUSDT"
    assert "GBPUSD" in config.SYMBOLS


def test_route_sync_end_to_end_pill_labels(monkeypatch):
    """Полный путь роут -> LLM -> уровни: пилюли как на макете (+diff prob%).

    Данные и LLM мокаются — сеть не нужна. Проверяем, что панель получает
    готовые к отрисовке уровни: +0.5 61.3%, +0.19 47.1%, -1.3 55.6%.
    """
    df = _df(n=50)
    df["close"] = 100.0
    monkeypatch.setattr(aibt, "get_series_df", lambda *a, **k: df)
    monkeypatch.setattr("app_pkg.ai.context.get_series_df", lambda *a, **k: df)
    monkeypatch.setattr(aibt, "compact_snapshot",
                        lambda *a, **k: {"t": {"close": 100.0}})
    monkeypatch.setattr(aibt, "_llm_request", lambda *a, **k: json.dumps({
        "levels": [
            {"side": "UP", "price": 100.5, "probability": 0.613},
            {"side": "UP", "price": 100.19, "probability": 0.471},
            {"side": "DOWN", "price": 98.7, "probability": 0.556},
        ],
    }))
    monkeypatch.setattr(aibt, "_structure_levels", lambda *a, **k: [])
    monkeypatch.setattr(aibt.db, "db_save_ai_backtest", lambda *a, **k: 1)

    app = create_app()
    app.config["TESTING"] = True
    resp = app.test_client().post("/api/ai-backtest", json={
        "symbol": "BTCUSDT", "timeframe": "1H", "mode": "live", "sync": True})

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["price"] == pytest.approx(100.0)
    assert data["error"] is None
    ups = [lv for lv in data["levels"] if lv["side"] == "UP"]
    downs = [lv for lv in data["levels"] if lv["side"] == "DOWN"]
    assert [round(lv["price"], 2) for lv in ups] == [100.19, 100.5]
    assert round(ups[1]["diff"], 2) == 0.5
    assert ups[1]["probability"] == pytest.approx(0.613)
    assert downs[0]["diff"] == pytest.approx(-1.3)
    assert downs[0]["probability"] == pytest.approx(0.556)


# -------------------------------------- структурная база уровней (логика)
def _mk_struct(cp):
    """Структурные уровни как у structure_levels_from_df (формат парсера)."""
    return [
        {"side": "UP", "price": 102.3, "probability": 0.35,
         "diff": 2.3, "diff_pct": 2.3},
        {"side": "UP", "price": 107.8, "probability": 0.22,
         "diff": 7.8, "diff_pct": 7.8},
        {"side": "DOWN", "price": 95.1, "probability": 0.33,
         "diff": -4.9, "diff_pct": -4.9},
        {"side": "DOWN", "price": 90.4, "probability": 0.21,
         "diff": -9.6, "diff_pct": -9.6},
    ]


def test_levels_for_slice_grounds_fake_grid_by_structure(monkeypatch):
    """Формульная сетка LLM (равный шаг) НЕ рисуется: цены — из структуры.

    Регрессия жалобы «уровни одинаковые для каждого теста»: модель при T=0
    возвращает равномерную сетку (шаг ~ATR, линейный спад вероятности) без
    связи с реальными экстремумами. Структурная база задаёт цены; LLM-уровень,
    совпавший со структурой по цене, лишь подтверждает её (+0.02 к шансу).
    """
    calls = {}

    def fake_llm(system, messages, **kwargs):
        calls["n"] = calls.get("n", 0) + 1
        return json.dumps({"targets": [
            # равномерная сетка: шаг 4, никак не связана со структурой.
            {"side": "UP", "price": 104.0, "probability": 0.33},
            {"side": "UP", "price": 108.0, "probability": 0.27},
            {"side": "UP", "price": 112.0, "probability": 0.20},
            {"side": "DOWN", "price": 96.0, "probability": 0.31},
            {"side": "DOWN", "price": 92.0, "probability": 0.24},
            {"side": "DOWN", "price": 88.0, "probability": 0.18},
        ]})

    monkeypatch.setattr(aibt, "_llm_request", fake_llm)
    monkeypatch.setattr(aibt, "_structure_levels", lambda *a, **k: _mk_struct(100.0))
    monkeypatch.setattr(aibt, "_slice_price", lambda *a, **k: 100.0)
    monkeypatch.setattr(aibt, "charon_prompt_text", lambda: "RULES")
    monkeypatch.setattr(aibt, "compact_snapshot",
                        lambda *a, **k: {"t": {"close": 100.0}})

    levels, price, err = aibt.levels_for_slice("BTCUSDT", "1H")
    assert err is None and price == pytest.approx(100.0)
    ups = [lv for lv in levels if lv["side"] == "UP"]
    downs = [lv for lv in levels if lv["side"] == "DOWN"]
    # Цены — из структуры, а не из сетки LLM (104/112/96/92/88 не попали).
    assert [lv["price"] for lv in ups] == pytest.approx([102.3, 107.8])
    assert [lv["price"] for lv in downs] == pytest.approx([95.1, 90.4])
    # Совпавший по цене уровень (108 ≈ 107.8 в пределах 0.3%) подтверждён
    # (+0.02); неподтверждённые остаются со структурным шансом.
    assert ups[1]["probability"] == pytest.approx(0.24)
    assert ups[0]["probability"] == pytest.approx(0.35)
    assert calls["n"] >= 1  # LLM всё ещё вызывается (для вердикта)


def test_flat_verdict_with_structure_returns_structure(monkeypatch):
    """Сортировка уровней в blend: UP по возрастанию, DOWN по убыванию."""
    levels = aibt._blend_levels(_mk_struct(100.0), [], 100.0)
    ups = [lv["price"] for lv in levels if lv["side"] == "UP"]
    downs = [lv["price"] for lv in levels if lv["side"] == "DOWN"]
    assert ups == pytest.approx([102.3, 107.8])
    assert downs == pytest.approx([95.1, 90.4])

    monkeypatch.setattr(aibt, "_llm_request", lambda *a, **k: json.dumps({
        "pu": 0.0, "pd": 0.0, "pf": 0.0, "sig": "F", "tg": []}))
    monkeypatch.setattr(aibt, "_structure_levels", lambda *a, **k: _mk_struct(100.0))
    monkeypatch.setattr(aibt, "compact_snapshot",
                        lambda *a, **k: {"t": {"close": 100.0}})

    levels, price, err = aibt.levels_for_slice("BTCUSDT", "1H",
                                               current_price=100.0)
    # Плоский вердикт + структура: уровни рисуются из логики, без ошибки.
    assert err is None and price == pytest.approx(100.0)
    assert {lv["price"] for lv in levels} == {102.3, 107.8, 95.1, 90.4}
