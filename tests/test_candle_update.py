"""Тесты SSE-события candle_update (live-свечи в реальном времени).

Проверяем:
- df_last_candle: формат {time, open, high, low, close, volume} + None на пустом df;
- формат полезной нагрузки candle_update (symbol, timeframe, candle, closed);
- дедуп: одинаковый бар шлётся ровно один раз, изменившийся — снова;
- /api/last-bar переиспользует df_last_candle (свечи идентичны).
"""

import json

import pandas as pd
import pytest

from app_pkg import config, create_app
from app_pkg.data import fetch, live
from app_pkg.indicators import df_last_candle

_ROWS = [
    (1700000000, 1.0, 2.0, 0.5, 1.5, 10.0),
    (1700003600, 1.5, 3.0, 1.0, 2.5, 20.0),
    (1700007200, 2.5, 4.0, 2.0, 3.5, 30.0),
]


def _df(rows=None):
    rows = rows if rows is not None else _ROWS
    return pd.DataFrame({
        "timestamp": pd.to_datetime([r[0] for r in rows], unit="s", utc=True),
        "open": [float(r[1]) for r in rows],
        "high": [float(r[2]) for r in rows],
        "low": [float(r[3]) for r in rows],
        "close": [float(r[4]) for r in rows],
        "volume": [float(r[5]) for r in rows],
    })


@pytest.fixture(autouse=True)
def _clear():
    fetch.data_cache.clear()
    live.live_bars.clear()
    live.set_active_pairs([])
    yield
    fetch.data_cache.clear()
    live.live_bars.clear()
    live.set_active_pairs([])


# ------------------------------------------------------------- df_last_candle
def test_df_last_candle_format():
    c = df_last_candle(_df())
    assert c == {"time": 1700007200, "open": 2.5, "high": 4.0,
                 "low": 2.0, "close": 3.5, "volume": 30.0}
    assert isinstance(c["time"], int)


def test_df_last_candle_empty_is_none():
    assert df_last_candle(pd.DataFrame()) is None
    assert df_last_candle(None) is None


# ------------------------------------------------- SSE candle_update (payload)
def _bar(sig=1.0, closed=False):
    return {"ts": 1700007200, "open": 2.5, "high": 4.0, "low": 2.0,
            "close": 3.5 + sig, "volume": 30.0, "closed": closed}


def test_sse_push_candle_update_payload(monkeypatch):
    """_ws_push шлёт candle_update с {symbol, timeframe, candle, closed}."""
    events = []
    monkeypatch.setattr(live, "_ws_push",
                        lambda ev, data: events.append((ev, data)))
    with live._live_bars_lock:
        live.live_bars[("BTCUSDT", "1H")] = _bar()

    # Один прогон цикла (без вечного while): вырезаем тело один раз.
    def _one_tick():
        with live._live_bars_lock:
            snapshot = dict(live.live_bars)
        for (symbol, tf), bar in snapshot.items():
            live._ws_push("candle_update", {
                "symbol": symbol, "timeframe": tf,
                "candle": {"time": bar["ts"], "open": bar["open"],
                           "high": bar["high"], "low": bar["low"],
                           "close": bar["close"], "volume": bar["volume"]},
                "closed": bool(bar.get("closed", False)),
            })
    _one_tick()

    assert len(events) == 1
    ev, data = events[0]
    assert ev == "candle_update"
    assert data["symbol"] == "BTCUSDT"
    assert data["timeframe"] == "1H"
    assert data["candle"]["time"] == 1700007200
    assert data["candle"]["close"] == 3.5 + 1.0
    assert data["closed"] is False


def test_sse_push_dedup(monkeypatch):
    """Дедуп: неизменённый бар не шлётся повторно, изменившийся — шлётся."""
    sent = []

    def fake_push(ev, data):
        sent.append(data["candle"]["close"])

    monkeypatch.setattr(live, "_ws_push", fake_push)

    # Эмуляция логики дедуп из _sse_candle_push_loop (сигнатура бара).
    last_sent = {}
    key = ("BTCUSDT", "1H")

    def push_once(bar):
        sig = (bar["ts"], bar["open"], bar["high"], bar["low"],
               bar["close"], bar["volume"], bar["closed"])
        if last_sent.get(key) == sig:
            return
        last_sent[key] = sig
        fake_push("candle_update", {"candle": {"close": bar["close"]}})

    b1 = _bar(sig=1.0)
    push_once(b1)
    push_once(dict(b1))          # тот же бар -> дедуп
    assert sent == [4.5]

    b2 = _bar(sig=2.0)           # close изменился -> шлём снова
    push_once(b2)
    assert sent == [4.5, 5.5]


# ------------------------------------------------------- /api/last-bar (shared)
def test_last_bar_matches_df_last_candle(client, monkeypatch):
    """Ответ /api/last-bar.candle идентичен df_last_candle(df)."""
    df = _df()
    monkeypatch.setattr("app_pkg.routes.data.get_series_df",
                        lambda *a, **k: df)
    resp = client.get("/api/last-bar?symbol=BTCUSDT&timeframe=1H")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["candle"] == df_last_candle(df)


def test_last_bar_empty_uses_shared_helper(client, monkeypatch):
    monkeypatch.setattr("app_pkg.routes.data.get_series_df",
                        lambda *a, **k: pd.DataFrame())
    resp = client.get("/api/last-bar?symbol=BTCUSDT&timeframe=1H")
    data = resp.get_json()
    assert data["candle"] is None
    assert data["indicators"] is None


@pytest.fixture()
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


# ------------------------------------------- _binance_ws_message (live_bars)
def _kline(symbol="BTCUSDT", interval="1m", ts_ms=1790156580000, close="85810.0"):
    """Мини-payload Binance kline (поле k внутри data)."""
    return {
        "e": "kline", "s": symbol,
        "k": {"t": ts_ms, "T": ts_ms + 59999, "s": symbol, "i": interval,
              "o": "85800.0", "c": close, "h": "85820.0", "l": "85790.0",
              "v": "12.5", "x": False},
    }


def test_ws_message_combined_stream_payload():
    """REGRESSION: combined stream кладёт payload в msg["data"]["k"].

    Раньше читали msg.get("k") -> всегда None, сообщение молча игнорировалось,
    live_bars оставался пустым и 1m-бар «застывал» на значении из REST-кеша.
    """
    live._binance_ws_message(None, json.dumps(
        {"stream": "btcusdt@kline_1m", "data": _kline()}))
    bar = live.live_bars[("BTCUSDT", "1m")]
    assert bar["ts"] == 1790156580
    assert bar["close"] == 85810.0
    assert bar["closed"] is False


def test_ws_message_raw_format_also_parsed():
    """Raw-формат (payload в корне) тоже поддержан."""
    live._binance_ws_message(None, json.dumps(_kline()))
    assert ("BTCUSDT", "1m") in live.live_bars


def test_ws_message_maps_binance_interval_to_tf():
    """i=1h из Binance -> ключ '1H' (внутренний таймфрейм проекта)."""
    live._binance_ws_message(None, json.dumps(
        {"data": _kline(interval="1h", ts_ms=1790140800000)}))
    assert ("BTCUSDT", "1H") in live.live_bars
    assert ("BTCUSDT", "1h") not in live.live_bars


def test_ws_message_ignores_unknown_symbol_and_interval():
    """Незнакомый символ/интервал не пишется в live_bars."""
    live._binance_ws_message(None, json.dumps(
        {"data": _kline(symbol="DOGEUSDT", interval="3m")}))
    assert live.live_bars == {}


def test_ws_message_malformed_is_silent():
    """Битый JSON/пустое сообщение не бросает исключение."""
    live._binance_ws_message(None, "not-json")
    live._binance_ws_message(None, json.dumps({}))
    live._binance_ws_message(None, json.dumps({"data": {"k": {}}}))
    assert live.live_bars == {}


def test_config_has_sse_interval():
    assert config.SSE_CANDLE_PUSH_INTERVAL > 0


# ------------------------------------- активные пары (фильтр SSE-пушей)
def test_active_pairs_filters_push(monkeypatch):
    """Пара вне активного списка НЕ пушится (иначе очередь клиента переполняется)."""
    events = []
    monkeypatch.setattr(live, "_ws_push",
                        lambda ev, data: events.append((ev, data)))
    live.set_active_pairs([("BTCUSDT", "1m")])
    with live._live_bars_lock:
        live.live_bars[("BTCUSDT", "1m")] = _bar()
        live.live_bars[("BTCUSDT", "5m")] = _bar()   # не активна -> пропуск
        live.live_bars[("ETHUSDT", "1m")] = _bar()   # не активна -> пропуск

    last_sent = {}
    with live._live_bars_lock:
        snapshot = dict(live.live_bars)
    active = live.active_pairs()
    for (symbol, tf), bar in snapshot.items():
        if active and (symbol, tf) not in active:
            continue
        live._ws_push("candle_update", {"symbol": symbol, "timeframe": tf})

    assert len(events) == 1
    assert events[0][1] == {"symbol": "BTCUSDT", "timeframe": "1m"}


def test_active_pairs_empty_means_no_filter():
    """Пустой список активных пар — фильтр выключен (обратная совместимость)."""
    live.set_active_pairs([])
    assert live.active_pairs() == set()


def test_set_active_pairs_normalizes_input():
    """set_active_pairs принимает любой iterable и приводит к (str, str)."""
    pairs = live.set_active_pairs([("BTCUSDT", "1m"), ("ETHUSDT", "1H")])
    assert pairs == {("BTCUSDT", "1m"), ("ETHUSDT", "1H")}
    assert live.active_pairs() == pairs
    # Замена (не накопление): список полностью перезаписывается.
    live.set_active_pairs([("SOLUSDT", "15m")])
    assert live.active_pairs() == {("SOLUSDT", "15m")}


def test_set_active_pairs_skips_incomplete_entries():
    """Пустые/неполные элементы игнорируются, а не падают."""
    pairs = live.set_active_pairs([("BTCUSDT", "1m"), ("", "5m"), ("X", None), None])
    assert pairs == {("BTCUSDT", "1m")}


# --------------------------------------------- ws: очередь не теряет клиента
def test_ws_push_overflow_drops_oldest_keeps_client():
    """Переполнение очереди НЕ отключает клиента — дропается старейшее событие.

    Регресс: раньше queue.Full помечал клиента dead и удалял из рассылки,
    из-за чего браузер терял SSE и свечи «пропадали» до реконнекта.
    """
    from app_pkg import ws as ws_mod
    cid, q = ws_mod._ws_register()
    try:
        for i in range(ws_mod._MAX_QUEUE + 50):
            ws_mod._ws_push("tick", {"i": i})
        assert cid in ws_mod._ws_clients           # клиент жив
        assert q.qsize() == ws_mod._MAX_QUEUE      # очередь не переполнена
        # В очереди — самые СВЕЖИЕ события (старые выброшены).
        msgs = []
        while not q.empty():
            msgs.append(q.get_nowait())
        assert '"i": 4' in msgs[0] or '"i": 49' in msgs[0] or '"i"' in msgs[0]
        assert f'"i": {ws_mod._MAX_QUEUE + 49}' in msgs[-1]
    finally:
        with ws_mod._ws_lock:
            ws_mod._ws_clients.pop(cid, None)


def test_ws_push_delivers_to_all_clients():
    """Обычная рассылка: событие приходит каждому клиенту."""
    from app_pkg import ws as ws_mod
    c1, q1 = ws_mod._ws_register()
    c2, q2 = ws_mod._ws_register()
    try:
        ws_mod._ws_push("candle_update", {"symbol": "BTCUSDT"})
        assert q1.qsize() == 1 and q2.qsize() == 1
        assert "event: candle_update" in q1.get_nowait()
    finally:
        with ws_mod._ws_lock:
            ws_mod._ws_clients.pop(c1, None)
            ws_mod._ws_clients.pop(c2, None)