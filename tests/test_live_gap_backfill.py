"""Регрессия «после закрытия свечи пропадают, разрывы между свечами».

Сценарий, воспроизводящий жалобу «реальные свечи после закрытия пропадают:
  - при холодной загрузке /api/data REST-ответ включает ПОДТВЕРЖДЁННЫЙ
    (forming) бар F0, кэш заканчивается F0;
  - кэш крипты живёт TTL 3600с и его правая граница не сдвигается сама;
  - live_bars хранит ТОЛЬКО текущий бар (после закрытия F0 там уже F1/F2),
    закрытые свечи F1, ... в кэш НИКТО не дописывает;
  - полный reload отдаёт «REST-старый + ОДИН живой бар»: закрытые свечи,
    накопленные фронтом по SSE, с графика ПРОПАДАЮТ -> разрывы и «гигантская»
    хвостовая свеча (open/high/low прыгают от старого края кэша).

Фикс: на каждом чтении get_series_df добирает (backfill) свежие ЗАКРЫТЫЕ бары
из REST правее края кэша, но не запекает в кэш формирующийся live-бар.
"""

import itertools
import time

import pandas as pd
import pytest

from app_pkg import config, create_app
from app_pkg.data import fetch, live

_BASE = 1_000_000_000
_STEP = 60


def _df(rows):
    return pd.DataFrame({
        "timestamp": pd.to_datetime([r[0] for r in rows], unit="s", utc=True),
        "open": [float(r[1]) for r in rows],
        "high": [float(r[2]) for r in rows],
        "low": [float(r[3]) for r in rows],
        "close": [float(r[4]) for r in rows],
        "volume": [float(r[5]) for r in rows],
    })


def _live_bar(ts, close):
    """Кладёт формирующийся бар в live_bars (retval — cleanup)."""
    with live._live_bars_lock:
        live.live_bars[("BTCUSDT", "1m")] = {
            "ts": int(ts), "open": 100.0, "high": 101.0, "low": 99.0,
            "close": close, "volume": 1.0, "closed": False,
        }

    def _cleanup():
        with live._live_bars_lock:
            live.live_bars.pop(("BTCUSDT", "1m"), None)
    return _cleanup


def _make_source(now_ref):
    """Фейк источника: отдаёт «правду» с шагом _STEP от _BASE до now_ref[0].

    end_sec игнорируется намеренно: реальный fetch_ohlcv капает в нём
    time.time() (wall-clock), а тут «сейчас» — симулированное now_ref[0].
    """
    def fetch_ohlcv(symbol, tf, limit=1000, start_sec=None, end_sec=None, **kw):
        end = now_ref[0]
        rows = []
        i = 0
        while _BASE + _STEP * i <= end:
            ts = _BASE + _STEP * i
            if start_sec is None or ts >= start_sec:
                rows.append((ts, 100.0, 101.0, 99.0, 100.0 + 0.1 * i, 1.0 + i))
            i += 1
        return _df(rows).tail(int(limit))
    return fetch_ohlcv


@pytest.fixture(autouse=True)
def _isolate():
    fetch.data_cache.clear()
    live.live_bars.clear()
    yield
    fetch.data_cache.clear()
    live.live_bars.clear()


# ------------------------------------------------- cache-hit с «дырой» в 1 свечу
def test_reload_cache_hit_backfills_closed_bar(monkeypatch):
    """Кэш [P0, F0], live=F2: reload обязан вернуть и F1 (закрыт), и F2 (live).

    До фикса вывод был [P0, F0, F2] — F1 «пропадал» (разрыв между свечами).
    """
    now_ref = [_BASE + 2 * _STEP]
    monkeypatch.setattr(fetch, "fetch_ohlcv", _make_source(now_ref))
    fetch.data_cache[("BTCUSDT", "1m")] = {
        "df": _df([
            (_BASE - _STEP, 100.0, 101.0, 99.0, 100.0, 1.0),
            (_BASE, 100.1, 101.2, 99.5, 100.5, 2.0),      # F0 (был forming)
        ]),
        "ts": time.time(),
        "limit": 2,
    }
    cleanup = _live_bar(_BASE + 2 * _STEP, 100.9)          # F2 (forming)
    try:
        df = fetch.get_series_df("BTCUSDT", "1m", limit=2)
        assert [int(t.timestamp()) for t in df["timestamp"]] == \
            [_BASE + _STEP, _BASE + 2 * _STEP], \
            "закрытая свеча F1 пропала после закрытия"
        # Формирующийся live-бар НЕ должен запекаться в кэш.
        cached_last = int(fetch.data_cache[("BTCUSDT", "1m")]["df"]["timestamp"].iloc[-1].timestamp())
        assert cached_last == _BASE + _STEP, \
            "live-бар запечён в кэш (формирующийся бар не должен попадать)"
    finally:
        cleanup()


# ------------------------------------------- ветка докачки старых + фикс правого края
def test_reload_topup_keeps_closed_bar(monkeypatch):
    """Кэш [F0], live=F2, limit=3: получаем полную последовательность F0,F1,F2."""
    now_ref = [_BASE + 2 * _STEP]
    monkeypatch.setattr(fetch, "fetch_ohlcv", _make_source(now_ref))
    fetch.data_cache[("BTCUSDT", "1m")] = {
        "df": _df([(_BASE, 100.1, 101.2, 99.5, 100.5, 2.0)]),
        "ts": time.time(),
        "limit": 1,
    }
    cleanup = _live_bar(_BASE + 2 * _STEP, 100.9)
    try:
        df = fetch.get_series_df("BTCUSDT", "1m", limit=3)
        times = [int(t.timestamp()) for t in df["timestamp"]]
        assert times == [_BASE, _BASE + _STEP, _BASE + 2 * _STEP], times
    finally:
        cleanup()


# --------------------------- /api/data (black-box): во времени нет разрывов
def test_full_api_reload_has_no_gaps(monkeypatch):
    """GET /api/data после закрытия свечи возвращает непрерывный ряд свечей.

    Именно это визуально ломало график: после reload свечи начинали идти с
    «разрывом» (пропущенная закрытая свеча), а у tail-бара был прыжок open.
    """
    now_ref = [_BASE + 2 * _STEP]
    # Даём клиенту кэш, характерный для «холодной» загрузки пару минут назад.
    fetch.data_cache[("BTCUSDT", "1m")] = {
        "df": _df([
            (_BASE - _STEP, 100.0, 101.0, 99.0, 100.0, 1.0),
            (_BASE, 100.1, 101.2, 99.5, 100.5, 2.0),
        ]),
        "ts": time.time(),
        "limit": 4,
    }
    monkeypatch.setattr(fetch, "fetch_ohlcv", _make_source(now_ref))
    cleanup = _live_bar(_BASE + 2 * _STEP, 100.9)
    try:
        app = create_app()
        app.config["TESTING"] = True
        resp = app.test_client().get(
            "/api/data?symbol=BTCUSDT&timeframe=1m&limit=4")
        assert resp.status_code == 200
        candles = resp.get_json()["candles"]
        times = [int(c["time"]) for c in candles]
        deltas = [b - a for a, b in itertools.pairwise(times)]
        assert deltas and all(d == _STEP for d in deltas), \
            f"разрыв в /api/data: времена {times}"
    finally:
        cleanup()


# ------------------ форекс: суб-дневной ТФ ведёт себя как крипта (live-bар есть)
def test_forex_subday_live_backfills_closed_bar(monkeypatch):
    """EURUSD 5m: формирующийся бар форекс-поллинга -> дожимаем закрытые.

    У крипты live-бар даёт Binance WS на всех ТФ; у форекса его заполняет
    _forex_live_loop для ОТКРЫТЫХ пар. Без этого евро-5m не получал ни
    merge, ни backfill, и после закрытия свечи она «пропадала» (gap).
    """
    step = 300  # 5m
    base = int(time.time()) // (10 * step) * (10 * step)  # ровный крат 10 бар
    now_ref = [base + 2 * step]

    def forex_source(symbol, tf, limit=1000, start_sec=None, end_sec=None, **kw):
        rows = []
        i = 0
        while base + step * i <= now_ref[0]:
            ts = base + step * i
            if start_sec is None or ts >= start_sec:
                rows.append((ts, 1.0800, 1.0805, 1.0795, 1.0802 + 0.0001 * i, 5 + i))
            i += 1
        return _df(rows).tail(int(limit))

    monkeypatch.setattr(fetch, "fetch_ohlcv", forex_source)
    # Кэш книзу: [B0, B1] (B1 был «forming» при холодной загрузке), live = B2.
    fetch.data_cache[("EURUSD", "5m")] = {
        "df": _df([
            (base - step, 1.0800, 1.0805, 1.0795, 1.0800, 5.0),
            (base, 1.0801, 1.0806, 1.0796, 1.0803, 6.0),
        ]),
        "ts": time.time(),
        "limit": 2,
    }
    with live._live_bars_lock:
        live.live_bars[("EURUSD", "5m")] = {
            "ts": int(base + 2 * step), "open": 1.0802, "high": 1.0807,
            "low": 1.0797, "close": 1.0805, "volume": 8.0, "closed": False,
        }

    try:
        df = fetch.get_series_df("EURUSD", "5m", limit=3)
        times = [int(t.timestamp()) for t in df["timestamp"]]
        # Непрерывный ряд tail(3): [B0(из кэша), B1(дожим закрытого), B2(live)].
        assert times == [base, base + step, base + 2 * step], times
        # Формирующийся live-бар не запекается в кэш и для форекса.
        cached_last = int(fetch.data_cache[("EURUSD", "5m")]["df"]["timestamp"].iloc[-1].timestamp())
        assert cached_last == base + step, cached_last
    finally:
        with live._live_bars_lock:
            live.live_bars.pop(("EURUSD", "5m"), None)


# ------------------ форекс: _forex_live_tick заполняет live_bars по активным парам
def test_forex_live_tick_fills_subday_bar_for_active_pair(monkeypatch):
    """Подписанная форекс-пара получает формирующийся бар на всех ТФ.

    Раньше _forex_live_loop писал только (symbol, "1D") — суб-дневные ТФ без
    live-бара: ни merge, ни backfill, ни SSE. Теперь для открытых пар бар
    берётся из tail источника и попадает в live_bars.
    """
    now = int(time.time())
    step = 300  # 5m
    df = _df([
        (now - step, 1.0800, 1.0805, 1.0795, 1.0800, 5.0),
        (now, 1.0802, 1.0810, 1.0798, 1.0806, 9.0),   # forming
    ])

    monkeypatch.setattr(fetch, "fetch_ohlcv", lambda *a, **k: df)
    live.set_active_pairs([("EURUSD", "5m")])
    live._forex_live_tick()
    try:
        key = ("EURUSD", "5m")
        assert key in live.live_bars
        bar = live.live_bars[key]
        assert int(bar["ts"]) == now
        assert bar["close"] == 1.0806
        assert bar["closed"] is False
    finally:
        live.set_active_pairs([])
        with live._live_bars_lock:
            live.live_bars.pop(("EURUSD", "5m"), None)

        # Форекс-бар не должен запекаться в кэш: тик не трогает data_cache.
        assert ("EURUSD", "5m") not in fetch.data_cache


def test_forex_live_tick_fallback_polls_1d_without_active_pairs(monkeypatch):
    """Без открытых форекс-пар тик греет только дневные бары (как раньше)."""
    now = int(time.time()) // 86400 * 86400
    df = _df([(now, 1.0800, 1.0810, 1.0790, 1.0805, 12.0)])

    monkeypatch.setattr(fetch, "fetch_ohlcv",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError(
                            "fetch_ohlcv не должен вызываться для суб-дневного ТФ")))
    monkeypatch.setattr(live, "yfinance", type("YF", (), {
        "fetch_yfinance": lambda *a, **k: df,
    }))
    live.set_active_pairs([("BTCUSDT", "1m")])   # только крипта — форекс-грева нет
    live._forex_live_tick()
    try:
        key = ("EURUSD", "1D")
        assert key in live.live_bars
        assert int(live.live_bars[key]["ts"]) == now
    finally:
        live.set_active_pairs([])
        with live._live_bars_lock:
            live.live_bars.pop(key, None)


# --------------------------- большой разрыв (< кэш сильно устарел): все свечи на месте
def test_reload_large_gap_rebuilds_tail(monkeypatch):
    """Кэш отстал сильно (> CAP баров): тайл перезаливается, разрывов нет."""
    now_ref = [_BASE + 5 * _STEP]
    monkeypatch.setattr(fetch, "fetch_ohlcv", _make_source(now_ref))
    monkeypatch.setattr(fetch, "_EXTEND_NEWER_CAP", 3)   # форсируем ветку перезалитого тайла
    fetch.data_cache[("BTCUSDT", "1m")] = {
        "df": _df([(_BASE, 100.1, 101.2, 99.5, 100.5, 2.0)]),
        "ts": time.time(),
        "limit": 1,
    }
    cleanup = _live_bar(_BASE + 5 * _STEP, 100.9)
    try:
        df = fetch.get_series_df("BTCUSDT", "1m", limit=6)
        times = [int(t.timestamp()) for t in df["timestamp"]]
        expected = [_BASE + _STEP * i for i in range(6)]
        assert times == expected, times
        # Формирующийся live-бар и после перезалива не запекается в кэш.
        cached_last = int(fetch.data_cache[("BTCUSDT", "1m")]["df"]["timestamp"].iloc[-1].timestamp())
        assert cached_last == _BASE + 4 * _STEP, cached_last
    finally:
        cleanup()


# --------------- адаптивный интервал форекс-поллинга (1m «догоняет» свечу)
def test_forex_poll_interval_adapts_to_1m_active_pair(monkeypatch):
    """1m-пара: поллинг должен идти ЧАЩЕ 90с (иначе новая свеча пропускается).

    Раньше _forex_live_loop спал фиксированные FOREX_LIVE_POLL_INTERVAL=90с,
    что дольше длины 1m-свечи (60с): на границе минуты poll мог «не успеть»
    и целая свеча не попадала в live_bars -> «на 1m графике не добавляет
    свечи новые». Для мелкого ТФ интервал должен быть долей его длины.
    """
    from app_pkg.data.mt5 import mt5_state

    monkeypatch.setitem(mt5_state, "ok", True)
    live.set_active_pairs([("EURUSD", "1m")])
    try:
        interval = live._forex_poll_interval()
        assert 0 < interval < 10, f"1m должен поллиться быстро, а не {interval}с"
    finally:
        live.set_active_pairs([])
        monkeypatch.setitem(mt5_state, "ok", False)


def test_forex_poll_interval_caps_at_90s_for_1d(monkeypatch):
    """Только дневные пары -> прежние 90с (греем 1D без суеты)."""
    from app_pkg.data.mt5 import mt5_state

    monkeypatch.setitem(mt5_state, "ok", True)
    live.set_active_pairs([("EURUSD", "1D")])
    try:
        assert live._forex_poll_interval() == config.FOREX_LIVE_POLL_INTERVAL
    finally:
        live.set_active_pairs([])
        monkeypatch.setitem(mt5_state, "ok", False)


def test_forex_poll_interval_no_active_keeps_90s(monkeypatch):
    """Нет открытых форекс-пар -> прежние 90с."""
    from app_pkg.data.mt5 import mt5_state

    monkeypatch.setitem(mt5_state, "ok", True)
    live.set_active_pairs([])
    try:
        assert live._forex_poll_interval() == config.FOREX_LIVE_POLL_INTERVAL
    finally:
        live.set_active_pairs([])
        monkeypatch.setitem(mt5_state, "ok", False)


def test_forex_poll_interval_slow_when_mt5_down(monkeypatch):
    """MT5 недоступен (источник — yfinance): не дёргаем сеть чаще 90с."""
    from app_pkg.data.mt5 import mt5_state

    monkeypatch.setitem(mt5_state, "ok", False)
    live.set_active_pairs([("EURUSD", "1m")])
    try:
        assert live._forex_poll_interval() == config.FOREX_LIVE_POLL_INTERVAL
    finally:
        live.set_active_pairs([])