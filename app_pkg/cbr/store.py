"""Запись снимков в CBR-БД: вектор 44 фич + хэш-дедуп + режим/сессия.

Порядок 44 фич (`FEATURE_ORDER`) совпадает бит-в-бит с колонками
`labels.charon_features` (marker-порядок: 39 числовых полей блоков +
clock.hour_utc/dow/london_ny_overlap/market_open/session_code) — ВАЖНО
не импортировать labels на уровне модуля: labels -> market_snapshot ->
cbr.store образует цикл. Соответствие проверяется тестом
test_cbr::test_feature_order_matches_labels.

snapshot_to_vector читает ТРИ формата источника:
  - flat     — блоки в плоских ключах "block.field" (строки charon_features
               / произвольные); session из clock.session_code;
  - compact  — схема v3 compact_snapshot (короткие ключи t/v/tr/mo/vl/rg/
               div/cnd/c; сессия — человеческое имя в c.ses);
  - raw      — блоки get_raw_market_data (technicals/volume/... + clock;
               сессия — имя в clock.session).
Отсутствующее поле -> 0.0: важен не «мнимый ноль», а стабильность:
одно и то же состояние рынка всегда даёт один и тот же float32-вектор и
один и тот же feature_hash (дедуп).

classify_regime — детерминированный режим из 3 фич (hurst/adx/ema20_50_ratio):
рахustный тренд (hurst>0.55 И adx>25) -> trend_up/trend_down по ema-сплиту;
mean-reverting (hurst<0.45) -> reversal; иначе flat.
"""

import hashlib
import json
import logging
import sqlite3
import threading

import numpy as np

from app_pkg import config, utils

log = logging.getLogger(__name__)

# Индекс ATR в 44-векторе: technicals.atr — 2-я фича (ATR_INDEX=1).
ATR_INDEX = 1

# --------------------------------------------------------------- порядок 44
# (raw_block, raw_field, compact_block, compact_key). raw_block.field —
# точное имя колонки charon_features.
FEATURE_ORDER = [
    ("technicals", "rsi", "t", "rsi"),
    ("technicals", "atr", "t", "atr"),
    ("technicals", "bb_pct_b", "t", "bb"),
    ("technicals", "sma20_diff_pct", "t", "sma"),
    ("technicals", "atr_percentile", "t", "ap"),
    ("technicals", "dist_to_high_pct", "t", "dh"),
    ("technicals", "dist_to_low_pct", "t", "dl"),
    ("volume", "rel_vol", "v", "rv"),
    ("volume", "obv_slope", "v", "obv"),
    ("volume", "vol_zscore", "v", "vz"),
    ("volume", "vol_percentile", "v", "vp"),
    ("volume", "vwap_dev", "v", "vwd"),
    ("trend", "adx", "tr", "adx"),
    ("trend", "plus_di", "tr", "pdi"),
    ("trend", "minus_di", "tr", "mdi"),
    ("trend", "di_spread", "tr", "ds"),
    ("trend", "ema20_50_ratio", "tr", "er"),
    ("trend", "reg_slope_20", "tr", "rs"),
    ("momentum", "macd", "mo", "macd"),
    ("momentum", "macd_signal", "mo", "ms"),
    ("momentum", "macd_hist", "mo", "mh"),
    ("momentum", "macd_hist_slope", "mo", "mhs"),
    ("momentum", "rsi_slope", "mo", "rsi_s"),
    ("volatility", "bb_width", "vl", "bw"),
    ("volatility", "bb_width_pct", "vl", "bwp"),
    ("volatility", "hv20", "vl", "hv"),
    ("volatility", "atr_change_pct", "vl", "atc"),
    ("volatility", "atr_ratio_short_long", "vl", "atr_r"),
    ("regime", "hurst", "rg", "h"),
    ("regime", "autocorr_lag1", "rg", "ac1"),
    ("regime", "efficiency_ratio", "rg", "er"),
    ("divergence", "rsi_price", "div", "rsi"),
    ("divergence", "macd_price", "div", "macd"),
    ("divergence", "obv_price", "div", "obv"),
    ("candle", "body_ratio", "cnd", "br"),
    ("candle", "upper_wick_ratio", "cnd", "uw"),
    ("candle", "lower_wick_ratio", "cnd", "lw"),
    ("candle", "engulfing", "cnd", "eng"),
    ("candle", "pinbar", "cnd", "pin"),
    ("clock", "hour_utc", "c", "h"),
    ("clock", "dow", "c", "dow"),
    ("clock", "london_ny_overlap", "c", "ovl"),
    ("clock", "market_open", "c", "open"),
    ("clock", "session_code", "c", "ses"),
]
assert len(FEATURE_ORDER) == 44

# Имена колонок charon_features в порядке вектора (для маппинга flat-формата).
FEATURE_NAMES = [f"{block}.{field}" for block, field, _, _ in FEATURE_ORDER]

# Индексы фич, нужные классификаторам (см. FEATURE_ORDER).
_IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}
_IDX_ADX = _IDX["trend.adx"]
_IDX_HURST = _IDX["regime.hurst"]
_IDX_EMA_RATIO = _IDX["trend.ema20_50_ratio"]

# Сессионные коды совпадают с labels._SESSION_CODE (дублируются специально,
# чтобы cbr.store не импортировал labels — см. докстринг модуля).
_SESSION_CODE = {"asia": 0, "london": 1, "london_ny": 2, "ny": 3,
                 "off_hours": 4, "weekend": 5}
_SESSION_NAME = {v: k for k, v in _SESSION_CODE.items()}


# ------------------------------------------------------------ конвертация
def _fnum(value):
    """float-значение или None (NaN/Inf не должны попасть в вектор)."""
    v = utils._clean(value)
    return v if v is not None else None


def snapshot_to_vector(snapshot):
    """Снимок рынка -> np.ndarray float32 (44,).

    Поддержаны flat ("block.field"), compact (короткие ключи v3) и raw
    (технические блоки market_snapshot) форматы; отсутствующее поле -> 0.0.
    Кидает TypeError на не-dict.
    """
    if not isinstance(snapshot, dict):
        raise ValueError(  # noqa: TRY004 — контракт c тестами
            "snapshot must be a dict, got "
            f"{type(snapshot).__name__}")
    vec = np.zeros(len(FEATURE_ORDER), dtype=np.float32)

    def _block(compact_key, raw_key):
        if not isinstance(snapshot, dict):
            return {}
        for key in (compact_key, raw_key):
            blk = snapshot.get(key)
            if isinstance(blk, dict):
                return blk
        return {}

    for i, (raw_block, raw_field, c_key, c_field) in enumerate(FEATURE_ORDER):
        if raw_block == "clock" and raw_field == "session_code":
            continue  # сессия обрабатывается отдельно ниже
        val = None
        # 1) плоский формат: точная колонка charon_features
        val = snapshot.get(f"{raw_block}.{raw_field}")
        if val is not None:
            val = _fnum(val)
        # 2) compact-блок (короткие ключи)
        if val is None:
            val = _fnum(_block(c_key, raw_block).get(c_field))
        # 3) raw-блок (полные имена полей)
        if val is None:
            val = _fnum(_block(c_key, raw_block).get(raw_field))
        if val is not None:
            vec[i] = val
    # clock.session_code: плоский -> число; compact/raw -> имя -> код.
    ses = snapshot.get("clock.session_code")
    if ses is None:
        ses = _block("c", "clock").get("ses")
    if ses is None:
        ses = _block("c", "clock").get("session")
    code = utils._clean(ses)
    if code is not None:
        vec[_IDX["clock.session_code"]] = code
    elif isinstance(ses, str):
        vec[_IDX["clock.session_code"]] = _SESSION_CODE.get(
            ses.strip().lower(), -1.0)
    return vec


def _snapshot_close(snapshot):
    """Цена close из снимка (compact t.close / raw technicals.close)."""
    for key in ("t", "technicals"):
        blk = snapshot.get(key)
        if isinstance(blk, dict):
            val = _fnum(blk.get("close"))
            if val is not None:
                return val
    return None


def _snapshot_session(snapshot):
    """(имя_сессии, код) из снимка; (None, None), если нет данных."""
    blk = None
    for key in ("c", "clock"):
        cand = snapshot.get(key)
        if isinstance(cand, dict):
            blk = cand
            break
    if blk is None:
        return None, None
    name = None
    for key in ("ses", "session"):
        val = blk.get(key)
        if isinstance(val, str) and val.strip():
            val = val.strip().lower()
            if val in _SESSION_CODE:
                name = val
                break
    code = utils._clean(blk.get("session_code"))
    if name is not None:
        return name, _SESSION_CODE[name]
    if code is not None:
        return _SESSION_NAME.get(int(code)), int(code)
    if isinstance(blk.get("session_code"), (int, float)):
        return _SESSION_NAME.get(int(blk["session_code"])), int(blk["session_code"])
    return None, None


# ------------------------------------------------------------- классификатор
def classify_regime(features):
    """Режим из вектора: trend_up/trend_down (hurst>0.55 + adx>25, H>0.45),
    reversal (hurst<0.45), иначе flat."""
    f = np.asarray(features, dtype="float64")
    hurst = float(f[_IDX_HURST]) if len(f) > _IDX_HURST else 0.0
    adx = float(f[_IDX_ADX]) if len(f) > _IDX_ADX else 0.0
    ema_ratio = float(f[_IDX_EMA_RATIO]) if len(f) > _IDX_EMA_RATIO else 1.0
    if hurst > 0.55 and adx > 25.0:
        return "trend_up" if ema_ratio > 1.0 else "trend_down"
    if hurst < 0.45:
        return "reversal"
    return "flat"


def _atr_pct(features, entry_price):
    """ATR(14)/entry по вектору; None, если atr или entry невалидны."""
    entry = utils._clean(entry_price)
    atr = float(features[ATR_INDEX]) if len(features) > ATR_INDEX else 0.0
    if entry is None or entry <= 0 or atr <= 0:
        return None
    return atr / entry


# ---------------------------------------------------------------- запись
def _levels_payload(levels):
    """[{delta, prob}] -> JSON-текст для levels_json (None без данных)."""
    items = []
    for lv in levels or []:
        if not isinstance(lv, dict):
            continue
        delta = utils._clean(lv.get("delta"))
        prob = utils._clean(lv.get("prob"))
        if delta is None and prob is None:
            continue
        items.append({
            "delta": round(delta, 6) if delta is not None else None,
            "prob": round(prob, 4) if prob is not None else None,
        })
    return json.dumps(items, ensure_ascii=False, separators=(",", ":")) \
        if items else None


def store_snapshot(conn, snapshot, entry_price=None, commit=True):
    """Записывает снимок в CBR-БД; возвращает id строки или None (дубликат).

    Дубликат определяется по feature_hash (md5 "symbol:tf:ts:" +
    float32-вектора) — INSERT OR IGNORE. entry_price: явный аргумент > snapshot.entry_price >
    close снимка (t.close/technicals.close). atr_pct считается ТОЛЬКО если
    atr и entry валидны (для backfill). source по умолчанию 'backtest'.
    Кидает ValueError на отсутствие symbol/timeframe/ts.
    """
    symbol = str(snapshot.get("symbol") or "").strip().upper()
    timeframe = str(snapshot.get("timeframe") or "").strip()
    if not symbol or not timeframe:
        raise ValueError("snapshot must have symbol and timeframe")
    ts = utils._clean(snapshot.get("ts"))
    if ts is None:
        raise ValueError("snapshot must have numeric ts")
    features = snapshot_to_vector(snapshot)

    entry = utils._clean(entry_price) if entry_price is not None else None
    if entry is None:
        entry = utils._clean(snapshot.get("entry_price"))
    if entry is None:
        entry = _snapshot_close(snapshot)
    atr_pct = _atr_pct(features, entry)

    est_regime = classify_regime(features)
    session_name, _ = _snapshot_session(snapshot)

    verdict = snapshot.get("verdict")
    if verdict is None:
        sig = snapshot.get("sig")
        verdict = sig if sig in ("L", "S", "F", "FLAT", "NULL") else None
    if verdict is not None:
        verdict = str(verdict).strip().upper()
        if verdict not in ("L", "S", "F", "FLAT", "NULL"):
            verdict = None
    if verdict == "FLAT":
        verdict = "F"
    if verdict == "NULL":
        verdict = None

    levels_json = _levels_payload(snapshot.get("levels"))
    source = str(snapshot.get("source") or "backtest").strip() or "backtest"
    # Уникальность по (symbol, tf, ts), не по вектору: одинаковые features на
    # разных барах — разные записи (у них разный outcome).
    feature_hash = hashlib.md5(
        f"{symbol}:{timeframe}:{int(ts)}:".encode() + features.tobytes()
    ).hexdigest()

    cur = conn.execute(
        "INSERT OR IGNORE INTO snapshots "
        "(symbol, timeframe, ts, features, feature_hash, regime, session, "
        " verdict, levels_json, entry_price, atr_pct, source) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (symbol, timeframe, int(ts),
         json.dumps([float(v) for v in features.tolist()],
                    ensure_ascii=False, separators=(",", ":")),
         feature_hash, est_regime, session_name,
         verdict, levels_json, entry, atr_pct, source),
    )
    if commit:
        conn.commit()
    return cur.lastrowid if cur.rowcount else None


# ------------------------------------------------------------ подключение
_conn_cache = {"path": None, "conn": None}
_conn_lock = threading.Lock()


def get_cbr_conn():
    """Ленивое глобальное CBR-подключение (WAL, check_same_thread=False).

    При смене config.CBR_DB_PATH соединение переоткрывается — тесты
    monkeypatch'ят путь на tmp-файл и получают изолированную БД.
    """
    path = str(config.CBR_DB_PATH)
    with _conn_lock:
        cached = _conn_cache["conn"]
        if cached is not None and _conn_cache["path"] == path:
            try:
                cached.execute("SELECT 1").fetchone()
                return cached
            except sqlite3.Error:
                pass  # оборванный коннект (файл удалён) — переоткрываем
        from app_pkg.cbr import schema
        conn = schema.init_db(path=path)
        _conn_cache["path"] = path
        _conn_cache["conn"] = conn
        return conn


def cbr_enabled():
    """Гейт записи снимков из pipeline (config.CBR_ENABLED)."""
    return bool(config.CBR_ENABLED)