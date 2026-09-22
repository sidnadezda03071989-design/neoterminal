# -*- coding: utf-8 -*-
"""Макро-срез: ставки/индекс доллара (FRED) и календарь экономики (Finnhub).

get_macro_snapshot(symbol) -> dict
    Серии FRED DGS10 (US 10Y), FEDFUNDS (ставка ФРС), DTWEXBGS (DXY broad):
      {"us10y","fed_rate","dxy"} каждый {"value","prev_value","change","date"}
      + "ok","ts". Без FRED_API_KEY или при ошибке — блоки null и "ok": false.
    Кэш 24 часа.

get_econ_calendar(ccy, ts_override=None) -> dict
    Finnhub /calendar/economic на период от базового времени (ts_override или
    системные часы) и +1 день; фильтр impact в (high,medium) и country в
    (ccy, US):
      {"events":[{time_unix,event,country,impact,estimate,actual}],
       "high_impact_in_2h": 0|1, "next_event_in_min": int|null, "ok","ts"}.
    ts_override — время среза (replay-safe): окно «сегодня» и флаги считаются
    от него, а не от системных часов. Без ключа/ошибка — events [], флаг 0,
    null. Кэш 10 минут. Не-валюта (не 3 буквы) -> {} (не применимо).
"""

import logging
import threading
import time
from datetime import datetime, timedelta, timezone

import requests

from app_pkg import config
from app_pkg.utils import _clean

log = logging.getLogger(__name__)

REQUEST_TIMEOUT = 5.0
MACRO_CACHE_TTL = 86400.0   # 24 часа
CALENDAR_CACHE_TTL = 600.0  # 10 минут

_FRED_SERIES = {
    "us10y": "DGS10",
    "fed_rate": "FEDFUNDS",
    "dxy": "DTWEXBGS",
}
# DTWEXBGS — месячная серия: последний пункт ещё не опубликован («.»), тянем
# больше наблюдений и отбрасываем нечисловые.
_FRED_LIMIT = 12

_CACHE = {}
_CACHE_LOCK = threading.Lock()


def _num(value, digits=4):
    v = _clean(value)
    return round(v, digits) if v is not None else None


def _parse_obs_value(obs):
    """Значение наблюдения FRED: «.»/пусто/None -> None."""
    if not isinstance(obs, dict):
        return None
    raw = obs.get("value")
    if raw in (None, "", ".", "NaN"):
        return None
    return _num(raw)


def _blank_item():
    return {"value": None, "prev_value": None, "change": None, "date": None}


def _blank_macro():
    return {"us10y": _blank_item(), "fed_rate": _blank_item(),
            "dxy": _blank_item(), "ok": False, "ts": None}


def _fetch_fred_series(series_id):
    """Последние два непустых значения серии: value/prev_value/change/date."""
    if not config.FRED_API_KEY:
        return None
    resp = requests.get(
        f"{config.FRED_BASE_URL}/series/observations",
        params={"series_id": series_id, "api_key": config.FRED_API_KEY,
                "file_type": "json", "sort_order": "desc",
                "limit": _FRED_LIMIT},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    obs = resp.json().get("observations") or []
    values = []
    for row in obs:
        val = _parse_obs_value(row)
        if val is not None:
            values.append({"value": val, "date": row.get("date")})
            if len(values) == 2:
                break
    if not values:
        return None
    latest, prev = values[0], (values[1] if len(values) > 1 else None)
    return {
        "value": latest["value"],
        "prev_value": prev["value"] if prev else None,
        "change": _num(latest["value"] - prev["value"]) if prev else None,
        "date": latest.get("date"),
    }


def _fetch_macro(symbol):
    """Живой макро-срез (без кеша). symbol зарезервирован под контракт."""
    out = {}
    for key, series_id in _FRED_SERIES.items():
        item = _fetch_fred_series(series_id)
        out[key] = item if item is not None else _blank_item()
    out["ok"] = True
    out["ts"] = int(time.time())
    return out


def get_macro_snapshot(symbol):
    """Макро-срез: TTL-кеш 24ч -> FRED (без exception)."""
    if not config.FRED_API_KEY:
        log.info("FRED_API_KEY не задан: макро-блок в null-схеме")
        return _blank_macro()

    now = time.time()
    with _CACHE_LOCK:
        item = _CACHE.get("macro:global")
        if item and (now - item["ts"]) < MACRO_CACHE_TTL:
            return item["value"]

    try:
        payload = _fetch_macro(str(symbol or "").upper())
    except Exception as exc:  # noqa: BLE001 — сеть/HTTP/JSON: деградация
        log.warning("macro snapshot failed: %s", exc)
        payload = _blank_macro()

    with _CACHE_LOCK:
        _CACHE["macro:global"] = {"value": payload, "ts": time.time()}
    return payload


def _is_ccy(ccy):
    """Валидный код валюты: ровно 3 буквы (EUR, USD, JPY...)."""
    return isinstance(ccy, str) and len(ccy) == 3 and ccy.isalpha()


def _parse_event_time(raw):
    """Finnhub время «YYYY-MM-DD HH:MM[:SS]» -> Unix-секунды (UTC)."""
    if not raw or not isinstance(raw, str):
        return None
    try:
        iso = raw.strip().replace(" ", "T")
        if iso.endswith("Z"):
            iso = iso[:-1]
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except ValueError:
        return None


def _blank_calendar(base_ts=None):
    return {
        "events": [], "high_impact_in_2h": 0, "next_event_in_min": None,
        "ok": False, "ts": int(base_ts) if base_ts is not None else None,
    }


def _fetch_calendar(ccy, base_ts):
    """Живой календарь Finnhub на (дата base_ts .. +1 день), без кеша."""
    base = datetime.fromtimestamp(float(base_ts), tz=timezone.utc)
    from_date = base.date()
    to_date = from_date + timedelta(days=1)
    resp = requests.get(
        f"{config.FINNHUB_BASE_URL}/calendar/economic",
        params={"token": config.FINNHUB_API_KEY,
                "from": from_date.isoformat(), "to": to_date.isoformat()},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    raw = resp.json().get("economicCalendar") or []

    events = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        country = str(row.get("country") or "").upper()
        impact = str(row.get("impact") or "").lower()
        if impact not in ("high", "medium") or country not in (ccy, "US"):
            continue
        time_unix = _parse_event_time(row.get("time"))
        if time_unix is None:
            continue
        events.append({
            "time_unix": time_unix,
            "event": str(row.get("event") or ""),
            "country": country,
            "impact": impact,
            "estimate": _num(row.get("estimate")),
            "actual": _num(row.get("actual")),
        })
    events.sort(key=lambda e: e["time_unix"])

    high_in_2h = 0
    upcoming = [e for e in events if e["time_unix"] >= base_ts]
    if any(e["impact"] == "high" and (e["time_unix"] - base_ts) <= 7200
           for e in upcoming):
        high_in_2h = 1
    next_min = None
    if upcoming:
        next_min = int((upcoming[0]["time_unix"] - base_ts) // 60)

    return {
        "events": events, "high_impact_in_2h": high_in_2h,
        "next_event_in_min": next_min, "ok": True, "ts": int(base_ts),
    }


def get_econ_calendar(ccy, ts_override=None):
    """Экономический календарь по валюте: TTL-кеш 10 мин -> Finnhub.

    ts_override (replay-safe) — базовое время среза вместо системных часов.
    """
    ccy = str(ccy or "").upper()
    if not _is_ccy(ccy):
        return {}
    if not config.FINNHUB_API_KEY:
        log.info("FINNHUB_API_KEY не задан: календарь в пустой схеме")
        return _blank_calendar(ts_override)

    base_ts = float(ts_override) if ts_override is not None else time.time()
    key = f"calendar:{ccy}:{int(base_ts // CALENDAR_CACHE_TTL)}"
    now = time.time()
    with _CACHE_LOCK:
        item = _CACHE.get(key)
        if item and (now - item["ts"]) < CALENDAR_CACHE_TTL:
            return item["value"]

    try:
        payload = _fetch_calendar(ccy, base_ts)
    except Exception as exc:  # noqa: BLE001 — сеть/HTTP/JSON: деградация
        log.warning("econ calendar %s failed: %s", ccy, exc)
        payload = _blank_calendar(base_ts)

    with _CACHE_LOCK:
        _CACHE[key] = {"value": payload, "ts": time.time()}
    return payload