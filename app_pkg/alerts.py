"""Алерты: проверка условий по последней свече и доставка уведомлений.

Каналы: browser (SSE через ws._ws_push), webhook (requests POST),
telegram (api.telegram.org).
"""

import logging
import threading
import time

from app_pkg import config
from app_pkg.data.fetch import get_series_df
from app_pkg.db import (
    db_commit,
    db_get_alerts,
    db_mark_alert_triggered,
)
from app_pkg.ws import _ws_push

log = logging.getLogger(__name__)


def _check_alert(alert_row):
    """True, если условие алерта выполнено на последней свече."""
    symbol = alert_row.get("symbol")
    condition = alert_row.get("condition")
    value = alert_row.get("value")
    if not symbol or condition is None or value is None:
        return False
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False

    df = get_series_df(symbol, config.ALERT_CHECK_TIMEFRAME, limit=2,
                       history_limit=config.ALERTS_HISTORY)
    if df is None or df.empty:
        return False
    price = float(df["close"].iloc[-1])
    cond = str(condition).strip().lower()

    if cond in ("above", ">"):
        return price > value
    if cond in ("below", "<"):
        return price < value
    if cond in (">=",):
        return price >= value
    if cond in ("<=",):
        return price <= value
    if cond == "cross":
        if len(df) < 2:
            return False
        prev = float(df["close"].iloc[-2])
        return (prev <= value < price) or (prev >= value > price)
    return False


def _trigger_alert(alert_row):
    """Доставка уведомления по каналу алерта."""
    payload = {
        "id": alert_row.get("id"),
        "symbol": alert_row.get("symbol"),
        "condition": alert_row.get("condition"),
        "value": alert_row.get("value"),
        "message": (
            f"Alert: {alert_row.get('symbol')} "
            f"{alert_row.get('condition')} {alert_row.get('value')}"
        ),
    }
    channel = (alert_row.get("channel") or "browser").lower()
    destination = alert_row.get("destination") or ""

    if channel == "browser":
        _ws_push("alert_triggered", payload)
    elif channel in ("webhook", "http"):
        try:
            import requests
            requests.post(destination, json={"text": payload["message"]}, timeout=10)
        except Exception as exc:  # noqa: BLE001
            log.warning("Webhook alert failed: %s", exc)
    elif channel == "telegram":
        try:
            import requests
            token, chat_id = destination.split(":", 1)
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            requests.post(url, json={"chat_id": chat_id, "text": payload["message"]},
                          timeout=10)
        except Exception as exc:  # noqa: BLE001
            log.warning("Telegram alert failed: %s", exc)
    else:
        _ws_push("alert_triggered", payload)


def _alerts_loop():
    """Фоновый цикл проверки алертов каждые ALERT_CHECK_INTERVAL_SECONDS.

    Сработавшие помечаются через db_mark_alert_triggered (без коммита),
    в конце прохода выполняется ОДИН conn.commit() через db_commit().
    """
    while True:
        try:
            alerts = db_get_alerts(active_only=True)
            for a in alerts:
                if a.get("triggered"):
                    continue
                try:
                    if _check_alert(a):
                        _trigger_alert(a)
                        db_mark_alert_triggered(a["id"])
                except Exception as exc:  # noqa: BLE001
                    log.debug("alert %s check error: %s", a.get("id"), exc)
                    continue
            db_commit()
        except Exception as exc:  # noqa: BLE001
            log.warning("Alerts loop error: %s", exc)
        time.sleep(config.ALERT_CHECK_INTERVAL_SECONDS)


def start_alert_thread():
    """Запустить фоновый цикл алертов (daemon)."""
    th = threading.Thread(target=_alerts_loop, name="alerts-loop", daemon=True)
    th.start()
    return th