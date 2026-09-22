"""Микроструктура стакана крипты: спред, дисбаланс, крупные сделки (Binance spot).

get_micro_snapshot(symbol) -> dict
    Только крипта (суффиксы USDT/BUSD/USDC) — иначе {} (не применимо).
    Схема:
      {"spread_norm", "ob_imb_10", "bid_sum_10", "ask_sum_10",
       "large_trades_ratio", "ok", "ts"} — только числа/None/bool.
    Ошибка сети -> поля null + "ok": false, БЕЗ exception. Кэш 30 секунд.

Источники (REST, бесплатные, как и WS depth20@100ms):
    api.binance.com/api/v3/depth?limit=20        — лучшие 20 уровней bids/asks
    api.binance.com/api/v3/aggTrades?limit=60    — последние агрегированные сделки

Производные:
    spread_norm      = (best_ask - best_bid) / mid — нормированный спред
    ob_imb_10        = (bid_sum - ask_sum) / (bid_sum + ask_sum) по ТОП-10 уровней
                       (+1 = стакан перегружен бидами, -1 = аски доминируют)
    bid_sum_10/ask_sum_10 = суммарный нотионал (цена × объём) ТОП-10 уровней
    large_trades_ratio = доля нотионала «крупных» сделок (>= 3× медианного
                       нотионала) в общем нотионале последних 60 aggTrades.
"""

import logging
import threading
import time

from app_pkg.data.binance import _binance_get
from app_pkg.utils import _clean

log = logging.getLogger(__name__)

REQUEST_TIMEOUT = 5.0
CACHE_TTL = 30.0  # глубина стакана чувствительна к задержке: 30 сек, не 5 мин

# Уровни глубины, по которым считается дисбаланс (из ответа depth limit=20).
_OB_LEVELS = 10
_DEPTH_LIMIT = 20
# Порог «крупной» сделки: не меньше N × медианного нотионала aggTrades.
_LARGE_TRADES_MULT = 3.0
_TRADES_LIMIT = 60
# Клип дисбаланса: редкие «магниты» (стакан на 99% на одной стороне) не
# уводят долю в крайние значения входа модели.
_IMB_CLIP = 0.95

_CRYPTO_SUFFIXES = ("USDT", "BUSD", "USDC")


def _is_crypto(symbol):
    s = str(symbol or "").upper()
    return s.endswith(_CRYPTO_SUFFIXES)


def _blank():
    return {
        "spread_norm": None, "ob_imb_10": None, "bid_sum_10": None,
        "ask_sum_10": None, "large_trades_ratio": None, "ok": False,
        "ts": None,
    }


def _compute(bids, asks, trades):
    """Свёртка сырых ответов Binance -> поля блока (без сети)."""
    out = {
        "spread_norm": None, "ob_imb_10": None, "bid_sum_10": None,
        "ask_sum_10": None, "large_trades_ratio": None,
    }
    try:
        bb = _clean(bids[0][0])
        ba = _clean(asks[0][0])
        if bb is not None and ba is not None and bb > 0 and ba > 0 and ba >= bb:
            mid = (bb + ba) / 2.0
            if mid > 0:
                out["spread_norm"] = round((ba - bb) / mid, 6)

        bid_sum, ask_sum = 0.0, 0.0
        for i in range(min(_OB_LEVELS, len(bids))):
            p, q = _clean(bids[i][0]), _clean(bids[i][1])
            if p is not None and q is not None and p > 0:
                bid_sum += p * q
        for i in range(min(_OB_LEVELS, len(asks))):
            p, q = _clean(asks[i][0]), _clean(asks[i][1])
            if p is not None and q is not None and p > 0:
                ask_sum += p * q
        out["bid_sum_10"] = round(bid_sum, 0)
        out["ask_sum_10"] = round(ask_sum, 0)
        total = bid_sum + ask_sum
        if total > 0:
            imb = (bid_sum - ask_sum) / total
            imb = max(-_IMB_CLIP, min(_IMB_CLIP, imb))
            out["ob_imb_10"] = round(imb, 4)
    except (IndexError, TypeError, ValueError):
        pass

    try:
        notional = []
        for t in trades or []:
            p, q = _clean(t.get("p")), _clean(t.get("q"))
            if p is not None and q is not None and p > 0 and q > 0:
                notional.append(p * q)
        if notional:
            med = sorted(notional)[len(notional) // 2]
            if med and med > 0:
                total = float(sum(notional))
                large = float(sum(n for n in notional if n >= _LARGE_TRADES_MULT * med))
                if total > 0:
                    ratio = large / total
                    out["large_trades_ratio"] = round(
                        max(0.0, min(1.0, ratio)), 4)
    except (TypeError, ValueError):
        pass
    return out


def _fetch_micro(symbol):
    """Живой срез стакана для торговой пары (без кеша, без exception наружу)."""
    depth = _binance_get("/api/v3/depth", {"symbol": symbol,
                                           "limit": _DEPTH_LIMIT}).json()
    trades = _binance_get("/api/v3/aggTrades",
                          {"symbol": symbol, "limit": _TRADES_LIMIT}).json()
    bids = depth.get("bids") or []
    asks = depth.get("asks") or []
    agg = trades if isinstance(trades, list) else []

    out = _compute(bids, asks, agg)
    out["ok"] = True
    out["ts"] = int(time.time())
    return out


# Кэш геттера: снимок сам не кэширует.
_CACHE = {}
_CACHE_LOCK = threading.Lock()


def get_micro_snapshot(symbol):
    """Срез микроструктуры по крипте: TTL-кеш 30 сек -> сеть (без exception).

    Не крипта -> {} (снимок ставит "applicable": false).
    """
    symbol = str(symbol or "").upper()
    if not _is_crypto(symbol):
        return {}

    now = time.time()
    with _CACHE_LOCK:
        item = _CACHE.get(symbol)
        if item and (now - item["ts"]) < CACHE_TTL:
            return item["value"]

    try:
        payload = _fetch_micro(symbol)
    except Exception as exc:  # noqa: BLE001 — сеть/HTTP/JSON: деградация
        log.warning("micro snapshot %s failed: %s", symbol, exc)
        payload = _blank()

    with _CACHE_LOCK:
        _CACHE[symbol] = {"value": payload, "ts": time.time()}
    return payload