"""Blueprint: GET /api/charon_signal — сигнал Charon по кнопке.

Обновление ТОЛЬКО по явному запросу пользователя (кнопка «Запросить сигнал»):
никаких таймеров/автообновлений на сервере. Возвращает результат
детерминированных правил 1-20 (app_pkg.ai.apply_rules) для текущей пары
symbol+tf без обращения к LLM.
"""

import logging
from concurrent.futures import ThreadPoolExecutor

from flask import Blueprint, jsonify, request

from app_pkg import config
from app_pkg.ai.apply_rules import apply_all_rules
from app_pkg.data.market_snapshot import compact_snapshot

bp = Blueprint("charon", __name__)
log = logging.getLogger(__name__)

# Сборка снимка уходит в отдельный пул с таймаутом: медленный форекс-источник
# (MT5/yfinance) не должен держать thread Waitress. По таймауту/ошибке —
# {"error": ...} без фиктивного сигнала. Снимок собирается только для правил
# 1-20 (rules_only: без macro/derivatives/volume/volatility/regime/context/
# candle и их внешних источников) и при upto_sec (реплей) — из исторического
# среза без обращения к live-данным, поэтому <1с на тёплом сервере.
_SIGNAL_TIMEOUT_SEC = 60.0
_signal_exec = ThreadPoolExecutor(max_workers=2, thread_name_prefix="charon-signal")


def _signal(symbol: str, tf: str, upto_sec=None) -> dict:
    snap = compact_snapshot(symbol, tf, upto_sec=upto_sec, rules_only=True)
    return apply_all_rules(snap)


@bp.route("/api/charon_signal", methods=["GET"])
def charon_signal():
    """Правила Charon 1-20 для symbol+tf (только по клику клиента).

    upto_sec (опционально) — timestamp текущего бара реплея: снимок строится
    по историческому срезу из replay (без взгляда в будущее), а не по live.
    """
    symbol = str(request.args.get("symbol") or "").upper()
    tf = str(request.args.get("tf") or "").strip()
    upto_arg = request.args.get("upto_sec")
    if symbol not in config.SYMBOLS:
        return jsonify({"error": f"Invalid symbol: {symbol}"}), 400
    if tf not in config.TF_SECONDS:
        return jsonify({"error": f"Invalid timeframe: {tf}"}), 400
    upto_sec = None
    if upto_arg:
        try:
            upto_sec = float(upto_arg)
        except (TypeError, ValueError):
            return jsonify({"error": "Invalid upto_sec"}), 400
    future = _signal_exec.submit(_signal, symbol, tf, upto_sec)
    try:
        result = future.result(timeout=_SIGNAL_TIMEOUT_SEC)
    except Exception:  # noqa: BLE001 — таймаут/битый источник
        log.warning("charon_signal timeout/error for %s %s (limit %.0fs)",
                    symbol, tf, _SIGNAL_TIMEOUT_SEC)
        future.cancel()
        return jsonify({"error": "timeout"}), 504
    return jsonify(result)