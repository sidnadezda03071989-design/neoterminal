"""Детерминированные уровни и калибровка вероятностей достижения."""
from __future__ import annotations

import json
import time
from typing import Any

from app_pkg.ai.calibration import CALIBRATOR

BB_OVERSOLD = 0.125
BB_OVERBOUGHT = 0.875
LEVEL_REACH = "LEVEL_REACH"
LEVEL_HORIZON_BARS = 20
MIN_LEVEL_SAMPLES = 50
LEVEL_PRIORS = {
    "VAH": 0.30,
    "VAL": 0.40,
    "POC": 0.30,
    "STRUCT_HIGH": 0.25,
    "STRUCT_LOW": 0.25,
}
_LEVEL_HISTORY_CACHE = {"stamp": 0.0, "rows": None}


def _num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _prob(value: Any, default: float = 0.0) -> float:
    result = _num(value)
    if result is None:
        return default
    if result > 1.0:
        result /= 100.0
    return max(0.0, min(1.0, result))


def _block(snap: dict, name: str) -> dict:
    block = snap.get(name) if isinstance(snap, dict) else None
    return block if isinstance(block, dict) else {}


def _val(snap: dict, block: str, key: str) -> float | None:
    return _num(_block(snap, block).get(key))


def _trend_sign(snap: dict) -> int:
    """Правило 16: знак тренда при tr.adx>20 (+1 pdi>mdi, -1 mdi>pdi)."""
    adx = _val(snap, "tr", "adx")
    pdi = _val(snap, "tr", "pdi")
    mdi = _val(snap, "tr", "mdi")
    if adx is None or adx <= 20 or pdi is None or mdi is None:
        return 0
    if pdi > mdi:
        return 1
    if mdi > pdi:
        return -1
    return 0


def _gated(direction: int, amount: float, trend: int) -> float:
    """Правило 16: направленный вклад нейтрален, если идёт против тренда.

    direction: +1 для pu-ветки, -1 для pd-ветки. trend: знак тренда из
    _trend_sign (0 = тренд не задан).
    """
    if trend and direction != trend:
        return 0.0
    return amount


def _level_history() -> list[tuple[str, float, int]]:
    now = time.monotonic()
    cached = _LEVEL_HISTORY_CACHE.get("rows")
    if cached is not None and now - float(_LEVEL_HISTORY_CACHE.get("stamp", 0.0)) < 5.0:
        return cached
    try:
        from app_pkg import db
        conn = db._get_db()
        rows = conn.execute(
            "SELECT side, raw_prob, hit, context_json "
            "FROM charon_calibration_history "
            "WHERE side=? OR side IN (?,?,?,?,?)",
            (LEVEL_REACH, *LEVEL_PRIORS.keys()),
        ).fetchall()
    except Exception:
        rows = []
    parsed = []
    for row in rows:
        try:
            side = str(row["side"] or "").strip().upper()
            context = json.loads(row["context_json"] or "{}")
        except (TypeError, ValueError, KeyError, json.JSONDecodeError):
            continue
        if not isinstance(context, dict):
            continue
        level_type = str(context.get("level_type") or "").strip().upper()
        if not level_type and side in LEVEL_PRIORS:
            level_type = side
        if level_type not in LEVEL_PRIORS:
            continue
        try:
            horizon = int(context.get("horizon_bars", LEVEL_HORIZON_BARS))
        except (TypeError, ValueError):
            horizon = LEVEL_HORIZON_BARS
        if horizon != LEVEL_HORIZON_BARS:
            continue
        raw = _num(row["raw_prob"])
        if raw is None:
            continue
        parsed.append((level_type, _prob(raw), 1 if row["hit"] else 0))
    _LEVEL_HISTORY_CACHE["stamp"] = now
    _LEVEL_HISTORY_CACHE["rows"] = parsed
    return parsed


def invalidate_level_history_cache() -> None:
    _LEVEL_HISTORY_CACHE["stamp"] = 0.0
    _LEVEL_HISTORY_CACHE["rows"] = None


def _level_calibration(raw: float, level_type: str,
                       history: list[tuple[str, float, int]]) -> tuple[float, int, float]:
    prior = LEVEL_PRIORS.get(level_type, 0.30)
    rows = [item for item in history if item[0] == level_type]
    samples = len(rows)
    empirical = (sum(item[2] for item in rows) + 1.0) / (samples + 2.0) \
        if samples else 0.0
    try:
        calibrated_value = CALIBRATOR.calibrate(_prob(raw), LEVEL_REACH)
        calibrated = _num(calibrated_value)
    except Exception:
        calibrated = None
    if calibrated is None:
        calibrated = empirical
    if samples < MIN_LEVEL_SAMPLES:
        return prior, samples, 0.0
    if calibrated == _prob(raw):
        calibrated = empirical
    return calibrated, samples, empirical


def _level_record(price: float, side: str, level_type: str, raw: float,
                  history: list[tuple[str, float, int]]) -> dict:
    calibrated, samples, empirical = _level_calibration(raw, level_type, history)
    raw = round(_prob(raw), 4)
    calibrated = round(_prob(calibrated), 4)
    return {
        "price": round(price, 2),
        "side": side,
        "type": level_type,
        "raw_prob": raw,
        "calibrated_prob": calibrated,
        "probability": calibrated,
        "calibration_samples": samples,
        "calibration_min_samples": MIN_LEVEL_SAMPLES,
        "historical_hit_rate": round(empirical, 4) if samples else None,
        "horizon_bars": LEVEL_HORIZON_BARS,
    }


def _generate_tg_levels(snapshot: dict, entry_price: float) -> list:
    atr = _num(_block(snapshot, "t").get("atr")) or 0.0
    if atr <= 0 or entry_price <= 0:
        return []
    vp = _block(snapshot, "vp")
    poc = _num(vp.get("poc")) or 0.0
    vah = _num(vp.get("vah")) or 0.0
    val = _num(vp.get("val")) or 0.0
    history = _level_history()
    levels = []
    if vah > entry_price:
        distance = vah - entry_price
        raw = max(0.10, min(0.50, 0.35 * max(0.0, 1.0 - distance / (2.5 * atr))))
        levels.append((vah, "UP", "VAH", raw))
    if poc > entry_price and poc - entry_price <= 2.0 * atr:
        distance = poc - entry_price
        raw = max(0.15, min(0.55, 0.45 * max(0.0, 1.0 - distance / (2.0 * atr))))
        levels.append((poc, "UP", "POC", raw))
    struct_high = entry_price + 2.0 * atr
    levels.append((struct_high, "UP", "STRUCT_HIGH", 0.25))
    if 0 < val < entry_price:
        distance = entry_price - val
        raw = max(0.10, min(0.45, 0.30 * max(0.0, 1.0 - distance / (2.5 * atr))))
        levels.append((val, "DOWN", "VAL", raw))
    if poc < entry_price and entry_price - poc <= 2.0 * atr:
        distance = entry_price - poc
        raw = max(0.12, min(0.50, 0.40 * max(0.0, 1.0 - distance / (2.0 * atr))))
        levels.append((poc, "DOWN", "POC", raw))
    struct_low = entry_price - 1.5 * atr
    levels.append((struct_low, "DOWN", "STRUCT_LOW", 0.20))
    best = {}
    for price, side, level_type, raw in levels:
        key = round(price, 2)
        current = best.get(key)
        if current is None or raw > current[3]:
            best[key] = (price, side, level_type, raw)
    out = [_level_record(price, side, level_type, raw, history)
           for price, side, level_type, raw in best.values()]
    out.sort(key=lambda item: (0 if item["side"] == "UP" else 1,
                               item["price"] if item["side"] == "UP"
                               else -item["price"]))
    return out


def _adaptive_barriers(*args, **kwargs) -> None:
    del args, kwargs
    return None


def apply_all_rules(snapshot: dict) -> dict:
    pu, pd = _baseline_rules(snapshot)
    for rule_fn in (
        _rule_14_micro,
        _rule_15_risk_cap,
        _rule_16_counter_trend_veto,
        _rule_17_trend_mortality,
        _rule_18_momentum_acceleration,
        _rule_19_rsi_price_divergence,
        _rule_20_volatility_contraction,
        _rule_21_funding_extreme,
        _rule_22_oi_divergence,
        _rule_23_cvd_trend,
        _rule_24_taker_imbalance,
    ):
        pu, pd = rule_fn(pu, pd, snapshot)
    pu = _prob(pu)
    pd = _prob(pd)
    entry_price = _num(_block(snapshot, "t").get("close")) or 0.0
    levels = _generate_tg_levels(snapshot, entry_price)
    return {
        "levels": levels,
        "_debug": {"pu": round(pu, 4), "pd": round(pd, 4)},
    }



def _baseline_rules(snapshot: dict) -> tuple[float, float]:
    """Правила 1-13 из charon_prompt.txt -> (pu, pd).

    Переносятся только ветки, меняющие pu/pd; остальные правила на pu/pd
    не влияют. Правило 16 применено inline к направленным веткам 2/3/13
    (см. _gated).
    """
    pu = 0.0
    pd = 0.0
    trend = _trend_sign(snapshot)

    # 1 base.
    se = _block(snapshot, "se")
    sharpe = _num(se.get("sharpe"))
    wr = _num(se.get("wr"))
    rsi = _val(snapshot, "t", "rsi")
    bb = _val(snapshot, "t", "bb")
    p_block = se.get("p")
    oversold = _num(p_block.get("oversold")) if isinstance(p_block, dict) else None
    if (sharpe is not None and sharpe > 0 and wr is not None
            and ((oversold is not None and rsi is not None and rsi < oversold)
                 or (bb is not None and bb < BB_OVERSOLD))):
        pu = wr
    if (sharpe is not None and sharpe > 0
            and bb is not None and bb > BB_OVERBOUGHT and wr is not None):
        pd = wr

    # 2 crowd.
    long_pct = _val(snapshot, "s", "long_pct")
    fng = _val(snapshot, "s", "fng")
    if (long_pct is not None and long_pct > 0.70
            and fng is not None and fng > 0.60):
        pu += _gated(-1, -0.10, trend)
        pd += _gated(-1, 0.05, trend)
    if fng is not None:
        if fng < 0.25:
            pu += _gated(1, 0.05, trend)
        if fng > 0.80:
            pd += _gated(-1, 0.05, trend)

    # 3 taker.
    tbs = _val(snapshot, "s", "tbs")
    pdi = _val(snapshot, "tr", "pdi")
    mdi = _val(snapshot, "tr", "mdi")
    if tbs is not None and pdi is not None and mdi is not None:
        if tbs > 0.60 and pdi > mdi:
            pu += _gated(1, 0.05, trend)
        if tbs < 0.40 and mdi > pdi:
            pd += _gated(-1, 0.05, trend)

    # 5 trend (единственный авторитет направления).
    adx = _val(snapshot, "tr", "adx")
    if adx is not None and pdi is not None and mdi is not None:
        if adx > 25 and pdi > mdi:
            pu += 0.10
        if adx > 25 and mdi > pdi:
            pd += 0.10

    # 6 momentum.
    mh = _val(snapshot, "mo", "mh")
    mhs = _val(snapshot, "mo", "mhs")
    if mh is not None and mhs is not None:
        if mh > 0 and mhs > 0:
            pu += 0.10
        if mh < 0 and mhs < 0:
            pd += 0.10

    # 9 mtf confluence (только confluence-ветки).
    mtf = _block(snapshot, "mtf")
    h4 = mtf.get("4h") or {}
    d1 = mtf.get("1d") or {}
    h4_adx = _num(h4.get("adx"))
    if (h4.get("trend") == "up" and d1.get("trend") == "up"
            and ((adx is not None and adx >= 25)
                 or (h4_adx is not None and h4_adx >= 25))):
        pu += 0.10
    if h4.get("trend") == "down" and d1.get("trend") == "down":
        pd += 0.10

    # 13 news tiebreaker.
    ns_avg = _val(snapshot, "ns", "avg")
    if ns_avg is not None:
        if ns_avg < -0.3:
            pd += _gated(-1, 0.05, trend)
        if ns_avg > 0.3:
            pu += _gated(1, 0.05, trend)

    return pu, pd


def _rule_14_micro(pu: float, pd: float, snapshot: dict) -> tuple[float, float]:
    """Правило 14. mcr present: obi>0.3 -> pu+=.05; obi<-0.3 -> pd+=.05.

    mcr.sp>0.002 -> conf-=.1 и mcr.ltr>0.5 -> k*=1.1 на pu/pd не влияют.
    """
    if not _block(snapshot, "mcr"):
        return pu, pd
    obi = _val(snapshot, "mcr", "obi")
    trend = _trend_sign(snapshot)
    if obi is not None:
        if obi > 0.3:
            pu += _gated(1, 0.05, trend)
        if obi < -0.3:
            pd += _gated(-1, 0.05, trend)
    return pu, pd


def _rule_15_risk_cap(pu: float, pd: float, snapshot: dict) -> tuple[float, float]:
    """Правило 15. Cumulative k cap 2.0 — ограничивает k, pu/pd не трогает."""
    return pu, pd


def _rule_16_counter_trend_veto(pu: float, pd: float,
                                snapshot: dict) -> tuple[float, float]:
    """Правило 16. Вето применено inline в _gated (ветки 2/3/13/14)."""
    return pu, pd


def _rule_17_trend_mortality(pu: float, pd: float,
                             snapshot: dict) -> tuple[float, float]:
    """Правило 17. adx_max_50>30 AND adx_slope<-0.5 AND adx>20 -> pu-=.10, pd+=.10."""
    tr = _block(snapshot, "tr")
    adx_max_50 = _num(tr.get("adx_max_50"))
    adx_slope = _num(tr.get("adx_slope"))
    adx = _num(tr.get("adx"))
    if adx_max_50 is None or adx_slope is None or adx is None:
        return pu, pd
    if adx_max_50 > 30 and adx_slope < -0.5 and adx > 20:
        pu -= 0.10
        pd += 0.10
    return pu, pd


def _rule_18_momentum_acceleration(pu: float, pd: float,
                                   snapshot: dict) -> tuple[float, float]:
    """Правило 18. rsi_delta>0 AND adx_delta>0 AND adx>25 -> pu+=.05."""
    adx_delta = _val(snapshot, "tr", "adx_delta")
    rsi_delta = _val(snapshot, "t", "rsi_delta")
    adx = _val(snapshot, "tr", "adx")
    if adx_delta is None or rsi_delta is None or adx is None:
        return pu, pd
    if rsi_delta > 0 and adx_delta > 0 and adx > 25:
        pu += 0.05
    return pu, pd


def _rule_19_rsi_price_divergence(pu: float, pd: float,
                                  snapshot: dict) -> tuple[float, float]:
    """Правило 19. +1 bearish -> pu-=.10,pd+=.05; -1 bullish -> pd-=.10,pu+=.05."""
    val = _val(snapshot, "div", "rsi_price_div")
    if val is None:
        return pu, pd
    if val == 1:
        pu -= 0.10
        pd += 0.05
    elif val == -1:
        pd -= 0.10
        pu += 0.05
    return pu, pd


def _rule_20_volatility_contraction(pu: float, pd: float,
                                    snapshot: dict) -> tuple[float, float]:
    """Правило 20. |bb_pct_slope|<0.02 AND atr_slope<0 -> pu*=0.9, pd*=0.9."""
    bb_slope = _val(snapshot, "t", "bb_pct_slope")
    atr_slope = _val(snapshot, "t", "atr_slope")
    if bb_slope is None or atr_slope is None:
        return pu, pd
    if abs(bb_slope) < 0.02 and atr_slope < 0:
        pu *= 0.9
        pd *= 0.9
    return pu, pd


# ---------------------------------------------------------------------------
# Правила 21-24: Деривативы (Funding, OI, CVD, Taker)
# ---------------------------------------------------------------------------


def _d_val(snapshot: dict, key: str) -> float | None:
    """Достать значение из блока d (деривативы).

    Проверяет, что блок d существует и ключ не None.
    """
    d = snapshot.get("d")
    if not isinstance(d, dict):
        return None
    return _num(d.get(key))


def _rule_21_funding_extreme(pu: float, pd: float,
                             snapshot: dict) -> tuple[float, float]:
    """Правило 21. funding_zscore >2.0 -> pu-=.10,pd+=.10; <-2.0 -> pd-=.10,pu+=.10."""
    fz = _d_val(snapshot, "funding_zscore")
    if fz is None:
        return pu, pd
    if fz > 2.0:
        pu -= 0.10
        pd += 0.10
    elif fz < -2.0:
        pu += 0.10
        pd -= 0.10
    return pu, pd


def _rule_22_oi_divergence(pu: float, pd: float,
                           snapshot: dict) -> tuple[float, float]:
    """Правило 22. oi_change_4 >0.05 -> pu+=.05,pd+=.05; <-0.05 -> pu*=0.9,pd*=0.9."""
    oic = _d_val(snapshot, "oi_change_4")
    if oic is None:
        return pu, pd
    if oic > 0.05:
        # OI growth: volume expansion, increase both sides (volatility)
        pu += 0.05
        pd += 0.05
    elif oic < -0.05:
        # OI decline: position closing, reduce conviction both sides
        pu *= 0.9
        pd *= 0.9
    return pu, pd


def _rule_23_cvd_trend(pu: float, pd: float,
                       snapshot: dict) -> tuple[float, float]:
    """Правило 23. cvd_slope >0.2 -> pu+=.05; <-0.2 -> pd+=.05."""
    cvd_s = _d_val(snapshot, "cvd_slope")
    if cvd_s is None:
        return pu, pd
    if cvd_s > 0.2:
        pu += 0.05
    elif cvd_s < -0.2:
        pd += 0.05
    return pu, pd


def _rule_24_taker_imbalance(pu: float, pd: float,
                             snapshot: dict) -> tuple[float, float]:
    """Правило 24. buy_sell_ratio >1.2 -> pu+=.03; <0.8 -> pd+=.03."""
    bsr = _d_val(snapshot, "buy_sell_ratio")
    if bsr is None:
        return pu, pd
    if bsr > 1.2:
        pu += 0.03
    elif bsr < 0.8:
        pd += 0.03
    return pu, pd
