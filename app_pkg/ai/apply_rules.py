"""Детерминированный путь применения правил Charon 1-20.

Без LLM. Без побочных эффектов. Логика правил 1-13 и 14-20 перенесена
построчно из config/charon_prompt.txt; пороги не менялись.
"""
from __future__ import annotations

from typing import Any

from app_pkg.ai.calibration import CALIBRATOR

# Порог направления из блока Decision промпта.
SIG_THRESHOLD = 0.20
# bb-порог перепроданности правила 1 (0.5 - 1.5*0.25).
BB_OVERSOLD = 0.125
BB_OVERBOUGHT = 0.875


def _num(value: Any) -> float | None:
    """float или None (None/нечисло/'')."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _block(snap: dict, name: str) -> dict:
    block = snap.get(name)
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


def apply_all_rules(snapshot: dict) -> dict:
    """Применяет правила 1-20 к snapshot.

    Возвращает {"pu": float, "pd": float, "sig": str, "fired": list[int], "raw": {...}}.
    Порядок: baseline (1-13) -> 14 -> 15 -> 16 -> 17 -> 18 -> 19 -> 20.
    В конце clamp pu/pd к [0, 1].

    fired — номера правил, которые изменили pu или pd (правила, влияющие
    только на conf/k, сюда не попадают).

    Калибровка: сырые pu/pd (raw) прогоняются через CALIBRATOR.calibrate для
    сторонов UP/DOWN. Пока в charon_calibration_history меньше MIN_SAMPLES (50)
    на сторону — калибровщик возвращает raw как есть (fallback), т.е. поведение
    идентично исходному детерминированному. Как только история накопится,
    pu/pd возвращаются уЖЕ калиброванными, а исходные значения кладутся в .raw
    (нужно для обратной записи pu_raw/pd_raw в историю калибровки).
    """
    pu, pd = _baseline_rules(snapshot)
    fired: list[int] = []
    for rule_num, rule_fn in [
        (14, _rule_14_micro),
        (15, _rule_15_risk_cap),
        (16, _rule_16_counter_trend_veto),
        (17, _rule_17_trend_mortality),
        (18, _rule_18_momentum_acceleration),
        (19, _rule_19_rsi_price_divergence),
        (20, _rule_20_volatility_contraction),
    ]:
        pu_before, pd_before = pu, pd
        pu, pd = rule_fn(pu, pd, snapshot)
        if (pu, pd) != (pu_before, pd_before):
            fired.append(rule_num)
    pu_raw = max(0.0, min(1.0, pu))
    pd_raw = max(0.0, min(1.0, pd))
    # Калибровка (fallback на raw при нехватке истории — см. докстринг).
    pu_cal = CALIBRATOR.calibrate(pu_raw, "UP")
    pd_cal = CALIBRATOR.calibrate(pd_raw, "DOWN")
    if _hard_flat(snapshot):
        sig = "FLAT"
    else:
        diff = pu_cal - pd_cal
        if diff > SIG_THRESHOLD:
            sig = "LONG"
        elif diff < -SIG_THRESHOLD:
            sig = "SHORT"
        else:
            sig = "FLAT"
    return {
        "pu": round(pu_cal, 4), "pd": round(pd_cal, 4), "sig": sig,
        "fired": fired,
        "raw": {"pu": round(pu_raw, 4), "pd": round(pd_raw, 4)},
    }


def _baseline_rules(snapshot: dict) -> tuple[float, float]:
    """Правила 1-13 из charon_prompt.txt -> (pu, pd).

    Переносятся только ветки, меняющие pu/pd; правила 4/7/8/10/11 меняют
    conf/k/sig и на pu/pd не влияют. Правило 16 применено inline к
    направленным веткам 2/3/13 (см. _gated).
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


def _hard_flat(snapshot: dict) -> bool:
    """HARD F (правила 11/12/13 + Decision-1): любое условие -> sig F."""
    se = _block(snapshot, "se")
    sharpe = _num(se.get("sharpe"))
    pf = _num(se.get("pf"))
    n = _num(se.get("n"))
    tsh = _num(se.get("tsh"))
    dd = _num(se.get("dd"))
    wr = _num(se.get("wr"))
    if sharpe is not None and sharpe < 0:
        return True
    if pf is not None and pf < 1.0:
        return True
    if n is not None and n < 20:
        return True
    if tsh is not None and sharpe is not None and tsh < 0.6 * sharpe:
        return True
    if dd is not None and dd > 0.5:
        return True
    if wr is not None and wr < 0.3:
        return True
    if _val(snapshot, "cal", "hi2h") == 1:
        return True
    if _val(snapshot, "c", "open") == 0:
        return True
    ns_avg = _val(snapshot, "ns", "avg")
    return bool(ns_avg is not None and ns_avg < -0.5)
