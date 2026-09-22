# -*- coding: utf-8 -*-
"""Детерминированный фильтр сигналов поверх вердикта LLM.

Правила (спека «MTF как обязательный фильтр») применяются ПОСЛЕ ответа
модели — это не промпт, а жёсткая арбитражная логика на исходном
compact_snapshot (app_pkg.data.market_snapshot.compact_snapshot):

  1. MTF — обязательный фильтр: 4h против сигнала режет сторону
     (pu *= 0.7 при LONG против "down", pd *= 0.7 при SHORT против "up").
  2. MTF consensus — сколько ТФ смотрят в ту же сторону, что и сигнал:
     +0.05 pu/pd за каждый согласный ТФ. alignment = +1 если 1h и 4h
     одного знака, -1 при расхождении (порядок, описание в metadata).
  3. Горизонт сигнала: horizon_bars (сколько баров актуален) и
     generated_at (unix-метка выдачи). Без них сигнал не использовать.
  4. Confidence — агрегат правил (база + вклад каждого сработавшего
     правила). Если confidence < SIGNAL_FILTER_CONFIDENCE_MIN — сигнал
     НЕ торгуем (sig => "F"), даже если pu/pd > 0.6.
  5. ADX-фильтр: tr.adx < 20 (флэт) -> pu/=0.6, pd/=0.6.
  6. VWAP + OBV подтверждение: LONG c v.vwd>0 и v.obv>0 -> pu += 0.05;
     зеркально для SHORT (vwd<0, obv<0) -> pd += 0.05.

Толерантность к отсутствию данных как в compact_snapshot: нет блока
mtf/tr/v — соответствующие правила не срабатывают (missing = no data).

Точки входа:
  • filter_verdict(verdict, snapshot) — вердикт {sig,pu,pd,pf,tg,conf}
    -> отфильтрованный вердикт + блок "filter" с метаданными;
  • filter_levels(levels, snapshot) — уровни {side, price, probability}
    -> вероятности с теми же правилами (сторона уровня = сторона сигнала);
  • compute_mtf_meta(snapshot) — consensus/alignment/4h отдельно.
"""

import logging
import time

from app_pkg import config

log = logging.getLogger(__name__)

# Порядок ТФ консенсуса = порядок снимка (market_snapshot._MTF_TFS).
MTF_TFS = ("15m", "1h", "4h", "1d")


def _clean(value):
    """Число из сырого значения; None на нечисле/None (как utils._clean)."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp01(value):
    """Клампинг вероятности в 0..1; None -> 0."""
    if value is None:
        return 0.0
    return min(1.0, max(0.0, value))


def compute_mtf_meta(snapshot=None):
    """Сводка мульти-ТФ для правил 1-2: {consensus, alignment, 4h, ...}.

    consensus — сколько ТФ из MTF_TFS смотрят «вверх» (tr==1, из 4);
    alignment — +1 если 1h и 4h одного знака (up/up, down/down), -1 при
    расхождении, None если хотя бы один из них flat/нет данных; 4h —
    trend 4h ("up"/"down"/flat), потребитель обязательного правила 1.
    up_tfs/down_tfs — сколько ТФ смотрит вверх/вниз (для согласия по
    стороне сигнала). None, если в снимке нет блока mtf (правила молчат).
    """
    if not snapshot:
        return None
    mtf = snapshot.get("mtf") or {}
    if not mtf:
        return None
    trends = {}
    seen = 0
    for tf in MTF_TFS:
        trend = (mtf.get(tf) or {}).get("trend")
        if trend in ("up", "down"):
            seen += 1
            trends[tf] = trend
    h1 = trends.get("1h")
    h4 = trends.get("4h")
    alignment = None
    if h1 and h4:
        alignment = 1 if h1 == h4 else -1
    return {
        "consensus": sum(1 for t in trends.values() if (t or "") == "up"),
        "alignment": alignment,
        "4h": h4,
        "up_tfs": sum(1 for t in trends.values() if (t or "") == "up"),
        "down_tfs": sum(1 for t in trends.values() if (t or "") == "down"),
        "seen": seen,
    }


def _score_confidence(sig, meta, adx, flags):
    """Агрегат правил (п.4): база + вклад каждого сработавшего правила.

    sig — сторона ДО блокировки (L/S, у F вкладов нет). meta — метрика
    MTF-блока, adx — значение tr.adx (None если нет), flags — сработавшие
    ветки. Сумма клампится 0..1 и округляется до 2 знаков.
    """
    delta = config.SIGNAL_FILTER_CONFIDENCE_BASE
    if sig in ("L", "S"):
        if "mtf_4h_against" in flags:
            delta += config.SIGNAL_FILTER_CONF_AGAINST_4H
        if "mtf_consensus" in flags:
            agreed = (meta.get("up_tfs")
                      if sig == "L" else meta.get("down_tfs"))
            delta += config.SIGNAL_FILTER_CONF_MTF_AGREE * int(agreed or 0)
        if meta.get("alignment") == -1:
            delta += config.SIGNAL_FILTER_CONF_ALIGNMENT_DIVERGE
        if "adx_flat" in flags:
            delta += config.SIGNAL_FILTER_CONF_ADX_FLAT
        elif adx is not None and adx >= config.SIGNAL_FILTER_ADX_TREND_THRESHOLD:
            delta += config.SIGNAL_FILTER_CONF_ADX_TREND
        if "vwap_obv_confirm" in flags:
            delta += config.SIGNAL_FILTER_CONF_VWAP_OBV
    return round(_clamp01(delta), 2)


def _attach_filter(out, meta=None, confidence=None, flags=None, applied=None,
                   generated_at=None, horizon_bars=None):
    """Вердикт + блок "filter" (consensus/alignment/горизонт/confidence)."""
    out["filter"] = {
        "consensus": (meta or {}).get("consensus"),
        "alignment": (meta or {}).get("alignment"),
        "horizon_bars": int(horizon_bars
                            or config.SIGNAL_FILTER_HORIZON_BARS),
        "generated_at": int(generated_at if generated_at is not None
                            else time.time()),
        "confidence": (float(confidence) if confidence is not None
                       else config.SIGNAL_FILTER_CONFIDENCE_BASE),
        "flags": list(flags or []),
        "applied": list(applied or []),
    }
    return out


def filter_verdict(verdict, snapshot=None, generated_at=None,
                   horizon_bars=None):
    """Применяет правила 1-6 к вердикту LLM; возвращает новый dict.

    verdict — {sig:"L|S|F", pu, pd, pf, tg, conf}. На выходе те же ключи
    с подстроенными pu/pd/conf (conf = сторона сигнала: pu для L, pd для
    S, pf для F), sig может стать "F" по п.4, плюс блок "filter":
    {consensus, alignment, horizon_bars, generated_at, confidence, flags,
    applied}. None-вход -> None; сигнал без стороны (плохой sig) просто
    дополняется метаданными без изменения вероятностей.
    """
    if not verdict:
        return None
    out = dict(verdict)
    sig = str(out.get("sig") or "").strip().upper()
    if sig not in ("L", "S", "F"):
        return _attach_filter(out)

    meta = compute_mtf_meta(snapshot) or {}
    snapshot = snapshot or {}
    tr = snapshot.get("tr") or {}
    v = snapshot.get("v") or {}
    pu = _clamp01(out.get("pu"))
    pd_ = _clamp01(out.get("pd"))
    pf = _clamp01(out.get("pf"))
    flags = []
    applied = []

    # 1. MTF обязательный фильтр: 4h против сигнала.
    h4 = meta.get("4h")
    if sig == "L" and h4 == "down":
        pu *= config.SIGNAL_FILTER_MTF_AGAINST_MULT
        flags.append("mtf_4h_against")
        applied.append(f"mtf.4h.down->pu*={config.SIGNAL_FILTER_MTF_AGAINST_MULT}")
    elif sig == "S" and h4 == "up":
        pd_ *= config.SIGNAL_FILTER_MTF_AGAINST_MULT
        flags.append("mtf_4h_against")
        applied.append(f"mtf.4h.up->pd*={config.SIGNAL_FILTER_MTF_AGAINST_MULT}")

    # 2. Consensus: +0.05 за каждый согласный ТФ.
    agreed = (meta.get("up_tfs") if sig == "L" else meta.get("down_tfs")) or 0
    agreed = int(agreed)
    if agreed:
        bonus = round(agreed * config.SIGNAL_FILTER_MTF_CONSENSUS_STEP, 4)
        if sig == "L":
            pu += bonus
        else:
            pd_ += bonus
        flags.append("mtf_consensus")
        applied.append(f"mtf.consensus+{bonus}")

    # 5. ADX-фильтр на вход: флэт < 20 -> режем обе стороны.
    adx = _clean(tr.get("adx"))
    if adx is not None:
        if adx < config.SIGNAL_FILTER_ADX_FLAT_THRESHOLD:
            pu *= config.SIGNAL_FILTER_ADX_FLAT_MULT
            pd_ *= config.SIGNAL_FILTER_ADX_FLAT_MULT
            flags.append("adx_flat")
            applied.append(f"tr.adx<{config.SIGNAL_FILTER_ADX_FLAT_THRESHOLD}"
                           f"->pu,pd*={config.SIGNAL_FILTER_ADX_FLAT_MULT}")
        elif adx >= config.SIGNAL_FILTER_ADX_TREND_THRESHOLD:
            flags.append("adx_trend")

    # 6. VWAP + OBV подтверждение (SHORT — зеркально LONG-ветке спеки).
    vwd = _clean(v.get("vwd"))
    obv = _clean(v.get("obv"))
    if (sig == "L" and vwd is not None and vwd > 0
            and obv is not None and obv > 0):
        pu += config.SIGNAL_FILTER_VWAP_OBV_STEP
        flags.append("vwap_obv_confirm")
        applied.append("v.vwd>0&v.obv>0->pu"
                       f"+={config.SIGNAL_FILTER_VWAP_OBV_STEP}")
    elif (sig == "S" and vwd is not None and vwd < 0
          and obv is not None and obv < 0):
        pd_ += config.SIGNAL_FILTER_VWAP_OBV_STEP
        flags.append("vwap_obv_confirm")
        applied.append("v.vwd<0&v.obv<0->pd"
                       f"+={config.SIGNAL_FILTER_VWAP_OBV_STEP}")

    # 4. Confidence-агрегат; ниже порога — не торгуем даже при pu>0.6.
    confidence = _score_confidence(sig, meta, adx, flags)
    if (sig in ("L", "S")
            and confidence < config.SIGNAL_FILTER_CONFIDENCE_MIN):
        sig = "F"
        flags.append("confidence_blocked")
        applied.append(f"confidence({confidence})"
                       f"<{config.SIGNAL_FILTER_CONFIDENCE_MIN}->sig=F")

    out["sig"] = sig
    out["pu"] = round(_clamp01(pu), 2)
    out["pd"] = round(_clamp01(pd_), 2)
    out["pf"] = round(_clamp01(pf), 2)
    out["conf"] = round(
        _clamp01(pu if sig == "L" else pd_ if sig == "S" else pf), 4)
    return _attach_filter(out, meta, confidence, flags, applied,
                          generated_at=generated_at, horizon_bars=horizon_bars)


def filter_levels(levels, snapshot=None):
    """Вероятности уровней -> с учётом правил 1/2/5/6 по стороне уровня.

    Уровень — мини-сигнал со своей стороной (side UP/DOWN): к нему
    применяются множители MTF-обязательного (0.7 против 4h), ADX<20
    (0.6) и аддитивные бонусы consensus (0.05 за согласный ТФ) и
    VWAP+OBV (0.05). price/diff/diff_pct и прочие ключи уровня сохраняются.
    Возвращает тот же список с новыми dict; вероятности клампится 0..1
    и округляются до 4 знаков. None -> []. Отсутствующие блоки — no-op.
    """
    if not levels:
        return list(levels or [])
    meta = compute_mtf_meta(snapshot) or {}
    snapshot = snapshot or {}
    tr = snapshot.get("tr") or {}
    v = snapshot.get("v") or {}
    h4 = meta.get("4h")
    adx = _clean(tr.get("adx"))
    vwd = _clean(v.get("vwd"))
    obv = _clean(v.get("obv"))
    adx_flat = (adx is not None
                and adx < config.SIGNAL_FILTER_ADX_FLAT_THRESHOLD)
    compiled = []
    for lv in levels:
        if not isinstance(lv, dict):
            compiled.append(lv)
            continue
        side = str(lv.get("side") or "").strip().upper()
        prob = round(_clamp01(lv.get("probability")), 4)
        changed = False
        if side == "UP" and h4 == "down":
            prob = round(prob * config.SIGNAL_FILTER_MTF_AGAINST_MULT, 4)
            changed = True
        elif side == "DOWN" and h4 == "up":
            prob = round(prob * config.SIGNAL_FILTER_MTF_AGAINST_MULT, 4)
            changed = True
        if adx_flat:
            prob = round(prob * config.SIGNAL_FILTER_ADX_FLAT_MULT, 4)
            changed = True
        if side == "UP":
            agreed = int(meta.get("up_tfs") or 0)
            if agreed:
                prob = round(
                    prob + agreed * config.SIGNAL_FILTER_MTF_CONSENSUS_STEP, 4)
                changed = True
            if vwd is not None and vwd > 0 and obv is not None and obv > 0:
                prob = round(prob + config.SIGNAL_FILTER_VWAP_OBV_STEP, 4)
                changed = True
        elif side == "DOWN":
            agreed = int(meta.get("down_tfs") or 0)
            if agreed:
                prob = round(
                    prob + agreed * config.SIGNAL_FILTER_MTF_CONSENSUS_STEP, 4)
                changed = True
            if vwd is not None and vwd < 0 and obv is not None and obv < 0:
                prob = round(prob + config.SIGNAL_FILTER_VWAP_OBV_STEP, 4)
                changed = True
        item = dict(lv)
        if changed:
            item["probability"] = round(_clamp01(prob), 4)
        compiled.append(item)
    return compiled