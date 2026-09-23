"""Детерминированные уровни из структуры свечей (AI Backtest).

Зачем: LLM при T=0 и лимите ~120 токенов рисует формульную сетку — уровни
через РАВНЫЙ шаг (~ATR) с линейно убывающей вероятностью, одинаковую для
каждого теста независимо от реальных экстремумов. Уровни в AI Backtest
должны отражать фактическую структуру рынка: свинг-хай/лоу (фракталы
±2 бара), пивоты последнего бара, VWAP, при нехватке кандидатов — якоря
на слайсе ATR. Чистая математика по OHLCV: без сети, без LLM, без рандома —
один и тот же df даёт один и тот же результат, разные df — разные уровни.

Формат элемента совпадает с parse_probability_levels: {"side": "UP"|"DOWN",
"price", "probability", "diff", "diff_pct"}.
"""

import math

# Свинг дальше STRUCT_SWING_RANGE_ATR · ATR от цены не считаем значимым уровнем.
STRUCT_SWING_RANGE_ATR = 2.5
# Если реальных свингов на стороне меньше MIN_PER_SIDE, добиваем до минимума
# якорями на FILL_STEPS_ATR · ATR (всегда «логика», никогда не шаблон).
FILL_STEPS_ATR = (0.75, 1.5, 2.25, 3.0)
MIN_PER_SIDE = 2
# Вероятности в пределах [MIN_PROB, MAX_PROB] — согласовано с фронтендом
# (ai_prob_zones.js помнит MIN_PROB=0.10).
MIN_PROB = 0.10
MAX_PROB = 0.50


def _finite(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def atr_last(high, low, close, period=14):
    """Последнее значение ATR(period) (Wilder, рекурсия через EMA).

    None при нехватке баров (< period+1) или нечисловых данных.
    """
    try:
        high = [_finite(x) for x in high]
        low = [_finite(x) for x in low]
        close = [_finite(x) for x in close]
    except Exception:  # noqa: BLE001
        return None
    n = len(high)
    if n < period + 1:
        return None
    trs = []
    prev_c = None
    for h, l, c in zip(high, low, close):
        if None in (h, l, c):
            trs = []
            prev_c = None
            continue
        if prev_c is None:
            prev_c = c
            continue
        trs.append(max(h - l, abs(h - prev_c), abs(l - prev_c)))
        prev_c = c
    if len(trs) < period:
        return None
    atr = sum(trs[:period]) / float(period)
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / float(period)
    return atr


def _fractals(high, low, window=2):
    """Свинг-хай/лоу: бар выше ВСЕХ соседей ±window (фрактал с точным >).
    Возвращает (swing_highs, swing_lows) — списки цен."""
    n = len(high)
    sw_high, sw_low = [], []
    for i in range(window, n - window):
        left, right = i - window, i + window + 1
        if high[i] > max(high[left:i]) and high[i] > max(high[i + 1:right]):
            sw_high.append(high[i])
        if low[i] < min(low[left:i]) and low[i] < min(low[i + 1:right]):
            sw_low.append(low[i])
    return sw_high, sw_low


def _dedupe(cands, tol):
    """Убрать почти совпадающие по цене кандидаты (max по несущественно)."""
    out = []
    for p in sorted(cands, reverse=True):
        if all(abs(p - q) > tol for q in out):
            out.append(p)
    return out


def _last_vwap(high, low, close, volume, bars=40):
    """VWAP по последним bars барам (типичная цена). None, если нет объёмов."""
    volume = [_finite(x) for x in volume]
    if not volume or not any(v and v > 0 for v in volume[-bars:]):
        return None
    total = 0.0
    weight = 0.0
    for h, l, c, v in zip(high[-bars:], low[-bars:], close[-bars:], volume[-bars:]):
        if None in (h, l, c, v) or v <= 0:
            continue
        total += (h + l + c) / 3.0 * v
        weight += v
    return total / weight if weight > 0 else None


def _mk_side(side, prices, cp):
    """Цены стороны -> уровни с вероятностью по дистанции (ближний сильнее)."""
    out = []
    n = len(prices)
    dmax = abs(prices[-1] - cp) if n > 1 else 0.0
    for idx, p in enumerate(prices):
        d = abs(p - cp)
        if n > 1 and dmax > 0:
            prob = 0.22 + 0.35 * (1.0 - d / dmax)
        else:
            prob = 0.30
        prob = min(MAX_PROB, max(MIN_PROB, prob))
        out.append({
            "side": side,
            "price": round(p, 8),
            "probability": round(prob, 4),
            "diff": round(p - cp, 8),
            "diff_pct": round((p - cp) / cp * 100.0, 4),
        })
    return out


def structure_levels_from_df(df, current_price, max_per_side=5,
                             atr_period=14):
    """Уровни из структуры свечей: свинги + VWAP + пивот + ATR-якоря.

    df: pandas.DataFrame с колонками open/high/low/close/volume.
    current_price: цена среза (якорь разбивки на UP/DOWN и нормализация).
    max_per_side: потолок уровней в каждую сторону (как PROB_LEVELS_MAX).
    Возвращает список уровней (формат parse_probability_levels) или [],
    если данных недостаточно.
    """
    try:
        cp = _finite(current_price)
        if cp is None or cp <= 0 or df is None or df is None or len(df) < 10:
            return []
        high = [_finite(x) for x in df["high"]]
        low = [_finite(x) for x in df["low"]]
        close = [_finite(x) for x in df["close"]]
        if len(high) != len(df) or not all(close):
            return []
    except Exception:  # noqa: BLE001
        return []

    atr = atr_last(high, low, close, atr_period)
    if atr is None or atr <= 0:
        return []
    tol = max(atr * 0.12, abs(cp) * 0.0025)

    sw_high, sw_low = _fractals(high, low)

    def _add_side(price, up_cands, dn_cands):
        if price is None:
            return
        if price > cp and (price - cp) <= STRUCT_SWING_RANGE_ATR * atr:
            up_cands.append(price)
        elif cp - price <= STRUCT_SWING_RANGE_ATR * atr:
            dn_cands.append(price)

    up_cands, dn_cands = [], []
    for p in sw_high:
        if p > cp and (p - cp) <= STRUCT_SWING_RANGE_ATR * atr:
            up_cands.append(p)
    for p in sw_low:
        if p < cp and (cp - p) <= STRUCT_SWING_RANGE_ATR * atr:
            dn_cands.append(p)
    vwap = _last_vwap(high, low, close, df["volume"])
    _add_side(vwap, up_cands, dn_cands)
    pivot = (high[-1] + low[-1] + close[-1]) / 3.0
    _add_side(pivot, up_cands, dn_cands)

    def _build(side_cands, side, below):
        cands = _dedupe(side_cands, tol)
        if not below:
            p_list = sorted(cands, reverse=False)
        else:
            p_list = sorted(cands, reverse=True)
        p_list = p_list[:max_per_side]
        # Гарантия MIN_PER_SIDE: добиваем ближайшими ATR-якорями.
        k = 0
        while len(p_list) < MIN_PER_SIDE and k < len(FILL_STEPS_ATR):
            p = cp - FILL_STEPS_ATR[k] * atr if below else cp + FILL_STEPS_ATR[k] * atr
            if p > 0 and all(abs(p - q) > tol for q in p_list):
                p_list.append(p)
                p_list = (sorted(p_list, reverse=bool(below))
                          )[:max_per_side]
            k += 1
        return _mk_side(side, p_list[:max_per_side], cp)

    ups = _build(up_cands, "UP", below=False)
    downs = _build(dn_cands, "DOWN", below=True)
    return ups + downs