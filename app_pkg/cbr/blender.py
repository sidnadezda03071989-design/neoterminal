"""Blender Charon + CBR (Этап 2): смешивание вероятностей направлений.

Решает, насколько доверять CBR-сигналу (k-NN winrate) против Charon-only
вердикта, по числу примеров n, confidence и РЕЖИМУ рынка:

  n < 10                        -> CBR не набрал: только Charon (w1=1.0, w2=0.0)
  n 10..30                      -> CBR слабый:  w1=0.6, w2=0.4
  n > 30 и conf > 0.6           -> CBR сильный: w1=0.3, w2=0.7
  regime trend_up/trend_down    -> CBR там слабый (сигнал размазан): w1=0.8, w2=0.2
  regime reversal               -> CBR там сильнее (variance-чек):     w1=0.4, w2=0.6

Правила режима имеют приоритет над числовыми (после проверки на недостаток
данных): разбиение A/B/C по режимам показывает edge CBR именно в reversal.

Результат:
  {"final_pu": w1*pu + w2*up, "final_pd": w1*pd + w2*down,
   "w1": w1, "w2": w2, "source": "charon_only"|"blended"|"cbr_only"}
"""

from app_pkg import config

TREND_REGIMES = ("trend_up", "trend_down")
REVERSAL_REGIME = "reversal"

# Пороги из config (для читаемости и переопределения в тестах).
MIN_SAMPLES = None        # лениво из config.CBR_BLEND_MIN_SAMPLES
STRONG_SAMPLES = None     # config.CBR_BLEND_STRONG_SAMPLES
STRONG_CONF = None        # config.CBR_BLEND_STRONG_CONF


def _thresholds():
    return (int(config.CBR_BLEND_MIN_SAMPLES),
            int(config.CBR_BLEND_STRONG_SAMPLES),
            float(config.CBR_BLEND_STRONG_CONF))


def _cbr_n(cbr):
    """Число примеров CBR (insufficient_data -> 0)."""
    if not isinstance(cbr, dict) or cbr.get("insufficient_data"):
        return 0
    try:
        return int(cbr.get("n", 0) or 0)
    except (TypeError, ValueError):
        return 0


def blend(charon_pu, charon_pd, cbr, regime):
    """Смешивает Charon-вердикт и CBR-сводку в финальные pu/pd.

    charon_pu/charon_pd — вероятности вверх/вниз от Charon (0..1).
    cbr — dict с ключами winrate_up/winrate_down/n/confidence (схема
    get_similar_summary) или {"n": .., "insufficient_data": True}.
    regime — 'trend_up'|'trend_down'|'reversal'|'flat'.
    """
    pu = float(charon_pu or 0.0)
    pd = float(charon_pd or 0.0)
    n = _cbr_n(cbr)
    up = float((cbr or {}).get("winrate_up", 0.0) or 0.0)
    dn = float((cbr or {}).get("winrate_down", 0.0) or 0.0)
    conf = float((cbr or {}).get("confidence", 0.0) or 0.0)
    min_n, strong_n, strong_c = _thresholds()

    # 1) CBR не набрал минимума примеров -> Charon alone.
    if n < min_n:
        w1, w2, source = 1.0, 0.0, "charon_only"
    # 2) Режимные правила (variance-чек: CBR силён в reversal, слаб в тренде).
    elif regime == REVERSAL_REGIME:
        w1, w2, source = 0.4, 0.6, "blended"
    elif regime in TREND_REGIMES:
        w1, w2, source = 0.8, 0.2, "blended"
    # 3) CBR сильный (много примеров + высокая уверенность).
    elif n > strong_n and conf > strong_c:
        w1, w2, source = 0.3, 0.7, "cbr_only"
    # 4) CBR обычный (10..30 примеров или слабая уверенность).
    else:
        w1, w2, source = 0.6, 0.4, "blended"

    return {
        "final_pu": round(w1 * pu + w2 * up, 4),
        "final_pd": round(w1 * pd + w2 * dn, 4),
        "w1": w1, "w2": w2, "source": source,
    }