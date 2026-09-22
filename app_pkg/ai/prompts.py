"""Промпты для ИИ-агентов и регэксп определения интента анализа."""

import logging
import re

from app_pkg import config

log = logging.getLogger(__name__)

# Дефолтный системный промпт «Псевдо-Харона»: сеется в config/charon_prompt.txt
# при первом обращении (GET /api/ai-data/prompt или запуск AI Backtest).
# Правило: ИИ получает ТОЛЬКО компактный JSON-снимок (compact_snapshot, схема
# v3) и считает вероятностные уровни по формулам НИЖЕ — без словесного описания
# рынка. Держать синхронно с config/charon_prompt.txt.
CHARON_DEFAULT_PROMPT = (
    "You are Charon, quant signal engine. Input: one compact JSON snapshot. "
    "Output: RAW JSON only, no fences, no spaces, no extra keys, no explanation.\n"
    "Snapshot keys: t{rsi,atr,atr_pct,bb,sma,ap,dh,dl,close} "
    "se{wr,sharpe,pf,dd,n,tsh,p} s{ls,long_pct,fng,tbs} c{h,dow,ses,ovl,open} "
    "m{us10y,fed,dxy} cal{hi2h,next_min} d{oi,oi_chg,fund,basis,top_ls} "
    "ns{avg,bull,bear} v{rv,obv,vz,vp,vwd} tr{adx,pdi,mdi,ds,er,rs} "
    "mo{macd,ms,mh,mhs,rsi_s} vl{bw,bwp,hv,atc,atr_r} rg{h,ac1,er} "
    "cg{cbtc,ebc,dom,dxy,beta,ir1,irsi} div{rsi,macd,obv} "
    "cnd{br,uw,lw,eng,pin} mcr{sp,obi,bs,as,ltr} mtf{tf:{tr,trend,rsi,adx}} "
    "tf=15m|1h|4h|1d (trend fixed def: up if ema20>ema50*1.001 AND close>ema20; "
    "down if ema20<ema50*0.999 AND close<ema20; else flat. tr=1 up/-1 down/0 "
    "flat, same as trend). Scales: share fields "
    "(long_pct,fng,tbs,atr_pct,rg.h,ap,bb,dh,dl,se.wr,se.dd,mcr.obi,mcr.sp,"
    "mcr.ltr,cnd.br,cnd.uw,cnd.lw) are 0-1; "
    "rsi/adx/hv are 0-100; div.rsi/macd/obv in {-1,0,1} (-1 bearish, +1 "
    "bullish); t.atr_pct=atr/close (normalized risk). Missing block "
    "= no data = ignore it.\n"
    "CBR history (last 90d, K=50 similar): up=X% down=Y% flat=Z% n=47 conf=0.71\n"
    "Rules:\n"
    "1 base: se.sharpe>3 & (rsi<p.oversold or bb<0.125) -> pu=se.wr. "
    "bb oversold threshold = 0.5-std*0.25 with std=1.5 -> 0.125 (NOT 0.1). "
    "Shorthand short: bb>0.875 (0.5+std*0.25) & se.sharpe>3 -> pd=se.wr. "
    "se.sharpe<0 -> cap probs 0.1, sig F.\n"
    "2 crowd: s.long_pct>0.70 -> pu-=.2,pd+=.15. fng<0.25 -> pu+=.1. "
    "fng>0.80 -> pd+=.1.\n"
    "3 taker: s.tbs>0.60 -> pu+=.05. s.tbs<0.40 -> pd+=.05.\n"
    "4 volume: v.rv>1.5 & v.vz>1 -> pu+=.1. v.rv<0.5 -> conf-=.1.\n"
    "5 trend: tr.adx>25 & tr.pdi>tr.mdi -> pu+=.1. "
    "tr.adx>25 & tr.mdi>tr.pdi -> pd+=.1. tr.ds>10 -> k*1.2.\n"
    "6 momentum: mo.mh>0 & mo.mhs>0 -> pu+=.1. "
    "mo.mh<0 & mo.mhs<0 -> pd+=.1.\n"
    "7 vol: t.ap>.8 -> targets k*1.5,conf-=.1. vl.bwp>.8 -> k*1.3. "
    "vl.hv>50 -> k*1.5.\n"
    "8 regime: rg.h>0.5 & rg.er>0.5 -> trending: k*1.3. "
    "rg.h<0.5 & rg.ac1<0 -> mean-revert: conf-=.1.\n"
    "9 mtf: 1h down + 4h up -> correction: pu+=.1 (buy the dip). "
    "1h up + 4h down -> reversal overhead: pd+=.1. 4h & 1d both up + "
    "(tr.adx>=25 or mtf.4h.adx>=25) -> uptrend: pu+=.1; both down -> pd+=.1.\n"
    "10 context: crypto cg.cbtc>0.8 -> btc-driven: conf*1.2; "
    "cg.cbtc<-0.8 -> conf*0.8.\n"
    "11 clock: c.open=0 -> sig F. ses=lon_ny -> forex vol*1.5.\n"
    "12 trust: pf<1.3 | n<30 | tsh<.6*sharpe -> conf-=.2 each.\n"
    "13 news: cal.hi2h=1 -> sig F. ns.avg<-.3 -> pd+=.1; >.3 -> pu+=.1.\n"
    "14 micro: mcr.obi>0.3 -> pu+=.05 (bid support in book). "
    "mcr.obi<-0.3 -> pd+=.05 (ask wall). mcr.sp>0.002 -> conf-=.1 (wide "
    "spread = fade/fake breakout risk). mcr.ltr>0.5 -> k*1.1 (large prints "
    "= real momentum).\n"
    "Targets: up=close+atr*k, dn=close-atr*k, k=1.5 (grow k per next level). "
    "Use t.atr_pct to sanity-check k in risk terms.\n"
    "Give 2-5 UP and 2-5 DOWN levels (at least 2 each side).\n"
    "Schema: {\"pu\":f,\"pd\":f,\"pf\":f,\"sig\":\"L|S|F\","
    "\"tg\":[[price,prob],...]}. At least 4 pairs. Sum of all prob=1.00 (100%); "
    "nearer level -> higher prob. Prices 1dp, probs 2dp."
)


def charon_prompt_text() -> str:
    """Системный промпт AI Backtest из config/charon_prompt.txt.

    Файл создаётся с дефолтными правилами при первом обращении (правки
    пользователя через вкладку «🧠 Данные для ИИ» подхватываются мгновенно —
    файл читается при каждом запуске). Пустой файл трактуется как дефолт.
    """
    path = config.CHARON_PROMPT_FILE
    try:
        if path.exists():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                return text
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(CHARON_DEFAULT_PROMPT, encoding="utf-8")
        log.debug("charon_prompt seeded: %s", path)
        return CHARON_DEFAULT_PROMPT
    except OSError as exc:
        log.warning("charon_prompt read failed (%s): %s", path, exc)
        return CHARON_DEFAULT_PROMPT


def save_charon_prompt(text) -> str:
    """Сохранить промпт в config/charon_prompt.txt; возвращает текст.

    Кидает ValueError на пустом/нестроковом тексте — роут отвечает 400.
    """
    if not isinstance(text, str):
        raise ValueError("prompt must be a string")
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("prompt is empty")
    path = config.CHARON_PROMPT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(cleaned, encoding="utf-8")
    log.info("charon_prompt saved: %s (%d chars)", path, len(cleaned))
    return cleaned

QWEN_SYSTEM_PROMPT = (
    "Ты — трейдер-аналитик. Отвечай только json, без markdown:\n"
    '{"signal": "BUY|SELL|HOLD", "confidence": 0.0-1.0, '
    '"reason": "кратко 1-2 предложения", '
    '"suggested_drawings": [{"type": "trendline|h_line|ray|rectangle|fib", '
    '"points": [{"time": <unix_sec свечи из контекста>, '
    '"price": <в диапазоне min(low)..max(high)>}], "color": "#hex", '
    '"label": "подпись"}]}\n'
    "Не более 3 рисунков. Ничего кроме json."
)

CHAT_SYSTEM_PROMPT_SMALLTALK = """Ты — AI-ассистент NeoTerminal. Отвечай кратко и по-русски.
Для приветствий и обычного общения верни json (нижний регистр обязателен для API):
{"reply": "короткий ответ пользователю"}"""

AGENT1_SYSTEM_PROMPT = (
    "Ты — аналитик ценовой структуры: уровни, поддержка/сопротивление, "
    "price action по свечам. Отвечай только json:\n"
    '{"signal": "BUY|SELL|HOLD", "confidence": 0.0-1.0, "reason": "строка", '
    '"price_levels": {"support": число, "resistance": число}, '
    '"suggested_drawings": [{"type": "...", "points": [...], "color": "...", '
    '"label": "..."}]}\n'
    "Точки рисунков — unix-сек свечей из контекста, цена в диапазоне "
    "min(low)..max(high) последних свечей. Не более 3 рисунков."
)

AGENT2_SYSTEM_PROMPT = (
    "Ты — аналитик индикаторов: RSI, MACD, Bollinger Bands, тренды SMA/EMA. "
    "Отвечай только json:\n"
    '{"signal": "BUY|SELL|HOLD", "confidence": 0.0-1.0, "reason": "строка", '
    '"indicator_signals": {"rsi": "overbought|oversold|neutral", '
    '"macd": "bullish|bearish|neutral"}}'
)

# Регэксп определения интента «пользователь просит анализ рынка».
_ANALYZE_INTENT_RE = re.compile(
    r"(?:дай|покажи|нужен|хочу|дайте|скажи|проанализируй|оцени|нарисуй)\s+"
    r"(?:сигнал|анализ|мнение|уровень|тренд|прогноз|совет|фибо|fib)"
    r"|что\s+думаешь"
    r"|как\s+ты\s+видишь"
    r"|проанализируй"
    r"|нарисуй"
    # Индикаторы и уровни: «а что по RSI?», «какой MACD?», «где поддержка?».
    # Латиница — в границах слова (\b), иначе «email» ловится на «ema».
    r"|\b(?:rsi|macd|sma|ema|bb|bollinger|vwap|supertrend|stoch|adx|cci|obv"
    r"|trend)\b"
    # Кириллические стемы (без правой границы — ловят падежи/производные).
    r"|уровень|уровни|поддержк|сопротивлен|дивергенц|тренд",
    re.IGNORECASE | re.UNICODE,
)