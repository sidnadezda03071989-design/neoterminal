"""Phase 2: сравнение вердиктов Charon «до/после» динамики на 3 барах.

Для каждого бара из data/phase2_bars.json строятся два снимка:
  * after  — текущий compact_snapshot (с полями динамики);
  * before — тот же снимок без полей динамики (симуляция старой версии).
И два системных промпта:
  * after  — config/charon_prompt.txt (правила 1-20);
  * before — scripts/charon_prompt_before_dynamics.txt (правила 1-16, HEAD).

Для бара C дополнительно:
  * повтор after-снимка (стабильность LLM);
  * прогон after-снимка с промптом без правила 17 (изоляция вклада правила).

Провайдер может отвечать HTTP 429 — каждый вызов ретраится с backoff.
Для BTCUSDT 1h в БД нет scanner-stats (блок se отсутствует) — флаг
--inject-se подставляет валидный se (одинаковый для before/after), чтобы
базовое правило 1 могло сработать; иначе вердикты схлопываются в F.

Запуск:
    python scripts/compare_charon_before_after.py --inject-se --repeats 3
"""
import argparse
import copy
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app_pkg import config
from app_pkg.ai import ai_backtest as aibt
from app_pkg.data.market_snapshot import compact_snapshot

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BARS_PATH = os.path.join(REPO, "data", "phase2_bars.json")
BEFORE_PROMPT_PATH = os.path.join(REPO, "scripts",
                                  "charon_prompt_before_dynamics.txt")
OUT_PATH = os.path.join(REPO, "reports", "phase2_before_after.json")

# Валидный scanner-edge (из живой 15m-статистики BTCUSDT) — только при
# --inject-se, чтобы у бара было чем наполнить базовое правило 1.
DEFAULT_SE = {"wr": 0.61, "sharpe": 4.9, "pf": 1.49, "dd": 0.01,
              "n": 33, "tsh": 4.9}

# Поля динамики, отсутствовавшие до этой фичи (Фаза 1).
DYNAMICS = {
    "t": ["rsi_delta", "rsi_slope", "rsi_pct50", "atr_delta", "atr_slope",
          "bb_pct_delta", "bb_pct_slope"],
    "tr": ["adx_delta", "adx_slope", "adx_pct50", "adx_max_50"],
    "mo": ["rsi_slope_short", "hist_delta", "macd_cross"],
    "div": ["price_slope", "rsi_price_div"],
}


def strip_dynamics(snap: dict) -> dict:
    out = copy.deepcopy(snap)
    for block, keys in DYNAMICS.items():
        for key in keys:
            if isinstance(out.get(block), dict):
                out[block].pop(key, None)
    return out


def drop_rule17(prompt: str) -> str:
    return re.sub(r"(?m)^17\b.*(?:\n|$)", "", prompt)


def call_llm(system: str, snapshot: dict, price, model, usage: dict,
             retries: int = 6):
    payload = json.dumps({"current_price": price, **snapshot},
                         ensure_ascii=False, separators=(",", ":"))
    for attempt in range(retries):
        try:
            raw = aibt._llm_verdict(system, payload, model, usage)
            return aibt.parse_verdict(raw)
        except RuntimeError as exc:
            if "rate limit" not in str(exc) or attempt == retries - 1:
                raise
            time.sleep(4 * (attempt + 1))
    return None


def verdict_fields(verdict) -> dict:
    if not verdict:
        return {"pu": None, "pd": None, "sig": None}
    return {"pu": verdict.get("pu"), "pd": verdict.get("pd"),
            "sig": verdict.get("sig")}


def summarize(verdicts) -> dict:
    sigs = [(v or {}).get("sig") for v in verdicts]
    return {
        "n": len(verdicts),
        "pu": [v.get("pu") for v in verdicts if v],
        "pd": [v.get("pd") for v in verdicts if v],
        "sig": sigs,
        "sig_mode": Counter(sigs).most_common(1)[0][0] if sigs else None,
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=None)
    p.add_argument("--bars", default=BARS_PATH)
    p.add_argument("--inject-se", action="store_true", dest="inject_se")
    p.add_argument("--repeats", type=int, default=1)
    args = p.parse_args()

    repeats = max(1, int(args.repeats))
    with open(args.bars, encoding="utf-8") as fh:
        bars_doc = json.load(fh)
    prompt_after = config.CHARON_PROMPT_FILE.read_text(encoding="utf-8")
    prompt_before = Path(BEFORE_PROMPT_PATH).read_text(encoding="utf-8")
    prompt_no17 = drop_rule17(prompt_after)
    assert "trend mortality" not in prompt_no17, "правило 17 не вырезано"

    symbol, tf = bars_doc["symbol"], bars_doc["timeframe"]
    usage_total = {}
    results = []

    for name, meta in bars_doc["bars"].items():
        if meta is None:
            print(f"[skip] {name}: нет бара")
            continue
        ts = meta["ts"]
        price = aibt._slice_price(symbol, tf, ts)
        snap_after = compact_snapshot(symbol, tf, upto_sec=ts)
        if args.inject_se and not snap_after.get("se"):
            snap_after["se"] = dict(DEFAULT_SE)
        snap_before = strip_dynamics(snap_after)

        usage = {}
        v_before = [call_llm(prompt_before, snap_before, price, args.model, usage)
                    for _ in range(repeats)]
        v_after = [call_llm(prompt_after, snap_after, price, args.model, usage)
                   for _ in range(repeats)]
        aibt._add_usage(usage_total, usage)

        row = {"bar": name, "ts": ts, "price": price,
               "tr_after": {k: (snap_after.get("tr") or {}).get(k)
                            for k in ("adx", "adx_slope", "adx_max_50")},
               "before": summarize(v_before), "after": summarize(v_after)}
        results.append(row)
        print(f"[llm] {name} before={row['before']['sig']} "
              f"after={row['after']['sig']}")

    # --- изоляция правила 17 на баре C (after-снимок, ±правило) ---
    c_row = next((r for r in results if r["bar"].startswith("C")), None)
    stability = None
    isolation = None
    if c_row is not None:
        ts = c_row["ts"]
        price = aibt._slice_price(symbol, tf, ts)
        snap_after = compact_snapshot(symbol, tf, upto_sec=ts)
        if args.inject_se and not snap_after.get("se"):
            snap_after["se"] = dict(DEFAULT_SE)

        usage = {}
        v_with = [call_llm(prompt_after, snap_after, price, args.model, usage)
                  for _ in range(repeats)]
        v_without = [call_llm(prompt_no17, snap_after, price, args.model, usage)
                     for _ in range(repeats)]
        aibt._add_usage(usage_total, usage)
        stability = {"with_rule17": summarize(v_with)}
        isolation = {"with_rule17": summarize(v_with),
                     "without_rule17": summarize(v_without)}

    doc = {"symbol": symbol, "timeframe": tf, "model": args.model,
           "temperature": aibt.VERDICT_TEMPERATURE,
           "repeats": repeats, "inject_se": bool(args.inject_se),
           "note": bars_doc.get("note"), "results": results,
           "stability_c": stability, "isolation_c": isolation,
           "usage_total": usage_total}
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=2)

    # --- таблица ---
    print("\n" + "-" * 96)
    print(f"{'Бар':<15}| {'pu до':>16} | {'pu после':>16} | "
          f"{'sig до':>10} | {'sig после':>10}")
    print("-" * 96)
    for r in results:
        b, a = r["before"], r["after"]
        print(f"{r['bar']:<15}| {b['pu']!s:>16} | {a['pu']!s:>16} | "
              f"{b['sig']!s:>10} | {a['sig']!s:>10}")
    print("-" * 96)
    print(f"Изоляция правила 17 (C): {json.dumps(isolation, ensure_ascii=False)}")
    print(f"LLM usage total: {usage_total}")
    print(f"[saved] {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
