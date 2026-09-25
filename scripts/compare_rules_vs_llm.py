"""Один бар BTCUSDT 1H: сравнить rules-путь с LLM-путём (1 LLM-вызов)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent.parent
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

from app_pkg.ai.ai_backtest import (
    _llm_verdict,
    _verdict_from_obj,
    parse_verdict,
)
from app_pkg.ai.apply_rules import apply_all_rules
from app_pkg.ai.prompts import charon_prompt_text
from app_pkg.ai.signal_filter import filter_verdict
from app_pkg.data.market_snapshot import compact_snapshot

TF = "1H"
SYMBOL = "BTCUSDT"


def main() -> int:
    snap = compact_snapshot(SYMBOL, TF)
    rules = apply_all_rules(snap)
    print("RULES:", rules)

    system = charon_prompt_text()
    payload = json.dumps(snap, ensure_ascii=False, separators=(",", ":"))
    usage: dict = {}
    raw = _llm_verdict(system, payload, None, usage)
    llm = parse_verdict(raw)
    print("LLM raw text:", (raw or "")[:400])
    print("LLM:  ", llm)
    print("usage:", usage)

    if llm and rules:
        d_pu = abs(rules["pu"] - (llm.get("pu") or 0.0))
        d_pd = abs(rules["pd"] - (llm.get("pd") or 0.0))
        print(f"Δpu={d_pu:.3f}  Δpd={d_pd:.3f}")
        print("OK" if max(d_pu, d_pd) < 0.15 else "MISMATCH")

        rules_v = _verdict_from_obj(rules)
        rf = filter_verdict(rules_v, snap)
        lf = filter_verdict(llm, snap)
        keys = ("sig", "pu", "pd", "conf")
        print("post-filter RULES:",
              {k: rf.get(k) for k in keys})
        print("post-filter LLM:  ",
              {k: lf.get(k) for k in keys})
    return 0


if __name__ == "__main__":
    sys.exit(main())
