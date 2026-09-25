"""Phase 2: верификация charon_signal после урезания снимка до правил 1-20.

Сравнивает на одном баре (upto_sec) ДВА пути:
  * old — компакт-снимок ПОЛНЫЙ (compact_snapshot без rules_only) ->
          apply_all_rules  (как было до изменения);
  * new — снимок ТОЛЬКО для правил (compact_snapshot rules_only=True) ->
          apply_all_rules  (как стало: endpoint /api/charon_signal).

Критерий: pu/pd/sig/fired совпали. Если не совпали — поодиночке возвращаем
каждый из «выкинутых» блоков (m/d/v/vl/rg/cg/cnd) в new-снимок и ищем, какой
блок меняет вердикт — его нужно вернуть в rules_only-путь.

Запуск:
    python scripts/verify_charon_upto_rules.py            # бар A_trend
    python scripts/verify_charon_upto_rules.py --bar B_flat
    python scripts/verify_charon_upto_rules.py --ts 1782824400
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app_pkg.ai.apply_rules import apply_all_rules
from app_pkg.data.market_snapshot import compact_snapshot

REPO = Path(__file__).resolve().parents[1]
BARS_PATH = REPO / "data" / "phase2_bars.json"

# Блоки, которые не читает ни одно правило 1-20 (см. apply_rules.py) — они
# исключены из rules_only-снимка. MTF (правило 9) НЕ в списке.
REMOVED = ("m", "d", "v", "vl", "rg", "cg", "cnd")


def verdict(snap: dict) -> dict:
    return apply_all_rules(snap)


def fields(v: dict) -> tuple:
    return (v.get("pu"), v.get("pd"), v.get("sig"), tuple(v.get("fired") or []))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbol", default=None)
    p.add_argument("--tf", default=None)
    p.add_argument("--ts", type=float, default=None)
    p.add_argument("--bar", default="A_trend",
                   help="бар из data/phase2_bars.json (если --ts нет)")
    args = p.parse_args()

    if args.symbol and args.tf:
        symbol, tf = args.symbol.upper(), args.tf.strip()
    elif BARS_PATH.exists():
        doc = json.loads(BARS_PATH.read_text(encoding="utf-8"))
        symbol, tf = doc["symbol"], doc["timeframe"]
        if args.ts is None:
            meta = doc["bars"].get(args.bar)
            if not meta:
                print(f"[!] нет бара {args.bar!r} в {BARS_PATH}")
                return 2
            args.ts = meta["ts"]
    else:
        print("укажите --symbol/--tf или --ts")
        return 2
    ts = float(args.ts)

    t0 = time.time()
    snap_old = compact_snapshot(symbol, tf, upto_sec=ts)
    t_old = time.time() - t0
    t0 = time.time()
    snap_new = compact_snapshot(symbol, tf, upto_sec=ts, rules_only=True)
    t_new = time.time() - t0

    v_old = verdict(snap_old)
    v_new = verdict(snap_new)

    old_keys = set(snap_old)
    new_keys = set(snap_new)
    removed = old_keys - new_keys
    missing = set(REMOVED) - removed
    print(f"symbol={symbol} tf={tf} upto_sec={int(ts)}")
    print(f"  old блоков: {len(snap_old)}   new блоков: {len(snap_new)}")
    print(f"  убрано из new: {sorted(removed)}")
    print(f"  блоки REMOVED не оказались в old: {sorted(missing) or '-'}")
    print(f"  время old={t_old:.2f}s new={t_new:.2f}s "
          f"(new быстрее в {t_old/t_new:.1f}x)")
    print(f"  old pu/pd/sig/fired: {v_old}")
    print(f"  new pu/pd/sig/fired: {v_new}")

    if fields(v_old) == fields(v_new):
        print("  РЕЗУЛЬТАТ: вердикты совпали — урезание безопасно.")
        return 0

    print("  РЕЗУЛЬТАТ: вердикты РАЗОШЛИСЬ — ищем виновный блок.")
    found = []
    for block in sorted(REMOVED):
        merged = dict(snap_new)
        merged[block] = snap_old[block]
        v = verdict(merged)
        if fields(v) == fields(v_old):
            found.append(block)
            print(f"    {block}: вердикт восстановлен -> этот блок КРИТИЧЕН, "
                  f"нужно вернуть в rules_only")
    if not found:
        print("    ни один выкинутый блок не восстанавливает вердикт -> "
              "ситуация неоднозначна, проверьте mtf/порядок вручную.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())