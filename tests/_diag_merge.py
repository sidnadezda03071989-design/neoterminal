"""Диагностика рассинхрона предпоследней/последней свечи.

Сравнивает в ОДИН момент времени:
  - /api/data        -> candles[-2], candles[-1]  (то, что рисует график)
  - /api/last-bar    -> candle                    (то, что тикает)

Ожидание: last-bar должен СОВПАДАТЬ с candles[-1] (time и close).
Если расходятся — где-то теряется live-мерж и свеча «зависает».

Запуск: python -u tests/_diag_merge.py
"""

import json
import socket
import time
import urllib.request

socket.setdefaulttimeout(30)
BASE = "http://127.0.0.1:5000"
SYMBOL = "BTCUSDT"
TF = "1m"
ROUNDS = 6


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=25) as r:
        return json.loads(r.read())


def main():
    bad = 0
    for i in range(ROUNDS):
        now = int(time.time())
        d = get(f"/api/data?symbol={SYMBOL}&timeframe={TF}&limit=6")
        lb = get(f"/api/last-bar?symbol={SYMBOL}&timeframe={TF}")
        cs = d["candles"]
        bar = lb.get("candle") or {}

        prev, last = cs[-2], cs[-1]
        print(f"--- round {i} (now={now}) ---")
        print(f"  data candles[-2]: time={prev['time']} age={now - prev['time']:>4} "
              f"o={prev['open']} h={prev['high']} l={prev['low']} c={prev['close']}")
        print(f"  data candles[-1]: time={last['time']} age={now - last['time']:>4} "
              f"o={last['open']} h={last['high']} l={last['low']} c={last['close']}")
        print(f"  last-bar candle: time={bar.get('time')} "
              f"age={now - int(bar.get('time', 0)):>4} "
              f"o={bar.get('open')} h={bar.get('high')} "
              f"l={bar.get('low')} c={bar.get('close')}")

        # 1) Последний бар данных обязан совпадать с last-bar по времени.
        if bar.get("time") != last["time"]:
            print(f"  !! MISMATCH time: data[-1]={last['time']} last-bar={bar.get('time')}")
            bad += 1
        # 2) Бар не должен «отставать» больше чем на 1 интервал (60с).
        if now - last["time"] > 120:
            print(f"  !! STALE: candles[-1] старше 120с (age={now - last['time']})")
            bad += 1
        # 3) Соседние бары должны различаться ровно на 60с.
        step = last["time"] - prev["time"]
        if step != 60:
            print(f"  !! STEP != 60s: {step}")
            bad += 1
        # 4) high >= max(o,c), low <= min(o,c) — иначе битый бар.
        for name, c in (("data[-2]", prev), ("data[-1]", last), ("last-bar", bar)):
            if not (c["high"] >= max(c["open"], c["close"])
                    and c["low"] <= min(c["open"], c["close"])):
                print(f"  !! BROKEN OHLC {name}: {c}")
                bad += 1
        time.sleep(4)

    print(f"\nVERDICT: {'OK: merge consistent' if bad == 0 else f'{bad} problems found'}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
