"""E2E-проверка live SSE: непрерывность и точность пушей.

Отличие от _e2e_sse_live.py: пишет временную линию (ts, close) каждого
события, чтобы увидеть, где были паузы и сколько реально разных close.

Запуск: python -u tests/_e2e_sse_timeline.py
"""

import collections
import json
import socket
import threading
import time
import urllib.request

socket.setdefaulttimeout(90)
BASE = "http://127.0.0.1:5000"
SYMBOL = "BTCUSDT"
TF = "1m"
LISTEN_SEC = 70  # > 1 минуты, чтобы гарантированно увидеть смену бара


def subscribe(symbol, tf):
    body = json.dumps({"symbol": symbol, "timeframe": tf}).encode()
    req = urllib.request.Request(
        BASE + "/api/live/subscribe", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())


def main():
    print("subscribe ->", subscribe(SYMBOL, TF), flush=True)
    samples = []
    events = collections.Counter()
    stop = [False]

    def sse():
        try:
            req = urllib.request.Request(BASE + "/ws",
                                         headers={"Accept": "text/event-stream"})
            with urllib.request.urlopen(req, timeout=LISTEN_SEC + 15) as resp:
                buf = b""
                event = data = None
                t0 = time.time()
                while not stop[0] and time.time() - t0 < LISTEN_SEC:
                    ch = resp.read(1)
                    if not ch:
                        break
                    buf += ch
                    while b"\n\n" in buf:
                        raw, buf = buf.split(b"\n\n", 1)
                        for line in raw.decode("utf-8", "replace").splitlines():
                            if line.startswith("event: "):
                                event = line[7:]
                            elif line.startswith("data: "):
                                data = line[6:]
                        if event:
                            events[event] += 1
                        if event == "candle_update" and data:
                            try:
                                d = json.loads(data)
                                c = d.get("candle", {})
                                samples.append((round(time.time(), 2),
                                                d.get("symbol"),
                                                d.get("timeframe"),
                                                c.get("time"),
                                                c.get("close")))
                            except Exception:
                                pass
                        event = data = None
        except Exception as exc:  # noqa: BLE001
            print("SSE ERR:", type(exc).__name__, exc, flush=True)

    th = threading.Thread(target=sse, daemon=True)
    th.start()
    th.join(timeout=LISTEN_SEC + 17)
    stop[0] = True

    print("events:", dict(events), flush=True)
    pairs = collections.Counter((s[1], s[2]) for s in samples)
    print("pairs pushed:", dict(pairs), flush=True)
    print(f"total {SYMBOL}/{TF}:", len(samples), flush=True)
    if not samples:
        print("VERDICT: NO EVENTS")
        return 1

    t0 = samples[0][0]
    print("\ntimeline (t+sec, bar_time, close):", flush=True)
    for s in samples:
        print(f"  +{s[0] - t0:6.2f}  {s[3]}  {s[4]}", flush=True)

    gaps = [samples[i + 1][0] - samples[i][0] for i in range(len(samples) - 1)]
    bars = collections.Counter(s[3] for s in samples)
    closes_by_bar = collections.defaultdict(set)
    for s in samples:
        closes_by_bar[s[3]].add(s[4])
    print("\n-- stats --", flush=True)
    print("bars seen:", dict(bars), flush=True)
    for b, cs in closes_by_bar.items():
        print(f"  bar {b}: {len(cs)} unique closes", flush=True)
    print("max gap(s):", round(max(gaps), 2) if gaps else 0,
          "| avg gap(s):", round(sum(gaps) / len(gaps), 2) if gaps else 0, flush=True)
    print("gaps > 5s:", [round(g, 2) for g in gaps if g > 5], flush=True)

    ok = pairs == {(SYMBOL, TF)} and len(bars) >= 1 and all(g < 6 for g in gaps)
    print("VERDICT:", "OK: continuous single-pair push" if ok else "CHECK", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
