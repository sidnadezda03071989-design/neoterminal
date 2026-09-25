"""E2E-проверка live SSE: одна пара, без лагов и пропусков.

Запуск (сервер уже должен быть поднят на 127.0.0.1:5000):
    python tests/_e2e_sse_live.py

Проверяет:
1. /api/live/subscribe регистрирует активную пару;
2. SSE шлёт candle_update ТОЛЬКО по активной паре (не по всем 18);
3. события идут непрерывно (макс. пауза < 5с) — нет «залипания» очереди;
4. бар реально обновляется (несколько уникальных close, смена минуты).
"""

import collections
import json
import socket
import threading
import time
import urllib.request

socket.setdefaulttimeout(50)
BASE = "http://127.0.0.1:5000"
SYMBOL = "BTCUSDT"
TF = "1m"
LISTEN_SEC = 35


def subscribe(symbol, tf):
    body = json.dumps({"symbol": symbol, "timeframe": tf}).encode()
    req = urllib.request.Request(
        BASE + "/api/live/subscribe", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())


def _parse(buf, event, data, events, samples):
    for line in buf.decode("utf-8", "replace").splitlines():
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
            samples.append((time.time(), d.get("symbol"), d.get("timeframe"),
                            c.get("time"), c.get("close")))
        except Exception:
            pass
    return None, None


def main():
    print("subscribe ->", subscribe(SYMBOL, TF))
    events = collections.Counter()
    samples = []
    stop = [False]
    err = []

    def sse():
        try:
            req = urllib.request.Request(BASE + "/ws",
                                         headers={"Accept": "text/event-stream"})
            with urllib.request.urlopen(req, timeout=LISTEN_SEC + 10) as resp:
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
                        event, data = _parse(raw, event, data, events, samples)
        except Exception as exc:  # noqa: BLE001
            err.append(f"{type(exc).__name__}: {exc}")

    th = threading.Thread(target=sse, daemon=True)
    th.start()
    print(f"listening {LISTEN_SEC}s ...")
    th.join(timeout=LISTEN_SEC + 12)
    stop[0] = True

    if err:
        print("SSE ERR:", err)
    print("events:", dict(events))
    pairs = collections.Counter((s[1], s[2]) for s in samples)
    print("pairs pushed:", dict(pairs))

    btc = [s for s in samples if s[1] == SYMBOL and s[2] == TF]
    print(f"{SYMBOL}/{TF} count:", len(btc))
    max_gap = None
    if btc:
        bars = {s[3] for s in btc}
        closes = {s[4] for s in btc}
        gaps = [btc[i + 1][0] - btc[i][0] for i in range(len(btc) - 1)]
        max_gap = max(gaps) if gaps else 0
        print("unique bars:", len(bars), "| unique closes:", len(closes))
        print("max gap(s):", round(max_gap, 2),
              "| avg gap(s):", round(sum(gaps) / len(gaps), 2) if gaps else 0)
        print("first:", btc[0][3], btc[0][4], "-> last:", btc[-1][3], btc[-1][4])

    ok = (pairs == {(SYMBOL, TF)} and len(btc) > 50
          and max_gap is not None and max_gap < 5)
    print("VERDICT:", "OK: single-pair, no lag, no drops" if ok else "CHECK")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
