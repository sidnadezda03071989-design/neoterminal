# -*- coding: utf-8 -*-
"""SSE-рассылка событий клиентам (event-stream).

Формат события: "event: {event}\ndata: {json}\n\n".
"""

import json
import queue
import threading
import time

_ws_clients = {}  # client_id -> queue.Queue(maxsize=100)
_ws_lock = threading.Lock()
_ws_counter = 0


def _ws_push(event, data):
    """Отправить SSE-событие всем подключённым клиентам."""
    msg = f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
    with _ws_lock:
        dead = set()
        for cid, q in list(_ws_clients.items()):
            try:
                q.put_nowait(msg)
            except queue.Full:
                dead.add(cid)
        for cid in dead:
            _ws_clients.pop(cid, None)


def _ws_register():
    """Зарегистрировать нового клиента; вернуть (client_id, queue)."""
    global _ws_counter
    q = queue.Queue(maxsize=100)
    with _ws_lock:
        _ws_counter += 1
        cid = _ws_counter
        _ws_clients[cid] = q
    return cid, q


def _ws_unregister(cid):
    """Удалить клиента по client_id."""
    with _ws_lock:
        _ws_clients.pop(cid, None)


def event_stream():
    """Generator SSE-ответа с keepalive.

    Сообщение connected кладётся ТОЛЬКО в собственную очередь клиента,
    не через _ws_push (иначе оно улетит всем).
    """
    cid, q = _ws_register()
    try:
        q.put_nowait(
            f"event: connected\ndata: {json.dumps({'ts': time.time()}, ensure_ascii=False)}\n\n"
        )
        while True:
            try:
                msg = q.get(timeout=15)
                yield msg
            except queue.Empty:
                yield ": keepalive\n\n"
    finally:
        _ws_unregister(cid)