"""SSE-рассылка событий клиентам (event-stream).

Формат события: "event: {event}\ndata: {json}\n\n".
"""

import json
import queue
import threading
import time

# client_id -> queue.Queue. Размер очереди ограничен, но при переполнении
# клиент НЕ отключается (см. _ws_push) — иначе редкий читатель терял
# соединение навсегда и свечи «пропадали» на графике.
_MAX_QUEUE = 500
_ws_clients = {}
_ws_lock = threading.Lock()
_ws_counter = 0


def _ws_push(event, data):
    """Отправить SSE-событие всем подключённым клиентам.

    При переполнении очереди выбрасываем САМОЕ СТАРОЕ сообщение и кладём
    новое, а не помечаем клиента мёртвым. Раньше `queue.Full` удалял клиента
    из рассылки: браузер, читающий поток с задержкой, терял соединение и
    свечи переставали приходить до реконнекта (5→10→20→40→60с) — это и
    выглядело как «свечи лагают и пропадают».
    """
    msg = f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
    with _ws_lock:
        for _cid, q in list(_ws_clients.items()):
            try:
                q.put_nowait(msg)
            except queue.Full:
                # Дропаем старейшее событие (клиент отстал) и кладём свежее:
                # актуальный бар важнее пропущенного исторического.
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    q.put_nowait(msg)
                except queue.Full:
                    pass  # гонка с читателем — не критично, шлём в след. тик


def _ws_register():
    """Зарегистрировать нового клиента; вернуть (client_id, queue)."""
    global _ws_counter
    q = queue.Queue(maxsize=_MAX_QUEUE)
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