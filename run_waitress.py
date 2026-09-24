"""NeoTerminal — production-запуск через Waitress."""

import threading

from waitress import serve

import app as appmod
from app_pkg import config
from app_pkg.data import mt5
from app_pkg.data.live import start_background_threads

# SSE (/ws) держит по одному соединению на каждый открытый таб; Waitress
# с 8 потоками быстро исчерпывал пул под нагрузкой (браузер + фоновые запросы).
_THREADS = 16

if __name__ == "__main__":
    start_background_threads()
    # MT5 (форекс) поднимается в фоне: если терминал закрыт, SDK запустит его
    # сам по MT5_PATH — форекс оживёт через ~10-30 сек, старт сервера не ждёт.
    threading.Thread(target=mt5._mt5_ensure, name="mt5-autostart", daemon=True).start()
    print(
        f"Waitress on http://localhost:{config.PORT}  (threads={_THREADS})",
        flush=True,
    )
    serve(appmod.app, host=config.HOST, port=config.PORT, threads=_THREADS)
