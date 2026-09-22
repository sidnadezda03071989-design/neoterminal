"""NeoTerminal — точка входа (dev-сервер Flask)."""

import threading

from app_pkg import config, create_app
from app_pkg.data import mt5
from app_pkg.data.live import start_background_threads

app = create_app()

if __name__ == "__main__":
    start_background_threads()
    # MT5 (форекс) поднимается в фоне, старт dev-сервера не блокируем.
    threading.Thread(target=mt5._mt5_ensure, name="mt5-autostart", daemon=True).start()
    app.run(host=config.HOST, port=config.PORT, debug=False, threaded=True)

