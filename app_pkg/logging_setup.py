"""Настройка логирования NeoTerminal.

setup_logging() инициализирует корневой логгер один раз при старте приложения.
"""

import logging
import sys

_LOGGER_CONFIGURED = False  # защита от повторной инициализации


def setup_logging(level: int = logging.INFO) -> None:
    """Настраивает единый формат логов: %(asctime)s [%(levelname)s] %(name)s: %(message)s."""
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root = logging.getLogger()
    root.setLevel(level)
    for old in list(root.handlers):
        root.removeHandler(old)
    root.addHandler(handler)
    _LOGGER_CONFIGURED = True