"""Тесты загрузки .env (python-dotenv, app_pkg/__init__.py).

Проверяем:
  - load_dotenv() по временному .env в tmp_path: os.getenv видит значение
    (реальный .env проекта не трогается — chdir в tmp_path);
  - при импорте app_pkg реальный .env корня проекта реально применён
    (тест пропускается, если .env отсутствует).
"""

import os
from pathlib import Path

import pytest

from app_pkg import config


def test_load_dotenv_reads_tmp_env(tmp_path, monkeypatch):
    """Временный .env в tmp_path -> os.getenv видит значение. Реальный .env не трогаем."""
    key = "ENV_LOADING_TEST_KEY"
    monkeypatch.delenv(key, raising=False)

    env_file = tmp_path / ".env"
    env_file.write_text(f"{key}=hello-from-tmp-env\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)  # cwd больше не корень проекта
    from dotenv import load_dotenv
    load_dotenv(env_file)

    assert os.getenv(key) == "hello-from-tmp-env"

    # подчищаем за собой, чтобы значение не текло в другие тесты
    monkeypatch.delenv(key, raising=False)


def test_app_pkg_init_applied_project_env():
    """app_pkg/__init__.py при импорте применил реальный <project>/.env.

    Тест пропускается, если .env в корне проекта отсутствует (CI и т.п.).
    """
    env_file = Path(config.BASE_DIR) / ".env"
    if not env_file.exists():
        pytest.skip("нет реального .env в корне проекта")

    # первая непустая строка VAR=... из файла
    first = next(
        line for line in env_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#") and "=" in line
    )
    var_name = first.split("=", 1)[0].strip()

    # load_dotenv в app_pkg/__init__.py выполнился при импорте пакета —
    # переменная должна быть видна в os.environ (если не перекрыта env).
    assert os.getenv(var_name) is not None, (
        f"{var_name} из .env не видна: load_dotenv не сработал при импорте app_pkg"
    )


def test_groq_api_key_from_env(monkeypatch):
    """GROQ_API_KEY (fallback) читается из .env через os.getenv."""
    import importlib

    monkeypatch.setenv("GROQ_API_KEY", "gsk-test-fallback-key")

    importlib.reload(config)
    try:
        assert config.GROQ_API_KEY == "gsk-test-fallback-key"
        # QWEN_* больше не алиас на Groq — значения читаются из env/.env
        # (проверяется в test_qwen_api_key_read_from_env_and_dotenv ниже).
    finally:
        importlib.reload(config)  # восстановить реальный конфиг


def test_qwen_api_key_read_from_env_and_dotenv(monkeypatch):
    """QWEN_API_KEY_1 читается из env и совпадает со значением из .env.

    Skip, если .env нет или QWEN_API_KEY_* в нём не задан.
    """
    import importlib

    env_file = Path(config.BASE_DIR) / ".env"
    if not env_file.exists():
        pytest.skip("нет реального .env в корне проекта")

    values = {}
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        values[name.strip()] = value.strip()
    dotenv_key = values.get("QWEN_API_KEY_1") or values.get("QWEN_API_KEY")
    if not dotenv_key:
        pytest.skip("QWEN_API_KEY_* не задан в .env")

    # 1) значение подхватывается именно из окружения (перекрывает .env)
    monkeypatch.setenv("QWEN_API_KEY_1", "sk-ws-test-qwen-key")
    importlib.reload(config)
    try:
        assert config.QWEN_API_KEY_1 == "sk-ws-test-qwen-key"
        assert config.QWEN_MODEL_1  # модель непустая (qwen-plus по умолчанию)
    finally:
        monkeypatch.delenv("QWEN_API_KEY_1", raising=False)
        importlib.reload(config)

    # 2) и совпадает со строкой из .env
    assert config.QWEN_API_KEY_1 == dotenv_key


def test_deepseek_api_key_from_env(monkeypatch):
    """DEEPSEEK_API_KEY (основной провайдер) читается из .env через os.getenv."""
    import importlib

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-aitunnel-test-key")

    importlib.reload(config)
    try:
        assert config.DEEPSEEK_API_KEY == "sk-aitunnel-test-key"
        assert config.llm_enabled() is True
        assert config.deepseek_enabled() is True
        assert config.any_llm_enabled() is True
    finally:
        importlib.reload(config)  # восстановить реальный конфиг
