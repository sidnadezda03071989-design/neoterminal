# -*- coding: utf-8 -*-
"""Разовый импорт прогоночного CSV сканера в SQLite (scan_results).

Читает scan_<run_id>.csv — выгрузку GET /api/scan/<run_id>/export.csv
(колонки: symbol, timeframe, strategy, params, train_sharpe, test_sharpe,
combined_sharpe, winrate, max_dd, profit_factor, trades), конвертирует
строки в список словарей и сохраняет через db.db_save_scan_results
(executemany, одна транзакция). Нужно, чтобы карточка «Статистика стратегий»
вкладки «Новости» (GET /api/scanner/stats) и ИИ-контекст видели прогон без
повторного запуска сканера (который может идти часами).

Запуск из корня проекта:
    python scripts/import_scan_csv.py scan_14a188075cc9454696280d7b36e4add1.csv

run_id по умолчанию берётся из имени файла (scan_<run_id>.csv); можно задать
явно: --run-id 14a188075cc9454696280d7b36e4add1.
ВНИМАНИЕ: повторный импорт того же файла ДУБЛИРУЕТ строки прогона. Для
перезаливки: --replace (сначала удаляет строки run_id, потом вставляет).
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

import pandas as pd

_BASE_DIR = Path(__file__).resolve().parent.parent
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

from app_pkg import db  # (после sys.path bootstrap)

_RUN_ID_RE = re.compile(r"scan_([0-9a-fA-F]{16,})\.csv$")


def csv_to_rows(csv_path) -> list:
    """CSV -> список словарей (строки для db_save_scan_results).

    pandas читает колонки export.csv; NaN -> None (пустая метрика),
    params (JSON-строка) -> dict. Значения колонок не приводим к float/int
    вручную — db_save_scan_results сам нормализует типы (в т.ч. numpy).
    """
    frame = pd.read_csv(csv_path)
    frame.columns = [str(c).strip() for c in frame.columns]
    rows = frame.to_dict(orient="records")
    for row in rows:
        for key, value in list(row.items()):
            if isinstance(value, float) and math.isnan(value):
                row[key] = None
        raw_params = row.get("params")
        if isinstance(raw_params, str):
            try:
                row["params"] = json.loads(raw_params)
            except (TypeError, ValueError):
                row["params"] = {}
    return rows


def run_id_from_filename(csv_path):
    """run_id из имени файла scan_<run_id>.csv; None, если не похоже."""
    match = _RUN_ID_RE.search(Path(csv_path).name)
    return match.group(1) if match else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Импорт scan_<run_id>.csv в таблицу scan_results.")
    parser.add_argument("csv", help="путь к scan_<run_id>.csv")
    parser.add_argument(
        "--run-id", default=None,
        help="run_id прогона (по умолчанию — из имени файла)")
    parser.add_argument(
        "--replace", action="store_true",
        help="удалить строки этого run_id перед вставкой (перезаливка)")
    args = parser.parse_args()

    run_id = args.run_id or run_id_from_filename(args.csv)
    if not run_id:
        sys.exit("Не удалось определить run_id из имени файла — "
                 "задайте --run-id явно")

    rows = csv_to_rows(args.csv)
    if not rows:
        sys.exit(f"CSV пуст или не читается: {args.csv}")

    if args.replace:
        removed = db.db_clear_scan_run(run_id)
        print(f"[import] удалено старых строк run_id={run_id}: {removed}")

    inserted = db.db_save_scan_results(run_id, rows)
    print(f"[import] run_id={run_id}: вставлено {inserted}/{len(rows)} строк "
          f"в scan_results ({db.config.DB_PATH})")

    best = max(
        (r for r in rows if r.get("combined_sharpe") not in (None, "")),
        key=lambda r: float(r["combined_sharpe"]), default=None)
    if best:
        print(f"[import] лучшая строка CSV: {best.get('symbol')} "
              f"{best.get('timeframe')} {best.get('strategy')} "
              f"combined_sharpe={best.get('combined_sharpe')} — "
              f"проверка: GET /api/scanner/stats?symbol={best.get('symbol')}"
              f"&timeframe={best.get('timeframe')}")


if __name__ == "__main__":
    main()