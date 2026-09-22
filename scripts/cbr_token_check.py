"""Честная sanity-проверка token-diet CBR-API: SELECT LIMIT 50 vs API.

LLM не грузит всю таблицу — наивно он дёрнул бы SELECT * LIMIT 50 по
релевантному фильтру (последние window_days). Сравниваем apples-to-apples:
  estimate_llm_naive_tokens — токены на сыром SELECT (features не читается);
  estimate_cbr_api_tokens   — токены на блоке get_stats_for_llm +
                              format_for_prompt.

Запуск из корня проекта:
    python scripts/cbr_token_check.py --symbol BTCUSDT --tf 1H
    python scripts/cbr_token_check.py --symbol BTCUSDT --tf 1H --db data/cbr.db
"""

import argparse
import sys
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent.parent
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

from app_pkg import config, utils
from app_pkg.cbr import query_api, schema


def canonical_tf(tf):
    """Приводит таймфрейм к канонической форме config.TF_SECONDS
    (1h/1H->1H, 5m->5m), иначе возвращает как есть."""
    tf = str(tf or "").strip()
    for canon in config.TF_SECONDS:
        if canon.lower() == tf.lower():
            return canon
    return tf


def estimate_llm_naive_tokens(conn, symbol, tf, window_days=90, limit=50):
    """Сколько токенов LLM потратит, если наивно дёрнет SELECT.

    LLM не грузит всю таблицу. Он грузит SELECT * LIMIT 50 по релевантному
    фильтру (последние window_days). features BLOB не считается — LLM его
    не читает: оценка ~4 токена на поле × число видимых полей × число строк.
    """
    now = int(utils.now_sec())
    rows = conn.execute(
        "SELECT * FROM snapshots "
        "WHERE symbol=? AND timeframe=? AND ts > ? "
        "ORDER BY ts DESC LIMIT ?",
        (str(symbol).upper(), canonical_tf(tf),
         now - int(window_days) * 86400, int(limit)),
    ).fetchall()
    n_fields_visible = 20  # без features и служебных колонок
    return len(rows) * n_fields_visible * 4


def estimate_cbr_api_tokens(conn, symbol, tf, window_days=90):
    """Сколько токенов через CBR API (get_stats_for_llm + format_for_prompt)."""
    stats = query_api.get_stats_for_llm(conn, str(symbol).upper(),
                                        canonical_tf(tf),
                                        window_days=window_days)
    prompt = query_api.format_for_prompt(stats)
    return max(1, len(prompt) // 4)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Token-diet проверка CBR-API (SELECT LIMIT 50 vs API).")
    parser.add_argument("--db", default=None,
                        help="путь к CBR-БД (по умолчанию config.CBR_DB_PATH)")
    parser.add_argument("--symbol", default="BTCUSDT", help="символ")
    parser.add_argument("--tf", default="1H", help="таймфрейм")
    parser.add_argument("--window-days", type=int, default=90,
                        help="окно релевантного фильтра (default: 90)")
    parser.add_argument("--limit", type=int, default=50,
                        help="LIMIT наивного SELECT (default: 50)")
    args = parser.parse_args()

    conn = schema.init_db(args.db or config.CBR_DB_PATH)
    naive = estimate_llm_naive_tokens(conn, args.symbol, args.tf,
                                      window_days=args.window_days,
                                      limit=args.limit)
    api = estimate_cbr_api_tokens(conn, args.symbol, args.tf,
                                  window_days=args.window_days)
    print(f"LLM naive (SELECT LIMIT 50): ~{naive} токенов")
    print(f"CBR API (get_stats_for_llm): ~{api} токенов")
    print(f"Экономия: x{naive / max(api, 1):.1f}")


if __name__ == "__main__":
    main()