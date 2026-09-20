p = r'c:\neoterminal\app_pkg\routes\backtest.py'
lines = open(p, encoding='utf-8').readlines()
# Fix docstring tp_sl section (lines 75-79, 0-indexed 74-78)
lines[74] = '    TP/SL (BLOCK-42): чекбокса «TP/SL уровни» больше нет — прогон ВСЕГДА\n'
lines[75] = '    идёт с уровнями config.BACKTEST_SL_ATR / BACKTEST_RR (SL×ATR для риска,\n'
lines[76] = '    TP = SL×R/R). Выход ТОЛЬКО по касанию линии — ровно как сканер. Поэтому\n'
lines[77] = '    число сделок здесь совпадает с колонкой TEST/TRAIN/FULL Trades панели\n'
lines[78] = '    1:1, а у каждой сделки есть tp_price/sl_price/tp_time/sl_time (IN/OUT/TP/SL).\n'
# Fix indentation on line 95
lines[94] = '        return jsonify({"error": "limit must be > 0"}), 400\n'
open(p, 'w', encoding='utf-8').writelines(lines)
print('done')
