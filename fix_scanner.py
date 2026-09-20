p = r'c:\neoterminal\app_pkg\ai\scanner.py'
lines = open(p, encoding='utf-8').readlines()
# Replace lines 176-182 (indices 175-181) with clean version (7 lines: 5 comment + blank)
new_block = [
    '        # BLOCK-42: сканер оценивает стратегию с ВЫХОДОМ ТОЛЬКО по линиям TP/SL\n',
    '        # (config.BACKTEST_SL_ATR × ATR для риска, config.BACKTEST_RR для R/R).\n',
    '        # Тот же самый прогон отдаёт /api/backtest/trades — поэтому число\n',
    '        # сделок в панели и число блоков на графике совпадают 1:1, а у каждой\n',
    '        # сделки есть tp_price/sl_price (IN/OUT/TP/SL на графике).\n',
]
lines[175:182] = new_block
open(p, 'w', encoding='utf-8').writelines(lines)
print('done')
