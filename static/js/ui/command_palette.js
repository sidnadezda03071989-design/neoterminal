// Command Palette (⌘K / Ctrl+K): реестр команд + fuzzy-поиск + UI + действия.
// Открытие/закрытие: window.__toggleCmdPalette (main.js), кнопка ⌘ K в topbar.

import { state } from '../state.js';
import { switchSymbol, setTimeframe, switchMode, setTool } from './toolbar.js';
import { toggleIndicator } from './indicators.js';
import { runAIAnalysis } from '../ai/analysis.js';
import { openChat, closeChat } from '../ai/chat.js';
import { loadLive } from '../data/live.js';
import { loadReplay } from '../data/replay.js';

const $ = (id) => document.getElementById(id);

// Переключение индикатора: реестр + чекбокс в меню индикаторов.
// id совпадает с ключом state.indicators и суффиксом #ind-<id>.
function toggleInd(id) {
  const current = state.indicators[id];
  toggleIndicator(id, !current);
  const cb = document.getElementById('ind-' + id);
  if (cb) cb.checked = !current;
}

// --------------------------------------------------------------- категории
const CATEGORY_LABELS = {
  symbol: 'Символ',
  timeframe: 'Таймфрейм',
  mode: 'Режим',
  tool: 'Инструмент',
  indicator: 'Индикатор',
  action: 'Действие',
};

// ---------------------------------------------------------- реестр команд
// {id, label, category, keywords, hint, run} — id совпадает с реальными
// ключами приложения: data-tool, state.indicators, значения select'ов.
const COMMANDS = [
  // --- symbol (значения #symbol-select) ---
  { id: 'BTCUSDT', label: 'BTCUSDT', category: 'symbol',
    keywords: ['btc', 'bitcoin', 'биткоин', 'крипта'],
    hint: 'Bitcoin', run: () => { switchSymbol('BTCUSDT'); } },
  { id: 'ETHUSDT', label: 'ETHUSDT', category: 'symbol',
    keywords: ['eth', 'ethereum', 'эфир', 'крипта'],
    hint: 'Ethereum', run: () => { switchSymbol('ETHUSDT'); } },
  { id: 'SOLUSDT', label: 'SOLUSDT', category: 'symbol',
    keywords: ['sol', 'solana', 'крипта'],
    hint: 'Solana', run: () => { switchSymbol('SOLUSDT'); } },
  { id: 'EURUSD', label: 'EURUSD', category: 'symbol',
    keywords: ['eur', 'eurusd', 'евро', 'forex'],
    hint: 'Евро/Доллар', run: () => { switchSymbol('EURUSD'); } },
  { id: 'GBPUSD', label: 'GBPUSD', category: 'symbol',
    keywords: ['gbp', 'sterling', 'фунт', 'forex'],
    hint: 'Фунт/Доллар', run: () => { switchSymbol('GBPUSD'); } },
  { id: 'USDCHF', label: 'USDCHF', category: 'symbol',
    keywords: ['chf', 'franc', 'франк', 'forex'],
    hint: 'Доллар/Франк', run: () => { switchSymbol('USDCHF'); } },
  { id: 'USDJPY', label: 'USDJPY', category: 'symbol',
    keywords: ['jpy', 'yen', 'иена', 'forex'],
    hint: 'Доллар/Иена', run: () => { switchSymbol('USDJPY'); } },
  { id: 'USDCAD', label: 'USDCAD', category: 'symbol',
    keywords: ['cad', 'loonie', 'канада', 'forex'],
    hint: 'Доллар/Канада', run: () => { switchSymbol('USDCAD'); } },
  { id: 'EURJPY', label: 'EURJPY', category: 'symbol',
    keywords: ['eurjpy', 'евро-иена', 'eur', 'jpy', 'forex'],
    hint: 'Евро/Иена', run: () => { switchSymbol('EURJPY'); } },
  { id: 'XAUUSD', label: 'XAUUSD', category: 'symbol',
    keywords: ['gold', 'золото', 'xau', 'металл'],
    hint: 'Золото', run: () => { switchSymbol('XAUUSD'); } },

  // --- timeframe (значения #timeframe-select) ---
  { id: '1m', label: '1m', category: 'timeframe',
    keywords: ['1m', 'm1', 'минута', 'минутный'],
    hint: '1 минута', run: () => { setTimeframe('1m'); } },
  { id: '5m', label: '5m', category: 'timeframe',
    keywords: ['5m', 'm5', 'пять минут'],
    hint: '5 минут', run: () => { setTimeframe('5m'); } },
  { id: '15m', label: '15m', category: 'timeframe',
    keywords: ['15m', 'm15', 'четверть часа'],
    hint: '15 минут', run: () => { setTimeframe('15m'); } },
  { id: '1H', label: '1H', category: 'timeframe',
    keywords: ['1h', 'h1', 'час', 'часовой'],
    hint: '1 час', run: () => { setTimeframe('1H'); } },
  { id: '4H', label: '4H', category: 'timeframe',
    keywords: ['4h', 'h4', 'четыре часа'],
    hint: '4 часа', run: () => { setTimeframe('4H'); } },
  { id: '1D', label: '1D', category: 'timeframe',
    keywords: ['1d', 'd1', 'день', 'дневной', 'daily'],
    hint: '1 день', run: () => { setTimeframe('1D'); } },

  // --- mode (#mode-toggle) ---
  { id: 'live', label: 'Live', category: 'mode',
    keywords: ['live', 'лайв', 'реальное время'],
    hint: 'Реальное время', run: () => { switchMode('live'); } },
  { id: 'replay', label: 'Replay', category: 'mode',
    keywords: ['replay', 'реплей', 'воспроизведение', 'возврат', 'бэктест'],
    hint: 'Воспроизведение', run: () => { switchMode('replay'); } },

  // --- tool (data-tool кнопок toolbar) ---
  { id: 'cursor', label: 'Cursor', category: 'tool',
    keywords: ['cursor', 'курсор', 'стрелка', 'выделение'],
    hint: 'Выделение и перемещение', run: () => { setTool('cursor'); } },
  { id: 'trendline', label: 'Trendline', category: 'tool',
    keywords: ['trendline', 'trend', 'трендовая', 'линия'],
    hint: 'Трендовая линия', run: () => { setTool('trendline'); } },
  { id: 'ray', label: 'Ray', category: 'tool',
    keywords: ['ray', 'луч'],
    hint: 'Луч', run: () => { setTool('ray'); } },
  { id: 'h_line', label: 'H-Line', category: 'tool',
    keywords: ['h_line', 'hline', 'горизонталь', 'уровень', 'line'],
    hint: 'Горизонтальная линия', run: () => { setTool('h_line'); } },
  { id: 'rectangle', label: 'Rectangle', category: 'tool',
    keywords: ['rectangle', 'rect', 'прямоугольник'],
    hint: 'Прямоугольник', run: () => { setTool('rectangle'); } },
  { id: 'fib', label: 'Fib', category: 'tool',
    keywords: ['fib', 'fibo', 'фибо', 'фибоначчи'],
    hint: 'Уровни Фибоначчи', run: () => { setTool('fib'); } },
  { id: 'pen', label: 'Pen', category: 'tool',
    keywords: ['pen', 'перо', 'кисть', 'freehand'],
    hint: 'Свободное рисование', run: () => { setTool('pen'); } },
  { id: 'text', label: 'Text', category: 'tool',
    keywords: ['text', 'текст', 'подпись', 'label'],
    hint: 'Текстовая метка', run: () => { setTool('text'); } },

  // --- indicator (ключи state.indicators; через toggleInd) ---
  { id: 'sma', label: 'SMA20', category: 'indicator',
    keywords: ['sma', 'sma20', 'ma', 'скользящая'],
    hint: 'Скользящая средняя 20', run: () => { toggleInd('sma'); } },
  { id: 'ema', label: 'EMA50', category: 'indicator',
    keywords: ['ema', 'ema50', 'экспоненциальная'],
    hint: 'Эксп. средняя 50', run: () => { toggleInd('ema'); } },
  { id: 'bb', label: 'Bollinger', category: 'indicator',
    keywords: ['bb', 'bollinger', 'боллинджер', 'полосы'],
    hint: 'Полосы Боллинджера', run: () => { toggleInd('bb'); } },
  { id: 'volume', label: 'Volume', category: 'indicator',
    keywords: ['volume', 'объём', 'объем'],
    hint: 'Объём', run: () => { toggleInd('volume'); } },
  { id: 'rsi', label: 'RSI', category: 'indicator',
    keywords: ['rsi', 'осциллятор'],
    hint: 'Индекс силы', run: () => { toggleInd('rsi'); } },
  { id: 'macd', label: 'MACD', category: 'indicator',
    keywords: ['macd'],
    hint: 'MACD', run: () => { toggleInd('macd'); } },
  { id: 'vwap', label: 'VWAP', category: 'indicator',
    keywords: ['vwap'],
    hint: 'Средняя по объёму', run: () => { toggleInd('vwap'); } },
  { id: 'supertrend', label: 'Supertrend', category: 'indicator',
    keywords: ['supertrend', 'супертренд', 'trend'],
    hint: 'Supertrend', run: () => { toggleInd('supertrend'); } },

  // --- action (кнопки toolbar/topbar) ---
  { id: 'ai-analysis', label: 'AI Analysis', category: 'action',
    keywords: ['ai', 'analysis', 'анализ', 'ии', 'сигнал'],
    hint: 'Запустить AI-анализ', run: () => { runAIAnalysis(); } },
  { id: 'clear-all', label: 'Clear All', category: 'action',
    keywords: ['clear', 'очистить', 'все', 'remove'],
    hint: 'Очистить все рисунки', run: () => { if (state.dm) state.dm.clearAll(); } },
  { id: 'erase-ai', label: 'Erase AI', category: 'action',
    keywords: ['erase', 'стереть', 'ai', 'удалить'],
    hint: 'Стереть AI-рисунки', run: () => { if (state.dm) state.dm.eraseAI(); } },
  { id: 'export', label: 'Export', category: 'action',
    keywords: ['export', 'экспорт', 'сохранить', 'json'],
    hint: 'Экспорт рисунков в JSON', run: () => { if (state.dm) state.dm.exportJSON(); } },
  { id: 'import', label: 'Import', category: 'action',
    keywords: ['import', 'импорт', 'загрузить', 'json'],
    hint: 'Импорт рисунков из JSON',
    run: () => { document.getElementById('import-btn')?.click(); } },
  { id: 'toggle-chat', label: 'Toggle Chat', category: 'action',
    keywords: ['chat', 'чат', 'toggle', 'открыть'],
    hint: 'Открыть/закрыть чат',
    run: () => {
      const p = document.getElementById('chat-panel');
      if (p && p.style.display === 'flex') closeChat();
      else openChat();
    } },
  { id: 'close-chat', label: 'Close Chat', category: 'action',
    keywords: ['chat', 'чат', 'close', 'закрыть'],
    hint: 'Закрыть чат', run: () => { closeChat(); } },
  { id: 'refresh-data', label: 'Refresh Data', category: 'action',
    keywords: ['refresh', 'update-refresh', 'reload', 'обновить', 'перезагрузить'],
    hint: 'Обновить данные графика',
    run: () => {
      if (state.mode === 'live') loadLive(true);
      else loadReplay();
    } },
];

// ------------------------------------------------------------ fuzzy-поиск
// Буквы query должны идти в text по порядку (подпоследовательность).
// score = 1 / (1 + количество пропущенных символов между совпадениями).
// Регистронезависимо. Возвращает score (>0) или null, если не совпало.
export function fuzzyMatch(query, text) {
  const q = String(query || '').toLowerCase();
  const t = String(text || '').toLowerCase();
  if (!q) return 1;
  let gaps = 0;
  let prev = -1;
  for (const ch of q) {
    const idx = t.indexOf(ch, prev + 1);
    if (idx === -1) return null;
    if (prev >= 0) gaps += idx - prev - 1;
    prev = idx;
  }
  return 1 / (1 + gaps);
}

// Поиск по реестру: label + keywords + id, лучший score, sort desc, топ 10.
// Пустой query — дефолтный список: топ 8 из категорий action + symbol.
export function searchCommands(query) {
  const q = String(query || '').trim();
  if (!q) {
    const actions = COMMANDS.filter((c) => c.category === 'action');
    const symbols = COMMANDS.filter((c) => c.category === 'symbol');
    return [...actions.slice(0, 4), ...symbols.slice(0, 4)];
  }
  const scored = [];
  for (const cmd of COMMANDS) {
    let best = null;
    for (const text of [cmd.label, ...cmd.keywords, cmd.id]) {
      const s = fuzzyMatch(q, text);
      if (s != null && (best == null || s > best)) best = s;
    }
    if (best != null) scored.push({ cmd, score: best });
  }
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, 10).map((s) => s.cmd);
}

// ------------------------------------------------------------------- UI
let _palette = null;
let _isOpen = false;
let _filtered = [];    // текущий отфильтрованный список команд
let _activeIndex = 0;  // индекс активного элемента в _filtered

function buildPalette() {
  // Guard: повторный вызов не создаёт дубликат палитры.
  const existing = document.getElementById('cmd-palette');
  if (existing) return existing;

  // (а) создать div, (б) вся разметка через innerHTML.
  const div = document.createElement('div');
  div.id = 'cmd-palette';
  div.innerHTML =
    '<input type="text" id="cmd-input" placeholder="Введите команду…" ' +
      'autocomplete="off" spellcheck="false">' +
    '<div id="cmd-list"></div>' +
    '<div class="cmd-footer">' +
      '<span><kbd>↑↓</kbd> навигация</span>' +
      '<span><kbd>Enter</kbd> выполнить</span>' +
      '<span><kbd>Esc</kbd> закрыть</span>' +
    '</div>';
  // (в) вставка в DOM — ДО навешивания слушателей.
  document.body.appendChild(div);

  // (г) ТОЛЬКО ТЕПЕРЬ слушатели. Элементы ищем внутри div (querySelector),
  // а не по document.getElementById — не зависим от момента вставки
  // и глобального ID-поиска (именно он возвращал null → TypeError на 225).
  const input = div.querySelector('#cmd-input');
  const list = div.querySelector('#cmd-list');

  input.addEventListener('input', () => {
    _activeIndex = 0;
    render();
  });
  input.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      moveActive(1);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      moveActive(-1);
    } else if (e.key === 'Enter') {
      e.preventDefault();
      executeActive();
    } else if (e.key === 'Escape') {
      e.preventDefault();
      closePalette();
    }
  });

  // Клик по элементу списка — выполнить + закрыть (делегирование).
  list.addEventListener('click', (e) => {
    const item = e.target.closest('.cmd-item');
    if (!item) return;
    const cmd = _filtered[Number(item.dataset.index)];
    if (cmd) runCommand(cmd);
  });

  // Клик вне палитры — закрыть. Клик по кнопке-тогглу игнорируем,
  // иначе mousedown закроет палитру, а click тут же снова откроет.
  document.addEventListener('mousedown', (e) => {
    if (!_isOpen) return;
    if (e.target.closest('#cmd-palette-btn')) return;
    if (_palette && !_palette.contains(e.target)) closePalette();
  });

  return div;
}

function render() {
  if (!_palette) return;
  const list = _palette.querySelector('#cmd-list');
  list.innerHTML = '';
  const inputValue = _palette.querySelector('#cmd-input').value;
  _filtered = searchCommands(inputValue);
  if (!_filtered.length) {
    const empty = document.createElement('div');
    empty.className = 'cmd-empty';
    empty.textContent = 'Ничего не найдено';
    list.appendChild(empty);
    return;
  }
  _activeIndex = Math.min(_activeIndex, _filtered.length - 1);
  _filtered.forEach((cmd, i) => {
    const item = document.createElement('div');
    item.className = 'cmd-item' + (i === _activeIndex ? ' active' : '');
    item.dataset.index = i;
    item.innerHTML =
      '<span class="cmd-category">' +
      (CATEGORY_LABELS[cmd.category] || cmd.category) + '</span>' +
      '<span class="cmd-label"></span>' +
      '<span class="cmd-hint">' + (cmd.hint || '') + '</span>';
    item.querySelector('.cmd-label').textContent = cmd.label;
    list.appendChild(item);
  });
  const active = list.querySelector('.cmd-item.active');
  if (active) active.scrollIntoView({ block: 'nearest' });
}

function moveActive(dir) {
  if (!_filtered.length) return;
  // Навигация с wrap: за границей списка перескакиваем на другой конец.
  _activeIndex = (_activeIndex + dir + _filtered.length) % _filtered.length;
  render();
}

function runCommand(cmd) {
  try {
    cmd.run();
  } catch (e) {
    console.error('command palette run:', e);
  }
  closePalette();
}

function executeActive() {
  const cmd = _filtered[_activeIndex];
  if (cmd) runCommand(cmd);
}

function openPalette() {
  if (!_palette) _palette = buildPalette();
  const input = _palette.querySelector('#cmd-input');
  input.value = '';       // при открытии query очищается
  _activeIndex = 0;
  render();
  _palette.classList.add('open');
  _isOpen = true;
  input.focus();
}

function closePalette() {
  if (!_palette) return;
  _palette.classList.remove('open');
  _isOpen = false;
}

export function toggleCommandPalette() {
  if (_isOpen) closePalette();
  else openPalette();
}

// Для ручной проверки в консоли: window.__cmdSearch('btc') → отфильтрованный
// список; window.__toggleCmdPalette() открывает палитру (main.js).
if (typeof window !== 'undefined') {
  window.__cmdSearch = searchCommands;
}

