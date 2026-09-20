/* Сканер стратегий (grid-search): панель, запуск /api/scan, SSE-прогресс
   (window.__scanProgress вешается в app_init.js), таблица результатов
   с фильтром min_sharpe, детали по клику, экспорт CSV и кнопки
   «Показать на графике»/«Скрыть сделки» (BLOCK-29b). */

import { state } from './state.js';
import { switchSymbol, setTimeframe } from './ui/toolbar.js';

const $ = (id) => document.getElementById(id);

/* Список символов зеркален config.SYMBOLS (app_pkg/config.py) и
   select#symbol-select в templates/index.html. */
const SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'EURUSD', 'GBPUSD',
  'USDCHF', 'USDJPY', 'USDCAD', 'EURJPY', 'XAUUSD'];
const STRATEGIES = [
  { id: 'sma_cross', label: 'SMA Cross' },
  { id: 'rsi_reversal', label: 'RSI Reversal' },
  { id: 'macd_cross', label: 'MACD Cross' },
  { id: 'ema_cross', label: 'EMA Cross' },
  { id: 'bb_reversal', label: 'BB Reversal' },
  { id: 'bb_breakout', label: 'BB Breakout' },
  { id: 'supertrend', label: 'Supertrend' },
  { id: 'stoch_reversal', label: 'Stoch Reversal' },
  { id: 'stoch_cross', label: 'Stoch Cross' },
  { id: 'cci_reversal', label: 'CCI Reversal' },
  { id: 'vwap_reversal', label: 'VWAP Reversal' },
  { id: 'adx_trend', label: 'ADX Trend' },
];
/* Фолбэк-константы оценки прогона: те же цифры приходят из /api/scan/grids
   (seconds_per_combination / fetch_seconds_per_symbol / workers /
   warn_seconds / hard_limit_seconds) — см. routes/scanner.py::_estimate_seconds.
   hard_limit_seconds — потолок длительности прогона (4 часа): больше —
   сервер вернёт 400, кнопка «Запустить» должна быть disabled. */
const ESTIMATE_DEFAULTS = {
  seconds_per_combination: 0.3, fetch_seconds_per_symbol: 15,
  workers: 8, max_combinations: 50000, warn_seconds: 1800,
  hard_limit_seconds: 14400, max_combinations_full: 2000,
};
/* Запасной поллинг: если SSE-событие scan_progress потерялось, дожидаемся
   конца прогона по stats.finished из GET /api/scan/<run_id>. */
const POLL_MS = 10000;

/* Вердикты backend: порядок сортировки таблицы и человекочитаемый вид. */
const VERDICT_ORDER = { stable: 0, weak: 1, overfit: 2, noise: 3, losing: 4 };
const VERDICT_TEXT = {
  stable: '✅ Стабильна',
  overfit: '⚠️ Переоптимизация',
  noise: '🎲 Шум',
  weak: '🟡 Слабо',
  losing: '❌ Убыточна',
};

const local = { runId: null, running: false, pollTimer: null,
  grids: null, summary: null, etaSeconds: null, wired: false,
  timeframes: [] };
/* Полный список ТФ: фолбэк для config_timeframes(), пока /api/scan/grids
   не загрузился (BLOCK-35). Раньше в фолбэке были только 15m/1H — на
   первом открытии панели чекбоксы строились ДО ответа grids, поэтому
   5m/4H/1D никогда не появлялись. */
const TF_ALL = ['5m', '15m', '1H', '4H', '1D'];
/* Дефолт-отмеченные ТФ в UI, если grids ещё не загрузились: повторяет
   config.SCAN_DEFAULT_TIMEFRAMES (отмеченные по умолчанию). */
const TF_FALLBACK = ['15m', '1H'];

/* ------------------------------------------------------- панель open/close */
export function openScanner() {
  const panel = $('scanner-panel');
  if (!panel) return;
  panel.style.display = 'block';
  _initFormDefaults();
  _ensureGrids();
}

export function closeScanner() {
  const panel = $('scanner-panel');
  if (panel) panel.style.display = 'none';
}

/* Чекбоксы символов и стратегий строим один раз из JS-списков. */
function _initFormDefaults() {
  const mkChecks = (boxId, items, defaults) => {
    const box = $(boxId);
    if (!box || box.children.length) return;
    for (const item of items) {
      const id = typeof item === 'string' ? item : item.id;
      const label = typeof item === 'string' ? item : item.label;
      const el = document.createElement('label');
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.value = id;
      cb.checked = defaults.includes(id);
      el.append(cb, document.createTextNode(label));
      box.appendChild(el);
    }
  };
  mkChecks('scan-symbols', SYMBOLS, ['BTCUSDT']);
  mkChecks('scan-strategies', STRATEGIES, ['rsi_reversal', 'macd_cross']);
  mkChecks('scan-timeframes', config_timeframes(), local.timeframes.length
    ? local.timeframes
    : ((local.grids && Array.isArray(local.grids.default_timeframes))
      ? local.grids.default_timeframes : TF_FALLBACK));
  // Любое изменение выбора символов/стратегий/ТФ пересчитывает counter,
  // оценку комбинаций/времени и состояние чекбоксов «Все».
  // wired: слушатели навешиваются один раз; _onSelectionChange продублирует
  // первый вызов (безопасный для инициализации).
  if (!local.wired) {
    for (const boxId of ['scan-symbols', 'scan-strategies', 'scan-timeframes']) {
      const box = $(boxId);
      if (box) box.addEventListener('change', _onSelectionChange);
    }
    const allStr = $('scan-all-strategies');
    const allTf = $('scan-all-timeframes');
    if (allStr) allStr.addEventListener('change', toggleAllStrategies);
    if (allTf) allTf.addEventListener('change', toggleAllTimeframes);
    const fullCb = $('scan-full-history');
    if (fullCb) fullCb.addEventListener('change', _onSelectionChange);
    local.wired = true;
  }
  _onSelectionChange();
  _syncAllStrategies();
  _syncAllTimeframes();
  _updateComboCount();
}

/* Любое изменение выбора (чекбоксы символов/стратегий/ТФ или «Все…»)
   пересчитывает counter, оценку комбинаций/времени и состояние
   чекбоксов «Все». */
function _onSelectionChange() {
  _syncAllStrategies();
  _syncAllTimeframes();
  _updateComboCount();
}

/* Чекбокс «Все стратегии»: checked, если отмечены все 12; indeterminate,
   если отмечена только часть (см. templates/index.html). */
function _syncAllStrategies() {
  const all = $('scan-all-strategies');
  const chosen = _checkedValues('scan-strategies').length;
  if (all) {
    all.checked = chosen > 0 && chosen === STRATEGIES.length;
    all.indeterminate = chosen > 0 && chosen < STRATEGIES.length;
  }
  const cnt = $('scan-strategies-count');
  if (cnt) cnt.textContent = `Выбрано: ${chosen} ${_pluralStrategies(chosen)}`;
}

/* Русская форма слова «стратегия» для counter. */
function _pluralStrategies(n) {
  const mod10 = n % 10;
  const mod100 = n % 100;
  if (mod10 === 1 && mod100 !== 11) return 'стратегия';
  if (mod10 >= 2 && mod10 <= 4 && (mod100 < 12 || mod100 > 14)) {
    return 'стратегии';
  }
  return 'стратегий';
}

/* Таймфреймы, доступные в чекбоксах: берём из /api/scan/grids.timeframes,
   пока гриды не загрузились — полный список TF_ALL (5m/15m/1H/4H/1D). */
function config_timeframes() {
  if (local.grids && Array.isArray(local.grids.timeframes)
      && local.grids.timeframes.length) {
    return local.grids.timeframes;
  }
  return TF_ALL.slice();
}

/* ---------- Таймфреймы: toggle-all + counter ---------- */

function _syncAllTimeframes() {
  const all = $('scan-all-timeframes');
  const box = $('scan-timeframes');
  const cnt = $('scan-timeframes-count');
  if (!box) return;
  const allTfs = config_timeframes();
  const checked = _checkedValues('scan-timeframes');
  const chosen = checked.length;
  if (all) {
    all.checked = chosen > 0 && chosen === allTfs.length;
    all.indeterminate = chosen > 0 && chosen < allTfs.length;
  }
  if (cnt) {
    cnt.textContent = `Выбрано: ${chosen} ${_pluralTimeframes(chosen)}`;
  }
}

function _pluralTimeframes(n) {
  if (n === 1) return 'ТФ';
  if (n >= 2 && n <= 4) return 'ТФ';
  return 'ТФ';
}

export function toggleAllTimeframes() {
  const all = $('scan-all-timeframes');
  const box = $('scan-timeframes');
  if (!all || !box) return;
  for (const cb of box.querySelectorAll('input[type="checkbox"]')) {
    cb.checked = all.checked;
  }
  _syncAllTimeframes();
}

/* Чекбокс «Все стратегии»: checked, если отмечены все 12; indeterminate,
   если отмечена только часть (см. templates/index.html). */
export function toggleAllStrategies() {
  const all = $('scan-all-strategies');
  const box = $('scan-strategies');
  if (!all || !box) return;
  for (const cb of box.querySelectorAll('input[type="checkbox"]')) {
    cb.checked = all.checked;
  }
  all.indeterminate = false;
  _onSelectionChange();
}

function _checkedValues(boxId) {
  const box = $(boxId);
  if (!box) return [];
  return Array.from(box.querySelectorAll('input:checked')).map((cb) => cb.value);
}

function _showError(msg) {
  const el = $('scan-error');
  if (!el) return;
  if (msg) {
    el.textContent = '⚠ ' + msg;
    el.style.display = 'block';
  } else {
    el.textContent = '';
    el.style.display = 'none';
  }
}

/* ------------------------------------------- конструктор вариаций (гриды) */
async function _ensureGrids() {
  if (local.grids) return;
  try {
    const resp = await fetch('/api/scan/grids');
    if (!resp.ok) return;
    local.grids = await resp.json();
    _rebuildTimeframeChecks();
    _buildConfigBlocks();
    _updateComboCount();
  } catch { /* конструктор просто останется пустым до следующего открытия */ }
}

/* Чекбоксы ТФ строятся из полного списка /api/scan/grids (BLOCK-35):
   на первом открытии панели grids ещё не загрузились, _initFormDefaults
   успевает построить чекбоксы из фолбэка, а после ответа grids они НЕ
   перестраивались — 5m/4H/1D пропадали из UI. Перестраиваем из полного
   списка, сохраняя текущий выбор пользователя. */
function _rebuildTimeframeChecks() {
  const box = $('scan-timeframes');
  const tfs = config_timeframes();
  if (!box || !tfs.length) return;
  const prev = new Set(_checkedValues('scan-timeframes'));
  const defaults = (local.grids && Array.isArray(local.grids.default_timeframes))
    ? local.grids.default_timeframes : TF_FALLBACK;
  box.innerHTML = '';
  for (const tf of tfs) {
    const el = document.createElement('label');
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.value = tf;
    cb.checked = prev.has(tf) || defaults.includes(tf);
    el.append(cb, document.createTextNode(tf));
    box.appendChild(el);
  }
  _syncAllTimeframes();
  _updateComboCount();
}

function _buildConfigBlocks() {
  const box = $('scan-config-blocks');
  if (!box || !local.grids) return;
  box.innerHTML = '';
  for (const [sid, meta] of Object.entries(local.grids.strategies)) {
    const block = document.createElement('div');
    block.className = 'scan-config-block';
    const title = document.createElement('div');
    title.className = 'scan-config-title';
    title.textContent = meta.label || sid;
    block.appendChild(title);
    for (const pname of meta.params) {
      const row = document.createElement('label');
      row.className = 'scan-config-row';
      const name = document.createElement('span');
      name.textContent = pname + ':';
      const inp = document.createElement('input');
      inp.type = 'text';
      inp.dataset.strategy = sid;
      inp.dataset.param = pname;
      const defaults = (meta.default_grid && meta.default_grid[pname]) || [];
      inp.value = defaults.join(', ');
      inp.placeholder = 'напр. 5, 10, 15';
      inp.addEventListener('input', _updateComboCount);
      row.append(name, inp);
      block.appendChild(row);
    }
    box.appendChild(block);
  }
}

/* '5, 10, x' -> [5, 10] — split -> trim -> number -> filter invalid.
   limits из /api/scan/grids.param_limits: у std/multiplier/vwap_threshold
   границы дробные (разрешаем float), у CCI oversold — отрицательные.
   Без границ поведение прежнее: целое > 0. */
function _parseValues(str, limits) {
  return String(str || '').split(',')
    .map((s) => s.trim())
    .filter(Boolean)
    .map(Number)
    .filter((n) => _valueOk(n, limits));
}

function _valueOk(n, limits) {
  if (!Number.isFinite(n)) return false;
  if (!Array.isArray(limits) || limits.length !== 2) {
    return Number.isInteger(n) && n > 0;
  }
  const [lo, hi] = limits;
  const allowFloat = !Number.isInteger(lo) || !Number.isInteger(hi);
  if (!allowFloat && !Number.isInteger(n)) return false;
  return n >= lo && n <= hi;
}

function _limitsFor(pname) {
  const lims = local.grids && local.grids.param_limits;
  return lims && lims[pname] ? lims[pname] : null;
}

/* Число комбинаций выбранного набора: Σ длин гридов стратегий (на
   один символ-ТФ). Учитывает свои параметры из конструктора, если он
   включён. Результат не умножает на символы/ТФ — это делают
   _updateComboCount/_currentEstimate, потому что для counter и estimate
   нужны понятные «стратегий × символов × ТФ». */
function _computedCombos(strategies, symbolsCount) {
  const useCustom = $('scan-use-custom') && $('scan-use-custom').checked;
  let total = 0;
  let badInput = false;
  for (const sid of strategies) {
    const meta = local.grids && local.grids.strategies[sid];
    const params = meta ? meta.params : [];
    let combos = 1;
    for (const pname of params) {
      let vals = null;
      if (useCustom) {
        const inp = document.querySelector(
          `#scan-config-blocks input[data-strategy="${sid}"][data-param="${pname}"]`);
        if (inp) {
          vals = _parseValues(inp.value, _limitsFor(pname));
          if (!vals.length && inp.value.trim() !== '') badInput = true;
        }
      }
      if (!vals || !vals.length) {
        vals = (meta && meta.default_grid && meta.default_grid[pname]) || [];
      }
      combos *= vals.length;
    }
    total += combos;
  }
  return { total: total * Math.max(Number(symbolsCount) || 0, 1), badInput };
}

/* Оценка длительности прогона (сек) — формула та же, что в бэкенде
   (routes/scanner.py::_estimate_seconds): комбинации × сек/воркер + фетч
   на каждый символ И таймфрейм (symbols × timeframes × сек). Полная история
   (20k) в ~4× медленнее 5k — умножаем итог на 4 (те же цифры, что ETA
   прогресса в ai/scanner.py::_eta_seconds). */

/* Чекбокс «Полная история»: по умолчанию включён (checked в HTML — дефолт
   backend config.SCAN_USE_FULL_HISTORY=True). */
function _isFullHistory() {
  const cb = $('scan-full-history');
  return !cb || cb.checked;
}

/* Порог «больше лимита»: при полной истории — max_combinations_full (меньше
   комбинаций за раз), иначе 50000 (SCAN_MAX_COMBINATIONS). */
function _maxCombos() {
  const cfg = Object.assign({}, ESTIMATE_DEFAULTS, local.grids || {});
  return _isFullHistory()
    ? (Number(cfg.max_combinations_full) ||
      ESTIMATE_DEFAULTS.max_combinations_full)
    : (Number(cfg.max_combinations) || ESTIMATE_DEFAULTS.max_combinations);
}

function _estimateSeconds(combos, symbolsCount, timeframesCount = 1) {
  const cfg = Object.assign({}, ESTIMATE_DEFAULTS, local.grids || {});
  const workers = Math.max(1, Number(cfg.workers) || 8);
  const timeframes = Math.max(1, Number(timeframesCount) || 1);
  const base = (combos * Number(cfg.seconds_per_combination)) / workers +
    symbolsCount * timeframes * Number(cfg.fetch_seconds_per_symbol);
  return _isFullHistory() ? base * 4 : base;
}

/* Секунды -> '45 сек' / '12 мин' / '4 ч 15 мин' / '1 ч'. */
function _formatDuration(seconds) {
  const sec = Math.max(0, Math.round(Number(seconds) || 0));
  if (sec < 60) return `${sec} сек`;
  const min = Math.round(sec / 60);
  if (min < 60) return `${min} мин`;
  const hours = Math.floor(min / 60);
  const mins = min % 60;
  return mins ? `${hours} ч ${mins} мин` : `${hours} ч`;
}

/* 24120 -> '24 120' (разделитель тысяч пробелом — как в подписи прогресса). */
function _fmtCount(n) {
  return String(Math.max(0, Math.round(Number(n) || 0)))
    .replace(/\B(?=(\d{3})+(?!\d))/g, ' ');
}

/* Текущий масштаб выбора: комбинации, оценка времени и лимиты сервера. */
function _currentEstimate() {
  const strategies = _checkedValues('scan-strategies');
  const symbols = _checkedValues('scan-symbols');
  const cfg = Object.assign({}, ESTIMATE_DEFAULTS, local.grids || {});
  const combos = _computedCombos(strategies, symbols.length).total;
  return {
    combos,
    seconds: _estimateSeconds(combos, symbols.length || 1),
    warnSeconds: Number(cfg.warn_seconds) || ESTIMATE_DEFAULTS.warn_seconds,
    hardSeconds: Number(cfg.hard_limit_seconds) ||
      ESTIMATE_DEFAULTS.hard_limit_seconds,
    maxCombos: _maxCombos(),
  };
}

function _updateComboCount() {
  const strategies = _checkedValues('scan-strategies');
  const { total, badInput } = _computedCombos(strategies);
  const symbols = _checkedValues('scan-symbols');
  const timeframes = _checkedValues('scan-timeframes');
  const tfCount = Math.max(1, timeframes.length);
  const symCount = Math.max(1, symbols.length);
  const maxCombos = _maxCombos();
  const el = $('scan-combo-count');
  if (el) {
    el.textContent = `Сгенерируется ${_fmtCount(total * symCount * tfCount)} комбинаций` +
      ` (${strategies.length} стр. × ${symCount} симв. × ${tfCount} ТФ)` +
      (badInput ? ' — есть некорректные значения' : '');
    el.classList.toggle('scan-combo-over', total * symCount * tfCount > maxCombos);
    el.classList.toggle('scan-combo-bad', badInput);
  }
  _updateEstimate(total, symCount, badInput, maxCombos);
}

/* Подсказка рядом с кнопкой «Запустить»:
   «Сгенерируется ~24 120 комбинаций (оценка ~30 мин)». Плюс предупреждения:
   жёлтое (> 30 мин — спросим confirm перед запуском) и красное (> 4 ч —
   объём слишком большой, кнопка «Запустить» disabled). */
function _updateEstimate(total, symbolsCount, badInput, maxCombos) {
  const el = $('scan-estimate');
  const cfg = Object.assign({}, ESTIMATE_DEFAULTS, local.grids || {});
  const seconds = _estimateSeconds(total, symbolsCount);
  const warnSeconds = Number(cfg.warn_seconds) || ESTIMATE_DEFAULTS.warn_seconds;
  const hardSeconds = Number(cfg.hard_limit_seconds) ||
    ESTIMATE_DEFAULTS.hard_limit_seconds;
  const over = total > maxCombos;
  const tooLong = !over && seconds > hardSeconds;
  const warn = !over && !tooLong && seconds > warnSeconds;
  if (el) {
    const mode = _isFullHistory()
      ? `полная история, ~${_formatDuration(seconds)}`
      : `оценка ~${_formatDuration(seconds)}`;
    let msg = `Сгенерируется ~${_fmtCount(total)} комбинаций (${mode})`;
    if (over) {
      msg += ` — больше лимита ${_fmtCount(maxCombos)}, сервер отклонит запуск`;
    }
    if (tooLong) {
      msg += ' ❌ Слишком большой объём (>4ч). ' +
        'Уменьшите символы/ТФ/стратегии.';
    }
    if (warn) {
      msg += ` ⚠️ Прогон займёт ~${_formatDuration(seconds)}. ` +
        'Убедитесь, что сервер не будет перезапущен.';
    }
    if (badInput) msg += ' — есть некорректные значения';
    el.textContent = msg;
    el.classList.toggle('scan-estimate-over', over || tooLong);
    el.classList.toggle('scan-estimate-warn', warn);
  }
  // Кнопка активна для warn-прогона (подтверждение спрашивается в confirm)
  // и disabled, если сервер всё равно отклонит запуск (лимит/4 часа).
  const runBtn = $('scan-run-btn');
  if (runBtn && !local.running) runBtn.disabled = over || tooLong;
}

export function toggleScanConfig() {
  const panel = $('scan-config-panel');
  if (!panel) return;
  const open = panel.style.display !== 'none';
  panel.style.display = open ? 'none' : 'block';
  const arrow = $('scan-config-arrow');
  if (arrow) arrow.textContent = open ? '▸' : '▾';
  if (!open) { _ensureGrids(); _updateComboCount(); }
}

export function resetScanConfig() {
  if (!local.grids) return;
  for (const inp of document.querySelectorAll('#scan-config-blocks input')) {
    const meta = local.grids.strategies[inp.dataset.strategy];
    const defaults = meta && meta.default_grid && meta.default_grid[inp.dataset.param];
    inp.value = (defaults || []).join(', ');
  }
  _updateComboCount();
}

/* Свои гриды только для выбранных стратегий; пустые параметры
   не передаются — сервер подставит дефолтные значения. */
function _collectCustomGrids(strategies) {
  const useCustom = $('scan-use-custom') && $('scan-use-custom').checked;
  if (!useCustom) return null;
  const grids = {};
  for (const sid of strategies) {
    const grid = {};
    document.querySelectorAll(
      `#scan-config-blocks input[data-strategy="${sid}"]`).forEach((inp) => {
      const vals = _parseValues(inp.value, _limitsFor(inp.dataset.param));
      if (vals.length) grid[inp.dataset.param] = vals;
    });
    if (Object.keys(grid).length) grids[sid] = grid;
  }
  return Object.keys(grids).length ? grids : null;
}

/* ---------------------------------------------------------------- прогресс */
/* Прогресс: «Прогон 24 120 комбинаций • Обработано 300 • Осталось ~2 ч 15 мин»
   (ETA приходит из SSE раз в SCAN_ETA_PUSH_EVERY комбинаций,
   см. ai/scanner.py). */
function _setProgress(done, total, etaSeconds, tfInfo) {
  const wrap = $('scan-progress');
  if (wrap) wrap.style.display = 'block';
  const bar = $('scan-progress-bar');
  if (bar) {
    bar.style.width = total ? Math.min(100, (done / total) * 100) + '%' : '0%';
  }
  const text = $('scan-progress-text');
  if (!text) return;
  if (!total) { text.textContent = '0/…'; return; }
  const eta = Number(etaSeconds);
  const parts = [`Прогон ${_fmtCount(total)} комбинаций`];
  /* Текущая пара (symbol, tf) приходит в SSE-событии (tf_index/tf_total/
     current_tf/current_symbol): при мульти-ТФ видно, где идёт скан. */
  if (tfInfo) parts.push(tfInfo);
  if (done) parts.push(`Обработано ${_fmtCount(done)}`);
  if (Number.isFinite(eta) && eta > 0) {
    parts.push(`Осталось ~${_formatDuration(eta)}`);
  }
  text.textContent = parts.join(' • ');
}

/* SSE 'scan_progress': app_init.js делает window.__scanProgress =
   updateScanProgress. Формат data: {run_id, done, total, current,
   eta_seconds?}. */
export function updateScanProgress(data) {
  if (!data) return;
  if (local.runId && data.run_id !== local.runId) return; // чужой прогон
  if (!local.runId && data.run_id) local.runId = data.run_id; // SSE обогнал POST
  const total = Number(data.total) || 0;
  const done = Number(data.done) || 0;
  // Пара (symbol, tf) упала: показываем красный блок, но прогресс НЕ
  // останавливаем — скан продолжается на следующих парах.
  if (data.last_error) {
    _showError(`Сбой на ${data.failed_at || '?'}: ${data.last_error}`);
  }
  // ETA приходит не в каждом событии — запоминаем последнюю, чтобы текст
  // не «мигал» между событиями без eta_seconds.
  const eta = Number(data.eta_seconds);
  if (Number.isFinite(eta) && eta > 0) local.etaSeconds = eta;
  /* Текущая пара «ТФ i/n: tf · symbol» — видна при скане по нескольким ТФ. */
  let tfInfo = null;
  if (data.tf_total > 1) {
    tfInfo = `ТФ ${data.tf_index}/${data.tf_total}`
      + (data.current_tf ? ': ' + data.current_tf : '')
      + (data.current_symbol ? ' · ' + data.current_symbol : '');
  }
  _setProgress(done, total, done >= total ? null : local.etaSeconds, tfInfo);
  if (total && done >= total) _finishScan();
}

function _stopPoll() {
  if (local.pollTimer) { clearInterval(local.pollTimer); local.pollTimer = null; }
}

function _startPoll() {
  _stopPoll();
  local.pollTimer = setInterval(async () => {
    if (!local.running || !local.runId) { _stopPoll(); return; }
    try {
      const resp = await fetch(`/api/scan/${local.runId}?min_sharpe=-999`);
      if (!resp.ok) return;
      const data = await resp.json();
      if (data.stats && data.stats.finished) {
        _finishScan();
      } else if (data.stats && data.stats.eta_seconds) {
        // Фолбэк ETA: SSE-событие потерялось — берём оценку из stats.
        local.etaSeconds = data.stats.eta_seconds;
        let tfInfo = null;
        const s = data.stats;
        if (s.tf_total > 1) {
          tfInfo = `ТФ ${s.tf_index}/${s.tf_total}`
            + (s.current_tf ? ': ' + s.current_tf : '')
            + (s.current_symbol ? ' · ' + s.current_symbol : '');
        }
        _setProgress(Number(s.done) || 0,
          Number(s.total) || 0, local.etaSeconds, tfInfo);
      }
    } catch { /* сеть могла моргнуть — повторим на следующем тике */ }
  }, POLL_MS);
}

function _finishScan() {
  _stopPoll();
  local.running = false;
  const runBtn = $('scan-run-btn');
  if (runBtn) { runBtn.disabled = false; runBtn.textContent = 'Запустить'; }
  loadResults();
}

/* ----------------------------------------------------------------- запуск */
export async function runScan() {
  if (local.running) return;
  // Данные графика ещё догружаются фоном — скан запускать рано.
  if (!state.progressiveLoaded) {
    alert('Данные графика ещё загружаются. Подождите.');
    return;
  }
  const symbols = _checkedValues('scan-symbols');
  const strategies = _checkedValues('scan-strategies');
  const timeframes = _checkedValues('scan-timeframes');
  _showError('');
  if (!symbols.length) { _showError('Выберите хотя бы один символ'); return; }
  if (!strategies.length) { _showError('Выберите хотя бы одну стратегию'); return; }
  if (!timeframes.length) { _showError('Выберите хотя бы один таймфрейм'); return; }

  // Защита от перегрузки до старта: больше лимита — не отправляем вовсе,
  // оценка > 4 часов — тоже отказ (сервер вернёт 400), оценка > 30 минут —
  // спрашиваем подтверждение. Те же проверки есть на сервере
  // (POST /api/scan: 400 / 400 / 200 + warning).
  const est = _currentEstimate();
  if (est.combos > est.maxCombos) {
    _showError(`Слишком много комбинаций: ${_fmtCount(est.combos)} > ` +
      `${_fmtCount(est.maxCombos)}. Сократите символы, ТФ или стратегии.`);
    return;
  }
  if (est.seconds > est.hardSeconds) {
    _showError(`❌ Слишком большой объём: оценка ~` +
      `${_formatDuration(est.seconds)} (>4ч). ` +
      'Уменьшите символы/ТФ/стратегии или запустите по частям.');
    return;
  }
  if (est.seconds > est.warnSeconds && !window.confirm(
    `⚠️ Прогон займёт ~${_formatDuration(est.seconds)} ` +
    `(${_fmtCount(est.combos)} комбинаций) — больше 30 минут. ` +
    'Убедитесь, что сервер не будет перезапущен. Продолжить?')) {
    return;
  }

  const runBtn = $('scan-run-btn');
  if (runBtn) { runBtn.disabled = true; runBtn.textContent = 'Выполняется…'; }
  local.running = true;
  local.runId = null;
  local.etaSeconds = null;
  _stopPoll();
  _setProgress(0, 0);
  _hideResults();

  try {
    const body = {
      symbols, timeframes, strategies,
      use_full_history: _isFullHistory(),
    };
    const customGrids = _collectCustomGrids(strategies);
    if (customGrids) body.custom_grids = customGrids;
    const resp = await fetch('/api/scan', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || 'HTTP ' + resp.status);
    local.runId = data.run_id;
    _setProgress(0, Number(data.total_combinations) || 0);
    _startPoll();
    if (data.warning) {
      // Сервер подтвердил долгий прогон (estimated_seconds > warn_seconds):
      // показываем жёлтое предупреждение рядом с кнопкой запуска.
      const estEl = $('scan-estimate');
      if (estEl) {
        const human = data.estimated_human ||
          _formatDuration(data.estimated_seconds);
        estEl.textContent = `⚠️ Прогон займёт ~${human}. ` +
          'Убедитесь, что сервер не будет перезапущен.';
        estEl.classList.remove('scan-estimate-over');
        estEl.classList.add('scan-estimate-warn');
      }
    }
  } catch (e) {
    _showError(e.message);
    local.running = false;
    _stopPoll();
    if (runBtn) { runBtn.disabled = false; runBtn.textContent = 'Запустить'; }
  }
}

function _hideResults() {
  const block = $('scan-results-block');
  if (block) block.style.display = 'none';
  const warning = $('scan-warning');
  if (warning) warning.style.display = 'none';
  const empty = $('scan-empty');
  if (empty) empty.style.display = 'none';
  const box = $('scan-results');
  if (box) box.innerHTML = '';
  const hint = $('scan-hint');
  if (hint) hint.style.display = 'none';
}

/* -------------------------------------------------------------- результаты */
export async function loadResults() {
  if (!local.runId) return;
  const inp = $('scan-min-sharpe');
  const parsed = inp ? parseFloat(inp.value) : NaN;
  const minSharpe = Number.isFinite(parsed) ? parsed : 0;
  let data;
  try {
    const resp = await fetch(`/api/scan/${local.runId}` +
      `?min_sharpe=${encodeURIComponent(minSharpe)}`);
    data = await resp.json();
    if (!resp.ok) throw new Error(data.error || 'HTTP ' + resp.status);
  } catch (e) {
    _showError(e.message);
    return;
  }
  const results = data.results || [];
  const warning = $('scan-warning');
  if (warning) {
    warning.textContent = data.warning || '';
    warning.style.display = data.warning ? 'block' : 'none';
  }
  const empty = $('scan-empty');
  if (empty) {
    empty.style.display = (!data.warning && !results.length) ? 'block' : 'none';
  }
  const hint = $('scan-hint');
  if (hint) hint.style.display = 'none';
  const block = $('scan-results-block');
  if (block) block.style.display = 'block';
  /* Сортировка по умолчанию: stable -> weak -> overfit -> noise -> losing,
     внутри группы — по combined_sharpe DESC. */
  results.sort((a, b) =>
    ((VERDICT_ORDER[a.verdict] ?? 9) - (VERDICT_ORDER[b.verdict] ?? 9)) ||
    ((Number(b.combined_sharpe) || 0) - (Number(a.combined_sharpe) || 0)));
  _renderResults(results);
  _loadSummary();
}

/* ------------------------------------------------------------- форматтеры */
function _fmtNum(v) {
  const n = Number(v);
  return v == null || !isFinite(n) ? '—' : n.toFixed(2);
}

function _fmtPct(v) {
  const n = Number(v);
  return v == null || !isFinite(n) ? '—' : (n * 100).toFixed(1) + '%';
}

/* Winrate с точностью до 0.01% (BLOCK-35): 1 знак схлопывал train/test
   с разным числом сделок (0.674 и 0.669 -> '67.4%' и '66.9%' различаются,
   а '67%' и '67%' — нет). */
function _fmtWinrate(v) {
  const n = Number(v);
  return v == null || !isFinite(n) ? '—' : (n * 100).toFixed(2) + '%';
}

function _fmtParams(params) {
  const p = params || {};
  const parts = Object.keys(p).map((k) => k + '=' + p[k]);
  return parts.length ? parts.join(' ') : '—';
}

function _rowClass(cs) {
  return cs >= 1.0 ? 'win' : cs >= 0 ? 'neutral' : 'lose';
}

function _detailLine(label, value) {
  const line = document.createElement('div');
  line.className = 'scan-detail-line';
  const l = document.createElement('span');
  l.textContent = label;
  const v = document.createElement('span');
  v.textContent = value;
  line.append(l, v);
  return line;
}

function _detailCol(title, m) {
  const col = document.createElement('div');
  col.className = 'scan-detail-col';
  const t = document.createElement('div');
  t.className = 'scan-detail-title';
  t.textContent = title;
  col.appendChild(t);
  const metrics = m || {};
  col.appendChild(_detailLine('Sharpe', _fmtNum(metrics.sharpe)));
  col.appendChild(_detailLine('Winrate', _fmtWinrate(metrics.winrate)));
  col.appendChild(_detailLine('Max DD', _fmtPct(metrics.max_dd)));
  col.appendChild(_detailLine('Trades',
    metrics.trades == null ? '—' : String(metrics.trades)));
  col.appendChild(_detailLine('Profit factor',
    metrics.profit_factor == null ? '—' : String(metrics.profit_factor)));
  col.appendChild(_detailLine('Return', _fmtPct(metrics.total_return)));
  return col;
}

function _renderResults(results) {
  const box = $('scan-results');
  if (!box) return;
  box.innerHTML = '';
  for (const r of results) {
    const train = r.train || {};
    const test = r.test || {};
    const cs = Number(r.combined_sharpe) || 0;
    const row = document.createElement('div');
    row.className = 'scan-result-row ' + _rowClass(cs);
    const trades = test.trades != null ? test.trades : r.total_trades;
    row.dataset.symbol = r.symbol || '';
    row.dataset.timeframe = r.timeframe || '';
    row.dataset.strategy = r.strategy || '';
    row.dataset.params = JSON.stringify(r.params || {});
    /* Счётчики сделок панели (BLOCK-33): по ним сверяем trades_full ответа
       endpoints /api/backtest/trades с TEST/TRAIN/FULL цифрами строки. */
    row.dataset.testTrades = test.trades != null ? String(test.trades) : '';
    row.dataset.trainTrades = train.trades != null ? String(train.trades) : '';
    row.dataset.fullTrades = r.total_trades != null ? String(r.total_trades) : '';
    const cells = [
      r.symbol || '—',
      r.timeframe || '—',
      r.strategy_label || r.strategy || '—',
      _fmtParams(r.params),
      _fmtNum(cs),
      _fmtNum(train.sharpe),
      _fmtNum(test.sharpe),
      trades == null ? '—' : String(trades),
      VERDICT_TEXT[r.verdict] || (r.verdict || '—'),
    ];
    cells.forEach((txt, idx) => {
      const s = document.createElement('span');
      s.textContent = txt;
      if (idx === 8 && r.verdict) {
        s.className = 'scan-verdict-cell scan-verdict-' + r.verdict;
        s.title = r.verdict_text || '';
      }
      row.appendChild(s);
    });
    /* Клик по строке — детали: train/test метрики прогона. */
    const details = document.createElement('div');
    details.className = 'scan-result-details';
    details.style.display = 'none';
    const grid = document.createElement('div');
    grid.className = 'scan-detail-grid';
    grid.append(_detailCol('Train (70%)', train), _detailCol('Test (30%)', test));
    details.appendChild(grid);
    /* Кнопки «Показать на графике»/«Скрыть сделки» — внутри раскрытой
       панели (BLOCK-29b). Клик обрабатывается делегированием на
       #scan-results, см. _showTradesForRow. */
    const actions = document.createElement('div');
    actions.className = 'scan-row-actions';
    /* Переключатель dataset «Показать на графике» (BLOCK-33): TEST (30%) —
       дефолт, TRAIN (70%), FULL. Радио общее на все строки (одно имя). */
    const radioWrap = document.createElement('div');
    radioWrap.className = 'scan-dataset-radio';
    for (const [val, label] of [['test', 'TEST (30%)'],
      ['train', 'TRAIN (70%)'], ['full', 'FULL']]) {
      const lab = document.createElement('label');
      const inp = document.createElement('input');
      inp.type = 'radio';
      inp.name = 'scan-dataset';
      inp.value = val;
      if (val === 'test') inp.checked = true;
      lab.append(inp, document.createTextNode(' ' + label));
      radioWrap.appendChild(lab);
    }
    actions.appendChild(radioWrap);
    const showBtn = document.createElement('button');
    showBtn.className = 'btn scan-show-trades-btn';
    showBtn.dataset.action = 'show-trades';
    showBtn.textContent = '📊 Показать на графике';
    const hideBtn = document.createElement('button');
    hideBtn.className = 'btn scan-hide-trades-btn';
    hideBtn.dataset.action = 'hide-trades';
    hideBtn.textContent = '✕ Скрыть сделки';
    actions.append(showBtn, hideBtn);
    details.appendChild(actions);
    row.addEventListener('click', () => {
      const open = details.style.display !== 'none';
      details.style.display = open ? 'none' : 'block';
      row.classList.toggle('expanded', !open);
    });
    box.append(row, details);
  }
}

/* ---------- Кнопки «Показать на графике»/«Скрыть сделки» (BLOCK-29b) ----------
   Делегирование на #scan-results: кнопки лежат в .scan-result-details (это
   сосед .scan-result-row, а не его потомок), поэтому строку берём через
   previousElementSibling, а не closest. */
(function () {
  const box = $('scan-results');
  if (!box) return;
  /* Радио dataset (BLOCK-33) встречается в каждой раскрытой панели с одним
     name, но вне формы — браузер может не связывать их в группу. Гарантируем
     единственный checked обработчиком change на контейнере. */
  box.addEventListener('change', function (e) {
    const t = e.target;
    if (!t || t.type !== 'radio' || t.name !== 'scan-dataset') return;
    document.querySelectorAll('input[name="scan-dataset"]').forEach((r) => {
      r.checked = (r === t);
    });
  });
  box.addEventListener('click', function (e) {
    const btn = e.target.closest('[data-action]');
    if (!btn) return;
    if (btn.dataset.action === 'hide-trades') {
      if (state.backtestRenderer) state.backtestRenderer.clear();
      return;
    }
    if (btn.dataset.action !== 'show-trades') return;
    const details = btn.closest('.scan-result-details');
    const row = btn.closest('.scan-result-row') ||
      (details ? details.previousElementSibling : null);
    if (!row) return;
    _showTradesForRow(row, btn);
  });
})();

async function _waitForData(timeoutMs = 120000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    if (state.progressiveLoaded === true) return true;
    await new Promise(r => setTimeout(r, 300));
  }
  return false;
}

/* BLOCK-36: зум применяется только ПОСЛЕ полного завершения фоновой
   прогрессивной догрузки (state.progressiveLoaded), плюс кадр на отрисовку.
   Иначе fitChartToData после progressive перебивает setVisibleRange и
   на экране остаются 2 блока из 97. */
async function _applyZoomWhenReady(trades, dataset) {
  const start = Date.now();
  while (!state.progressiveLoaded && Date.now() - start < 180000) {
    await new Promise(r => setTimeout(r, 300));
  }
  await new Promise(r => requestAnimationFrame(r));
  await new Promise(r => setTimeout(r, 100));

  // Авто-зум на ПОСЛЕДНИЕ 50 сделок: entry/exit_time — unix-секунды
  // (НЕ миллисекунды), плюс отступ 1 час. 200 блоков сжимали barSpacing
  // до ~0.5px (свечи исчезали); 50 — компромисс «и сделки видны, и свечи».
  // Сами блоки рисуются по ВСЕМ сделкам (появятся при zoom-out).
  const MAX_TRADES_IN_VIEW = 50;
  const visible = trades.length > MAX_TRADES_IN_VIEW
    ? trades.slice(-MAX_TRADES_IN_VIEW) : trades;
  const first = visible[0];
  const last = visible[visible.length - 1];
  if (!state.chart || !first || !last) return;
  const from = first.entry_time - 3600;
  const to = (last.exit_time || last.entry_time) + 3600;
  state.chart.timeScale().setVisibleRange({ from, to });
  console.log('[scan] FINAL zoom:',
    new Date(from * 1000).toISOString(), '→',
    new Date(to * 1000).toISOString(),
    `(${visible.length} of ${trades.length}, dataset=${dataset})`);
}

async function _showTradesForRow(row, btn) {
  const symbol = row.dataset.symbol;
  const timeframe = row.dataset.timeframe;
  const strategy = row.dataset.strategy;
  let params = {};
  try { params = JSON.parse(row.dataset.params || '{}'); } catch (e) { /* ignore */ }
  if (!symbol || !timeframe || !strategy) {
    console.warn('Недостаточно данных строки для показа сделок', row.dataset);
    return;
  }

  const originalText = btn.textContent;
  btn.disabled = true;
  btn.textContent = '⏳ Загрузка...';

  /* Dataset «Показать на графике» (BLOCK-33): radio в раскрытой панели,
     дефолт — test (последние 30% истории, цифра = TEST Trades сканера). */
  const datasetRadio = document.querySelector('input[name="scan-dataset"]:checked');
  const dataset = datasetRadio ? datasetRadio.value : 'test';
  /* BLOCK-37: TP/SL больше не опция — /api/backtest/trades считает ВСЕГДА с
     уровнями config.BACKTEST_TP_ATR/SL_ATR (2×ATR / 1×ATR), ровно как сканер.
     Поэтому trades на графике == Trades в панели (1:1), а чекбокс
     «TP/SL уровни» из панели убран. */
  console.log('[scan] request dataset=' + dataset);

  /* BLOCK-36: замораживаем pollLive на время fetch + render + zoom,
     чтобы фоновый полл не сбросил видимое окно посреди рендера сделок.
     Разморозка — через 2 сек после завершения (finally ниже). */
  state._suspendPoll = true;
  try {
    /* BLOCK-36: дёргаем график только при РЕАЛЬНОЙ смене symbol/tf.
       Повторный клик «Показать» на том же символе не должен запускать
       loadLive(true) → progressive → fitChartToData, который перебивает
       setVisibleRange и оставляет на экране 2 блока из 97. */
    const symSel = document.getElementById('symbol-select');
    const tfSel = document.getElementById('timeframe-select');
    const currentSym = symSel ? symSel.value : '';
    const currentTf = tfSel ? tfSel.value : '';

    const symChanged = currentSym !== symbol;
    const tfChanged = currentTf !== timeframe;

    if (symChanged || tfChanged) {
      if (symChanged) switchSymbol(symbol);
      if (tfChanged) setTimeframe(timeframe);
      console.log('[scan] chart switched:', { symChanged, tfChanged });
      // Смена символа/ТФ инициирует новую прогрессивную догрузку: ждём её
      // полного завершения (state.progressiveLoaded), иначе сделки нарисуем
      // на недогруженных свечах и lightweight-charts сожмёт свечи.
      const ready = await _waitForData();
      if (!ready) {
        alert('Данные не загрузились за 2 мин. Повторите.');
        return;
      }
    } else {
      console.log('[scan] chart unchanged, skipping reload');
    }

    const resp = await fetch('/api/backtest/trades', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      /* БЕЗ limit: визуализация идёт по ВСЕЙ истории сделок (backend берёт
         BACKTEST_MAX_CANDLES=20000). Хардкод limit:1000 резал сделки —
         видели 32 блока вместо 500+. */
      body: JSON.stringify({
        symbol, timeframe, strategy, params, dataset,
      }),
    });
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    const data = await resp.json();

    const trades = data.trades_full || [];
    const metrics = data.metrics || {};
    const totalTrades = metrics.total_trades != null ? metrics.total_trades : trades.length;
    console.log('trades received:', trades.length,
      'trades_used:', data.candles_used,
      'range:', data.bars_from, '→', data.bars_to,
      'dataset:', data.dataset || dataset);
    if (trades.length < totalTrades) {
      console.warn(`WARN: total_trades=${totalTrades} but trades_full.length=${trades.length}`);
    }
    /* Сверка с панелью (BLOCK-37): панель и этот эндпоинт считают ОДИН прогон
       (TP/SL всегда включены), поэтому trades_full обязан совпадать с колонкой
       TEST/TRAIN/FULL Trades. Любое расхождение — реальный баг. */
    const panelCount = dataset === 'test' ? row.dataset.testTrades
      : dataset === 'train' ? row.dataset.trainTrades : row.dataset.fullTrades;
    if (panelCount) {
      if (String(trades.length) === String(panelCount)) {
        console.log(`[scan] OK: trades_full=${trades.length} == панель ${dataset.toUpperCase()} (${panelCount})`);
      } else {
        console.warn(`[scan] trades_full=${trades.length} != панель ${dataset.toUpperCase()}=${panelCount}`);
      }
    }

    if (trades.length === 0) {
      console.warn('Нет сделок для', symbol, timeframe, strategy);
      btn.textContent = '⚠ Нет сделок';
      setTimeout(() => { btn.textContent = originalText; btn.disabled = false; }, 2000);
      return;
    }

    if (!state.backtestRenderer) {
      throw new Error('backtestRenderer не инициализирован');
    }
    state.backtestRenderer.render(trades);

    // BLOCK-36: zoom применяем только ПОСЛЕ завершения фоновой прогрессивной
    // догрузки — иначе fitChartToData перебивает setVisibleRange (см. выше).
    await _applyZoomWhenReady(trades, dataset);

    btn.textContent = '✓ ' + trades.length + ' сделок';
    setTimeout(() => { btn.textContent = originalText; btn.disabled = false; }, 2000);
  } catch (e) {
    console.error('show trades failed:', e);
    btn.textContent = '⚠ Ошибка: ' + e.message;
    setTimeout(() => { btn.textContent = originalText; btn.disabled = false; }, 3000);
  } finally {
    // BLOCK-36: pollLive размораживаем через 2 сек после рендера,
    // чтобы он не перебил setVisibleRange/рисование блоков.
    setTimeout(() => { state._suspendPoll = false; }, 2000);
  }
}

/* ---------------------------------------------------------------- экспорт */
export function exportCsv() {
  if (!local.runId) { _showError('Сначала запустите скан'); return; }
  const a = document.createElement('a');
  a.href = `/api/scan/${local.runId}/export.csv`;
  a.download = `scan_${local.runId}.csv`;
  document.body.appendChild(a);
  a.click();
  a.remove();
}

/* ------------------------------------------------- сводка и отчёт (ЧАСТЬ C) */
async function _loadSummary() {
  if (!local.runId) return;
  try {
    const resp = await fetch(`/api/scan/${local.runId}/summary`);
    if (!resp.ok) return;
    local.summary = await resp.json();
    renderSummary(local.summary);
  } catch { /* сводка не критична — таблица уже отрисована */ }
}

export function renderSummary(summary) {
  const wrap = $('scan-summary-wrap');
  const box = $('scan-summary');
  if (!wrap || !box || !summary) return;
  box.innerHTML = '';
  const bv = summary.by_verdict || {};
  const rows = [
    ['📊 Обработано', `${summary.kept} из ${summary.total_combinations}`, null],
    ['✅ Стабильных', String(bv.stable || 0), 'scan-verdict-stable'],
    ['⚠️ Переоптимизация', String(bv.overfit || 0), 'scan-verdict-overfit'],
    ['🎲 Мало сделок', String(bv.noise || 0), 'scan-verdict-noise'],
    ['❌ Убыточных', String(bv.losing || 0), 'scan-verdict-losing'],
  ];
  if (bv.weak) rows.push(['🟡 Слабых', String(bv.weak), 'scan-verdict-weak']);
  for (const [label, value, cls] of rows) {
    const row = document.createElement('div');
    row.className = 'scan-summary-row';
    const l = document.createElement('span');
    l.textContent = label;
    const v = document.createElement('span');
    v.textContent = value;
    if (cls) v.className = cls;
    row.append(l, v);
    box.appendChild(row);
  }
  const recs = summary.recommendations || [];
  if (recs.length) {
    const ul = document.createElement('ul');
    ul.className = 'scan-summary-recs';
    for (const rec of recs) {
      const li = document.createElement('li');
      li.textContent = rec;
      ul.appendChild(li);
    }
    box.appendChild(ul);
  }
  wrap.style.display = 'block';
}

/* Сводка -> буфер обмена как человекочитаемый markdown. */
export function copySummaryReport() {
  const s = local.summary;
  if (!s) { _showError('Нет отчёта — сначала запустите скан'); return; }
  const bv = s.by_verdict || {};
  const lines = ['## Сканер стратегий — отчёт', '',
    `Обработано: ${s.kept} из ${s.total_combinations}`];
  if (bv.stable) lines.push(`- ✅ Стабильных: ${bv.stable}`);
  if (bv.overfit) lines.push(`- ⚠️ Переоптимизация: ${bv.overfit}`);
  if (bv.noise) lines.push(`- 🎲 Мало сделок: ${bv.noise}`);
  if (bv.losing) lines.push(`- ❌ Убыточных: ${bv.losing}`);
  if (bv.weak) lines.push(`- 🟡 Слабых: ${bv.weak}`);
  const recs = s.recommendations || [];
  if (recs.length) {
    lines.push('', '### Рекомендации');
    for (const rec of recs) lines.push(`- ${rec}`);
  }
  const md = lines.join('\n');
  const btn = $('scan-copy-btn');
  const finish = (ok) => {
    if (btn) btn.textContent = ok ? '✅ Скопировано' : '📋 Отчёт для копирования';
    setTimeout(() => {
      if (btn) btn.textContent = '📋 Отчёт для копирования';
    }, 1500);
  };
  const fallback = () => {
    const ta = document.createElement('textarea');
    ta.value = md;
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    let ok = false;
    try { ok = document.execCommand('copy'); } catch { ok = false; }
    ta.remove();
    finish(ok);
  };
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(md).then(() => finish(true)).catch(fallback);
  } else {
    fallback();
  }
}
