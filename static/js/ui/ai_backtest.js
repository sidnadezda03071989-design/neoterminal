// Панель «🤖 AI Backtest»: вероятностные уровни «Псевдо-Харон».
//
// Уровни считаются по ТЕКУЩЕМУ срезу данных и работают и в реплее, и в live:
//   • live   — срез = последняя свеча (upto_sec не передаём);
//   • replay — срез = время барьера реплея (state.replay.time → upto_sec),
//              при сдвиге барьера уровни пересчитываются автоматически.
// Данные и контекст — ровно те же, что у Харона (build_multi_tf_context).
//
// Метрик стратегии (сделки/винрейт/sharpe/…) в панели НЕТ: это помощник
// вероятностей, а не прогон стратегии. Результат — зоны на графике
// (AIProbZonesRenderer) + краткий список уровней в панели.

import { state } from '../state.js';
import { AIProbZonesRenderer } from '../drawings/ai_prob_zones.js';

const $ = (id) => document.getElementById(id);

const SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'EURUSD', 'GBPUSD',
  'USDCHF', 'USDJPY', 'USDCAD', 'EURJPY', 'XAUUSD'];
const TIMEFRAMES = ['1m', '5m', '15m', '1H', '4H', '1D'];

const local = {
  runId: null, running: false, pollTimer: null,
  // Время среза последнего расчёта (якорь зон). Пересчёт — только по явному
  // нажатию «Запустить», без авто-реакции на движение барьера реплея.
  lastUpto: null,
};

const POLL_MS = 2000;

// Настройка «TP/SL по уровням»: живёт в localStorage, чисто клиентская.
const TP_SL_KEY = 'aibt_tpsl_enabled';

/* ------------------------------------------------------- панель open/close */
export function setAiProbZonesRenderer(chart, series, container) {
  // Якорь зон — полный ряд свечей (state.candles); в реплее вызывающий код
  // передаёт явный якорь-барьер (renderer.render(result, anchor)).
  const renderer = new AIProbZonesRenderer(
    chart, series, container, () => state.candles,
  );
  renderer.enableTpsl = _tpslEnabled();
  state.aiProbRenderer = renderer;
  return renderer;
}

export function openAiBacktest() {
  const panel = $('ai-backtest-panel');
  if (!panel) return;
  panel.style.display = 'block';
  _initFormDefaults();
  _syncModeHint();
  _initTpslToggle();
}

export function closeAiBacktest() {
  const panel = $('ai-backtest-panel');
  if (panel) panel.style.display = 'none';
  _stopPoll();
}

/* Тумблер «TP/SL по уровням»: состояние из localStorage, смена перерисовывает
   текущие зоны без повторного запроса к LLM. */
function _tpslEnabled() {
  const stored = localStorage.getItem(TP_SL_KEY);
  return stored != null ? stored === '1' : true;
}

function _initTpslToggle() {
  const t = $('aibt-tpsl-toggle');
  if (!t) return;
  if (!t.dataset.bound) {
    t.dataset.bound = '1';
    t.addEventListener('change', () => {
      localStorage.setItem(TP_SL_KEY, t.checked ? '1' : '0');
      if (state.aiProbRenderer) state.aiProbRenderer.setTpsl(t.checked);
    });
  }
  if (t.checked !== _tpslEnabled()) t.checked = _tpslEnabled();
}

function _initFormDefaults() {
  const fill = (id, val) => {
    const el = $(id);
    if (el && Array.from(el.options || []).some((o) => o.value === val)) {
      el.value = val;
    }
  };
  fill('aibt-symbol', state.symbol);
  fill('aibt-timeframe', state.timeframe);
}

/* Подпись режима: в реплее явно пишем, до какого времени считается срез. */
function _syncModeHint() {
  const el = $('aibt-mode-hint');
  if (!el) return;
  if (state.mode === 'replay') {
    const t = state.replay && state.replay.time;
    const txt = t
      ? new Date(t * 1000).toISOString().slice(0, 16).replace('T', ' ')
      : '—';
    el.textContent = `Режим: реплей · срез до ${txt}`;
  } else {
    el.textContent = 'Режим: реальное время · срез = последняя свеча';
  }
}

/* time (сек) барьера реплея или null в live. */
function _replayUpto() {
  if (state.mode !== 'replay') return null;
  const t = state.replay && state.replay.time;
  const n = Number(t);
  return Number.isFinite(n) && n > 0 ? Math.floor(n) : null;
}

/* Якорь зон для текущего режима: срез = последняя ВИДИМАЯ свеча.
   В live это последний бар state.candles; в реплее — бар на барьере. */
function _anchorForUpto(upto) {
  const arr = state.mode === 'replay' ? state.activeCandles : state.candles;
  if (!arr || !arr.length) return null;
  if (upto == null) {
    const c = arr[arr.length - 1];
    return { time: Number(c.time), price: Number(c.close) };
  }
  for (let i = arr.length - 1; i >= 0; i--) {
    if (Number(arr[i].time) <= upto) {
      return { time: Number(arr[i].time), price: Number(arr[i].close) };
    }
  }
  return { time: Number(arr[0].time), price: Number(arr[0].close) };
}

function _showError(msg) {
  const el = $('aibt-error');
  if (!el) return;
  if (msg) {
    el.textContent = '⚠ ' + msg;
    el.style.display = 'block';
  } else {
    el.textContent = '';
    el.style.display = 'none';
  }
}

/* ---------------------------------------------------------------- прогресс */
function _setProgress(done, total) {
  const wrap = $('aibt-progress');
  if (wrap) wrap.style.display = 'block';
  const bar = $('aibt-progress-bar');
  if (bar) bar.style.width = '100%';
  const text = $('aibt-progress-text');
  if (text) text.textContent = 'Считаю уровни…';
}

function _hideProgress() {
  const wrap = $('aibt-progress');
  if (wrap) wrap.style.display = 'none';
}

/* SSE 'ai_backtest_progress': app_init.js делает window.__aiBacktestProgress =
   updateAiBacktestProgress. Формат data: {run_id, done, total, current_bar,
   levels, finished?, error?}. */
export function updateAiBacktestProgress(data) {
  if (!data || !data.run_id) return;
  if (local.runId && data.run_id !== local.runId) return;
  if (!local.runId) local.runId = data.run_id;
  _setProgress(Number(data.done) || 0, Number(data.total) || 0);
  if (data.error) _showError(data.error);
  if (data.finished) _finish(data.run_id);
}

/* ---------------------------------------------------------------- результаты */
async function _finish(runId) {
  _stopPoll();
  try {
    const resp = await fetch(`/api/ai-backtest/${runId}`);
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || 'HTTP ' + resp.status);
    local.running = false;
    _setRunUi(false);
    _hideProgress();
    _applyResult(data);
  } catch (e) {
    _showError(e.message);
    local.running = false;
    _setRunUi(false);
    _hideProgress();
  }
}

/* Общий путь применения результата (и для sync, и для поллинга). */
function _applyResult(data) {
  const levels = (data && Array.isArray(data.levels)) ? data.levels : [];
  // Рисуем ТОЛЬКО уровни: снять блоки сделок обычного бэктеста, чтобы на
  // графике не оставалось «сделок» рядом с зонами вероятностей.
  if (state.backtestRenderer && state.backtestRenderer.clear) {
    state.backtestRenderer.clear();
  }
  const upto = data && data.upto_sec != null ? Number(data.upto_sec) : null;
  local.lastUpto = upto;
  // Зоны раньше списка: сводка TP/SL в панели читает renderer.tpsl —
  // тот же источник, что и линии на графике (единый набор).
  if (state.aiProbRenderer) {
    state.aiProbRenderer.enableTpsl = _tpslEnabled();
    state.aiProbRenderer.render(levels, _anchorForUpto(upto));
  }
  _renderLevels(levels, data && data.error, data && data.notice);
  // Причину отказа показываем явно (квота/rate limit/нет JSON) — «уровни не
  // получены» без деталей бесполезно.
  if (data && data.error) _showError(data.error);
  else _showError('');
}

/* Краткий список уровней в панели: +{diff} {prob}% (как на пилюлях графика).
   error — причина, если уровней нет (показываем её вместо заглушки);
   notice — нейтральное пояснение (напр. «рынок плоский») без красного ⚠.
   Источник — зафиксированные на графике уровни (renderer.levels: ближние
   уже сдвинуты на 0.5·ATR), чтобы панель и график не расходились. */
function _renderLevels(levels, error, notice) {
  const box = $('aibt-levels');
  if (!box) return;
  box.innerHTML = '';
  const disp = (state.aiProbRenderer && state.aiProbRenderer.levels.length)
    ? state.aiProbRenderer.levels : levels;
  if (!disp.length) {
    const empty = document.createElement('div');
    empty.className = 'aibt-levels-empty';
    if (error) empty.textContent = 'Уровни не получены — см. причину выше';
    else if (notice) empty.textContent = notice;
    else empty.textContent = 'Уровни не получены';
    box.appendChild(empty);
    return;
  }
  const sorted = [...disp]
    .sort((a, b) => Number(b.price) - Number(a.price));
  for (const lv of sorted) {
    const isUp = lv.side === 'UP';
    const diff = Number(lv.diff);
    const price = Number(lv.price);
    const prob = Number(lv.probability);
    if (!Number.isFinite(prob)) continue;
    const row = document.createElement('div');
    row.className = 'aibt-level-row ' + (isUp ? 'up' : 'down');
    const left = document.createElement('span');
    left.className = 'aibt-level-price';
    left.textContent = Number.isFinite(diff)
      ? (diff >= 0 ? '+' : '−') + _fmtPrice(Math.abs(diff))
      : _fmtPrice(price);
    const right = document.createElement('span');
    right.className = 'aibt-level-prob';
    right.textContent = Number.isFinite(prob)
      ? (prob * 100).toFixed(prob * 100 >= 100 ||
        prob * 100 === Math.round(prob * 100) ? 0 : 1) + '%'
      : '—';
    row.append(left, right);
    box.appendChild(row);
  }
  _renderTpslSummary(box);
}

/* Сводка потенциальных TP/SL в панели: читает готовый renderer.tpsl
   (тот же выбор направления+ATR-отбор, что рисуется на графике), поэтому
   цены панели и линий всегда совпадают. Показывается при включённой
   настройке. Направление: цель на стороне с большей вероятностью. */
function _renderTpslSummary(box) {
  if (!_tpslEnabled()) return;
  const r = state.aiProbRenderer;
  const tpsl = r && r.tpsl;
  if (!tpsl) return;
  const parts = [];
  if (tpsl.tp && Number.isFinite(Number(tpsl.tp.price))) {
    parts.push('TP ' + _fmtPrice(tpsl.tp.price));
  }
  if (tpsl.sl && Number.isFinite(Number(tpsl.sl.price))) {
    parts.push('SL ' + _fmtPrice(tpsl.sl.price));
  }
  if (!parts.length) return;
  const dir = tpsl.side === 'short' ? 'ШОРТ' : 'ЛОНГ';
  const sum = document.createElement('div');
  sum.className = 'aibt-tpsl-summary';
  sum.textContent = dir + ' · Потенциальные: ' + parts.join('  ·  ');
  box.appendChild(sum);
}

function _fmtPrice(v) {
  if (v == null || !Number.isFinite(Number(v))) return '—';
  const av = Math.abs(Number(v));
  const dg = av >= 1000 ? 0 : av >= 1 ? 1 : av >= 0.01 ? 3 : 6;
  return Number(v).toFixed(dg);
}

function _stopPoll() {
  if (local.pollTimer) { clearInterval(local.pollTimer); local.pollTimer = null; }
}

function _startPoll() {
  _stopPoll();
  local.pollTimer = setInterval(async () => {
    if (!local.running || !local.runId) { _stopPoll(); return; }
    try {
      const resp = await fetch(`/api/ai-backtest/${local.runId}`);
      if (!resp.ok) return;
      const data = await resp.json();
      if (data.status === 'finished') _finish(local.runId);
      else _setProgress(Number(data.done) || 0, Number(data.total) || 0);
    } catch { /* сеть могла моргнуть — повторим на следующем тике */ }
  }, POLL_MS);
}

function _setRunUi(running) {
  const runBtn = $('aibt-run-btn');
  if (!runBtn) return;
  runBtn.disabled = running;
  runBtn.textContent = running ? '⏳ Выполняется…' : '▶️ Запустить';
}

/* ------------------------------------------------------------------- запуск */
function _currentParams() {
  const symbol = $('aibt-symbol') ? $('aibt-symbol').value : state.symbol;
  const timeframe = $('aibt-timeframe') ? $('aibt-timeframe').value : state.timeframe;
  const model = $('aibt-model') ? $('aibt-model').value.trim() : '';
  return { symbol, timeframe, model };
}

/* Синхронный расчёт: ответ приходит сразу (без SSE/поллинга). */
async function _fetchLevelsSync(upto) {
  const { symbol, timeframe, model } = _currentParams();
  const resp = await fetch('/api/ai-backtest', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      symbol, timeframe, mode: state.mode, upto_sec: upto, sync: true,
      ...(model ? { model } : {}),
    }),
  });
  const data = await resp.json();
  if (!resp.ok) throw new Error(data.error || 'HTTP ' + resp.status);
  return data;
}

export async function runAiBacktest() {
  if (local.running) return;
  const { symbol, timeframe, model } = _currentParams();
  if (!symbol) { _showError('Выберите символ'); return; }
  if (!timeframe) { _showError('Выберите таймфрейм'); return; }

  _showError('');
  _syncModeHint();
  const runBtn = $('aibt-run-btn');
  if (runBtn) { runBtn.disabled = true; runBtn.textContent = '⏳ Выполняется…'; }
  _setProgress(0, 0);

  local.running = true;
  local.runId = null;
  _stopPoll();

  const upto = _replayUpto();
  try {
    const data = await _fetchLevelsSync(upto);
    local.running = false;
    _setRunUi(false);
    _hideProgress();
    _applyResult(data);
  } catch (e) {
    _showError(e.message);
    local.running = false;
    _setRunUi(false);
    _hideProgress();
  }
}

/* Смена symbol/tf/режима: зоны устарели — просто скрыть. */
export function resetAiProbZones() {
  local.lastUpto = null;
  local.runId = null;
  _hideProgress();
  _renderLevels([]);
  if (state.aiProbRenderer) state.aiProbRenderer.clear();
  _syncModeHint();
}

/* ------------------------------------------------------------------- helpers */
export function initAiBacktestSymbols() {
  const sel = $('aibt-symbol');
  if (!sel || sel.children.length) return;
  for (const s of SYMBOLS) {
    const opt = document.createElement('option');
    opt.value = s;
    opt.textContent = s;
    sel.appendChild(opt);
  }
  const tf = $('aibt-timeframe');
  if (tf && !tf.children.length) {
    for (const t of TIMEFRAMES) {
      const opt = document.createElement('option');
      opt.value = t;
      opt.textContent = t;
      tf.appendChild(opt);
    }
  }
  _initFormDefaults();
}

/* Спрятать зоны вероятностей (кнопка «✕ Скрыть зоны»). */
export function hideAiProbZones() {
  local.lastUpto = null;
  if (state.aiProbRenderer) state.aiProbRenderer.clear();
  _renderLevels([]);
}
