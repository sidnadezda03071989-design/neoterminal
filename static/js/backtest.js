import { state } from './state.js';

const $ = (id) => document.getElementById(id);

/* 'YYYY-MM-DD' из input[type=date] → unix-секунды (UTC полночь), null если пусто.
   endOfDay=true → 23:59:59, чтобы включить бары последней даты диапазона. */
function _dateToSec(value, endOfDay) {
  if (!value) return null;
  const t = new Date(value + 'T00:00:00Z').getTime();
  if (!Number.isFinite(t)) return null;
  return Math.floor(t / 1000) + (endOfDay ? 86399 : 0);
}

/* ------------------------------------------------------- панель open/close */
export function openBacktest() {
  const panel = $('backtest-panel');
  if (!panel) return;
  panel.style.display = 'block';
  _initFormDefaults();
}

export function closeBacktest() {
  const panel = $('backtest-panel');
  if (panel) panel.style.display = 'none';
}

export function toggleBacktest() {
  const panel = $('backtest-panel');
  if (!panel) return;
  if (panel.style.display === 'none' || !panel.style.display) openBacktest();
  else closeBacktest();
}

function _initFormDefaults() {
  const fill = (id, val) => {
    const el = $(id);
    if (el && Array.from(el.options || []).some((o) => o.value === val)) {
      el.value = val;
    }
  };
  fill('bt-symbol', state.symbol);
  fill('bt-timeframe', state.timeframe);
  const from = $('bt-from');
  const to = $('bt-to');
  if (from && !from.value) {
    from.value = new Date(Date.now() - 7 * 864e5).toISOString().slice(0, 10);
  }
  if (to && !to.value) to.value = new Date().toISOString().slice(0, 10);
}

function _showError(msg) {
  const el = $('bt-error');
  if (!el) return;
  if (msg) {
    el.textContent = '⚠ ' + msg;
    el.style.display = 'block';
  } else {
    el.textContent = '';
    el.style.display = 'none';
  }
}

/* ---------------------------------------------------------------- форматтеры */
function _fmtDateTime(sec) {
  const d = new Date(sec * 1000);
  const p = (n) => String(n).padStart(2, '0');
  return p(d.getUTCMonth() + 1) + '-' + p(d.getUTCDate()) +
    ' ' + p(d.getUTCHours()) + ':' + p(d.getUTCMinutes());
}

function _fmtPrice(v) {
  if (v == null || !isFinite(v)) return '—';
  const av = Math.abs(v);
  const dg = av >= 1000 ? 1 : av >= 1 ? 2 : av >= 0.01 ? 4 : 6;
  return Number(v).toFixed(dg);
}

function _fmtNum(v) {
  return Number(v).toLocaleString('en-US', { maximumFractionDigits: 2 });
}

/* 'YYYY-MM-DD' из unix-секунд (UTC), '—' если не число. */
function _fmtDateShort(sec) {
  if (sec == null || !Number.isFinite(+sec)) return '—';
  const d = new Date(+sec * 1000);
  const p = (n) => String(n).padStart(2, '0');
  return d.getUTCFullYear() + '-' + p(d.getUTCMonth() + 1) + '-' + p(d.getUTCDate());
}

/* Статус после прогона: сколько свечей реально прогнано, диапазон, сделок. */
function _showStatusSummary(data) {
  const status = $('bt-status');
  if (!status) return;
  const used = data.candles_used;
  const from = _fmtDateShort(data.bars_from);
  const to = _fmtDateShort(data.bars_to);
  const n = data.total_trades != null ? data.total_trades : (data.trades || []).length;
  if (used != null && from !== '—' && to !== '—') {
    status.textContent = `Прогнано ${used} свечей (${from} → ${to}), сделок: ${n}`;
  } else {
    status.textContent = `Сделок: ${n}`;
  }
  status.style.display = 'block';
}

/* ------------------------------------------------------------------- метрики */
function _addMetric(box, label, value, cls) {
  const cell = document.createElement('div');
  cell.className = 'bt-metric';
  const lbl = document.createElement('div');
  lbl.className = 'bt-metric-label';
  lbl.textContent = label;
  const val = document.createElement('div');
  val.className = 'bt-metric-value' + (cls ? ' ' + cls : '');
  val.textContent = value;
  cell.append(lbl, val);
  box.appendChild(cell);
}

function _renderMetrics(data, initialCash) {
  const box = $('bt-metrics');
  if (!box) return;
  box.innerHTML = '';
  const ret = data.total_return || 0;
  const curve = data.equity_curve || [];
  const finalEq = curve.length ? curve[curve.length - 1].equity : initialCash;
  _addMetric(box, 'Return',
    (ret >= 0 ? '+' : '') + (ret * 100).toFixed(2) + '%',
    ret >= 0 ? 'up' : 'down');
  _addMetric(box, 'Sharpe',
    data.sharpe_ratio == null ? '—' : Number(data.sharpe_ratio).toFixed(2),
    data.sharpe_ratio > 0 ? 'up' : 'down');
  _addMetric(box, 'Max DD',
    '-' + ((data.max_drawdown || 0) * 100).toFixed(2) + '%', 'down');
  _addMetric(box, 'Winrate',
    ((data.win_rate || 0) * 100).toFixed(1) + '%',
    (data.win_rate || 0) >= 0.5 ? 'up' : '');
  _addMetric(box, 'Сделок', String(data.total_trades || 0), '');
  _addMetric(box, 'Эквити', _fmtNum(finalEq),
    finalEq >= initialCash ? 'up' : 'down');
}

/* ------------------------------------------------------------ кривая эквити */
function _drawEquity(curve, initialCash) {
  const cv = $('bt-equity-canvas');
  if (!cv) return;
  const w = Math.max(80, cv.clientWidth || 360);
  const h = 120;
  const dpr = window.devicePixelRatio || 1;
  cv.width = Math.round(w * dpr);
  cv.height = Math.round(h * dpr);
  const ctx = cv.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  if (!curve || curve.length < 2) {
    ctx.fillStyle = '#787b86';
    ctx.font = '11px "Segoe UI", Tahoma, sans-serif';
    ctx.fillText('Нет данных', 10, h / 2);
    return;
  }
  let min = Infinity, max = -Infinity;
  for (const p of curve) {
    if (p.equity < min) min = p.equity;
    if (p.equity > max) max = p.equity;
  }
  min = Math.min(min, initialCash);
  max = Math.max(max, initialCash);
  if (max - min < 1e-9) { max += 1; min -= 1; }
  const pad = 8;
  const xAt = (i) => pad + (i / (curve.length - 1)) * (w - 2 * pad);
  const yAt = (v) => pad + (1 - (v - min) / (max - min)) * (h - 2 * pad);

  // пунктирная базовая линия — стартовый капитал
  ctx.strokeStyle = '#787b86';
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(pad, yAt(initialCash));
  ctx.lineTo(w - pad, yAt(initialCash));
  ctx.stroke();
  ctx.setLineDash([]);

  const up = curve[curve.length - 1].equity >= initialCash;
  const color = up ? '#26a69a' : '#ef5350';

  // линия эквити
  ctx.beginPath();
  curve.forEach((p, i) => {
    if (i) ctx.lineTo(xAt(i), yAt(p.equity));
    else ctx.moveTo(xAt(i), yAt(p.equity));
  });
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // полупрозрачная заливка под линией
  ctx.lineTo(xAt(curve.length - 1), h - pad);
  ctx.lineTo(xAt(0), h - pad);
  ctx.closePath();
  ctx.fillStyle = up ? 'rgba(38,166,154,0.12)' : 'rgba(239,83,80,0.12)';
  ctx.fill();

  // подписи max/min
  ctx.fillStyle = '#787b86';
  ctx.font = '10px "Segoe UI", Tahoma, sans-serif';
  ctx.fillText(_fmtNum(max), pad + 2, pad + 10);
  ctx.fillText(_fmtNum(min), pad + 2, h - pad - 2);
}

/* ------------------------------------------------------------------- сделки */
function _renderTrades(trades) {
  const box = $('bt-trades');
  if (!box) return;
  box.innerHTML = '';
  const head = document.createElement('div');
  head.className = 'bt-trades-head';
  ['Вход', 'Выход', 'Вх → Вых', 'PnL'].forEach((t) => {
    const s = document.createElement('span');
    s.textContent = t;
    head.appendChild(s);
  });
  box.appendChild(head);
  if (!trades.length) {
    const empty = document.createElement('div');
    empty.className = 'bt-trades-empty';
    empty.textContent = 'Сделок нет';
    box.appendChild(empty);
    return;
  }
  for (const t of trades) {
    const row = document.createElement('div');
    row.className = 'bt-trade-row';
    const pnl = t.pnl == null ? null : Number(t.pnl);
    const cells = [
      _fmtDateTime(t.entry_time),
      _fmtDateTime(t.exit_time),
      _fmtPrice(t.entry_price) + ' → ' + _fmtPrice(t.exit_price),
      pnl == null ? '—' : (pnl >= 0 ? '+' : '') + pnl.toFixed(2),
    ];
    cells.forEach((txt, i) => {
      const s = document.createElement('span');
      if (i === 3) s.className = 'bt-pnl ' + (pnl >= 0 ? 'up' : 'down');
      s.textContent = txt;
      row.appendChild(s);
    });
    box.appendChild(row);
  }
}

/* ------------------------------------------------------------------- запуск */
export async function runBacktest() {
  const strategy = $('bt-strategy') ? $('bt-strategy').value : 'sma_cross';
  const symbol = $('bt-symbol') ? $('bt-symbol').value : state.symbol;
  const timeframe = $('bt-timeframe') ? $('bt-timeframe').value : state.timeframe;
  const fromSec = _dateToSec($('bt-from') ? $('bt-from').value : '');
  const toSec = _dateToSec($('bt-to') ? $('bt-to').value : '', true);
  const initialCash = Number($('bt-cash') ? $('bt-cash').value : 0) || 10000;
  const limit = Number($('bt-limit') ? $('bt-limit').value : 0) || 5000;

  // валидация до fetch — ошибки только в красную плашку, никаких alert()
  if (!symbol) { _showError('Выберите символ'); return; }
  if (!timeframe) { _showError('Выберите таймфрейм'); return; }
  if (!fromSec || !toSec) { _showError('Укажите даты «С» и «По»'); return; }
  if (fromSec >= toSec) { _showError('Дата «С» должна быть раньше даты «По»'); return; }

  const runBtn = $('bt-run-btn');
  const status = $('bt-status');
  const results = $('bt-results');
  _showError('');
  if (status) status.style.display = 'block';
  if (runBtn) { runBtn.disabled = true; runBtn.textContent = 'Выполняется…'; }
  if (results) results.style.display = 'none';

  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 90000); // защита от вечного ожидания
  let ok = false;
  try {
    const resp = await fetch('/api/backtest', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        symbol: symbol,
        timeframe: timeframe,
        strategy: strategy,
        params: {},
        from: fromSec,
        to: toSec,
        initial_cash: initialCash,
        limit: limit,
      }),
      signal: ctrl.signal,
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || 'HTTP ' + resp.status);

    _renderMetrics(data, initialCash);
    if (results) results.style.display = 'block';
    _drawEquity(data.equity_curve || [], initialCash);
    _renderTrades(data.trades || []);
    // Авто-зум на ПОСЛЕДНИЕ 200 сделок (как в сканере): диапазон всех сделок
    // сжимал свечи. Блоки на графике рисует BacktestTradesRenderer по всем
    // сделкам — они появятся при zoom-out. Отступ 1 час по сторонам.
    const btTrades = data.trades || [];
    const VISIBLE_LIMIT = 200;
    const btVisible = btTrades.length > VISIBLE_LIMIT
      ? btTrades.slice(-VISIBLE_LIMIT) : btTrades;
    if (btVisible.length && state.chart) {
      const btFirst = btVisible[0];
      const btLast = btVisible[btVisible.length - 1];
      const btFrom = btFirst.entry_time - 3600;
      const btTo = (btLast.exit_time || btLast.entry_time) + 3600;
      state.chart.timeScale().setVisibleRange({ from: btFrom, to: btTo });
      console.log(`[backtest] zoomed to last ${btVisible.length} of ${btTrades.length} trades`);
    }
    _showStatusSummary(data);
    ok = true;
  } catch (e) {
    if (e.name === 'AbortError') {
      _showError('Превышено время ожидания (90с). Уменьшите период ' +
        'или выберите больший timeframe (15m/1H).');
    } else {
      _showError(e.message);
    }
    if (results) results.style.display = 'none';
  } finally {
    clearTimeout(timer);
    if (!ok && status) status.style.display = 'none';
    if (runBtn) { runBtn.disabled = false; runBtn.textContent = 'Запустить'; }
  }
}
