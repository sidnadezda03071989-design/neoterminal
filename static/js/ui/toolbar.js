import { state } from '../state.js';

const $ = (id) => document.getElementById(id);
let _changeHandler = null;

export function setSymbolChangeHandler(fn) { _changeHandler = fn; }

export function confirmedDanger(msg) { return window.confirm(msg); }

export function setTool(tool) {
  if (!tool) return;
  document.querySelectorAll('.tool-btn').forEach((b) => {
    b.classList.toggle('active', b.dataset.tool === tool);
  });
  const mBtn = $('measure-btn');
  if (mBtn) mBtn.classList.toggle('active', tool === 'measure');
  if (state.dm) state.dm.setTool(tool);
}

export function toggleMagnet() {
  if (!state.dm) return;
  state.dm.setMagnet(!state.dm.magnet);
  const btn = $('magnet-btn');
  if (btn) btn.classList.toggle('active', state.dm.magnet);
}

export function syncToolbarUI() {
  const mode = state.mode;
  const toggle = $('mode-toggle');
  if (toggle) {
    toggle.querySelectorAll('button').forEach((b) => {
      b.classList.toggle('active', b.dataset.mode === mode);
    });
  }
  const rb = $('replay-bar');
  if (rb) rb.style.display = mode === 'replay' ? 'flex' : 'none';
  const meaBtn = $('measure-btn');
  if (meaBtn) meaBtn.classList.toggle('active', !!(state.dm && state.dm.tool === 'measure'));
  const magBtn = $('magnet-btn');
  if (magBtn) magBtn.classList.toggle('active', !!(state.dm && state.dm.magnet));
}

export function switchMode(mode) {
  state.mode = mode;
  syncToolbarUI();
  if (_changeHandler) _changeHandler();
}

export function switchSymbol(sym) {
  const sel = $('symbol-select');
  if (!sel) return;
  /* BLOCK-36: тот же символ — не дёргаем _changeHandler, иначе loadLive(true)
     запустит progressive и fitChartToData перебьёт setVisibleRange сделок. */
  if (sel.value === sym) {
    console.log('[toolbar] switchSymbol: same value, skip');
    return;
  }
  sel.value = sym;
  if (_changeHandler) _changeHandler();
}

export function setTimeframe(tf) {
  const sel = $('timeframe-select');
  if (!sel) return;
  /* BLOCK-36: тот же ТФ — не дёргаем _changeHandler (см. switchSymbol). */
  if (sel.value === tf) {
    console.log('[toolbar] setTimeframe: same value, skip');
    return;
  }
  sel.value = tf;
  if (_changeHandler) _changeHandler();
}
