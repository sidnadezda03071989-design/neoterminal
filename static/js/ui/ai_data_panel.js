// Вкладка «🧠 Данные для ИИ»: сырые данные (Market Snapshot) + системный промпт.
//
// Слева — чистый JSON с цифрами для нейросети (GET /api/ai-data/raw):
// технические индикаторы, статистика сканера, сентимент. Без текста.
// Справа — системный промпт AI Backtest (config/charon_prompt.txt,
// GET /api/ai-data/prompt): правила анализа, редактируется и сохраняется
// кнопкой «💾 Сохранить промпт» (POST /api/ai-data/prompt).
//
// Панель использует классы .floating-panel / .hidden (не style.display —
// поэтому panel-active переключается здесь вручную). JSON обновляется при
// смене символа/таймфрейма (onSymbolTfChanged из app_init.js).

import { state } from '../state.js';

const $ = (id) => document.getElementById(id);
const panel = () => $('ai-data-panel');
const btn = () => $('ai-data-toggle-btn');

let _loading = false;

function _esc(s) {
  const d = document.createElement('div');
  d.textContent = String(s == null ? '' : s);
  return d.innerHTML;
}

function _setStatus(text, cls) {
  const el = $('ai-data-status');
  if (!el) return;
  el.textContent = text || '';
  el.className = 'ai-data-status' + (cls ? ' ' + cls : '');
}

function _syncButton() {
  const b = btn();
  if (b && panel()) b.classList.toggle('panel-active', !panel().classList.contains('hidden'));
}

export function openAiDataPanel() {
  const p = panel();
  if (!p) return;
  p.classList.remove('hidden');
  _syncButton();
  _updateSub();
  loadRaw();
  loadPrompt();
}

export function closeAiDataPanel() {
  const p = panel();
  if (!p) return;
  p.classList.add('hidden');
  _syncButton();
}

function _updateSub() {
  const el = $('ai-data-panel-sub');
  if (el) el.textContent = `${state.symbol} · ${state.timeframe}`;
}

/* Сырые данные текущей пары -> <pre><code>. */
export async function loadRaw() {
  const p = panel();
  if (!p || p.classList.contains('hidden')) return;
  if (_loading) return;
  _loading = true;
  const pre = $('ai-data-json');
  const sym = state.symbol;
  const tf = state.timeframe;
  if (pre) pre.innerHTML = '<code>загрузка…</code>';
  try {
    const resp = await fetch(
      `/api/ai-data/raw?symbol=${encodeURIComponent(sym)}&timeframe=${encodeURIComponent(tf)}`);
    const data = await resp.json();
    if (state.symbol !== sym || state.timeframe !== tf) return; // сменилось во время запроса
    if (!resp.ok) {
      if (pre) { pre.textContent = ''; pre.innerHTML = `<code>${_esc(data.error || 'HTTP ' + resp.status)}</code>`; }
      return;
    }
    if (pre) {
      pre.textContent = '';
      const code = document.createElement('code');
      code.textContent = JSON.stringify(data, null, 2);
      pre.appendChild(code);
    }
  } catch {
    if (pre && state.symbol === sym) {
      pre.textContent = '';
      pre.innerHTML = '<code>Не удалось загрузить данные</code>';
    }
  } finally {
    _loading = false;
  }
}

/* Системный промпт -> <textarea>. */
export async function loadPrompt() {
  const p = panel();
  if (!p || p.classList.contains('hidden')) return;
  const ta = $('ai-data-prompt');
  if (!ta) return;
  try {
    const resp = await fetch('/api/ai-data/prompt');
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || 'HTTP ' + resp.status);
    ta.value = data.prompt || '';
  } catch {
    _setStatus('Не удалось загрузить промпт', 'err');
  }
}

/* Сохранение промпта: POST /api/ai-data/prompt. */
export async function savePrompt() {
  const ta = $('ai-data-prompt');
  if (!ta) return;
  const prompt = ta.value;
  _setStatus('Сохраняю…', 'ok');
  try {
    const resp = await fetch('/api/ai-data/prompt', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || 'HTTP ' + resp.status);
    _setStatus('✓ Промпт сохранён', 'ok');
  } catch (e) {
    _setStatus('⚠ ' + e.message, 'err');
  }
}

/* Смена символа/таймфрейма: если панель открыта — обновляем JSON и подпись. */
export function onSymbolTfChanged() {
  const p = panel();
  if (!p || p.classList.contains('hidden')) return;
  _updateSub();
  loadRaw();
}

export function initAiDataUI() {
  const close = $('ai-data-close-btn');
  if (close) close.addEventListener('click', closeAiDataPanel);
  const save = $('ai-data-save-btn');
  if (save) save.addEventListener('click', savePrompt);
}