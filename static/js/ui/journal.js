// Торговый журнал: блокнот на общем компоненте notepad.js (произвольный
// текст + файлы/скриншоты). Дополнительно — вкладка «📋 все заметки»
// (таблица по символам), клик по строке переключает график и открывает
// заметку актива.

import { state } from '../state.js';
import { switchSymbol } from './toolbar.js';
import { openNotes } from './notes.js';
import { createNotepad } from './notepad.js';

const $ = (id) => document.getElementById(id);

const notepad = createNotepad({
  panelId: 'journal-panel',
  textareaId: 'journal-textarea',
  statusId: 'journal-status',
  dropzoneId: 'journal-dropzone',
  fileInputId: 'journal-file',
  screenshotsId: 'journal-screenshots',
  saveBtnId: 'journal-save-btn',
  endpoint: () => '/api/journal',
});

export function toggleJournal() {
  if (notepad.isOpen()) notepad.close();
  else openJournal();
}

export function openJournal() {
  const p = $('journal-panel');
  if (!p) return;
  p.style.display = 'block';
  showNotepad();
  notepad.open(null);
}

export function closeJournal() {
  const p = $('journal-panel');
  if (p) p.style.display = 'none';
}

function showNotepad() {
  const npv = $('journal-notepad-view');
  const lst = $('journal-list-view');
  if (npv) npv.style.display = 'flex';
  if (lst) lst.style.display = 'none';
}

function showList() {
  const npv = $('journal-notepad-view');
  const lst = $('journal-list-view');
  if (npv) npv.style.display = 'none';
  if (lst) lst.style.display = 'block';
  loadJournalList();
}

function fmtDate(iso) {
  if (!iso) return '—';
  const t = new Date(iso);
  if (Number.isNaN(t.getTime())) return String(iso);
  return t.toLocaleString('ru-RU', { dateStyle: 'short', timeStyle: 'short' });
}

async function loadJournalList() {
  const list = $('journal-list');
  const empty = $('journal-empty');
  try {
    const r = await fetch('/api/notes');
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const d = await r.json();
    const notes = d.notes || [];
    if (empty) empty.style.display = notes.length ? 'none' : 'block';
    if (!list) return;
    list.innerHTML = '';
    notes.forEach((n) => {
      const row = document.createElement('div');
      row.className = 'journal-row';
      row.innerHTML =
        '<span class="jr-symbol">' + n.symbol + '</span>' +
        '<span class="jr-date">' + fmtDate(n.updated_at) + '</span>' +
        '<span class="jr-text">' + (n.content_preview || '') + '</span>' +
        '<span class="jr-shots">' + (n.screenshots_count ? '📎 ' + n.screenshots_count : '') + '</span>';
      row.addEventListener('click', () => {
        const sym = n.symbol;
        switchSymbol(sym);
        openNotes(state.symbol || sym);
        closeJournal();
      });
      list.appendChild(row);
    });
  } catch (e) {
    console.error('journal list:', e);
    if (empty) {
      empty.style.display = 'block';
      empty.textContent = '⚠ Ошибка загрузки';
    }
  }
}

export function initJournalUI() {
  notepad.bind();
  const lstBtn = $('journal-list-btn');
  if (lstBtn) lstBtn.addEventListener('click', showList);
  const npBtn = $('journal-notepad-btn');
  if (npBtn) npBtn.addEventListener('click', showNotepad);
}