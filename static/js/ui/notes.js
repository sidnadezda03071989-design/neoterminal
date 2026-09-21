// Заметки по активу: блокнот (текст + файлы/скриншоты) на общем
// компоненте notepad.js, привязан к текущему символу state.symbol.

import { state } from '../state.js';
import { createNotepad } from './notepad.js';

const notepad = createNotepad({
  panelId: 'notes-panel',
  textareaId: 'notes-textarea',
  statusId: 'notes-status',
  dropzoneId: 'notes-dropzone',
  fileInputId: 'notes-file',
  screenshotsId: 'notes-screenshots',
  saveBtnId: 'notes-save-btn',
  labelId: 'notes-symbol-label',
  endpoint: (sym) => '/api/notes/' + encodeURIComponent(sym || state.symbol),
  title: (sym) => '📝 ' + sym,
});

export function toggleNotes() {
  if (notepad.isOpen()) notepad.close();
  else notepad.open(state.symbol);
}

export function openNotes(symbol) {
  notepad.open(symbol || state.symbol);
}

export function closeNotes() {
  notepad.close();
}

export function onSymbolChanged(newSym) {
  notepad.onSymbolChanged(newSym);
}

export function initNotesUI() {
  notepad.bind();
}