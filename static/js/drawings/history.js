// Undo/Redo: классический command-pattern stack для рисунков.
// Каждое действие — { type, before, after, id }:
//   'create'  → before=null,    after=shape
//   'delete'  → before=shape,   after=null
//   'update'  → before=shape,   after=shape (перемещение/изменение узла)
//   'clear'   → before=[shapes], after=null (массовое удаление)
// Стек ограничен MAX_HISTORY; новое действие очищает redo.

const MAX_HISTORY = 100;

let undoStack = [];
let redoStack = [];
let _subscriber = null;

// Колбэк вызывается после любого изменения стеков (обновление UI кнопок).
export function setHistorySubscriber(fn) {
  _subscriber = typeof fn === 'function' ? fn : null;
}

function notify() {
  if (_subscriber) _subscriber();
}

export function pushAction(action) {
  undoStack.push(action);
  if (undoStack.length > MAX_HISTORY) undoStack.shift();
  redoStack = [];
  notify();
}

// apply(action, 'undo') — применить before; apply(action, 'redo') — after.
export function undo(apply) {
  if (!undoStack.length) return false;
  const action = undoStack[undoStack.length - 1];
  try {
    apply(action, 'undo');
  } catch (e) {
    return false;
  }
  undoStack.pop();
  redoStack.push(action);
  notify();
  return true;
}

export function redo(apply) {
  if (!redoStack.length) return false;
  const action = redoStack[redoStack.length - 1];
  try {
    apply(action, 'redo');
  } catch (e) {
    return false;
  }
  redoStack.pop();
  undoStack.push(action);
  notify();
  return true;
}

export function canUndo() { return undoStack.length > 0; }
export function canRedo() { return redoStack.length > 0; }

export function clearHistory() {
  undoStack = [];
  redoStack = [];
  notify();
}