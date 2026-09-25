import { state } from '../state.js';
import { loadAlerts } from './alerts.js';

const $ = (id) => document.getElementById(id);

const TYPE_LABELS = {
  trendline: 'Трендовая линия',
  ray: 'Луч',
  h_line: 'Горизонтальная линия',
  rectangle: 'Прямоугольник',
  fib: 'Фибоначчи',
  pen: 'Кисть',
  text: 'Текст',
  hray: 'Горизонтальный луч',
  vline: 'Вертикальная линия',
  channel: 'Канал',
  long_position: 'Long-позиция',
  short_position: 'Short-позиция',
  arrow: 'Стрелка',
  position: 'Позиция',
};

const PANEL_IDS = {
  quotes: 'watchlist-panel',
  alerts: 'alerts-panel',
  objects: 'objects-panel',
};
const PANEL_TITLES = {
  quotes: 'Список котировок',
  alerts: 'Оповещения',
  objects: 'Дерево объектов',
};
const WIDTH_KEY = 'neoterminal_right_dock_width';
const DEFAULT_WIDTH = 420;
let activePanel = null;
const MIN_DOCK_WIDTH = 320;
const MAX_DOCK_WIDTH = 1200;
const MIN_CHART_WIDTH = 280;

let objectRefreshTimer = null;

export function isRightDockActive(panel) {
  return !!activePanel && activePanel === panel;
}

function _setPanel(panel) {
  activePanel = panel || null;
  for (const [name, id] of Object.entries(PANEL_IDS)) {
    const el = $(id);
    if (el) el.style.display = name === activePanel ? 'flex' : 'none';
  }
  document.querySelectorAll('.right-dock-tab').forEach((button) => {
    const active = button.dataset.rightPanel === activePanel;
    button.classList.toggle('panel-active', active);
    button.setAttribute('aria-pressed', active ? 'true' : 'false');
  });
  const title = $('right-dock-title');
  if (title) title.textContent = activePanel ? PANEL_TITLES[activePanel] : '';
  if (isRightDockActive('alerts')) loadAlerts();
  else if (isRightDockActive('objects')) loadObjectTree();
}

export function openRightDock(panel) {
  if (!PANEL_IDS[panel]) return;
  document.body.classList.add('right-dock-open');
  _setPanel(panel);
}

export function closeRightDock() {
  document.body.classList.remove('right-dock-open');
  _setPanel(null);
  if (objectRefreshTimer) {
    clearTimeout(objectRefreshTimer);
    objectRefreshTimer = null;
  }
}

export function toggleRightDock(panel) {
  if (isRightDockActive(panel)) closeRightDock();
  else openRightDock(panel);
}

function _formatDate(value) {
  if (!value) return '';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString('ru-RU', {
    day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit',
  });
}

function _renderObjectTree(drawings) {
  const tree = $('objects-tree');
  const count = $('objects-count');
  if (!tree) return;
  tree.innerHTML = '';
  const list = Array.isArray(drawings) ? drawings : [];
  if (count) count.textContent = String(list.length);
  if (!list.length) {
    const empty = document.createElement('div');
    empty.className = 'objects-empty';
    empty.textContent = 'Нарисованных объектов пока нет';
    tree.appendChild(empty);
    return;
  }

  const groups = new Map();
  for (const drawing of list) {
    const type = TYPE_LABELS[drawing.type] || drawing.type || 'Объект';
    const key = `${drawing.symbol || '—'} · ${drawing.timeframe || '—'} · ${type}`;
    if (!groups.has(key)) {
      groups.set(key, {
        symbol: drawing.symbol,
        timeframe: drawing.timeframe,
        type,
        items: [],
      });
    }
    groups.get(key).items.push(drawing);
  }

  for (const group of groups.values()) {
    const details = document.createElement('details');
    details.className = 'object-tree-group';
    details.open = true;
    const summary = document.createElement('summary');
    summary.textContent = `${group.symbol || '—'} · ${group.timeframe || '—'} · ${group.type}`;
    const badge = document.createElement('span');
    badge.className = 'object-tree-count';
    badge.textContent = String(group.items.length);
    summary.appendChild(badge);
    details.appendChild(summary);

    for (const drawing of group.items) {
      const row = document.createElement('div');
      row.className = 'object-tree-item';
      row.title = drawing.id || '';
      const name = document.createElement('div');
      name.className = 'object-tree-name';
      name.textContent = drawing.label || TYPE_LABELS[drawing.type] || drawing.type || 'Объект';
      const meta = document.createElement('div');
      meta.className = 'object-tree-meta';
      const author = drawing.created_by === 'ai' ? 'ИИ' : 'Пользователь';
      const points = Array.isArray(drawing.points) ? drawing.points.length : 0;
      meta.textContent = `${drawing.id || 'без ID'} · ${points} точек · ${author}`;
      const date = _formatDate(drawing.updated_at || drawing.created_at);
      if (date) meta.textContent += ` · ${date}`;
      row.append(name, meta);
      row.addEventListener('click', () => {
        if (!state.dm || drawing.symbol !== state.symbol || drawing.timeframe !== state.timeframe) return;
        state.dm.selectedId = drawing.id || null;
        if (typeof state.dm._fire === 'function') state.dm._fire();
      });
      details.appendChild(row);
    }
    tree.appendChild(details);
  }
}

export function loadObjectTree() {
  const tree = $('objects-tree');
  if (tree) tree.innerHTML = '<div class="objects-empty">Загрузка объектов…</div>';
  return fetch('/api/drawings')
    .then((response) => {
      if (!response.ok) throw new Error('HTTP ' + response.status);
      return response.json();
    })
    .then((data) => _renderObjectTree(data.drawings || []))
    .catch((error) => {
      console.error('object tree:', error);
      if (tree) tree.innerHTML = '<div class="objects-empty">Не удалось загрузить объекты</div>';
    });
}

export function scheduleObjectTreeRefresh() {
  if (!isRightDockActive('objects')) return;
  if (objectRefreshTimer) clearTimeout(objectRefreshTimer);
  objectRefreshTimer = setTimeout(() => {
    objectRefreshTimer = null;
    loadObjectTree();
  }, 250);
}

function _clampDockWidth(width) {
  const viewportMax = window.innerWidth - MIN_CHART_WIDTH;
  const max = Math.max(
    MIN_DOCK_WIDTH,
    Math.min(MAX_DOCK_WIDTH, viewportMax),
  );
  const min = Math.min(MIN_DOCK_WIDTH, max);
  return Math.max(min, Math.min(max, Math.round(width)));
}

export function initRightDockResizer() {
  const handle = $('right-dock-resizer');
  if (!handle || handle.dataset.ready === 'true') return;
  handle.dataset.ready = 'true';
  const root = document.documentElement;
  try {
    const saved = Number(localStorage.getItem(WIDTH_KEY));
    if (Number.isFinite(saved) && saved > 0) {
      root.style.setProperty('--right-dock-width', _clampDockWidth(saved) + 'px');
    }
  } catch (e) { /* localStorage may be unavailable */ }

  handle.addEventListener('pointerdown', (event) => {
    if (!isRightDockActive('quotes') && !isRightDockActive('alerts') && !isRightDockActive('objects')) return;
    if (event.button !== 0) return;
    event.preventDefault();

    const startX = event.clientX;
    let latestX = startX;
    let frame = 0;
    const startWidth = parseFloat(getComputedStyle(root).getPropertyValue('--right-dock-width')) || DEFAULT_WIDTH;
    const setWidth = () => {
      frame = 0;
      const width = _clampDockWidth(startWidth + (startX - latestX));
      root.style.setProperty('--right-dock-width', width + 'px');
    };

    document.body.classList.add('right-dock-resizing');

    // Pointer events приходят быстрее кадров браузера. Coalescing через rAF
    // исключает конкуренцию нескольких layout/resize за один кадр.
    const onMove = (moveEvent) => {
      latestX = moveEvent.clientX;
      if (!frame) frame = requestAnimationFrame(setWidth);
    };
    const onUp = () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
      window.removeEventListener('pointercancel', onUp);
      window.removeEventListener('blur', onUp);
      if (frame) {
        cancelAnimationFrame(frame);
        setWidth();
      }
      document.body.classList.remove('right-dock-resizing');
      try {
        localStorage.setItem(
          WIDTH_KEY,
          getComputedStyle(root).getPropertyValue('--right-dock-width').trim(),
        );
      } catch (e) { /* noop */ }
    };

    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    window.addEventListener('pointercancel', onUp);
    window.addEventListener('blur', onUp);
  });

  handle.addEventListener('dblclick', () => {
    root.style.setProperty('--right-dock-width', DEFAULT_WIDTH + 'px');
    try { localStorage.setItem(WIDTH_KEY, String(DEFAULT_WIDTH)); } catch (e) { /* noop */ }
  });
}
