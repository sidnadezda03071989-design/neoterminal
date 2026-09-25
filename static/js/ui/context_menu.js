import { state } from '../state.js';
import { chart, candleSeries, container } from '../chart/setup.js';
import { fitChartToData } from '../chart/series.js';
import { updateLegend } from './legend.js';
import { clearIndicators } from './indicators.js';
import { isRightDockActive } from './right_dock.js';

const $ = (id) => document.getElementById(id);
let initialized = false;
let lastCrosshairParam = null;
let contextPrice = null;
let drawingCount = 0;
let toastTimer = null;

function _finite(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function _formatPrice(value) {
  const price = _finite(value);
  if (price == null) return '';
  const abs = Math.abs(price);
  const digits = abs >= 1000 ? 2 : abs >= 1 ? 5 : 6;
  return String(Number(price.toFixed(digits)));
}

function _priceAtPoint(event) {
  if (!event || !container || !candleSeries) return null;
  try {
    const rect = container.getBoundingClientRect();
    const y = event.clientY - rect.top;
    const direct = candleSeries.coordinateToPrice(y);
    const price = _finite(direct);
    if (price != null) return price;
  } catch (e) { /* fallback to the last crosshair below */ }

  const param = lastCrosshairParam;
  if (param && param.point) {
    try {
      const price = _finite(candleSeries.coordinateToPrice(param.point.y));
      if (price != null) return price;
    } catch (e) { /* noop */ }
  }
  if (param && param.seriesData) {
    try {
      const candle = param.seriesData.get(candleSeries);
      return _finite(candle && candle.value && candle.value.close);
    } catch (e) { /* noop */ }
  }
  return null;
}

function _notify(message, isError = false) {
  const toast = $('context-toast');
  if (!toast) return;
  toast.textContent = message;
  toast.classList.toggle('error', isError);
  toast.classList.add('show');
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove('show'), 2600);
}

async function _copyText(value) {
  const text = String(value);
  if (navigator.clipboard && window.isSecureContext) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const textarea = document.createElement('textarea');
  textarea.value = text;
  textarea.style.position = 'fixed';
  textarea.style.opacity = '0';
  document.body.appendChild(textarea);
  textarea.focus();
  textarea.select();
  const copied = document.execCommand('copy');
  textarea.remove();
  if (!copied) throw new Error('copy failed');
}

function _indicatorCount() {
  return Object.values(state.indicators || {}).filter(Boolean).length;
}

function _setCounts() {
  const drawings = $('context-drawings-count');
  const indicators = $('context-indicators-count');
  if (drawings) drawings.textContent = String(drawingCount);
  if (indicators) indicators.textContent = String(_indicatorCount());
}

function _refreshDrawingCount() {
  // Счётчик в меню отражает все сохранённые объекты, как и кнопка очистки.
  return fetch('/api/drawings', { cache: 'no-store' })
    .then((response) => {
      if (!response.ok) throw new Error('HTTP ' + response.status);
      return response.json();
    })
    .then((data) => {
      drawingCount = Array.isArray(data.drawings) ? data.drawings.length : 0;
      _setCounts();
    })
    .catch(() => {
      drawingCount = state.dm && Array.isArray(state.dm.drawings)
        ? state.dm.drawings.length : 0;
      _setCounts();
    });
}

function _closeMenu() {
  const menu = $('chart-context-menu');
  if (menu) menu.classList.remove('open');
}

function _openMenu(event) {
  const menu = $('chart-context-menu');
  if (!menu) return;
  event.preventDefault();
  contextPrice = _priceAtPoint(event);
  drawingCount = state.dm && Array.isArray(state.dm.drawings)
    ? state.dm.drawings.length : drawingCount;
  _setCounts();

  menu.classList.add('open');
  menu.style.left = '0px';
  menu.style.top = '0px';
  menu.style.visibility = 'hidden';
  const rect = menu.getBoundingClientRect();
  const margin = 8;
  const left = Math.max(margin, Math.min(event.clientX, window.innerWidth - rect.width - margin));
  const top = Math.max(margin, Math.min(event.clientY, window.innerHeight - rect.height - margin));
  menu.style.left = `${left}px`;
  menu.style.top = `${top}px`;
  menu.style.visibility = '';
  void _refreshDrawingCount();
}

function _resetChart() {
  const candles = state.activeCandles && state.activeCandles.length
    ? state.activeCandles : state.candles;
  if (!candles || !candles.length) {
    _notify('Нет данных для сброса графика', true);
    return;
  }
  fitChartToData(candles);
  if (state.dm) {
    state.dm.selectedId = null;
    if (typeof state.dm._fire === 'function') state.dm._fire();
  }
  _notify('Состояние графика сброшено');
}

async function _copyCursorPrice() {
  const price = contextPrice;
  if (price == null) {
    _notify('Цена линии курсора недоступна', true);
    return;
  }
  try {
    await _copyText(_formatPrice(price));
    _notify(`Цена скопирована: ${_formatPrice(price)}`);
  } catch (e) {
    _notify('Не удалось скопировать цену', true);
  }
}

function _openAlertForm() {
  const symbol = $('alert-symbol');
  const value = $('alert-value');
  const condition = $('alert-condition');
  if (symbol) symbol.value = state.symbol;
  if (condition) condition.value = 'above';
  if (value) {
    value.value = contextPrice == null ? '' : _formatPrice(contextPrice);
  }

  const alertsButton = $('alerts-toggle-btn');
  if (alertsButton && !isRightDockActive('alerts')) alertsButton.click();
  setTimeout(() => {
    if (value) {
      value.focus();
      value.select();
    }
  }, 0);
  _notify(contextPrice == null
    ? 'Открыто оповещение — укажите цену'
    : `В поле оповещения подставлена цена ${_formatPrice(contextPrice)}`);
}

function _clearDrawings() {
  const dm = state.dm;
  if (!dm || typeof dm.clearAll !== 'function') {
    _notify('Объекты рисования недоступны', true);
    return;
  }
  if (drawingCount <= 0) {
    // На первом открытии меню серверный счётчик мог ещё не успеть прийти.
    void _refreshDrawingCount().then(() => {
      if (drawingCount > 0) _clearDrawings();
      else _notify('Объекты рисования отсутствуют');
    });
    return;
  }
  if (!window.confirm(`Удалить все объекты рисования (${drawingCount})?`)) return;
  Promise.resolve()
    .then(() => dm.clearAll())
    .then(() => {
      drawingCount = 0;
      _setCounts();
      _notify('Объекты рисования удалены');
    })
    .catch(() => _notify('Не удалось удалить объекты рисования', true));
}

function _clearIndicators() {
  const count = _indicatorCount();
  if (!count) {
    _notify('Индикаторы уже выключены');
    return;
  }
  if (!window.confirm(`Удалить все индикаторы (${count})?`)) return;
  clearIndicators();
  _setCounts();
  _notify('Индикаторы удалены');
}

function _handleAction(action) {
  _closeMenu();
  if (action === 'reset-chart') _resetChart();
  else if (action === 'copy-price') void _copyCursorPrice();
  else if (action === 'add-alert') _openAlertForm();
  else if (action === 'clear-drawings') _clearDrawings();
  else if (action === 'clear-indicators') _clearIndicators();
}

export function initContextMenu() {
  if (initialized) return;
  initialized = true;
  const menu = $('chart-context-menu');
  const chartWrap = container && container.closest('.chart-wrap');
  if (!menu || !chartWrap) return;

  chartWrap.addEventListener('contextmenu', _openMenu);
  menu.addEventListener('click', (event) => {
    const button = event.target.closest('[data-context-action]');
    if (!button) return;
    event.preventDefault();
    event.stopPropagation();
    _handleAction(button.dataset.contextAction);
  });

  document.addEventListener('pointerdown', (event) => {
    if (!menu.contains(event.target)) _closeMenu();
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') _closeMenu();
  });
  window.addEventListener('resize', _closeMenu);
  window.addEventListener('scroll', _closeMenu, true);
  window.addEventListener('blur', _closeMenu);

  try {
    chart.subscribeCrosshairMove((param) => {
      lastCrosshairParam = param;
      updateLegend(param);
    });
  } catch (e) { /* older chart build: copy still works from click coordinates */ }

  _setCounts();
}
