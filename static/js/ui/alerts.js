import { state } from '../state.js';
import { candleSeries } from '../chart/setup.js';
import { notificationsEnabled } from './settings.js';

const $ = (id) => document.getElementById(id);
let _alertsCache = [];
const _alertLines = new Map();

function _removeAlertLine(id) {
  const line = _alertLines.get(String(id));
  if (!line) return;
  try { candleSeries.removePriceLine(line); } catch (e) { /* noop */ }
  _alertLines.delete(String(id));
}

export function syncAlertLines() {
  const visible = _alertsCache.filter((alert) => alert.symbol === state.symbol);
  const visibleIds = new Set(visible.map((alert) => String(alert.id)));
  for (const id of Array.from(_alertLines.keys())) {
    if (!visibleIds.has(id)) _removeAlertLine(id);
  }

  const dashed = window.LightweightCharts?.LineStyle?.Dashed ?? 2;
  for (const alert of visible) {
    const id = String(alert.id);
    const value = Number(alert.value);
    if (_alertLines.has(id) || !Number.isFinite(value)) continue;
    try {
      const line = candleSeries.createPriceLine({
        price: value,
        color: '#ffffff',
        lineWidth: 1,
        lineStyle: dashed,
        axisLabelVisible: true,
        title: `🔔 ${value}`,
      });
      _alertLines.set(id, line);
    } catch (e) { /* chart may still be initializing */ }
  }
}

/* Лейблы условий для компактного отображения в списке. */
const CONDITION_LABELS = { above: '≥', below: '≤', cross: '×' };

/* --------------------------------------------------------------- список */
export function loadAlerts() {
  const list = $('alerts-list');
  if (!list) return Promise.resolve([]);
  return fetch('/api/alerts?active=0')
    .then((r) => r.json())
    .then((data) => {
      const alerts = Array.isArray(data.alerts) ? data.alerts : [];
      _alertsCache = alerts;
      syncAlertLines();
      const current = alerts.filter((a) => Number(a.triggered) !== 1 && Number(a.active) !== 0);
      const triggered = alerts.filter((a) => Number(a.triggered) === 1);
      list.innerHTML = '';

      const renderGroup = (title, rows, isTriggered) => {
        if (!rows.length) return;
        const heading = document.createElement('div');
        heading.className = 'alerts-section-title';
        heading.textContent = title;
        list.appendChild(heading);
        for (const a of rows) {
          const row = document.createElement('div');
          row.className = 'alert-item' + (isTriggered ? ' triggered' : '');
          const sym = document.createElement('span');
          sym.className = 'alert-symbol';
          sym.textContent = a.symbol;
          const cond = document.createElement('span');
          cond.className = 'alert-cond';
          cond.textContent = (CONDITION_LABELS[a.condition] || a.condition) + ' ' + a.value;
          const status = document.createElement('span');
          status.className = 'alert-status';
          status.textContent = isTriggered ? 'Сработало' : 'Текущее';
          const del = document.createElement('button');
          del.className = 'alert-delete';
          del.textContent = '✕';
          del.title = 'Удалить';
          del.addEventListener('click', () => deleteAlert(a.id));
          row.append(sym, cond, status, del);
          list.appendChild(row);
        }
      };

      renderGroup('Текущие оповещения', current, false);
      renderGroup('Сработавшие оповещения', triggered, true);
      if (!current.length && !triggered.length) {
        const empty = document.createElement('div');
        empty.className = 'alerts-empty';
        empty.textContent = 'Оповещений пока нет';
        list.appendChild(empty);
      }
      const countText = 'активных: ' + current.length + ' · сработало: ' + triggered.length;
      const cnt = $('alerts-count');
      if (cnt) cnt.textContent = countText;
    })
    .catch((e) => {
      console.error('alerts:', e);
      return [];
    });
}

/* ------------------------------------------------------------ удаление */
export async function deleteAlert(id) {
  try {
    const resp = await fetch('/api/alerts/' + id, { method: 'DELETE' });
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    await loadAlerts();
  } catch (e) {
    console.error('alerts delete:', e);
    showAlertToast({ message: '⚠ Ошибка удаления: ' + e.message });
  }
}

/* ---------------------------------------------------------- создание */
export async function createAlert() {
  const symSel = $('alert-symbol');
  const condSel = $('alert-condition');
  const valInp = $('alert-value');
  const chanSel = $('alert-channel');
  const destInp = $('alert-destination');
  if (!symSel || !condSel || !valInp || !chanSel) return;
  const value = valInp.value.trim();
  if (value === '' || isNaN(Number(value))) {
    showAlertToast({ message: '⚠ Укажите числовое значение' });
    return;
  }
  const body = {
    symbol: symSel.value,
    condition: condSel.value,
    value: Number(value),
    channel: chanSel.value,
    destination: destInp ? destInp.value.trim() : '',
  };
  try {
    const resp = await fetch('/api/alerts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const r = await resp.json();
    if (!resp.ok) throw new Error(r.error || 'HTTP ' + resp.status);
    valInp.value = '';
    await loadAlerts();
    showAlertToast({ message: '✅ Алерт создан: ' + body.symbol + ' ' + body.condition + ' ' + body.value });
    if (body.channel === 'browser' && notificationsEnabled() && window.Notification
        && Notification.permission !== 'granted') {
      requestNotificationPermission();
    }
  } catch (e) {
    showAlertToast({ message: '⚠ Ошибка: ' + e.message });
  }
}

/* ------------------------------------------------- browser notifications */
export function requestNotificationPermission() {
  if (!window.Notification || Notification.permission !== 'default') return;
  Notification.requestPermission().catch(() => {});
}

/* ---------------------------------------------------------------- тосты */
export function showAlertToast(data) {
  const box = $('alert-toasts');
  if (!box) return;
  let payload = data;
  if (typeof data === 'string') {
    try { payload = JSON.parse(data); } catch { payload = { message: data }; }
  }
  payload = payload || {};
  const msg = payload.message
    || ('Alert: ' + [payload.symbol, payload.condition, payload.value]
      .filter(v => v != null).join(' '));
  const toast = document.createElement('div');
  toast.className = 'alert-toast';
  const title = document.createElement('div');
  title.className = 'alert-toast-title';
  title.textContent = '🔔 Сработал алерт';
  const body = document.createElement('div');
  body.className = 'alert-toast-body';
  body.textContent = msg;
  toast.append(title, body);
  box.appendChild(toast);
  setTimeout(() => {
    toast.classList.add('out');
    setTimeout(() => toast.remove(), 300);
  }, 5000);
  if (notificationsEnabled() && window.Notification && Notification.permission === 'granted') {
    try { new Notification('NeoTerminal', { body: msg }); } catch { /* noop */ }
  }
}
