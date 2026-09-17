const $ = (id) => document.getElementById(id);

/* Лейблы условий для компактного отображения в списке. */
const CONDITION_LABELS = { above: '≥', below: '≤', cross: '×' };

/* ---------------------------------------------------------------- панель */
export function openAlerts() {
  const panel = $('alerts-panel');
  if (!panel) return;
  panel.style.display = 'block';
  loadAlerts();
}

export function closeAlerts() {
  const panel = $('alerts-panel');
  if (panel) panel.style.display = 'none';
}

export function toggleAlerts() {
  const panel = $('alerts-panel');
  if (!panel) return;
  if (panel.style.display === 'none') openAlerts();
  else closeAlerts();
}

/* --------------------------------------------------------------- список */
export function loadAlerts() {
  const list = $('alerts-list');
  if (!list) return;
  fetch('/api/alerts?active=1')
    .then(r => r.json())
    .then(data => {
      const alerts = data.alerts || [];
      list.innerHTML = '';
      for (const a of alerts) {
        const row = document.createElement('div');
        row.className = 'alert-item' + (Number(a.triggered) === 1 ? ' triggered' : '');
        const sym = document.createElement('span');
        sym.className = 'alert-symbol';
        sym.textContent = a.symbol;
        const cond = document.createElement('span');
        cond.className = 'alert-cond';
        cond.textContent = (CONDITION_LABELS[a.condition] || a.condition) + ' ' + a.value;
        const del = document.createElement('button');
        del.className = 'alert-delete';
        del.textContent = '✕';
        del.title = 'Удалить';
        del.addEventListener('click', () => deleteAlert(a.id));
        row.append(sym, cond, del);
        list.appendChild(row);
      }
      if (!alerts.length) {
        const empty = document.createElement('div');
        empty.className = 'alerts-empty';
        empty.textContent = 'Нет активных алертов';
        list.appendChild(empty);
      }
      const cnt = $('alerts-count');
      if (cnt) {
        const active = alerts.filter(a => Number(a.triggered) !== 1).length;
        cnt.textContent = 'активных: ' + active;
      }
    })
    .catch(e => console.error('alerts:', e));
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
    if (body.channel === 'browser' && window.Notification
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
  if (window.Notification && Notification.permission === 'granted') {
    try { new Notification('NeoTerminal', { body: msg }); } catch { /* noop */ }
  }
}