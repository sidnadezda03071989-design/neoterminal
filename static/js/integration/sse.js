let _sse = null;
let _retryCount = 0;
let _retryTimer = null;

import { loadAlerts } from '../ui/alerts.js';

const MAX_RETRIES = 10;
const BASE_DELAY_SEC = 5;
const MAX_DELAY_SEC = 60;

/* Экспоненциальная задержка: 5→10→20→40→60, дальше максимум 60с. */
function _retryDelaySec() {
  return Math.min(BASE_DELAY_SEC * 2 ** _retryCount, MAX_DELAY_SEC);
}

export function sseConnect() {
  if (_sse) return;
  try {
    _sse = new EventSource('/ws');
    _sse.addEventListener('connected', () => {
      if (_retryCount > 0) console.info('sse: соединение восстановлено');
      _retryCount = 0; // успешное подключение сбрасывает счётчик попыток
    });
    _sse.addEventListener('alert_triggered', (e) => {
      let data = null;
      try { data = JSON.parse(e.data); } catch { data = null; }
      if (window.__alertToast) window.__alertToast(data || e.data || 'alert');
      loadAlerts(); // перерисовать список (алерт помечен triggered)
    });
    _sse.addEventListener('scan_progress', (e) => {
      let data = null;
      try { data = JSON.parse(e.data); } catch { data = null; }
      if (window.__scanProgress) window.__scanProgress(data);
    });
    _sse.addEventListener('ai_backtest_progress', (e) => {
      let data = null;
      try { data = JSON.parse(e.data); } catch { data = null; }
      if (window.__aiBacktestProgress) window.__aiBacktestProgress(data);
    });
    _sse.addEventListener('candle_update', () => { /* placeholder */ });
    _sse.addEventListener('ai_analysis_done', () => { /* placeholder */ });
    _sse.onerror = () => {
      // readyState === 2 (CLOSED): браузер сам уже не переподключится —
      // надо сбросить _sse и запланировать реконнект вручную.
      if (!_sse || _sse.readyState !== 2) return; // 0/1 — браузер переподключит сам
      _sse.close();
      _sse = null;
      if (_retryCount >= MAX_RETRIES) {
        console.error('sse: не удалось переподключиться после ' + MAX_RETRIES + ' попыток, реконнект остановлен');
        return;
      }
      const delaySec = _retryDelaySec();
      _retryCount += 1;
      console.warn('sse: соединение закрыто, реконнект через ' + delaySec + 'с (попытка ' + _retryCount + '/' + MAX_RETRIES + ')');
      _retryTimer = setTimeout(() => { _retryTimer = null; sseConnect(); }, delaySec * 1000);
    };
  } catch (e) {
    console.error('sse:', e);
  }
}

export function wsConnect() { sseConnect(); }

export function wsDisconnect() {
  if (_retryTimer) { clearTimeout(_retryTimer); _retryTimer = null; } // отменить запланированный реконнект
  if (_sse) { _sse.close(); _sse = null; }
}
