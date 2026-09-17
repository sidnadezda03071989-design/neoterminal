import { state } from '../state.js';
import { captureChartScreenshot } from '../chart/setup.js';

const $ = (id) => document.getElementById(id);

function toTimeNum(v) {
  if (v == null) return null;
  if (typeof v === 'number') return Math.round(v);
  if (typeof v === 'object' && v.year != null) {
    return Math.round(new Date(Date.UTC(v.year, (v.month||1)-1, v.day||1)).getTime() / 1000);
  }
  return null;
}

export async function fetchChartContext() {
  const params = new URLSearchParams({
    symbol: state.symbol, timeframe: state.timeframe,
    mode: state.mode, limit: '10000',
  });
  if (state.mode === 'replay') {
    params.set('replay_index', String(state.replay.index));
    params.set('from', String(($('from-date') || {}).value || ''));
    params.set('to', String(($('to-date') || {}).value || ''));
  }
  try {
    const vr = state.chart ? state.chart.timeScale().getVisibleRange() : null;
    if (vr) {
      const vf = toTimeNum(vr.from); const vt = toTimeNum(vr.to);
      if (vf != null && vt != null) {
        params.set('visible_from', String(vf));
        params.set('visible_to', String(vt));
      }
    }
  } catch (e) {}
  const resp = await fetch('/api/chart-context?' + params.toString());
  if (!resp.ok) throw new Error('chart-context HTTP ' + resp.status);
  return resp.json();
}

export function renderAIResult(r) {
  const sigEl = $('ai-signal');
  if (sigEl) {
    sigEl.textContent = String(r.signal || 'HOLD');
    sigEl.className = 'ai-signal signal-' + String(r.signal || 'HOLD');
  }
  let conf = Number(r.confidence || 0);
  if (conf <= 1) conf *= 100;
  conf = Math.max(0, Math.min(100, conf));
  const fill = $('conf-fill'); if (fill) fill.style.width = conf + '%';
  const txt = $('conf-text'); if (txt) txt.textContent = 'уверенность ' + conf.toFixed(1) + '%';
  const reason = $('ai-reason'); if (reason) reason.textContent = String(r.reason || '');
  const meta = $('ai-meta');
  if (meta) {
    const parts = [`режим: ${state.mode} · ${state.symbol} ${state.timeframe}`];
    if (r.model) parts.push(`модель: ${r.model}`);
    if (r.fallback) parts.push('⚠ эвристика');
    if (r.vision_used === true) parts.push('👁 vision');
    meta.textContent = parts.join(' · ');
  }
  const mini = $('ai-indicators-mini');
  if (mini) {
    if (r.indicator_signals && typeof r.indicator_signals === 'object') {
      mini.style.display = 'flex';
      const is = r.indicator_signals;
      mini.innerHTML =
        '<span class="mini-tag">тренд: ' + (is.trend || '—') + '</span>' +
        '<span class="mini-tag">моментум: ' + (is.momentum || '—') + '</span>' +
        '<span class="mini-tag">MACD: ' + (is.macd || '—') + '</span>' +
        '<span class="mini-tag">BB: ' + (is.bb_position || '—') + '</span>';
    } else {
      mini.style.display = 'none';
    }
  }
}

export async function runAIAnalysis() {
  const btn = $('ai-analysis-btn');
  if (btn) { btn.disabled = true; btn.textContent = '… анализ …'; }
  const sigEl = $('ai-signal');
  if (sigEl) { sigEl.textContent = '…'; sigEl.className = 'ai-signal'; }
  try {
    const ctx = await fetchChartContext();
    const body = { ...ctx };
    if (state.mode === 'replay') {
      // ИИ не должен видеть данные после replay_index / правого края окна.
      body.replay_index = state.replay.index;
      let vt = null;
      try {
        const vr = state.chart ? state.chart.timeScale().getVisibleRange() : null;
        if (vr) vt = toTimeNum(vr.to);
      } catch (e) {}
      if (vt != null) body.visible_to = vt;
    }
    // Скриншот графика для Vision-анализа (null — если не удалось снять).
    body.screenshot = captureChartScreenshot();
    const resp = await fetch('/api/ai-analysis', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const r = await resp.json();
    renderAIResult(r);
    if (state.dm && r.drawings && Array.isArray(r.drawings)) state.dm.refresh(r.drawings);
  } catch (e) {
    const reason = $('ai-reason');
    if (reason) reason.textContent = 'Ошибка: ' + e.message;
    if (sigEl) sigEl.textContent = 'ERR';
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = '🧠 AI Analysis'; }
  }
}

export function updateAiDrawingsUI() {
  if (!state.dm) return;
  const ai = state.dm.aiDrawings ? state.dm.aiDrawings() : [];
  const cnt = $('ai-draw-count'); if (cnt) cnt.textContent = String(ai.length);
  const list = $('ai-draw-list');
  if (!list) return;
  list.innerHTML = '';
  ai.slice(0, 8).forEach((d) => {
    const li = document.createElement('li');
    li.textContent = (d.type || '') + (d.label ? ' — ' + d.label : '');
    list.appendChild(li);
  });
}
