import { state } from '../state.js';

const $ = (id) => document.getElementById(id);
let _busy = false;

function appendMessage(role, text) {
  const body = $('chat-body');
  if (!body) return null;
  const wrap = document.createElement('div');
  wrap.className = 'chat-msg ' + (role === 'user' ? 'user' : 'ai');
  const bubble = document.createElement('div');
  bubble.className = 'chat-bubble';
  bubble.textContent = text;
  wrap.appendChild(bubble);
  body.appendChild(wrap);
  body.scrollTop = body.scrollHeight;
  return wrap;
}

export function renderChatHistory(messages) {
  const body = $('chat-body');
  if (!body) return;
  body.innerHTML = '';
  (messages || []).forEach((m) => {
    if (m && (m.role === 'user' || m.role === 'assistant')) {
      appendMessage(m.role === 'user' ? 'user' : 'ai', m.content || '');
    }
  });
}

export function openChat() {
  const panel = $('chat-panel');
  if (!panel) return;
  panel.style.display = 'flex';
  fetch('/api/chat/history')
    .then(r => r.json())
    .then(d => renderChatHistory(d.messages || []))
    .catch(e => console.error('chat history:', e));
  const inp = $('chat-input'); if (inp) inp.focus();
}

export function closeChat() {
  const panel = $('chat-panel');
  if (panel) panel.style.display = 'none';
}

export async function sendChatMessage() {
  if (_busy) return;
  const inp = $('chat-input');
  if (!inp) return;
  const text = inp.value.trim();
  if (!text) return;
  inp.value = '';
  appendMessage('user', text);
  _busy = true;
  const btn = $('chat-send-btn'); if (btn) btn.disabled = true;
  try {
    const body = {
      message: text,
      symbol: state.symbol,
      timeframe: state.timeframe,
      mode: state.mode,
    };
    if (state.mode === 'replay') {
      // Время последней видимой свечи — чтобы анализ не видел будущее.
      const arr = (state.activeCandles && state.activeCandles.length)
        ? state.activeCandles : null;
      const c = arr ? arr[arr.length - 1]
        : (state.candles && state.candles[state.replay.index]) || null;
      const t = c && c.time != null ? Math.round(c.time) : null;
      if (t != null) body.replay_time = t;
    }
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const r = await resp.json();
    if (!resp.ok) throw new Error(r.error || ('HTTP ' + resp.status));
    appendMessage('ai', r.reply || '…');
    if (r.drawings && r.drawings.length && state.dm) {
      state.dm.refresh(r.all_drawings || r.drawings);
      updateAiDrawingsUI();
    }
  } catch (e) {
    appendMessage('ai', '⚠ Ошибка: ' + e.message);
  } finally {
    _busy = false;
    if (btn) btn.disabled = false;
  }
}

import { updateAiDrawingsUI } from './analysis.js';
