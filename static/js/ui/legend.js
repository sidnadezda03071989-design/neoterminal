import { state } from '../state.js';
import { candleSeries, smaSeries, emaSeries } from '../chart/setup.js';
import { formatInTz, activeOffsetSeconds } from './timezone.js';

const $ = (id) => document.getElementById(id);
const COLORS = { up:'#26a69a', down:'#ef5350', sma:'#2196f3', ema:'#ff9800' };

function fmt(n, digits) {
  if (n == null || !isFinite(n)) return '—';
  return Number(n).toFixed(digits == null ? (Math.abs(n) >= 1000 ? 2 : 5) : digits);
}

/* Метка времени свечи в легенде — в активном часовом поясе (как в TV). */
function fmtBarTime(sec) {
  if (!sec) return '';
  const d = formatInTz(sec, {
    year: '2-digit', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', hour12: false,
  });
  const off = activeOffsetSeconds();
  const sign = off < 0 ? '−' : '+';
  const abs = Math.abs(off);
  const utc = off === 0 ? 'UTC'
    : 'UTC' + sign + Math.floor(abs / 3600) + ':'
      + String(Math.floor((abs % 3600) / 60)).padStart(2, '0');
  return `<span class="lg-item lg-time">${d} <span class="lg-utc">${utc}</span></span>`;
}

export function updateLegend(crossParam) {
  const el = $('chart-legend');
  if (!el) return;
  let c = null, sma = null, ema = null;
  if (crossParam && crossParam.seriesData) {
    c = crossParam.seriesData.get(candleSeries);
    const sv = crossParam.seriesData.get(smaSeries);
    const ev = crossParam.seriesData.get(emaSeries);
    if (sv) sma = sv.value;
    if (ev) ema = ev.value;
  }
  if (!c) {
    const act = state.activeCandles || state.candles;  // в replay — активное окно
    if (act && act.length) c = act[act.length - 1];
  }
  if (!c) { el.innerHTML = ''; return; }
  const isUp = c.close >= c.open;
  const sign = isUp ? '+' : '';
  const upColor = isUp ? COLORS.up : COLORS.down;
  el.innerHTML =
    `<span class="lg-item" style="font-weight:700">${state.symbol} ${state.timeframe}</span>` +
    fmtBarTime(c.time) +
    `<div class="lg-item">${fmt(c.open,5)}</div>` +
    `<div class="lg-item">${fmt(c.high,5)}</div>` +
    `<div class="lg-item">${fmt(c.low,5)}</div>` +
    `<div class="lg-item" style="color:${upColor};font-weight:600">${sign}${fmt(c.close,5)}</div>` +
    `<span class="lg-item"><span class="lg-swatch" style="background:${COLORS.sma}"></span>SMA20 ${sma != null ? fmt(sma,5) : ''}</span>` +
    `<span class="lg-item"><span class="lg-swatch" style="background:${COLORS.ema}"></span>EMA50 ${ema != null ? fmt(ema,5) : ''}</span>`;
}
