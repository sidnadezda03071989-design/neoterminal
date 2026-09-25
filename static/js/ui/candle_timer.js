/* Таймер до закрытия свечи в нижнем таймере.
   Тикает локально раз в секунду (без сети): берёт время последней свечи
   из state.candles и длительность ТФ из config.TF_SECONDS.
   Отображается только мини-таймер на правой ценовой шкале (#price-timer):
   позиция обновляется по candleSeries.priceToCoordinate(close) каждый кадр. */

import { state } from '../state.js';
import { TF_SECONDS } from '../config.js';
import { candleSeries, container } from '../chart/setup.js';

const $ = (id) => document.getElementById(id);
// #price-timer лежит в .chart-wrap (position:relative); #chart-container —
// absolute внутри него (справа оставлена полоса кнопок), поэтому
// координата pane-y (цена) по-прежнему совпадает с top в chart-wrap.
const _wrap = $('chart-wrap') || container;

let _timer = null;
let _raf = null;           // rAF-цикл репозиции чипа на ценовой шкале
let _lastLeft = null;      // последний корректный остаток, для репозиции чипа
let _cachedTop = null;     // кэш top — не трогаем DOM, если позиция не изменилась
let _baseOffsetSec = 0;    // поправка часов: (now источника) − (now машины), сек

function _pad(n) { return String(n).padStart(2, '0'); }

/* Синхронизация таймера с «сейчас» сервера в эпохе источника данных:
   крипта — часы Binance (могут отставать от машины на ~16с!), форекс — часы
   сервера. Сервер шлёт `now` в SSE candle_update и в /api/last-bar, /api/data.
   _baseOffsetSec = serverNow − localNow; тикаем локально, но с поправкой. */
export function syncServerClock(serverNowSec) {
  const s = Number(serverNowSec);
  if (!Number.isFinite(s) || s <= 0) return;
  _baseOffsetSec = s - Date.now() / 1000;
}

function _nowSec() {
  return Date.now() / 1000 + _baseOffsetSec;
}

/* «MM:SS» (или «HH:MM:SS» для длинных ТФ). */
export function formatRemainder(sec) {
  const s = Math.max(0, Math.floor(sec));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = s % 60;
  return h > 0 ? h + ':' + _pad(m) + ':' + _pad(ss) : _pad(m) + ':' + _pad(ss);
}

/* Время (epoch-сек) закрытия бара с данным временем старта. */
export function barCloseTime(barTime) {
  const dur = TF_SECONDS[state.timeframe] || 60;
  return (barTime || 0) + dur;
}

function _tick() {
  const act = state.activeCandles && state.activeCandles.length
    ? state.activeCandles
    : state.candles;
  if (!act || !act.length || state.mode !== 'live') {
    _lastLeft = null;
    return;
  }
  const last = act[act.length - 1];
  if (!last || !Number.isFinite(Number(last.time))) {
    _lastLeft = null;
    return;
  }
  const nowSec = _nowSec();
  const close = barCloseTime(Number(last.time));
  const left = close - nowSec;
  if (left <= 0 || left > (TF_SECONDS[state.timeframe] || 60) + 1) {
    // До смены свечи «дожили» (или часы съехали) — до новой свечи
    // data/live.js price-timer будет скрыт.
    _lastLeft = null;
    return;
  }
  _lastLeft = left;
}

/* Позиционирует мини-таймер на правой ценовой шкале у цены последней свечи.
   Показываем только в live с валидным остатком И когда цена видна в
   видимой области прайс-шкалы (priceToCoordinate не null). */
function _placePriceChip() {
  const el = $('price-timer');
  if (!el) return;
  const act = state.activeCandles && state.activeCandles.length
    ? state.activeCandles
    : state.candles;
  const left = _lastLeft;
  const bad = state.mode !== 'live' || !act || !act.length
    || left == null || left <= 0;
  if (bad) {
    if (el.style.display !== 'none') { el.textContent = ''; el.style.display = 'none'; }
    return;
  }
  const last = act[act.length - 1];
  if (!last || typeof last.time !== 'number') {
    if (el.style.display !== 'none') { el.textContent = ''; el.style.display = 'none'; }
    return;
  }
  let y = null;
  try { y = candleSeries.priceToCoordinate(Number(last.close)); } catch (e) {}
  if (y == null || !Number.isFinite(y)) {
    if (el.style.display !== 'none') { el.textContent = ''; el.style.display = 'none'; }
    return;
  }
  // Чуть ниже точки цены (label самой цены выше него), но не за нижний край.
  let top = y + 18;
  const wrapH = _wrap.clientHeight || 400;
  const elH = el.offsetHeight || 19;
  if (top + elH > wrapH - 4) top = Math.max(0, wrapH - elH - 4);
  if (top !== _cachedTop) {
    el.style.top = top + 'px';
    _cachedTop = top;
  }
  if (el.textContent !== '⏱ ' + formatRemainder(left)) {
    el.textContent = '⏱ ' + formatRemainder(left);
  }
  if (el.style.display !== 'block') el.style.display = 'block';
}

function _rafLoop() {
  _placePriceChip();
  _raf = requestAnimationFrame(_rafLoop);
}

/* Запустить тикер (идемпотентно). */
export function startCandleTimer() {
  if (_timer) clearInterval(_timer);
  _tick();
  _timer = setInterval(_tick, 1000);
  if (_raf == null) _raf = requestAnimationFrame(_rafLoop);
}

export function stopCandleTimer() {
  if (_timer) { clearInterval(_timer); _timer = null; }
  if (_raf != null) { cancelAnimationFrame(_raf); _raf = null; }
  _cachedTop = null;
}
