import { state } from '../state.js';
import {
  HISTORY_LIMIT, INITIAL_CANDLES, FULL_CANDLES, PROGRESSIVE_LOAD_ENABLED,
  PROGRESSIVE_STEP, PROGRESSIVE_STEP_DELAY_MS,
} from '../config.js';
import { setAllData, showChartLoading, hideChartLoading, withPreservedView } from '../chart/series.js';

const $ = (id) => document.getElementById(id);

export function replaySetData(index) {
  const idx = Math.max(0, Math.min(index, state.candles.length));
  const candles = state.candles.slice(0, idx);
  const ind = {};
  for (const k of Object.keys(state.ind || {})) ind[k] = state.ind[k].slice(0, idx);
  state.activeCandles = candles;
  setAllData(candles, ind);
  updateReplayPos(idx);
}

export function updateReplayPos(idx) {
  const pos = $('replay-pos-text');
  const slider = $('progress-slider');
  if (pos) pos.textContent = idx + ' / ' + state.replay.total;
  if (slider) slider.value = String(idx);
}

let _loadSeq = 0;  // номер загрузки replay (отмена устаревших фоновых подмен)

async function _fetchReplayData(from, to, limit) {
  const url = '/api/replay-data?symbol=' + encodeURIComponent(state.symbol)
    + '&timeframe=' + encodeURIComponent(state.timeframe)
    + '&from=' + encodeURIComponent(from)
    + '&to=' + encodeURIComponent(to)
    + '&limit=' + limit;
  const resp = await fetch(url);
  if (!resp.ok) throw new Error('HTTP ' + resp.status);
  return resp.json();
}

export async function loadReplay() {
  stopReplay();
  showChartLoading();
  const from = ($('from-date') || {}).value || '';
  const to = ($('to-date') || {}).value || '';
  const fullLimit = HISTORY_LIMIT[state.timeframe] || FULL_CANDLES;
  const initialLimit = Math.min(INITIAL_CANDLES, fullLimit);
  _loadSeq++;
  const seq = _loadSeq;
  try {
    // Шаг 1: небольшой лимит → быстрый ответ → отрисовка.
    const data = await _fetchReplayData(from, to, initialLimit);
    if (!data.candles || !data.candles.length) {
      state.replay.total = 0;
      updateReplayPos(0);
      return;
    }
    state.candles = data.candles;
    state.ind = data.indicators;
    state.replay.total = data.candles.length;
    state.replay.index = 0;
    const slider = $('progress-slider');
    if (slider) { slider.max = String(Math.max(1, state.replay.total - 1)); slider.value = '0'; }
    replaySetData(0);
    // Шаг 2 (фон): догрузка полного объёма и подмена без потери вида.
    if (PROGRESSIVE_LOAD_ENABLED && initialLimit < fullLimit) {
      _loadReplayFullInBackground(seq, from, to, fullLimit);
    }
  } catch (e) {
    console.error('loadReplay:', e);
  } finally {
    hideChartLoading();  // гасим спиннер всегда, включая пустые данные и ошибки
  }
}

// Шаг 2 прогрессивной загрузки replay: итеративная догрузка чанками по
// PROGRESSIVE_STEP (backend докачивает только дельту). После каждого чанка
// подмена с сохранением вида, total и slider.max обновляются на каждом шаге.
async function _loadReplayFullInBackground(seq, from, to, fullLimit) {
  let currentLen = state.candles.length;
  try {
    for (
      let target = Math.min(INITIAL_CANDLES + PROGRESSIVE_STEP, fullLimit);
      target <= fullLimit;
      target += PROGRESSIVE_STEP
    ) {
      // Гонка: replay уже перезагрузили с другими параметрами — фон устарел.
      if (seq !== _loadSeq) return;
      const data = await _fetchReplayData(from, to, target);
      if (seq !== _loadSeq) return;
      if (!data.candles || !data.candles.length) break;
      // Нет прироста относительно уже нарисованного — дальше смысла нет
      // (покрывает и случай length === currentLen).
      if (data.candles.length <= currentLen) {
        console.debug('progressive(replay): stop at ' + data.candles.length + ' (source limit)');
        break;
      }
      const offset = Math.max(0, data.candles.length - currentLen);
      state.candles = data.candles;
      state.ind = data.indicators;
      state.replay.total = data.candles.length;  // total обновляется после каждого шага
      // Индекс ползунка должен указывать на те же бары, что были видны до подмены.
      state.replay.index = Math.min(state.replay.index + offset, state.replay.total - 1);
      const slider = $('progress-slider');
      if (slider) slider.max = String(Math.max(1, state.replay.total - 1));
      // Подмена без потери зума/скролла (добавленные бары лежат слева).
      withPreservedView(() => { replaySetData(state.replay.index); }, offset);
      currentLen = data.candles.length;
      console.debug('progressive(replay): ' + currentLen + '/' + fullLimit);
      // Чанк применён; если источник отдал меньше target (потолок истории),
      // следующие запросы будут холостыми — останавливаемся.
      if (data.candles.length < target) {
        console.debug('progressive(replay): stop at ' + currentLen + ' (source limit)');
        break;
      }
      await new Promise((r) => setTimeout(r, PROGRESSIVE_STEP_DELAY_MS));
    }
  } catch (e) {
    // Ошибка шага: просто прерываем цикл — уже нарисованные чанки остаются.
    console.warn('loadReplay(progressive):', e);
  }
}

export function stopReplay() {
  state.replay.playing = false;
  if (state.replay.timer) {
    clearInterval(state.replay.timer);
    state.replay.timer = null;
  }
}

export function playReplay() {
  if (state.replay.playing) return;
  if (state.replay.total <= 0 || !state.candles.length) return;
  if (state.replay.index >= state.replay.total - 1) state.replay.index = 0;
  state.replay.playing = true;
  const btn = $('play-btn'); if (btn) btn.classList.add('active');
  const tick = () => {
    if (state.replay.index >= state.replay.total - 1) {
      pauseReplay();
      return;
    }
    state.replay.index += 1;
    replaySetData(state.replay.index);
  };
  state.replay.timer = setInterval(tick, 1000 / state.replay.speed);
}

export function pauseReplay() {
  stopReplay();
  const btn = $('play-btn'); if (btn) btn.classList.remove('active');
}

export function resetReplay() {
  stopReplay();
  state.replay.index = 0;
  replaySetData(0);
}

export function replayStepBack() { pauseReplay(); if (state.replay.index > 0) { state.replay.index -= 1; replaySetData(state.replay.index); } }
export function replayStepForward() { pauseReplay(); if (state.replay.index < state.replay.total - 1) { state.replay.index += 1; replaySetData(state.replay.index); } }
export function replayStepBack10() { pauseReplay(); state.replay.index = Math.max(0, state.replay.index - 10); replaySetData(state.replay.index); }
export function replayStepForward10() { pauseReplay(); state.replay.index = Math.min(state.replay.total - 1, state.replay.index + 10); replaySetData(state.replay.index); }

export function applyReplayPreset(preset) {
  const now = new Date();
  const to = now.toISOString().slice(0, 10);
  let from;
  switch (preset) {
    case '30d': from = new Date(now.getTime() - 30 * 86400000); break;
    case '3m':  from = new Date(now.getTime() - 90 * 86400000); break;
    case '1y':  from = new Date(now.getTime() - 365 * 86400000); break;
    case '2y':  from = new Date(now.getTime() - 730 * 86400000); break;
    default:    from = new Date(now.getTime() - 7300 * 86400000);
  }
  const fd = $('from-date'); const td = $('to-date');
  if (td) td.value = to;
  if (fd) fd.value = from.toISOString().slice(0, 10);
  loadReplay();
}
