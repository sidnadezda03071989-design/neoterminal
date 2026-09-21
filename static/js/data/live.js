import { state } from '../state.js';
import {
  HISTORY_LIMIT, POLL_INTERVAL_MS, POLL_FULL_RELOAD_MS,
  INITIAL_CANDLES, FULL_CANDLES, PROGRESSIVE_LOAD_ENABLED,
  PROGRESSIVE_STEP, PROGRESSIVE_STEP_DELAY_MS,
} from '../config.js';
import {
  setAllData, updateAllLast, showChartLoading, hideChartLoading,
  withPreservedView,
} from '../chart/series.js';
import { updateLegend } from '../ui/legend.js';
import { sseConnect } from '../integration/sse.js';

const $ = (id) => document.getElementById(id);

// Быстрый поллинг: раз в POLL_INTERVAL_MS (из config.js) тянем только последний
// бар (/api/last-bar, без индикаторов), а полный /api/data — раз в
// POLL_FULL_RELOAD_MS (страховка от рассинхрона) — счётчик тиков.
const _FULL_RELOAD_EVERY = Math.max(1, Math.round(POLL_FULL_RELOAD_MS / POLL_INTERVAL_MS));

let _liveKey = '';    // "symbol|tf", к которому привязаны state.candles / state.ind
let _pollCount = 0;   // тиков быстрого поллинга с момента последнего полного reload
let _bgSeq = 0;       // номер фоновой прогрессивной загрузки (отмена устаревших)

export function setStatus(ok, text) {
  const dot = $('pulse-dot');
  const txt = $('status-text');
  if (!dot || !txt) return;
  dot.className = 'pulse-dot ' + (ok ? 'ok' : 'warn');
  txt.textContent = text || (ok ? 'live' : '');
}

/* Индикатор «загрузка: N/полный» в топбаре (рядом со статусом live). */
function _setLoadStatus(show, len, total) {
  const el = $('data-load-status');
  if (!el) return;
  if (!show) { el.style.display = 'none'; return; }
  el.style.display = 'inline-block';
  if (len != null && total != null) {
    el.textContent = `загрузка: ${len}/${total}`;
  }
}

export async function loadLive(fit) {
  if (state.mode !== 'live') return;  // фон не должен переписать данные реплея
  setStatus(true, 'загрузка…');
  const fullLimit = HISTORY_LIMIT[state.timeframe] || FULL_CANDLES;
  /* BLOCK-33: fit=true — это первый load / смена symbol или TF: рисуем
     быстро малую порцию (INITIAL_CANDLES), зум по умолчанию сбрасывается.
     fit=false — полный reload ТОГО ЖЕ symbol/tf (pollLive): тянем сразу весь
     объём (иначе подмена 20000→1000 свечей обрубит историю) и применяем в
     withPreservedView — видимый zoom/скролл не сбрасывается. */
  const initialLimit = fit
    ? Math.min(INITIAL_CANDLES, fullLimit)
    : fullLimit;
  // Новая фоновая загрузка отменяет предыдущую (устаревшая подмена запрещена).
  _bgSeq++;
  const seq = _bgSeq;
  try {
    // Шаг 1: небольшой лимит → сервер отвечает быстро → график виден за ~2с.
    const url = `/api/data?symbol=${encodeURIComponent(state.symbol)}&timeframe=${encodeURIComponent(state.timeframe)}&limit=${initialLimit}`;
    const resp = await fetch(url);
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    const data = await resp.json();
    if (!data.candles || !data.candles.length) {
      setStatus(false, 'нет данных');
      return;
    }
    state.candles = data.candles;
    state.ind = data.indicators;
    state.activeCandles = data.candles;
    if (fit) {
      // Первый load/смена symbol или TF — однократный fit (последние ~200
      // баров, авто-цена). Дальше график сам не двигается (BLOCK-34).
      state.chartNeedsFit = true;
      setAllData(data.candles, data.indicators);
    } else {
      // Полный reload того же symbol — сохраняем видимый zoom/скролл.
      withPreservedView(() => setAllData(data.candles, data.indicators));
    }
    setStatus(true, state.symbol + ' · live');
    updateLegend(null);
    state.first = false;
    // Полная загрузка фиксирует ключ и сбрасывает счётчик быстрых тиков.
    _liveKey = state.symbol + '|' + state.timeframe;
    _pollCount = 0;
    hideChartLoading();
    // Если полный объём уже получен шагом 1 — фоновая догрузка не нужна.
    state.progressiveLoaded = initialLimit >= fullLimit;
    // Шаг 2 (фон, если включён и полный объём больше начального):
    // не блокирует UI и не блокирует возврат из loadLive.
    if (PROGRESSIVE_LOAD_ENABLED && initialLimit < fullLimit) {
      _setLoadStatus(true, state.candles.length, fullLimit);
      _loadFullInBackground(seq, _liveKey, fullLimit);
    } else {
      _setLoadStatus(false);
    }
  } catch (e) {
    console.error('loadLive:', e);
    setStatus(false, 'ошибка данных');
    hideChartLoading();
    _setLoadStatus(false);
    // Разрешаем pollLive восстановиться полным reload.
    state.progressiveLoaded = true;
  }
}

// Шаг 2 прогрессивной загрузки: итеративная догрузка чанками по PROGRESSIVE_STEP.
// Backend на запрос с бóльшим limit докачивает только дельту (1 страница Binance
// на шаг), поэтому 5000→10000→15000→20000 — это 4 дешёвых запроса. После каждого
// чанка данные подменяются с сохранением видимого окна (zoom/скролл не сбрасываются
// и история «дорисовывается» влево на глазах), статус в топбаре обновляется.
// state.progressiveLoaded=true сигнализирует pollLive, что полный reload снова
// безопасен.
async function _loadFullInBackground(seq, key, fullLimit) {
  state.progressiveLoaded = false;
  let currentLen = state.candles.length;
  const step = Math.max(1, Math.round(PROGRESSIVE_STEP));
  const chunks = Math.max(1, Math.ceil(fullLimit / step));
  _setLoadStatus(true, currentLen, fullLimit);
  console.log(`[progressive] starting 0 → ${fullLimit} in ${chunks} chunks`);
  try {
    for (let target = step; target <= fullLimit; target += step) {
      // Гонка: символ/tf сменили, ушёл в replay/live-режим или уже стартовала
      // новая загрузка — фон устарел. Особенно важно не догружать чанки в
      // REPLAY-режиме: live-подмена переписывала state.candles на полный объём
      // и «топила» барьер реплея за экраном (view уезжала к концу данных).
      if (seq !== _bgSeq || key !== _liveKey || state.mode !== 'live') return;
      const url = `/api/data?symbol=${encodeURIComponent(state.symbol)}&timeframe=${encodeURIComponent(state.timeframe)}&limit=${target}`;
      const resp = await fetch(url);
      if (!resp.ok) throw new Error('HTTP ' + resp.status);
      const data = await resp.json();
      if (seq !== _bgSeq || key !== _liveKey || state.mode !== 'live') return;
      if (!data.candles || !data.candles.length) break;
      // Нет прироста относительно уже нарисованного — дальше смысла нет
      // (покрывает и случай length === currentLen).
      if (data.candles.length <= currentLen) {
        console.debug('progressive: stop at ' + data.candles.length + ' (source limit)');
        break;
      }
      // offset = дельта длины: добавленные бары лежат слева, видимое окно сдвигается.
      const offset = data.candles.length - currentLen;
      withPreservedView(() => {
        state.candles = data.candles;
        state.ind = data.indicators;
        state.activeCandles = data.candles;
        setAllData(data.candles, data.indicators);
      }, offset);
      updateLegend(null);
      currentLen = data.candles.length;
      _pollCount = 0;
      _setLoadStatus(true, currentLen, fullLimit);
      // Чанк применён; если источник отдал меньше target (потолок истории),
      // следующие запросы будут холостыми — останавливаемся.
      if (data.candles.length < target) {
        console.debug('progressive: stop at ' + currentLen + ' (source limit)');
        break;
      }
      await new Promise((r) => setTimeout(r, PROGRESSIVE_STEP_DELAY_MS));
    }
  } catch (e) {
    // Ошибка шага: прерываем цикл, но pollLive не должен застрять.
    console.warn('loadLive(progressive):', e);
  } finally {
    state.progressiveLoaded = true;
    _setLoadStatus(false);
  }
}

export async function pollLive() {
  /* BLOCK-36: на время рендера сделок бэктеста (fetch + render + zoom)
     полл заморожен — иначе обновление данных сбрасывает видимое окно
     (view) посреди setVisibleRange. Разморозка через 2 сек в scanner.js. */
  if (state._suspendPoll) {
    console.log('[live] poll suspended during render');
    return;
  }
  if (state.mode !== 'live') return;

  const key = state.symbol + '|' + state.timeframe;

  // Первый полл и смена symbol/tf — полный loadLive() (как и раньше).
  if (!state.candles.length || _liveKey !== key) {
    await loadLive(true);
    return;
  }

  // Страховка от рассинхрона: полный reload раз в POLL_FULL_RELOAD_MS.
  // Пока фоновая прогрессивная догрузка не завершилась, полный reload не
  // дёргаем — иначе он перезапустит загрузку и сотрёт ещё не подменённые данные.
  if (_pollCount >= _FULL_RELOAD_EVERY) {
    if (state.progressiveLoaded === true) {
      _pollCount = 0;
      // BLOCK-33: reload того же symbol/tf — с сохранением zoom (fit=false),
      // иначе каждые POLL_FULL_RELOAD_MS видимый диапазон сбрасывался.
      await loadLive(false);
      return;
    }
    // откладываем: полный reload сделает один из следующих тиков после фона
  }
  _pollCount++;

  try {
    const url = `/api/last-bar?symbol=${encodeURIComponent(state.symbol)}&timeframe=${encodeURIComponent(state.timeframe)}`;
    const resp = await fetch(url);
    if (!resp.ok) return;
    const data = await resp.json();
    if (state.mode !== 'live') return;  // ушли в реплей — ответ живого полла не применяем
    const bar = data && data.candle;
    if (!bar) return;

    const candles = state.candles;
    const last = candles[candles.length - 1];
    const ind = data.indicators || null;
    if (last && bar.time > last.time) {
      // Новый бар: дописываем его и свежие индикаторы (с /api/last-bar).
      candles.push(bar);
      if (ind) {
        for (const k of Object.keys(state.ind || {})) {
          const arr = state.ind[k];
          if (Array.isArray(arr) && k in ind && ind[k] != null) {
            arr.push({ time: bar.time, value: ind[k] });
          }
        }
      }
      state.activeCandles = candles;
      state.candlesPrevLength = candles.length;
    } else if (last && bar.time === last.time) {
      Object.assign(last, bar);  // текущий бар тикает — держим state в синхроне
      // Тот же бар: заменяем последние точки индикаторов свежими значениями.
      if (ind) {
        for (const k of Object.keys(state.ind || {})) {
          const arr = state.ind[k];
          if (Array.isArray(arr) && arr.length && k in ind && ind[k] != null) {
            arr[arr.length - 1] = { time: bar.time, value: ind[k] };
          }
        }
      }
    } else {
      return;  // bar.time < last.time (устаревший ответ) — не откатываем график
    }

    // Точечное обновление: updateAllLast() внутри вызывает candleSeries.update(last),
    // обновляет volume и последние точки индикаторов. Полного setAllData нет,
    // поэтому 18 индикаторов × HISTORY_LIMITS[tf] строк не пересчитываются.
    updateAllLast(candles, state.ind || {});
    setStatus(true, state.symbol + ' · live');
  } catch (e) {
    setStatus(false, 'сеть потеряна');
  }
}

export function startLivePolling() {
  stopLivePolling();
  state.livePollTimer = setInterval(pollLive, POLL_INTERVAL_MS);
  sseConnect();
}

export function stopLivePolling() {
  if (state.livePollTimer) {
    clearInterval(state.livePollTimer);
    state.livePollTimer = null;
  }
}
