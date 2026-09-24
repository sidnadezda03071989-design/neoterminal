import { state } from '../state.js';
import {
  HISTORY_LIMIT, POLL_INTERVAL_MS, POLL_FULL_RELOAD_MS,
  INITIAL_CANDLES, FULL_CANDLES, PROGRESSIVE_LOAD_ENABLED,
  PROGRESSIVE_STEP, PROGRESSIVE_STEP_DELAY_MS, TF_SECONDS,
} from '../config.js';
import {
  setAllData, updateAllLast, showChartLoading, hideChartLoading,
  withPreservedView,
} from '../chart/series.js';
import { updateLegend } from '../ui/legend.js';
import { sseConnect } from '../integration/sse.js';
import { startCandleTimer, syncServerClock } from '../ui/candle_timer.js';

const $ = (id) => document.getElementById(id);

// Быстрый поллинг: раз в POLL_INTERVAL_MS (из config.js) тянем только последний
// бар (/api/last-bar, без индикаторов), а полный /api/data — раз в
// POLL_FULL_RELOAD_MS (страховка от рассинхрона) — счётчик тиков.
const _FULL_RELOAD_EVERY = Math.max(1, Math.round(POLL_FULL_RELOAD_MS / POLL_INTERVAL_MS));

let _liveKey = '';    // "symbol|tf", к которому привязаны state.candles / state.ind
let _pollCount = 0;   // тиков быстрого поллинга с момента последнего полного reload
let _bgSeq = 0;       // номер фоновой прогрессивной загрузки (отмена устаревших)
let _loadingSeq = 0;  // номер loadLive: ответы старых запросов не должны
                       // «выигрывать» у более нового (иначе setAllData получает
                       // устаревшие OHLC и последняя свеча дёргается/меняет форму).
let _RELOAD_IN_FLIGHT = false; // защита от параллельных полных reload'ов
let _freshLiveBar = null;      // последний свежий live-бар {time,OHLCV} из
                               // SSE/last-bar; нужен, чтобы setAllData (граница
                               // минуты/фоновая догрузка) не «откатил»/не съел
                               // уже нарисованный бар отставшим ответом /api/data.

// Защита от «пробела»: когда новый бар приходит с ПРЫЖКОМ больше 1 интервала
// (напр. пропущена 1m-свеча, gap 2 минуты), нельзя просто push — lightweight-charts
// сожмёт дыру и создаст визуальный разрыв «свеча пропала → потом снова».
// Тогда делаем полный reload /api/data, который догрузит пропущенные бары и
// пересчитает индикаторы. _gapReloadGuard не даёт зациклиться (реload сам роняет
// живой бар в следующий poll).
let _gapReloadGuard = 0; // timestamp lastTriggered (ms) — кулдаун 3с
let _barSyncTimer = null; // отложенный reload последнего бара (/api/last-bar)
let _barSyncRunning = false; // не допускать параллельных sync-запросов

/* Добавить НОВЫЙ бар (SSE или /api/last-bar) с защитой от gap.
   Возвращает true, если бар применён (push) / дид reload; false — если нет. */
function _pushNewBar(bar, ind) {
  const candles = state.candles;
  if (!Array.isArray(candles) || candles.length === 0) return false;
  const last = candles[candles.length - 1];
  if (!last || typeof last.time !== 'number') return false;

  // Локальное время бара: иначе (часто на длинных/дробных интервалах) новый бар
  // может попасть не в хвост массива. updateAllLast() берёт ровно candles[-1],
  // и chart получает последнюю «старую» свечу вместо свежего бара.
  const t = Number(bar.time);
  if (!Number.isFinite(t) || t <= last.time) return false;
  bar = { ...bar, time: Math.floor(t) };

  const step = TF_SECONDS[state.timeframe] || 60;
  const delta = bar.time - last.time;
  // Прыжок не = 1 интервалу (пропало SSE-событие / сервер молчал на границе
  // минут) — НО бар всё равно рисуем: не «прячем» свечу. Заполнить дыру
  // попросим полный reload в фоне (см. ниже), когда он действительно нужен.
  const isGap = delta !== step && delta > 0;
  const doReload = isGap && state.mode === 'live' && (Date.now() - _gapReloadGuard) > 3000;
  if (doReload) {
    _gapReloadGuard = Date.now();
    console.warn(`[live] gap ${delta}s != step ${step}s — bar drawn, backfill reload`, bar.time, last.time);
    _pollCount = _FULL_RELOAD_EVERY;  // следующий pollLive сделает loadLive(false)
  }

  // Всегда рисуем новый бар (в т.ч. с gap), чтобы не терять свечу на графике.
  candles.push(bar);
  if (ind) {
    for (const k of Object.keys(state.ind || {})) {
      const arr = state.ind[k];
      if (Array.isArray(arr) && k in ind && ind[k] != null) {
        arr.push({ time: bar.time, value: ind[k] });
      }
    }
  }
  return true;
}

/* Регистрируем открытую пару на сервере: SSE-пуши шлются только по ней.
   Иначе сервер заливает события по всем 18 парам, очередь клиента
   переполняется и свечи на графике отстают/пропадают. */
function _registerLivePair() {
  try {
    fetch('/api/live/subscribe', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ symbol: state.symbol, timeframe: state.timeframe }),
    }).catch(() => {});
  } catch (e) { /* noop: подписка не критична */ }
}

/* Умный доскролл к правому краю: НЕ двигаем график, если пользователь
   прокрутил в историю (иначе каждый новый бар «плассит» видимую область
   влево). Скроллим только когда видимый правый край близок к последнему
   бару (т.е. юзер смотрит живой конец). chart сюда не импортируем —
   берём из state.chart (выставлен в main.js), чтобы не плодить зависимости. */
function _scrollToRealTime() {
  const chart = state.chart;
  if (!chart) return;  // chart ещё не готов
  const candles = state.candles;
  if (!Array.isArray(candles) || candles.length === 0) return;
  const last = candles[candles.length - 1];
  if (!last || typeof last.time !== 'number') return;
  const step = TF_SECONDS[state.timeframe] || 60;
  let to = null;
  try {
    // getVisibleRange() возвращает {from,to} в ЕДИНИЦАХ ВРЕМЕНИ (epoch, как
    // time последнего бара). getVisibleLogicalRange() даёт логический индекс
    // (тысячи), который < epoch (~1.7млрд) — сравнение с edge всегда было
    // ложным, и доскролл к реальному времени не срабатывал.
    const vis = chart.timeScale().getVisibleRange();
    if (vis && typeof vis.to === 'number') to = vis.to;
  } catch (e) {}
  // Если не удалось узнать видимое окно — оставляем как было (не скроллим,
  // чтобы не «плассить» по умолчанию).
  if (to === null) return;
  const edge = last.time - step * 0.45;
  // Правый край видимого окна близок к последнему бару -> юзер смотрит живой конец.
  if (to >= edge) chart.timeScale().scrollToRealTime();
}

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
  _registerLivePair();  // сервер будет пушить SSE только по этой паре
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
  const loadId = ++_loadingSeq;
  const wasInFlight = _RELOAD_IN_FLIGHT;
  _RELOAD_IN_FLIGHT = true;
  try {
    // Шаг 1: небольшой лимит → сервер отвечает быстро → график виден за ~2с.
    const url = `/api/data?symbol=${encodeURIComponent(state.symbol)}&timeframe=${encodeURIComponent(state.timeframe)}&limit=${initialLimit}`;
    const resp = await fetch(url);
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    const data = await resp.json();
    // Синхронизируем часы таймера по «сейчас» сервера (эпоха источника).
    syncServerClock(data && data.now);
    // Ответ устарел (пока шёл fetch, стартовал более новый loadLive) — не
    // применяем: иначе setAllData(data.candles) затрёт более свежий хвост
    // и последняя свеча «дрожит»/меняет форму.
    if (loadId !== _loadingSeq) return;
    if (state.mode !== 'live') return;  // ушли в реплей во время fetch
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
    // Хвост мог обновиться из SSE во время fetch — вернуть свежий бар в
    // state.candles и на график, чтобы setAllData не «откатил» последнюю свечу.
    _reapplyFreshLiveBar();
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
  } finally {
    if (!wasInFlight) _RELOAD_IN_FLIGHT = false;
  }
}

/* Обновляет/добавляет хвостовой LIVE-бар у state.candles и рисует только его.

   Главный баг-фикс: раньше SSE/last-bar мержил новый бар в state.candles,
   но updateAllLast() брал candles[-1] и мог отрисовать уже готовый бар.
   Если GET /api/data (полный reload или прогрессивная догрузка) стартовал ДО
   мержа и завершился ПОСЛЕ, setAllData(data.candles) затирал уже отрисованный
   новый бар данными кеша (без него) — свеча «прыгала», меняла форму или
   пропадала. Этот свежий бар дописывается в state.candles и state.ind, чтобы
   setAllData/индикаторы уже учитывали его, а сам бар сразу рисуется точечно.
   Вызывать из applyLiveCandleUpdate / pollLive после _pushNewBar / _mergeBar. */
function _applyFreshLiveBar(bar, { pushed = false, indicators = null } = {}) {
  if (!bar || state.mode !== 'live') return;
  const candles = state.candles;
  if (!Array.isArray(candles) || candles.length === 0) return;
  const last = candles[candles.length - 1];
  if (!last || last.time !== bar.time) return;
  if (indicators) _replaceIndicatorTail(bar, indicators);
  state.activeCandles = candles;
  state.candlesPrevLength = candles.length;
  if (pushed) _scrollToRealTime();
  updateAllLast(candles, state.ind || {});
}

/* Заменяет/добавляет последнюю точку каждой серии индикатора значениями
   последнего live-бара. Раньше этот код был дублирован в pollLive — вынесен
   в общую функцию, чтобы SSE/last-bar обновляли хвост одинаково. */
function _replaceIndicatorTail(bar, ind) {
  if (!ind) return;
  for (const k of Object.keys(state.ind || {})) {
    const arr = state.ind[k];
    if (!Array.isArray(arr) || arr.length === 0) continue;
    if (!(k in ind) || ind[k] == null) continue;
    arr[arr.length - 1] = { time: bar.time, value: ind[k] };
  }
}

/* После setAllData (полный reload / прогрессивная догрузка) вернуть свежий
   live-бар, если он НОВЕЕ последнего, что пришло из /api/data. Без этого
   ответ кеша (отставший на тик) затирал уже нарисованную свечу — визуально
   свеча «меняет форму» или пропадает/появляется вновь. */
function _reapplyFreshLiveBar() {
  if (state.mode !== 'live' || !_freshLiveBar) return;
  const candles = state.candles;
  if (!Array.isArray(candles) || candles.length === 0) return;
  const tail = candles[candles.length - 1];
  if (!tail || typeof tail.time !== 'number') return;
  if (tail.time < _freshLiveBar.time) {
    candles.push({ ..._freshLiveBar });
    _syncIndicatorTail(_freshLiveBar);
    state.activeCandles = candles;
    state.candlesPrevLength = candles.length;
    updateAllLast(candles, state.ind || {});
    _scrollToRealTime();
  } else if (tail.time === _freshLiveBar.time) {
    _mergeBar(tail, _freshLiveBar);
    updateAllLast(candles, state.ind || {});
  }
}

/* Отложенный sync последнего бара после нового бара из /api/last-bar.
   Серверу нужно время финализировать индикаторы закрытого бара; тут лишь
   точечно подтягиваем /api/last-bar (без полного /api/data), чтобы не
   затирать свежий хвост отставшим ответом и не дёргать zoom. */
function _scheduleBarSync(bar) {
  if (!bar || state.mode !== 'live') return;
  if (_barSyncTimer) clearTimeout(_barSyncTimer);
  _barSyncTimer = setTimeout(() => { _barSyncTimer = null; Promise.resolve(_syncLastBarNow()).catch(() => {}); }, 1200);
}

function _syncLastBarNow() {
  if (state.mode !== 'live' || _barSyncRunning) return Promise.resolve();
  _barSyncRunning = true;
  return (async () => {
    try {
      const url = `/api/last-bar?symbol=${encodeURIComponent(state.symbol)}&timeframe=${encodeURIComponent(state.timeframe)}`;
      const resp = await fetch(url);
      if (!resp.ok) return;
      const data = await resp.json();
      syncServerClock(data && data.now);
      if (state.mode !== 'live') return;
      const bar = data && data.candle;
      if (!bar) return;
      const t = Math.floor(Number(bar.time));
      if (!Number.isFinite(t)) return;
      const freshBar = { ...bar, time: t };
      if (!_acceptsBar(freshBar)) return;
      const candles = state.candles;
      const last = candles[candles.length - 1];
      if (!last) return;
      if (freshBar.time > last.time) {
        _freshLiveBar = { ...freshBar };
        const pushed = _pushNewBar(freshBar, null);
        _syncIndicatorTail(freshBar);
        _applyFreshLiveBar(freshBar, { pushed });
      } else if (freshBar.time === last.time) {
        _mergeBar(last, freshBar);
        _freshLiveBar = { ...last };
        _applyFreshLiveBar(freshBar);
      }
    } catch (e) { /* noop: sync-сбой не критичен — догонит pollLive/SSE */ }
    finally { _barSyncRunning = false; }
  })();
}

/* Полный фоновый reload (шаг 2/прогрессивная догрузка): игнорирует устаревшие
   ответы fetch'а. Без seq ответ старого запроса мог примениться ПОСЛЕ свежего
   и setAllData(data.candles) мог откатить/«съесть» последнюю свечу. */
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
      // Ответ /api/data отстаёт от уже нарисованного хвоста — НЕ затираем им
      // более свежий state.candles (иначе последняя свеча «откатывается»/пропадает).
      if (!Array.isArray(state.candles) || state.candles.length === 0) break;
      const stateLast = state.candles[state.candles.length - 1];
      const respLast = data.candles[data.candles.length - 1];
      if (stateLast && respLast && respLast.time < stateLast.time) break;
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
      // Хвост мог быть обновлён/добавлен SSE во время fetch — дописываем в
      // state.candles и сразу рисуем, чтобы setAllData не «откатил» бар.
      _reapplyFreshLiveBar();
      updateLegend(null);
      currentLen = state.candles.length;
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
  // RELOAD_IN_FLIGHT: только один полный reload одновременно. Если ответ
  // подвис (лаг сеть), повторный loadLive без seq мог бы примениться позже
  // и затереть уже нарисованные новые свечи устаревшим ответом.
  // Пока идёт прогрессивная догрузка, полный reload откладываем: он
  // перезапустит загрузку и сотрёт ещё не подменённые чанки.
  if (_pollCount >= _FULL_RELOAD_EVERY) {
    if (state.progressiveLoaded === true && !_RELOAD_IN_FLIGHT) {
      _pollCount = 0;
      // BLOCK-33: reload того же symbol/tf — с сохранением zoom (fit=false),
      // иначе каждые POLL_FULL_RELOAD_MS видимый диапазон сбрасывался.
      await loadLive(false);
      return;
    }
    if (!_RELOAD_IN_FLIGHT) {
      // Не «прожигаем» тик без reload: если _RELOAD_IN_FLIGHT снялся
      // между проверками, следующий tick pollLive сразу сделает reload.
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
    syncServerClock(data && data.now);
    if (state.mode !== 'live') return;  // ушли в реплей — ответ живого полла не применяем
    const bar = data && data.candle;
    if (!bar) return;
    const barTime = Math.floor(Number(bar.time));
    if (!Number.isFinite(barTime)) return;
    // НЕ мутируем ответ fetch напрямую: тот же объект может лежать в
    // _freshLiveBar/очереди событий; правим локальную копию freshBar.
    const freshBar = { ...bar, time: barTime };
    // Отбрасываем ответ, «отставший» от уже нарисованного бара: иначе свеча
    // откатывается назад и мигает (см. _acceptsBar).
    if (!_acceptsBar(freshBar)) return;

    const candles = state.candles;
    const last = candles[candles.length - 1];
    const ind = data.indicators || null;
    if (last && freshBar.time > last.time) {
      // Новый бар: применяем с защитой от gap (_pushNewBar ВСЕГДА рисует бар;
      // gap лишь просит полный reload в фоне, не «пряча» свечу).
      _freshLiveBar = { ...freshBar };
      const pushed = _pushNewBar(freshBar, ind);
      _applyFreshLiveBar(freshBar, { pushed });
      _scheduleBarSync(freshBar);
    } else if (last && freshBar.time === last.time) {
      _mergeBar(last, freshBar);  // текущий бар тикает — накапливаем экстремумы
      _freshLiveBar = { ...last };
      // Тот же бар: заменяем последние точки индикаторов свежими значениями.
      if (ind) _replaceIndicatorTail(freshBar, ind);
      _applyFreshLiveBar(freshBar);
    } else {
      return;  // freshBar.time < last.time (устаревший ответ) — не откатываем график
    }

    setStatus(true, state.symbol + ' · live');
  } catch (e) {
    setStatus(false, 'сеть потеряна');
  }
}

/* Мгновенное применение свечи из SSE-события candle_update.
   Аналог тела pollLive, но без fetch: данные приходят push'ем с сервера
   (см. app_pkg/data/live.py, _sse_candle_push_loop).
   payload = {symbol, timeframe, candle:{time,open,high,low,close,volume}, closed}. */
export function applyLiveCandleUpdate(payload) {
  if (!payload || state.mode !== 'live') return;
  syncServerClock(payload.now);
  if (payload.symbol !== state.symbol || payload.timeframe !== state.timeframe) return;
  const raw = payload.candle;
  if (!raw) return;
  const rawTime = Number(raw.time);
  if (!Number.isFinite(rawTime)) return;
  const freshBar = { ...raw, time: Math.floor(rawTime) };
  if (state._suspendPoll) return;
  // Устаревшее SSE-событие (доставшееся из очереди после долгой паузы) не
  // должно откатывать уже более свежий нарисованный бар.
  if (!_acceptsBar(freshBar)) return;

  const candles = state.candles;
  if (!Array.isArray(candles) || candles.length === 0) return;
  const last = candles[candles.length - 1];

  if (freshBar.time > last.time) {
    // Новый бар: дозаписываем (_pushNewBar ВСЕГДА рисует, never отбрасывает).
    _freshLiveBar = { ...freshBar };
    const pushed = _pushNewBar(freshBar, null);
    _syncIndicatorTail(freshBar);
    _applyFreshLiveBar(freshBar, { pushed });
    _scheduleBarSync(freshBar);
  } else if (freshBar.time === last.time) {
    _mergeBar(last, freshBar);  // текущий бар тикает — накапливаем экстремумы, не теряем high/low
    _freshLiveBar = { ...last };
    _applyFreshLiveBar(freshBar);
  } else {
    return;  // устаревший бар — не откатываем
  }

  setStatus(true, state.symbol + ' · live');

  // Закрытый бар: сбрасываем счётчик, чтобы следующий tick pollLive
  // подтянул /api/data с уже финализированными индикаторами (MACD/RSI и т.д.).
  if (payload.closed) _pollCount = _FULL_RELOAD_EVERY;
}

/* Продлевает все серии индикаторов до времени нового бара, чтобы длины
   совпадали с candles. Точка берётся как последнее известное значение
   (пересчёт индикаторов придёт полным /api/data после closed-бара). */
function _syncIndicatorTail(bar) {
  const ind = state.ind;
  if (!ind) return;
  for (const k of Object.keys(ind)) {
    const arr = ind[k];
    if (!Array.isArray(arr) || arr.length === 0) continue;
    const tail = arr[arr.length - 1];
    if (!tail || typeof tail.time !== 'number') continue;
    if (tail.time < bar.time) {
      arr.push({ time: bar.time, value: tail.value });
    } else if (tail.time === bar.time) {
      // уже есть точка этого бара — оставляем как есть
    }
  }
}

/* Монотонность времени последнего бара.

   ГЛАВНАЯ ЗАЩИТА ОТ «10.07 / 10.12»: в один и тот же объект
   candles[last] пишут ТРИ источника — SSE candle_update, /api/last-bar
   и полный /api/data (reload раз в POLL_FULL_RELOAD_MS, кэш крипты TTL
   ровно 3600с). Если reload отдаёт бар, отставший от уже нарисованного
   (например, кэш ещё содержит предыдущий тик), последняя свеча визуально
   ОТКАТЫВАЕТСЯ назад — «10.07», потом SSE догоняет — «10.12». Это и
   воспринимается как лаг и «пропадание» свечи.

   Правило: время последнего бара может только РАСТИ. Ответ с меньшим
   временем отбрасывается целиком; ответ с тем же временем проходит через
   _mergeBar (накопление экстремумов), никогда не затирая целиком.
   Возвращает true, если данные применимы. */
function _acceptsBar(bar) {
  const candles = state.candles;
  if (!Array.isArray(candles) || candles.length === 0) return true;
  const last = candles[candles.length - 1];
  if (!last || typeof last.time !== 'number') return true;
  if (bar.time < last.time) {
    console.debug('[live] drop stale bar', bar.time, '<', last.time);
    return false;
  }
  return true;
}

/* Мержит live-бар в существующий бар того же времени.
   ВАЖНО: не Object.assign — он перезаписывает high/low целиком, и если
   очередной SSE-тик пришёл с уже другим (или частично снятым) значением,
   на графике появляется «дрожание» 10.07 → 10.12. Для high/low берём
   экстремум накопления, остальные поля — свежие из источника. */
function _mergeBar(target, bar) {
  target.open = bar.open;
  target.high = Math.max(Number(target.high), Number(bar.high));
  target.low = Math.min(Number(target.low), Number(bar.low));
  target.close = bar.close;
  target.volume = bar.volume;
  return target;
}

export function startLivePolling() {
  stopLivePolling();
  state.livePollTimer = setInterval(pollLive, POLL_INTERVAL_MS);
  sseConnect();
  startCandleTimer();  // таймер закрытия свечи (тик каждую секунду)
}

export function stopLivePolling() {
  if (state.livePollTimer) {
    clearInterval(state.livePollTimer);
    state.livePollTimer = null;
  }
  if (_barSyncTimer) {
    clearTimeout(_barSyncTimer);
    _barSyncTimer = null;
  }
  // Новые ответы после stop не должны «выигрывать» у более новых load'ов.
  _loadingSeq++;
  _RELOAD_IN_FLIGHT = false;
}
