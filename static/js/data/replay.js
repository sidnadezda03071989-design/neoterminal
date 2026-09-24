import { state } from '../state.js';
import {
  HISTORY_LIMIT, INITIAL_CANDLES, FULL_CANDLES, PROGRESSIVE_LOAD_ENABLED,
  PROGRESSIVE_STEP, PROGRESSIVE_STEP_DELAY_MS,
} from '../config.js';
import { setAllData, showChartLoading, hideChartLoading, withPreservedView } from '../chart/series.js';
import { setReplayBarrier, setReplayBarrierPixel, getReplayBarrierTime, getReplayBarrierPixel, chart, container } from '../chart/setup.js';
import { formatInTz } from '../ui/timezone.js';

const $ = (id) => document.getElementById(id);

// Дата из input type=date ("YYYY-MM-DD") → epoch-секунды (бэкенд ждёт int).
// endOfDay: 'to' берём как полночь СЛЕДУЮЩЕГО дня, чтобы день был включён.
function _dateInputEpoch(id, endOfDay = false) {
  const v = ($(id) || {}).value || '';
  if (!v) return '';
  const base = Date.parse(v + 'T00:00:00Z');
  if (isNaN(base)) return '';
  const t = endOfDay ? base + 86400000 : base;
  return String(Math.floor(t / 1000));
}

export function replaySetData(index, skipBarrier) {
  const idx = Math.max(0, Math.min(index, state.candles.length));
  // Данные на графике НЕ режем: всегда полный набор, «будущее» правее линии
  // скрывает шторка (ReplayBarrierPrimitive). Меняем только логику/барьер/тексты,
  // поэтому вьюпорт и ось не трогаются ни при драге, ни при плее — график не
  // «улетает» и не «следует» за вновь открываемыми барами.
  state.activeCandles = state.candles.slice(0, idx);
  // Время барьера = время последней видимой свечи; правее — скрыто шторкой.
  const prev = idx > 0 ? state.candles[idx - 1] : null;
  state.replay.time = prev ? prev.time : null;
  // skipBarrier: только пересчёт логики (refreshIndicatorsUpto) — не трогать
  // барьер, иначе во время драга линия «отвалится» от мыши.
  // idx как якорь правого края: парковка линии совпадает со стартом Play-глиссады.
  if (!skipBarrier) {
    const b = _barrierBoundary(idx);
    setReplayBarrier(b.time, idx, b.tNext);
  }
  const timeTxt = $('replay-time-text');
  if (timeTxt) {
    timeTxt.textContent = state.replay.time
      ? formatInTz(state.replay.time, {
          year:'numeric', month:'2-digit', day:'2-digit',
          hour:'2-digit', minute:'2-digit', hour12:false,
        }).replace(',', '')
      : '';
  }
  updateReplayPos(idx);
  _followBarrier();
}

export function updateReplayPos(idx) {
  const pos = $('replay-pos-text');
  const slider = $('progress-slider');
  if (pos) pos.textContent = idx + ' / ' + state.replay.total;
  if (slider) slider.value = String(idx);
}

let _loadSeq = 0;  // номер загрузки replay (отмена устаревших фоновых подмен)

async function _fetchReplayData(from, to, limit, uptoSec) {
  let url = '/api/replay-data?symbol=' + encodeURIComponent(state.symbol)
    + '&timeframe=' + encodeURIComponent(state.timeframe)
    + '&from=' + encodeURIComponent(from)
    + '&to=' + encodeURIComponent(to)
    + '&limit=' + limit;
  if (uptoSec != null) url += '&upto_sec=' + encodeURIComponent(uptoSec);
  const resp = await fetch(url);
  if (!resp.ok) throw new Error('HTTP ' + resp.status);
  return resp.json();
}

// Последний индекс свечи с time <= t (данные отсортированы по возрастанию).
function _nearestIndexLE(candles, t) {
  let lo = 0, hi = candles.length - 1, ans = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (candles[mid].time <= t) { ans = mid; lo = mid + 1; }
    else hi = mid - 1;
  }
  return ans;
}

// Граница «между свечами» для барьера: { time(last visible), tNext(first
// covered) }. Шторка/линия встают на СТЫКЕ (середина между центрами), поэтому
// видимая свеча ВСЕГДА целая. idx===0 → полная шторка; idx>=total → всё
// раскрыто (виртуальная граница справа от последней свечи, маска пустая).
function _barrierBoundary(idx) {
  const n = state.candles.length;
  if (idx <= 0 || n === 0) return { time: null, tNext: null };
  const t = state.candles[idx - 1].time;
  if (idx < n) return { time: t, tNext: state.candles[idx].time };
  const step = n >= 2 ? state.candles[n - 1].time - state.candles[n - 2].time : 60;
  return { time: t, tNext: t + step };
}

// X между временами двух свечей, линейная интерполяция по frac (0..1).
// ТОЛЬКО timeToCoordinate: в lightweight-charts v5 дробный логический индекс
// (x.5) даёт 0, а не координату между центрами баров.
function _timeInterpX(t0, t1, frac) {
  try {
    const ts = chart.timeScale();
    const x0 = ts.timeToCoordinate(t0);
    const x1 = ts.timeToCoordinate(t1);
    if (x0 == null || x1 == null || !isFinite(x0) || !isFinite(x1)) return null;
    return x0 + (x1 - x0) * frac;
  } catch (e) { return null; }
}

// Авто-follow вьюпорта за барьером. Реплей живёт «шторкой»: окно во время Play
// СТОИТ на месте, маска скользит внутри окна и открывает свечи слева направо,
// а view пододвигается ТОЛЬКО когда барьер вышел за ПРАВУЮ кромку (иначе
// скрытое «будущее» оказалось бы видимым). Левый выход при Play НЕ трогаем —
// иначе пан/скролл мышью «дергается и возвращается». force=true — разовый
// показ барьера из любого места (старт Play с начала сессии, скраб/шаги).
function _setBarrierView(force) {
  if (_grab) return;
  try {
    const ts = chart.timeScale();
    const R = ts.getVisibleLogicalRange();
    if (!R || !isFinite(R.from) || !isFinite(R.to)) return;
    const width = Math.max(1, R.to - R.from);
    const n = state.candles.length;
    const ti = Math.max(0, Math.min(n - 1, state.replay.index - 1));
    if (ti > R.to + 1) {
      // Барьер вышел за правую кромку окна: «перелистнуть» страницу целиком —
      // новый экран начинается с текущей свечи (барьер у ЛЕВОГО края, как при
      // старте), маска снова скрывает остальное. Между перелистами окно стоит
      // НЕПОДВИЖНО, поэтому экран не «едет за графиком» и не дерётся с мышью.
      const from = ti;
      const to = Math.min(n - 1, ti + width);
      ts.setVisibleLogicalRange({ from, to });
    } else if (force || !state.replay.playing) {
      // Не-плей: явная смена индекса (скраб/шаг/старт) — показать окно от барьера.
      if (ti < R.from - 2) {
        ts.setVisibleLogicalRange({ from: ti, to: Math.min(n - 1, ti + width) });
      }
    }
  } catch (e) {}
}

function _followBarrier() { _setBarrierView(false); }
function _forceBarrierView() { _setBarrierView(true); }

export async function loadReplay(preserveTime) {
  stopReplay();
  showChartLoading();
  const from = _dateInputEpoch('from-date');
  const to = _dateInputEpoch('to-date', true);
  const fullLimit = HISTORY_LIMIT[state.timeframe] || FULL_CANDLES;
  // Реплей живёт на ВСЮ длину уже загруженного графика: если данные уже
  // догружены (live прогрессив докачал, а мы входим в реплей), стартуем сразу
  // с полного окна, а не режем до INITIAL_CANDLES. Свежий/пустой путь остаётся
  // прогрессивным (1000 → fullLimit фоном).
  const loaded = state.candles && state.candles.length > 0 ? state.candles.length : 0;
  const initialLimit = Math.max(INITIAL_CANDLES, Math.min(fullLimit, loaded));
  _loadSeq++;
  const seq = _loadSeq;
  // Запрос «с нуля» (пресет/Загрузить/включение режима): старт с середины.
  // preserveTime (смена symbol/tf): барьер остаётся на прежнем времени.
  if (!preserveTime) state.replay.time = null;
  state.replay.lastUpto = null;
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
    let kept = (preserveTime && state.replay.time != null)
      ? _nearestIndexLE(state.candles, state.replay.time) : -1;
    // preserveTime (смена symbol/tf): если барьер старше первого чанка
    // (например 1H→1m: INITIAL_CANDLES×1m не дотягивают до барьера на пару
    // дней назад), догоняем историю чанками, пока барьер не попадёт в данные.
    // Иначе kept=-1 → index сбрасывался в середину → «реплей обнулился».
    if (preserveTime && kept < 0 && state.replay.time != null && initialLimit < fullLimit) {
      let limit = initialLimit;
      while (limit < fullLimit && seq === _loadSeq) {
        limit = Math.min(limit + PROGRESSIVE_STEP, fullLimit);
        const more = await _fetchReplayData(from, to, limit);
        if (seq !== _loadSeq) return; // устарели — новый load уже идёт
        if (!more.candles || !more.candles.length) break;
        if (more.candles.length <= state.candles.length) break; // прироста нет
        state.candles = more.candles;
        state.ind = more.indicators;
        kept = _nearestIndexLE(state.candles, state.replay.time);
      }
    }
    state.replay.total = state.candles.length;
    // По умолчанию барьер ставится на ПОСЛЕДНЮЮ свечу (конец данных =
    // «в реальном времени»); в середину — только если барьер сохраняли и нашли.
    state.replay.index = kept >= 0 ? kept : Math.max(0, state.replay.total - 1);
    const slider = $('progress-slider');
    if (slider) {
      slider.max = String(Math.max(1, state.replay.total - 1));
      slider.value = String(state.replay.index);
    }
    // Самый важный шаг: рисуем ПОЛНЫЙ набор свечей и индикаторов ОДИН раз.
    // Дальше ничего не режется (реплей живёт шторкой), поэтому на оси ничего
    // не съезжает при драге/плее.
    setAllData(state.candles, state.ind);
    replaySetData(state.replay.index);
    // Смена symbol/tf (preserveTime): линия остаётся на прежнем времени.
    // Центрируем вьюпорт на барьере, чтобы линия не «уезжала» за экран
    // (иначе BLOCK-34-fit показывает конец данных и кажется, что реплей сброшен).
    if (preserveTime && kept >= 0) {
      try {
        chart.timeScale().setVisibleLogicalRange({
          from: Math.max(0, kept - 150),
          to: Math.min(state.replay.total - 1, kept + 50),
        });
      } catch (e) {}
    }
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
    // Стартуем ПЕРЕД текущим объёмом: после чейза preserveTime данные уже
    // больше INITIAL_CANDLES — иначе фон откатил бы нарезанный вид назад.
    for (
      let target = Math.min(Math.max(INITIAL_CANDLES, currentLen) + PROGRESSIVE_STEP, fullLimit);
      target <= fullLimit;
      target = Math.min(target + PROGRESSIVE_STEP, fullLimit)
    ) {
      // Гонка: replay уже перезагрузили с другими параметрами — фон устарел.
      if (seq !== _loadSeq) return;
      // Индикаторы успели пересчитать до барьера (пауза/шаг) — фон с полными
      // индикаторами (с «будущим») стал бы регрессией: останавливаемся.
      if (state.replay.lastUpto != null) return;
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
      // Подмена полных данных без потери зума/скролла (добавленные бары лежат
      // слева): распаковка новых баров в шкалу и НЕзависимая от них логика.
      // Шторка (барьер) при этом заново паркуется на правый край текущего бара.
      withPreservedView(() => {
        setAllData(state.candles, state.ind);
        replaySetData(state.replay.index);
      }, offset);
      currentLen = data.candles.length;
      console.debug('progressive(replay): ' + currentLen + '/' + fullLimit);
      // Чанк применён; если источник отдал меньше target (потолок истории),
      // следующие запросы будут холостыми — останавливаемся.
      if (data.candles.length < target) {
        console.debug('progressive(replay): stop at ' + currentLen + ' (source limit)');
        break;
      }
      // Дошли до полного объёма (fullLimit): это последний чанк. Без break
      // инкремент (см. for: Math.min(target+step, fullLimit)) зациклился бы
      // на fullLimit и гонял бы один и тот же запрос бесконечно.
      if (target >= fullLimit) break;
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
    cancelAnimationFrame(state.replay.timer);
    state.replay.timer = null;
  }
}

// Воспроизведение: playhead-ВРЕМЯ движется непрерывно (startT + elapsed×speed×tf),
// а линия глиссирует по пикселям от бара к бару через timeToCoordinate —
// никаких ступенек setInterval. Данные при плее НЕ нарезаются и не перекладываются:
// будущее скрыто шторкой, линия просто «отодвигает» её вправо по мере открытия баров.
export function playReplay() {
  if (state.replay.playing) return;
  if (state.replay.total <= 0 || !state.candles.length) return;
  if (state.replay.index >= state.replay.total - 1) state.replay.index = 0;
  state.replay.playing = true;
  const btn = $('play-btn'); if (btn) btn.classList.add('active');
  const tfSec = state.candles.length > 1
    ? (state.candles[1].time - state.candles[0].time) : 60;
  // Сброс индекса на 0 ДО расчёта startT: иначе Play с последней свечи
  // стартует с начального времени конца и мгновенно «доигрывает» до конца.
  if (state.replay.index === 0) {
    state.replay.index = 1;
    replaySetData(1); // первый бар открыт: шторка паркуется на центре бара 0
    _forceBarrierView(); // разовый показ старта (вьюпорт мог быть на другом краю)
  }
  const startT = (state.replay.time != null)
    ? state.replay.time
    : (state.candles[0] ? state.candles[0].time : 0);
  const t0 = performance.now();
  let lastCut = state.replay.index;
  const tick = (now) => {
    if (!state.replay.playing) return;
    // Непрерывное время playhead: скорость в свечах/сек, но визуально свечи
    // открываются СТРОГО ЦЕЛЫМИ — линия стоит на границе, а не скользит.
    const t = startT + ((now - t0) / 1000) * state.replay.speed * tfSec;
    const last = state.candles.length - 1;
    if (t >= state.candles[last].time) {
      // Вся история раскрыта: линия на правом краю последней свечи, маска
      // пустая — НИКАКОЙ «половинки» свечи. stopReplay без pauseReplay,
      // чтобы не перепарковать и не делать лишнего пересчёта индикаторов.
      state.replay.index = state.replay.total - 1;
      replaySetData(state.replay.index, true);
      const b = _barrierBoundary(state.replay.total);
      setReplayBarrier(b.time, state.replay.total, b.tNext);
      const btn = $('play-btn'); if (btn) btn.classList.remove('active');
      stopReplay();
      return;
    }
    // cut = первый бар после t: видны [0 .. cut-1] ЦЕЛЫМИ, линия на границе
    // cand(cut-1)/cand(cut). Частичные свечи исключены.
    const kk = Math.max(1, _nearestIndexLE(state.candles, t) + 1);
    if (kk !== lastCut) {
      lastCut = kk;
      state.replay.index = kk;
      replaySetData(kk, true); // обновляем логику на границе; шторку трогаем ниже
    }
    // Граница между последней видимой и первой скрытой (середина центров).
    const b = _barrierBoundary(kk);
    const c = b.tNext != null ? _timeInterpX(b.time, b.tNext, 0.5) : null;
    if (c != null && isFinite(c)) setReplayBarrierPixel(c);
    else setReplayBarrier(b.time, kk, b.tNext);
    state.replay.timer = requestAnimationFrame(tick);
  };
  state.replay.timer = requestAnimationFrame(tick);
}

export function pauseReplay() {
  const wasPlaying = state.replay.playing;
  stopReplay();
  const btn = $('play-btn'); if (btn) btn.classList.remove('active');
  // Пауза = момент «разбора»: линия из px-глиссады паркуется точно на границе
  // последнего видимого бара, индикаторы пересчитываются ровно до барьера.
  if (wasPlaying) {
    if (state.replay.time != null) {
      const b = _barrierBoundary(state.replay.index);
      setReplayBarrier(b.time, state.replay.index, b.tNext);
    }
    refreshIndicatorsUpto();
  }
}

// Фоновый пересчёт индикаторов строго до времени барьера (replay_time):
// /api/replay-data?upto_sec=… считает индикаторы по префиксу <= барьера.
// Текущие индикаторы все каузальные (rolling/ewm/cumsum в indicators.py),
// поэтому upto-значения совпадают со срезом полного ряда — слияние лишь
// закрепляет «пересчёт до линии» и страхует от будущих рекурсивных индикаторов.
// Хвост полного ряда НЕ трогаем: он правее барьера и никогда не рисуется,
// зато Play после паузы не теряет данные индикаторов.
export async function refreshIndicatorsUpto() {
  const t = state.replay.time;
  if (!t || state.mode !== 'replay') return;
  if (t === state.replay.lastUpto) return; // уже пересчитаны до этого времени
  const from = _dateInputEpoch('from-date');
  const to = _dateInputEpoch('to-date', true);
  const limit = Math.max(state.replay.total, 1);
  try {
    const data = await _fetchReplayData(from, to, limit, t);
    if (state.replay.time !== t) return; // барьер уехал — результат устарел
    if (!data.indicators) return;
    const res = data.indicators;
    for (const k of Object.keys(res)) {
      if (!Array.isArray(res[k])) continue;
      if (Array.isArray(state.ind?.[k]) && state.ind[k].length > res[k].length) {
        for (let i = 0; i < res[k].length && i < state.ind[k].length; i++) {
          state.ind[k][i] = res[k][i]; // обновляем только до линии
        }
      } else {
        state.ind[k] = res[k];
      }
    }
    state.replay.lastUpto = t;
    replaySetData(state.replay.index, true); // барьер не трогаем (важно при драге)
  } catch (e) { console.warn('replay(recompute indicators):', e); }
}

export function resetReplay() {
  stopReplay();
  state.replay.index = 0;
  state.replay.lastUpto = null;
  replaySetData(0);
}

export function replayStepBack() { pauseReplay(); if (state.replay.index > 0) { state.replay.index -= 1; replaySetData(state.replay.index); } refreshIndicatorsUpto(); }
export function replayStepForward() { pauseReplay(); if (state.replay.index < state.replay.total - 1) { state.replay.index += 1; replaySetData(state.replay.index); } refreshIndicatorsUpto(); }
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

/* ---------- Перетаскивание линии-барьера мышью ----------
   Работает с активным инструментом «Курсор» (default). Захват линии ±8px;
   перетаскивание — на всю ширину загруженных свечей. Пан графика на время
   драга глушится (handleScroll.pressedMouseMove=false), так что экран
   реагирует только на перетаскивание барьера.

   «Крюк»: линия привязана к логическому индексу под курсором и рисуется в
   пиксельном режиме (setReplayBarrierPixel) СТРОГО по X мыши — движение 1:1,
   без анимации, ускорения и интерполяции по времени (из-за гэпов выходных
   линия раньше улетала за мышью и пропадала). Данные НЕ режутся и НЕ
   перекладываются: будущее скрыто шторкой, поэтому вьюпорт не прыгает и
   свечи правее линии не могут вылезти. */

const GRAB_PX = 8;

// X видимой (отрисованной) линии — по ней же цепляем «крюк», поэтому захват
// совпадает с тем, что видно на экране (а не с целевой позицией анимации).
function _barrierX() {
  // Точный X только что нарисованной линии (совпадает с глазом даже во время
  // анимации/глиссады) — по нему же цепляется «крюк».
  const p = getReplayBarrierPixel();
  if (p != null) return p;
  const dt = getReplayBarrierTime();
  const t = dt != null ? dt : state.replay.time;
  if (t == null) return null;
  try { return chart.timeScale().timeToCoordinate(t); }
  catch (e) { return null; }
}

let _grab = null;  // { baseIdx, basePx, mousePx } — точка «крюка»

// X границы между свечами R-1 и R (R = число видимых свечей): по нему линия
// при драге открывает/закрывает свечи ЦЕЛЫМИ — как при воспроизведении, где
// шторка стоит на стыке баров, а не «режет» свечу под курсором пополам.
// Сначала — точный стык через timeToCoordinate (середина между центрами двух
// соседних баров; тот же расчёт, что у playReplay). Вне экрана/загруженных
// данных — оценка логическим индексом с клампом по кромкам: слева всё скрыто
// (полная шторка), справа — всё раскрыто (маска пустая).
function _boundaryPixelFor(R) {
  const n = state.candles.length;
  if (n === 0) return 0;
  const W = container.clientWidth || 0;
  if (R <= 0) return 0;                 // сброс: шторка на всю панель
  let t0, t1;
  if (R < n) {
    t0 = state.candles[R - 1].time;
    t1 = state.candles[R].time;
  } else {
    t0 = state.candles[n - 1].time;     // всё раскрыто: правый край последней
    const step = n >= 2 ? state.candles[n - 1].time - state.candles[n - 2].time : 60;
    t1 = t0 + step;
  }
  const x = _timeInterpX(t0, t1, 0.5);
  if (x != null) return x;
  try {
    const ts = chart.timeScale();
    const c = ts.logicalToCoordinate(R); // центр cand(R) → граница на полбара левее
    if (c != null && isFinite(c)) return c - (ts.options().barSpacing || 6) / 2;
  } catch (e) {}
  try {
    const Rv = chart.timeScale().getVisibleLogicalRange();
    if (Rv) {
      if (R <= Rv.from + 0.5) return 0;  // граница ушла влево за экран: всё скрыто
      if (R >= Rv.to) return W;          // правее экрана: всё раскрыто
    }
  } catch (e) {}
  return W;
}

// Драг-«крюк»: свечи при перетаскивании появляются/исчезают СТРОГО ЦЕЛЫМИ —
// линия привязывается к ГРАНИЦЕ между барами R-1 и R (как при Play), а не к
// X мыши (иначе шторка резала бы свечу под курсором «по половинке»). Данные
// на графике НЕ режутся и не перекладываются — шторка просто «щелкает» по
// стыкам за мышью, поэтому вьюпорт не «улетает» и свечи правее не вылезают.
// isFinal — финальная парковка на границе свечи (time-режим, как в pauseReplay).
function _moveBarrierToPx(px, isFinal) {
  if (!_grab) return;
  let F = _grab.baseIdx;
  try {
    const l = chart.timeScale().coordinateToLogical(px);
    if (l != null && isFinite(l)) {
      F = l;
    } else {
      let spacing = 6;
      try { spacing = chart.timeScale().options().barSpacing || 6; } catch (e) {}
      F = _grab.baseIdx + (px - _grab.mousePx) / Math.max(1, spacing);
    }
  } catch (e) {
    let spacing = 6;
    try { spacing = chart.timeScale().options().barSpacing || 6; } catch (e) {}
    F = _grab.baseIdx + (px - _grab.mousePx) / Math.max(1, spacing);
  }
  const n = state.candles.length;
  const R = Math.max(1, Math.min(Math.round(F), n));
  if (R !== state.replay.index) {
    state.replay.index = R;
    replaySetData(R, true); // обновляем логику/тексты; шторкой управляем через px ниже
  }
  // X линии — граница между барами (целые свечи), кламп по ширине панели.
  const W = container.clientWidth || 0;
  const bx = _boundaryPixelFor(R);
  if (isFinal) {
    const b = _barrierBoundary(R);
    if (b.time != null) setReplayBarrier(b.time, R, b.tNext);  // паркуемся на границе
  } else {
    const x = (bx != null && isFinite(bx))
      ? (W > 0 ? Math.max(0, Math.min(bx, W)) : bx)
      : (W > 0 ? Math.max(0, Math.min(px, W)) : px);
    setReplayBarrierPixel(x);  // шторка на стыке свечей: «щелкает» по барам, не режет
  }
}

export function initReplayBarrierDrag() {
  if (!container || !chart) return;

  // Курсор-подсказка: ew-resize при наведении на линию.
  container.addEventListener('pointermove', (e) => {
    if (state.mode !== 'replay') { container.style.cursor = ''; return; }
    if (_grab) return; // драгим барьер — курсор уже задан
    if (state.dm && state.dm.tool !== 'cursor') { container.style.cursor = ''; return; }
    const x = _barrierX();
    if (x == null) { container.style.cursor = ''; return; }
    const mx = e.clientX - container.getBoundingClientRect().left;
    container.style.cursor = Math.abs(mx - x) <= GRAB_PX ? 'ew-resize' : '';
  });

  container.addEventListener('pointerdown', (e) => {
    if (state.mode !== 'replay') return;
    if (state.dm && state.dm.tool !== 'cursor') return;
    const x = _barrierX();
    if (x == null) return;
    // Рект кэшируем ОДИН раз при захвате: x считается от одной базы,
    // иначе смещения между событиями ломают «крюк» (рисуем не там, где мышь).
    const rect = container.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    if (Math.abs(mx - x) > GRAB_PX) return;
    pauseReplay();  // остановка воспроизведения на время перетаскивания
    _loadSeq++;     // гасим фоновую прогрессивную подмену данных (иначе indices ползут)
    state.chartNeedsFit = false;  // ручной драг: никакой авто-подгонки вида (иначе «улетит»)
    e.preventDefault();  // гасим mousedown: пан/рисование не стартуют
    e.stopPropagation();
    _grab = { baseIdx: state.replay.index || 1, basePx: x, mousePx: mx, rect };
    try { chart.applyOptions({ handleScroll: { pressedMouseMove: false } }); } catch (err) {}
    _moveBarrierToPx(mx, false);  // линия сразу «цепляется» за курсор

    const cxp = (ev) => ev.clientX - _grab.rect.left;
    const onMove = (ev) => {
      _moveBarrierToPx(cxp(ev), false);
    };
    const onUp = (ev) => {
      document.removeEventListener('pointermove', onMove);
      document.removeEventListener('pointerup', onUp);
      document.removeEventListener('pointercancel', onUp);
      try { chart.applyOptions({ handleScroll: { pressedMouseMove: true } }); } catch (err) {}
      container.style.cursor = '';
      _moveBarrierToPx(cxp(ev), true);
      _grab = null;
      refreshIndicatorsUpto();
    };
    document.addEventListener('pointermove', onMove);
    document.addEventListener('pointerup', onUp);
    document.addEventListener('pointercancel', onUp);
  });
}
