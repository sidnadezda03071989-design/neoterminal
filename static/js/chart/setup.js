import { COLORS } from '../config.js';
import { state } from '../state.js';

// '#rrggbb' → [r, g, b].
function _hexToRgb(hex) {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex || '');
  if (!m) return [19, 23, 34];
  const n = parseInt(m[1], 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

export const $ = (id) => document.getElementById(id);
export const container = $('chart-container');

export const chart = LightweightCharts.createChart(container, {
  width: container.clientWidth || 800,
  height: container.clientHeight || 500,
  layout: {
    background: { type:'solid', color: COLORS.bg },
    textColor: COLORS.text,
    fontSize: 11,
    attributionLogo: false,
    panes: {
      separatorColor: COLORS.border,
      separatorHoverColor: 'rgba(255,255,255,0.20)',
      enableResize: true,
    },
  },
  grid: {
    vertLines: { color: COLORS.grid },
    horzLines: { color: COLORS.grid },
  },
  rightPriceScale: { borderColor: COLORS.border, autoScale: true },
  // BLOCK-34: график живёт своими руками. shiftVisibleRangeOnNewBar=false —
  // новый бар НЕ автоскроллит видимый диапазон; rightBarStaysOnScroll=true —
  // крайний бар не «уплывает» при прокрутке.
  timeScale: {
    borderColor: COLORS.border, timeVisible: true, secondsVisible: false,
    shiftVisibleRangeOnNewBar: false,
    rightBarStaysOnScroll: true,
  },
  handleScroll: {
    mouseWheel: true,
    pressedMouseMove: true,      // drag мышью работает
    horzTouchDrag: true,
    vertTouchDrag: false,        // вертикальный drag не нужен
  },
  handleScale: {
    axisPressedMouseMove: {      // НЕ даём двигать ТФ за ось времени
      time: false,
      price: true,               // ценовую шкалу разрешаем
    },
    axisDoubleClickReset: {
      time: false,
      price: true,
    },
    mouseWheel: true,
    pinch: true,
  },
  crosshair: {
    mode: LightweightCharts.CrosshairMode.Normal,
    vertLine: { color:'rgba(255,255,255,0.25)', width:1, style:3, labelBackgroundColor: COLORS.border },
    horzLine: { color:'rgba(255,255,255,0.25)', width:1, style:3, labelBackgroundColor: COLORS.border },
  },
  localization: { locale: 'ru-RU' },
});

setTimeout(() => chart.resize(container.clientWidth, container.clientHeight), 0);
new ResizeObserver(() => chart.resize(container.clientWidth, container.clientHeight))
  .observe(container);

function line(color, width) {
  return { color, lineWidth: width || 2, priceLineVisible:false, lastValueVisible:false };
}

// PANE 0
export const candleSeries = chart.addSeries(LightweightCharts.CandlestickSeries, {
  upColor:COLORS.up, downColor:COLORS.down,
  borderUpColor:COLORS.up, borderDownColor:COLORS.down,
  wickUpColor:COLORS.up, wickDownColor:COLORS.down,
}, 0);

export const smaSeries = chart.addSeries(LightweightCharts.LineSeries, line(COLORS.sma), 0);
export const emaSeries = chart.addSeries(LightweightCharts.LineSeries, line(COLORS.ema), 0);
export const bbUpSeries = chart.addSeries(LightweightCharts.LineSeries,
  { color:COLORS.bbUp, lineWidth:1, lineStyle:2, priceLineVisible:false, lastValueVisible:false }, 0);
export const bbMidSeries = chart.addSeries(LightweightCharts.LineSeries,
  { color:COLORS.bbMid, lineWidth:1, lineStyle:2, priceLineVisible:false, lastValueVisible:false }, 0);
export const bbLowSeries = chart.addSeries(LightweightCharts.LineSeries,
  { color:COLORS.bbLow, lineWidth:1, lineStyle:2, priceLineVisible:false, lastValueVisible:false }, 0);
export const vwapSeries = chart.addSeries(LightweightCharts.LineSeries, line(COLORS.sma), 0);
export const supertrendSeries = chart.addSeries(LightweightCharts.LineSeries, line('#4caf50'), 0);
export const stochKSeries = chart.addSeries(LightweightCharts.LineSeries, line('#ab47bc',1), 0);
export const stochDSeries = chart.addSeries(LightweightCharts.LineSeries, line('#ffa726',1), 0);
export const adxSeries = chart.addSeries(LightweightCharts.LineSeries, line('#00bcd4'), 0);
export const cciSeries = chart.addSeries(LightweightCharts.LineSeries, line('#f44336',1), 0);
export const obvSeries = chart.addSeries(LightweightCharts.LineSeries, line('#9c27b0',1), 0);
export const pivotSeries = chart.addSeries(LightweightCharts.LineSeries,
  { color:'#7e57c2', lineWidth:1, lineStyle:2, priceLineVisible:false, lastValueVisible:false }, 0);

// PANE 1
export const volumeSeries = chart.addSeries(LightweightCharts.HistogramSeries,
  { priceFormat:{type:'volume'}, priceLineVisible:false, lastValueVisible:false }, 1);

// PANE 2
export const rsiSeries = chart.addSeries(LightweightCharts.LineSeries, line(COLORS.rsi), 2);

// PANE 3
export const macdHistSeries = chart.addSeries(LightweightCharts.HistogramSeries,
  { priceLineVisible:false, lastValueVisible:false }, 3);
export const macdLineSeries = chart.addSeries(LightweightCharts.LineSeries, line(COLORS.macd), 3);
export const macdSignalSeries = chart.addSeries(LightweightCharts.LineSeries, line(COLORS.macdSignal), 3);

export function tunePanes() {
  try {
    const panes = chart.panes();
    if (panes[1]) panes[1].setHeight(state.indicators.volume ? 100 : 0);
    if (panes[2]) panes[2].setHeight(state.indicators.rsi ? 90 : 0);
    if (panes[3]) panes[3].setHeight(state.indicators.macd ? 100 : 0);
  } catch (e) { /* noop */ }
}
tunePanes();

export function updatePaneVisibility() { tunePanes(); }

try {
  rsiSeries.createPriceLine({ price:70, color:'rgba(239,83,80,0.35)', lineWidth:1, lineStyle:2, axisLabelVisible:false });
  rsiSeries.createPriceLine({ price:30, color:'rgba(38,166,154,0.35)', lineWidth:1, lineStyle:2, axisLabelVisible:false });
  macdHistSeries.createPriceLine({ price:0, color:COLORS.border, lineWidth:1, lineStyle:2, axisLabelVisible:false });
} catch (e) { /* noop */ }

// Replay-барьер + «шторка» будущего. Вертикальная пунктирная линия на всё
// дерево панелей + НЕПРОЗРАЧНАЯ заливка справа от линии (цвет фона панели),
// скрывающая ещё не сыгранные свечи и индикаторы. Данные при драге/плее
// НЕ режутся и не перекладываются — прячется только рисунок справа от линии,
// поэтому график не «улетает» и не «следует» за вновь открываемыми барами.
//
// ВАЖНО: координаты считаем ТОЛЬКО через timeToCoordinate. lightweight-charts
// v5 (5.2.1) для дробного логического индекса возвращает 0 ( 250.5 → 0,
// 998.5 → 0), поэтому logicalToCoordinate(x.5) использовать нельзя — линия
// «прилипала» к левому краю и закрывала весь график шторкой.
class ReplayBarrierPrimitive {
  constructor() {
    this._time = null;    // точное время барьера (сек) — режим `time`
    this._px = null;      // отрисованный X линии (режим `pixel`) — 1:1 с мышью при драге
    this._mode = 'time';  // 'time' | 'pixel'
    this._coverAll = false; // idx=0 (сброс): шторка на всю панель, линии нет
    this._none = false;     // live / всё раскрыто: маску и линию не рисуем
    this._boundary = null;  // время ГРАНИЦЫ (середина между последней видимой и первой скрытой)
    this._req = null;
    this._raf = null;
    // Кеш сетки BASE-канваса панели: чтобы фон справа от линии выглядел ТАК ЖЕ,
    // как слева (сетка не «пропадает»), мы сэмплируем реальную сетку панели и
    // рисуем её поверх шторки. _gridSig — признак «вид не изменился».
    this._gridSig = null;
    this._gridAt = 0;
    this._gridV = [];  // media-x вертикальных линий сетки
    this._gridH = [];  // media-y горизонтальных линий сетки
  }

  attached(param) {
    if (param && typeof param.requestUpdate === 'function') {
      this._req = param.requestUpdate.bind(param);
    }
  }

  detached() {
    this._cancelAnim();
    this._req = null;
  }

  _cancelAnim() {
    if (this._raf != null) {
      cancelAnimationFrame(this._raf);
      this._raf = null;
    }
  }

  // // Сетка «шторки»: фон справа от линии должен выглядеть так же, как слева.
  // Базовый канвас панели (под overlay) всё ещё содержит настоящую сетку lwc по
  // всей ширине — сэмплируем её и повторяем на замаскированной области. Чтобы
  // не читать пиксели каждый кадр, кешируем по признаку видимого диапазона.
  _drawMaskGrid(scope, w, h, xc) {
    try {
      const now = performance.now();
      const ts = chart.timeScale();
      let R = null;
      try { R = ts.getVisibleLogicalRange(); } catch (e) {}
      const px = scope.horizontalPixelRatio || 1;
      const py = scope.verticalPixelRatio || 1;
      const sig = [w, h, px, py,
        R ? R.from.toFixed(2) + '|' + R.to.toFixed(2) : '',
        h > 500 ? 'main' : 'sub'].join('/');
      if (this._gridSig !== sig || now - this._gridAt > 1500) {
        this._gridSig = sig;
        this._gridAt = now;
        this._hydrateGrid(scope, w, h, xc, px, py);
      }
      const ctx = scope.context;
      if (this._gridV.length || this._gridH.length) {
        ctx.strokeStyle = COLORS.grid;
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (const mx of this._gridV) {
          if (mx >= xc - 0.5 && mx <= w + 0.5) {
            ctx.moveTo(mx + 0.5, 0);
            ctx.lineTo(mx + 0.5, h);
          }
        }
        for (const my of this._gridH) {
          ctx.moveTo(xc, my + 0.5);
          ctx.lineTo(w, my + 0.5);
        }
        ctx.stroke();
      }
    } catch (e) {
      // Сетка — косметика: сбой ни в коем случае не должен ломать шторку/линию.
    }
  }

  // Сэмплирование реальной сетки: пунктир вертикалей по верхней строке базовой
  // канвасы (свечи её почти не перекрывают), горизонтали — по обилию цвета сетки
  // в строке (в обычной строке «сетка» бывает только на вертикальных линиях).
  _hydrateGrid(scope, w, h, xc, px, py) {
    this._gridV = [];
    this._gridH = [];
    try {
      const gw = Math.max(1, Math.round(w * px));
      const gh = Math.max(1, Math.round(h * py));
      let base = null;
      for (const c of container.querySelectorAll('canvas')) {
        if (Math.abs(c.width - gw) <= 2 && Math.abs(c.height - gh) <= 2) { base = c; break; }
      }
      if (!base) return;
      const ctx = base.getContext('2d', { willReadFrequently: true });
      const img = ctx.getImageData(0, 0, base.width, base.height).data;
      const [gr, gg, gb] = _hexToRgb(COLORS.grid);
      const tol = 9;
      const bw = base.width, bh = base.height;
      // Вертикали сетки: строка чуть ниже верхней кромки панели.
      const Y = Math.min(bh - 1, Math.max(0, Math.round(3 * py + 0.5)));
      const vs = [];
      for (let X = 0; X < bw; X++) {
        const i = (Y * bw + X) * 4;
        if (Math.abs(img[i] - gr) <= tol && Math.abs(img[i + 1] - gg) <= tol && Math.abs(img[i + 2] - gb) <= tol) {
          vs.push(X);
        }
      }
      // Группируем соседние колонки (1px-линия может захватить 2 пикселя DPR).
      const cols = [];
      for (let k = 0; k < vs.length; k++) {
        if (!cols.length || vs[k] - cols[cols.length - 1] > 2) cols.push(vs[k]);
      }
      this._gridV = cols.map((X) => X / px);
      // Горизонтали сетки: строка, где цвета сетки много (>8% ширины) —
      // в обычной строке сетка встречается лишь на редких вертикалях.
      const threshold = Math.max(4, Math.floor(bw * 0.08));
      const runs = [];
      for (let Y2 = 0; Y2 < bh; Y2++) {
        let cnt = 0;
        for (let X = 0; X < bw; X += 2) {
          const i = (Y2 * bw + X) * 4;
          if (Math.abs(img[i] - gr) <= tol && Math.abs(img[i + 1] - gg) <= tol && Math.abs(img[i + 2] - gb) <= tol) cnt++;
        }
        if (cnt >= threshold) {
          if (!runs.length || Y2 - runs[runs.length - 1].end > 3) runs.push({ start: Y2, end: Y2 });
          else runs[runs.length - 1].end = Y2;
        }
      }
      this._gridH = runs.map((r) => Math.round((r.start + r.end) / 2) / py);
    } catch (e) {
      this._gridV = [];
      this._gridH = [];
    }
  }

  // Пиксельный режим (драг/плей): линия+шторка ставятся напрямую по X без
  // анимации — движение в точности = движение мыши (крюк).
  setPixel(px) {
    this._cancelAnim();
    this._mode = 'pixel';
    this._px = (px == null) ? null : px;
    if (this._req) this._req();
  }

  // Точное время барьера → X на шкале (null, если вне загруженных свечей).
  _coord(t) {
    try { return chart.timeScale().timeToCoordinate(t); } catch (e) { return null; }
  }

  // X стыка между последней видимой и первой скрытой свечой. В v5
  // timeToCoordinate(midpoint) возвращает null — время между барами не мапится,
  // поэтому X берём как СРЕДНЕЕ реальных координат двух соседних баров.
  // Если первая скрытая свеча виртуальна (idx>=total: всё раскрыто) — линия
  // у правого края последней реальной свечи (x0 + barSpacing/2).
  _boundaryX() {
    if (this._mode !== 'time' || this._time == null || this._tNext == null) return null;
    try {
      const ts = chart.timeScale();
      const x0 = ts.timeToCoordinate(this._time);
      if (x0 != null) {
        const x1 = ts.timeToCoordinate(this._tNext);
        if (x1 != null) return x0 + (x1 - x0) / 2;
        try { return x0 + (ts.options().barSpacing || 6) / 2; } catch (e) { return x0; }
      }
    } catch (e) {}
    return null;
  }

  // X линии, как она реально нарисована сейчас (для захвата «крюком»):
  // режим time → стык свечей (граница); pixel → по пикселю.
  _drawnPx() {
    if (this._mode === 'time') {
      return (this._boundary != null) ? this._boundaryX() : null;
    }
    return this._px;
  }

  // Мгновенная установка барьера БЕЗ анимации/инерции. Линия+маска ставятся
  // СТРОГО на границу между свечами (середина между центрами cand(idx-1) и
  // cand(idx)), поэтому видимая свеча — всегда ЦЕЛАЯ, никогда не «половина».
  // tNext = время первой скрытой свечи; null при idx>=total (всё раскрыто:
  // линия на правом краю последней свечи, маска пустая); idx===0 → шторка
  // на всю панель (сброс реплея). t==null + idx==0 → полная шторка; иначе
  // t==null (live) → ничего не рисуем.
  setTime(t, idx, tNext) {
    this._cancelAnim();
    this._mode = 'time';
    this._time = (t == null) ? null : t;
    this._tNext = (tNext == null) ? null : tNext;
    this._px = null;
    this._coverAll = (idx === 0);
    this._none = false;
    if (this._coverAll) {
      this._boundary = null;
    } else if (t != null && tNext != null) {
      this._boundary = (t + tNext) / 2;
    } else {
      this._none = true; // live или всё раскрыто: маску и линию не рисуем
      this._boundary = null;
    }
    if (this._req) this._req();
  }

  updateAllViews() {}

  paneViews() {
    const self = this;
    return [{
      zOrder: () => 'top',
      renderer: () => ({
        draw: (target) => self._draw(target),
      }),
    }];
  }

  _draw(target) {
    let x;
    if (this._mode === 'pixel') {
      x = this._px;
    } else if (this._coverAll) {
      x = 0;
    } else if (this._none) {
      return;
    } else if (this._boundary != null) {
      x = this._boundaryX();
    } else {
      return;
    }
    if (x == null || !isFinite(x)) return;
    target.useMediaCoordinateSpace((scope) => {
      const ctx = scope.context;
      const w = scope.mediaSize.width;
      const h = scope.mediaSize.height;
      const xc = Math.max(0, Math.min(x, w));
      ctx.save();
      // «Шторка» будущего: скрываем всё правее линии цветом фона панели,
      // поэтому свечи впереди не показываются и не «вылезают» при драге.
      if (xc < w) {
        ctx.fillStyle = COLORS.bg;
        ctx.fillRect(xc, 0, w - xc, h);
        // Фон справа — поверх заливки рисуем ту же сетку, что и слева
        // (иначе замаскированная область выглядит «дырой» без сетки).
        this._drawMaskGrid(scope, w, h, xc);
      }
      // Линия барьера — только если она в пределах панели (и не «сброс»).
      if (!this._coverAll && x >= 0 && x <= w) {
        ctx.strokeStyle = 'rgba(239, 83, 80, 0.95)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([5, 4]);
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
        ctx.stroke();
        // Ручка-«наконечник» сверху replay-барьера.
        ctx.setLineDash([]);
        ctx.fillStyle = '#ef5350';
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x - 5, 8);
        ctx.lineTo(x + 5, 8);
        ctx.closePath();
        ctx.fill();
      }
      ctx.restore();
    });
  }
}

// Один инстанс на панель: каждый рисует линию+шторку на высоту своей панели,
// чтобы «будущее» скрывалось и у свечей, и у объёма/RSI/MACD.
const replayMasks = [];
function _attachReplayMask(series) {
  if (!series) return;
  const m = new ReplayBarrierPrimitive();
  replayMasks.push(m);
  try { series.attachPrimitive(m); } catch (e) { console.warn('replay mask attach failed:', e); }
}
_attachReplayMask(candleSeries);
_attachReplayMask(volumeSeries);
_attachReplayMask(rsiSeries);
_attachReplayMask(macdHistSeries);

// t = время последней видимой свечи (сек); idx — кол-во видимых свечей;
// tNext — время первой скрытой (idx>=total — ничего скрытого, маска пустая).
// Линия/шторка паркуются на ГРАНИЦЕ между свечами: свечи всегда целые.
export function setReplayBarrier(t, idx, tNext) {
  replayMasks.forEach((m) => m.setTime(t, idx, tNext));
}

// px = позиция линии в пикселях (драг/плей-режим, «крюк» за мышью).
export function setReplayBarrierPixel(px) {
  replayMasks.forEach((m) => m.setPixel(px));
}

// Текущее ТОЧНОЕ время барьера — для логики, не для захвата.
export function getReplayBarrierTime() {
  return replayMasks.length ? replayMasks[0]._time : null;
}

// Текущий ОТРИСОВАННЫЙ X линии (реальный, как на экране) — по нему «крюк»
// захвата всегда совпадает с видимой линией.
export function getReplayBarrierPixel() {
  for (const m of replayMasks) {
    const p = m._drawnPx();
    if (p != null) return p;
  }
  return null;
}

// Скриншот видимой части графика для Vision-анализа (base64 PNG без data:-префикса).
export function captureChartScreenshot() {
  try {
    const canvas = chart.takeScreenshot(true, true); // addTopLayer, includeCrosshair
    return canvas.toDataURL('image/png').split(',')[1]; // только base64
  } catch (e) {
    console.warn('screenshot failed:', e);
    return null;
  }
}
