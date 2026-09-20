import { COLORS } from '../config.js';
import { state } from '../state.js';

export const $ = (id) => document.getElementById(id);
export const container = $('chart-container');

export const chart = LightweightCharts.createChart(container, {
  width: container.clientWidth || 800,
  height: container.clientHeight || 500,
  layout: {
    background: { type:'solid', color: COLORS.bg },
    textColor: COLORS.text,
    fontSize: 11,
    panes: {
      separatorColor: '#363a45',
      separatorHoverColor: 'rgba(255,255,255,0.15)',
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
    vertLine: { color:'rgba(255,255,255,0.25)', width:1, style:3, labelBackgroundColor:'#4a4f5f' },
    horzLine: { color:'rgba(255,255,255,0.25)', width:1, style:3, labelBackgroundColor:'#4a4f5f' },
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
  rsiSeries.createPriceLine({ price:70, color:'rgba(255,80,80,0.35)', lineWidth:1, lineStyle:2, axisLabelVisible:false });
  rsiSeries.createPriceLine({ price:30, color:'rgba(80,255,140,0.35)', lineWidth:1, lineStyle:2, axisLabelVisible:false });
  macdHistSeries.createPriceLine({ price:0, color:'#363a45', lineWidth:1, lineStyle:2, axisLabelVisible:false });
} catch (e) { /* noop */ }

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
