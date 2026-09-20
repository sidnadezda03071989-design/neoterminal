import { state } from '../state.js';
import { chart } from './setup.js';
import {
  candleSeries, volumeSeries, smaSeries, emaSeries,
  bbUpSeries, bbMidSeries, bbLowSeries, rsiSeries,
  macdHistSeries, macdLineSeries, macdSignalSeries,
  vwapSeries, supertrendSeries, stochKSeries, stochDSeries,
  adxSeries, cciSeries, obvSeries, pivotSeries,
  updatePaneVisibility,
} from './setup.js';

/* Защита от null/NaN: lightweight-charts бросает "Value is null", если в
   setData/update попадает точка с пустым time или значением. _cleanData
   пропускает и свечи (термально закрытые OHLC — числовой close), и линейные
   точки (числовой value). */
function _isFiniteNumber(v) {
  return typeof v === 'number' && !isNaN(v) && isFinite(v);
}

function _cleanData(data) {
  if (!Array.isArray(data)) return [];
  return data.filter((d) => d && d.time != null &&
    (_isFiniteNumber(d.close) || _isFiniteNumber(d.value)));
}

export function safeSetData(series, data) {
  if (!series) return;
  try { series.setData(_cleanData(data)); }
  catch (e) { console.warn('[chart] setData skipped:', e.message); }
}

export function safeUpdate(series, point) {
  if (!series || !point) return;
  if (point.time == null) return;
  if (!_isFiniteNumber(point.close) && !_isFiniteNumber(point.value)) return;
  try { series.update(point); }
  catch (e) { console.warn('[chart] update skipped:', e.message); }
}

let _loadingTimer = null;

export function showChartLoading() {
  const el = document.getElementById('chart-loading');
  if (!el) return;
  if (_loadingTimer) clearTimeout(_loadingTimer);
  _loadingTimer = setTimeout(() => { el.classList.add('active'); }, 120);
}

export function hideChartLoading() {
  if (_loadingTimer) { clearTimeout(_loadingTimer); _loadingTimer = null; }
  const el = document.getElementById('chart-loading');
  if (!el) return;
  el.style.opacity = '0';
  setTimeout(() => { el.classList.remove('active'); el.style.opacity = ''; }, 180);
}

// Прогрессивная подмена данных: сохраняем видимое логическое окно до
// перерисовки и восстанавливаем после. offset — на сколько логических баров
// сдвинулось окно из-за догруженных слева свечей (full.length - initial.length),
// чтобы после подмены пользователь видел те же бары с тем же зумом.
// Использовать точечно — только для фоновой подмены, не для обычных setAllData.
export function withPreservedView(fn, offset = 0) {
  let range = null;
  try { range = chart.timeScale().getVisibleLogicalRange(); } catch (e) {}
  fn();
  if (range) {
    try {
      chart.timeScale().setVisibleLogicalRange({
        from: range.from + offset,
        to: range.to + offset,
      });
    } catch (e) {}
  }
}

export function toLineData(arr) {
  const out = [];
  for (const it of arr || []) {
    if (it && it.value != null) out.push({ time: it.time, value: it.value });
  }
  return out;
}

export function toVolumeData(candles) {
  return (candles || []).map((c) => ({
    time: c.time, value: c.volume,
    color: c.close >= c.open ? 'rgba(38,166,154,0.5)' : 'rgba(239,83,80,0.5)',
  }));
}

export function toMacdHist(arr) {
  return (arr || [])
    .filter(it => it.value != null)
    .map(it => ({ time: it.time, value: it.value,
      color: it.value >= 0 ? 'rgba(66,165,245,0.55)' : 'rgba(239,83,80,0.55)' }));
}

// Разовый fit после первой загрузки / смены symbol/tf (state.chartNeedsFit):
// показываем последние ~200 свечей и включаем авто-масштаб ценовой шкалы.
// Повторные вызовы невозможны — флаг снимается сразу (BLOCK-34).
export function fitChartToData(candles) {
  if (!Array.isArray(candles) || candles.length === 0) return;
  state.chartNeedsFit = false;
  try {
    const n = candles.length;
    chart.timeScale().applyOptions({ barSpacing: 6, rightOffset: 5 });
    chart.timeScale().setVisibleLogicalRange({
      from: Math.max(0, n - 200),
      to: n - 1,
    });
    candleSeries.priceScale().applyOptions({ autoScale: true });
    chart.priceScale('right').applyOptions({ autoScale: true });
  } catch (e) { /* noop: fit не критичен */ }
}

export function setAllData(candles, ind) {
  if (!Array.isArray(candles) || candles.length === 0) return;
  if (!ind || typeof ind !== 'object') ind = {};
  try { chart.timeScale().applyOptions({ barSpacing: 6, rightOffset: 5 }); } catch (e) {}
  safeSetData(candleSeries, candles);
  safeSetData(volumeSeries, state.indicators.volume ? toVolumeData(candles) : []);
  safeSetData(smaSeries, state.indicators.sma ? toLineData(ind.sma20) : []);
  safeSetData(emaSeries, state.indicators.ema ? toLineData(ind.ema50) : []);
  safeSetData(bbUpSeries, state.indicators.bb ? toLineData(ind.bb_up) : []);
  safeSetData(bbMidSeries, state.indicators.bb ? toLineData(ind.bb_mid) : []);
  safeSetData(bbLowSeries, state.indicators.bb ? toLineData(ind.bb_low) : []);
  safeSetData(rsiSeries, state.indicators.rsi ? toLineData(ind.rsi) : []);
  safeSetData(macdHistSeries, state.indicators.macd ? toMacdHist(ind.macd_hist) : []);
  safeSetData(macdLineSeries, state.indicators.macd ? toLineData(ind.macd) : []);
  safeSetData(macdSignalSeries, state.indicators.macd ? toLineData(ind.macd_signal) : []);
  safeSetData(vwapSeries, state.indicators.vwap ? toLineData(ind.vwap) : []);
  safeSetData(supertrendSeries, state.indicators.supertrend ? toLineData(ind.supertrend) : []);
  safeSetData(stochKSeries, state.indicators.stoch ? toLineData(ind.stoch_k) : []);
  safeSetData(stochDSeries, state.indicators.stoch ? toLineData(ind.stoch_d) : []);
  safeSetData(adxSeries, state.indicators.adx ? toLineData(ind.adx) : []);
  safeSetData(cciSeries, state.indicators.cci ? toLineData(ind.cci) : []);
  safeSetData(obvSeries, state.indicators.obv ? toLineData(ind.obv) : []);
  safeSetData(pivotSeries, state.indicators.pivot ? toLineData(ind.pivot) : []);
  updatePaneVisibility();

  // Fit — ТОЛЬКО когда его явно попросили (первый load / смена symbol-tf).
  // Прогрессивные чанки и полный reload того же symbol/tf флаг не ставят:
  // их видимый zoom/скролл сохраняется (BLOCK-34).
  if (state.chartNeedsFit === true) fitChartToData(candles);

  hideChartLoading();
}

export function updateAllLast(candles, ind) {
  if (!Array.isArray(candles) || candles.length === 0) return;
  const last = candles[candles.length - 1];
  safeUpdate(candleSeries, last);
  if (state.indicators.volume) {
    safeUpdate(volumeSeries, toVolumeData([last])[0]);
  }
  const map = {};
  if (state.indicators.sma) map.sma20 = smaSeries;
  if (state.indicators.ema) map.ema50 = emaSeries;
  if (state.indicators.bb) { map.bb_up = bbUpSeries; map.bb_mid = bbMidSeries; map.bb_low = bbLowSeries; }
  if (state.indicators.rsi) map.rsi = rsiSeries;
  if (state.indicators.macd) { map.macd = macdLineSeries; map.macd_signal = macdSignalSeries; }
  if (state.indicators.vwap) map.vwap = vwapSeries;
  if (state.indicators.supertrend) map.supertrend = supertrendSeries;
  if (state.indicators.stoch) { map.stoch_k = stochKSeries; map.stoch_d = stochDSeries; }
  if (state.indicators.adx) map.adx = adxSeries;
  if (state.indicators.cci) map.cci = cciSeries;
  if (state.indicators.obv) map.obv = obvSeries;
  if (state.indicators.pivot) map.pivot = pivotSeries;
  for (const [name, series] of Object.entries(map)) {
    const arr = ind[name];
    if (!arr || !arr.length) continue;
    const it = arr[arr.length - 1];
    if (!it || it.value == null) continue;
    safeUpdate(series, { time: it.time, value: it.value });
  }
}