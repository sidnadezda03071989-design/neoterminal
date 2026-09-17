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

export function setAllData(candles, ind) {
  try { chart.timeScale().applyOptions({ barSpacing: 6, rightOffset: 5 }); } catch (e) {}
  candleSeries.setData(candles);
  volumeSeries.setData(state.indicators.volume ? toVolumeData(candles) : []);
  smaSeries.setData(state.indicators.sma ? toLineData(ind.sma20) : []);
  emaSeries.setData(state.indicators.ema ? toLineData(ind.ema50) : []);
  bbUpSeries.setData(state.indicators.bb ? toLineData(ind.bb_up) : []);
  bbMidSeries.setData(state.indicators.bb ? toLineData(ind.bb_mid) : []);
  bbLowSeries.setData(state.indicators.bb ? toLineData(ind.bb_low) : []);
  rsiSeries.setData(state.indicators.rsi ? toLineData(ind.rsi) : []);
  macdHistSeries.setData(state.indicators.macd ? toMacdHist(ind.macd_hist) : []);
  macdLineSeries.setData(state.indicators.macd ? toLineData(ind.macd) : []);
  macdSignalSeries.setData(state.indicators.macd ? toLineData(ind.macd_signal) : []);
  vwapSeries.setData(state.indicators.vwap ? toLineData(ind.vwap) : []);
  supertrendSeries.setData(state.indicators.supertrend ? toLineData(ind.supertrend) : []);
  stochKSeries.setData(state.indicators.stoch ? toLineData(ind.stoch_k) : []);
  stochDSeries.setData(state.indicators.stoch ? toLineData(ind.stoch_d) : []);
  adxSeries.setData(state.indicators.adx ? toLineData(ind.adx) : []);
  cciSeries.setData(state.indicators.cci ? toLineData(ind.cci) : []);
  obvSeries.setData(state.indicators.obv ? toLineData(ind.obv) : []);
  pivotSeries.setData(state.indicators.pivot ? toLineData(ind.pivot) : []);
  updatePaneVisibility();

  // Сброс ценовой шкалы под новый диапазон (критично при смене символа)
  try { candleSeries.priceScale().applyOptions({ autoScale: true }); } catch (e) {}
  try { chart.priceScale('right').applyOptions({ autoScale: true }); } catch (e) {}

  // Жёсткий сброс timescale: показываем последние 200 баров
  try {
    const n = candles.length;
    if (n > 0) {
      chart.timeScale().applyOptions({ barSpacing: 6, rightOffset: 5 });
      chart.timeScale().setVisibleLogicalRange({
        from: Math.max(0, n - 200),
        to: n - 1,
      });
    }
  } catch (e) {}

  hideChartLoading();
}

export function updateAllLast(candles, ind) {
  const last = candles[candles.length - 1];
  if (!last) return;
  candleSeries.update(last);
  if (state.indicators.volume) {
    try { volumeSeries.update(toVolumeData([last])[0]); } catch (e) {}
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
    series.update({ time: it.time, value: it.value });
  }
}