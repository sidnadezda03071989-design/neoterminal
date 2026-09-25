import { state } from '../state.js';
import { setAllData } from '../chart/series.js';
import { updatePaneVisibility } from '../chart/setup.js';

const $ = (id) => document.getElementById(id);

export const indMap = {
  'ind-sma':'sma', 'ind-ema':'ema', 'ind-bb':'bb', 'ind-volume':'volume',
  'ind-rsi':'rsi', 'ind-macd':'macd', 'ind-vwap':'vwap',
  'ind-supertrend':'supertrend', 'ind-stoch':'stoch',
  'ind-adx':'adx', 'ind-cci':'cci', 'ind-obv':'obv',
};

export function toggleIndicator(name, visible) {
  state.indicators[name] = visible;
  // Всегда рисуем ПОЛНЫЙ массив: в replay «будущее» правее линии скрывает
  // шторка (ReplayBarrierPrimitive), поэтому пере-рендер среза не нужен и
  // сломал бы график (ось сжалась бы до активного окна).
  const candles = state.candles;
  if (!candles || !candles.length || !state.ind) return;
  const n = candles.length;
  const ind = {};
  for (const k of Object.keys(state.ind)) ind[k] = (state.ind[k] || []).slice(0, n);
  setAllData(candles, ind);
}

export function clearIndicators() {
  for (const name of Object.keys(state.indicators)) {
    state.indicators[name] = false;
  }
  for (const id of Object.keys(indMap)) {
    const checkbox = $(id);
    if (checkbox) checkbox.checked = false;
  }
  if (state.candles && state.candles.length && state.ind) {
    setAllData(state.candles, state.ind);
  } else {
    updatePaneVisibility();
  }
}
