import { state } from '../state.js';
import { setAllData } from '../chart/series.js';

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
