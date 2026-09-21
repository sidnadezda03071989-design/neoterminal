export const state = {
  symbol: 'BTCUSDT',
  timeframe: '1H',
  mode: 'live',
  candles: [],
  ind: null,
  activeCandles: [],
  livePollTimer: null,
  candlesPrevLength: 0,
  first: true,
  progressiveLoaded: false,  // фоновая прогрессивная догрузка завершилась (live/replay)
  // BLOCK-34: true в первый load и на смену symbol/tf — setAllData тогда
  // единственный раз делает fit (последние ~200 свечей + авто-цена);
  // прогрессивные чанки и reload того же symbol/tf флаг не выставляют.
  chartNeedsFit: true,
  replay: { index:0, total:0, playing:false, timer:null, speed:2,
    time:null,      // время барьера (replay_time) — последняя видимая свеча
    lastUpto:null }, // upto_sec, по которому уже пересчитаны индикаторы
  indicators: {
    sma:true, ema:true, bb:true, volume:true, rsi:false,
    macd:false, vwap:false, supertrend:false, stoch:false,
    adx:false, cci:false, obv:false,
  },
  dm: null,
};

let _currentLayout = '1';
export function setCurrentLayout(l) { _currentLayout = l; }
export function getCurrentLayout() { return _currentLayout; }
