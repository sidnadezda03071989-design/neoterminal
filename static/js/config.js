export const COLORS = {
  bg: '#131722', grid: '#1e222d', border: '#2a2e39',
  text: '#d1d4dc', up: '#26a69a', down: '#ef5350',
  sma: '#2196f3', ema: '#ff9800',
  bbUp: '#7e57c2', bbMid: '#7e57c2', bbLow: '#7e57c2',
  rsi: '#ab47bc', macd: '#42a5f5', macdSignal: '#ffa726',
};

export const HISTORY_LIMIT = {
  '1m':20000,'5m':20000,'15m':20000,'1H':20000,'4H':20000,'1D':20000,
};

export const REPLAY_DAYS = {
  '1m':3,'5m':14,'15m':30,'1H':180,'4H':730,'1D':7300,
};

// Прогрессивная загрузка: сначала быстро рисуем INITIAL_CANDLES свечей,
// затем фоном докачиваем FULL_CANDLES и подменяем данные без потери вида.
export const INITIAL_CANDLES = 1000;
export const FULL_CANDLES = 20000;
export const PROGRESSIVE_LOAD_ENABLED = true;
// Прогрессивная догрузка идёт чанками: backend на запрос с большим limit
// докачивает только дельту (по 1 странице Binance на шаг),
// т.е. 1000→2000→…→20000 = 19 дешёвых шагов.
export const PROGRESSIVE_STEP = 1000;
export const PROGRESSIVE_STEP_DELAY_MS = 60;  // пауза между чанками

// Live-поллинг: каждый тик тянет только последний бар (/api/last-bar),
// а полный /api/data — раз в POLL_FULL_RELOAD_MS (страховка от рассинхрона).
export const POLL_INTERVAL_MS = 2500;
export const POLL_FULL_RELOAD_MS = 60000;
