export const COLORS = {
  bg: '#131722', grid: '#1e222d', border: '#2a2e39',
  text: '#d1d4dc', up: '#26a69a', down: '#ef5350',
  sma: '#2196f3', ema: '#ff9800',
  bbUp: '#7e57c2', bbMid: '#7e57c2', bbLow: '#7e57c2',
  rsi: '#ab47bc', macd: '#2962ff', macdSignal: '#ff9800',
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
// т.е. 5000→10000→15000→20000 = 4 дешёвых шага (~1.5 мин вместо 8).
export const PROGRESSIVE_STEP = 5000;
export const PROGRESSIVE_STEP_DELAY_MS = 200;  // пауза между чанками

// Live-поллинг: каждый тик тянет только последний бар (/api/last-bar),
// а полный /api/data — раз в POLL_FULL_RELOAD_MS (страховка от рассинхрона).
// Основной канал обновления графика — SSE (событие candle_update, ~1с);
// HTTP-поллинг оставлен как fallback при обрыве SSE. Интервал уменьшен
// до 5000 мс: если SSE молчит (обрыв/старый сервер), график всё равно
// обновляется хотя бы раз в 5 секунд — новая 1m-свеча не «зависает».
export const POLL_INTERVAL_MS = 5000;
export const POLL_FULL_RELOAD_MS = 60000;

/* Длительности ТФ в секундах — для таймера закрытия свечи (как в TV).
   Держим здесь (а не на бэке), чтобы тикер работал локально и без сети. */
export const TF_SECONDS = {
  '1m': 60, '5m': 300, '15m': 900,
  '1H': 3600, '4H': 14400, '1D': 86400,
};
