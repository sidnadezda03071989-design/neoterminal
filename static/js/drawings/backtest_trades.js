// BacktestTradesRenderer — primitive для отрисовки сделок бэктеста на графике.
// Паттерн (paneViews/attached/detach/attachPrimitive) повторяет DrawingsManager.

// BLOCK-40: логика отрисовки позиции переписана с нуля. Вид СТРОГО как на
// макете long/short position:
//   • зелёная зона  — entry → TP, на всю ширину сделки (вход → выход);
//   • красная зона  — entry → SL, на ту же ширину;
//   • белая пунктирная вертикаль на баре входа — левая граница позиции;
//   • серая пунктирная линия от точки входа к точке выхода;
//   • подпись НАД позицией (BLOCK-41): чем закрылась — take/stop — и
//     результат в % со знаком ('take +0.42%' / 'stop -0.21%').
// Соотношение зон СТРОГО 2:1: TP = 2×ATR, SL = 1×ATR от входа
// (config.BACKTEST_TP_ATR / config.BACKTEST_SL_ATR). Высота зон зависит от
// волатильности: бэкенд отдаёт atr_at_entry (ATR на баре входа), по нему
// фронтенд восстанавливает TP/SL, если уровней нет. Сделки не перекрываются:
// правая граница блока зажимается по входу следующей сделки.
// Последний костыль: если ни tp_price/sl_price, ни atr_at_entry нет (легаси) —
// зоны рисуются фиксированными смещениями 40px (TP) / 20px (SL) от входа.

const GREEN_FILL = 'rgba(38, 166, 154, 0.35)';  // зона профита (entry → TP)
const RED_FILL = 'rgba(239, 83, 80, 0.35)';     // зона риска (entry → SL)
const ENTRY_LINE = 'rgba(255, 255, 255, 0.55)'; // пунктирная вертикаль входа
const EXIT_LINE = 'rgba(158, 165, 178, 0.9)';   // пунктир вход → выход
const GREEN_TEXT = '#26a69a';                   // подпись TP / профит, %
const RED_TEXT = '#ef5350';                     // подпись SL / убыток, %
const NEUTRAL_TEXT = '#9ea5b2';                 // pnl == 0

const MIN_WIDTH = 20; // мин ширина позиции, px (сделка без координаты выхода)

// Множители для зон, когда tp_price/sl_price нет на бэкенде: TP/SL
// восстанавливаются из atr_at_entry — SL = sl_atr×ATR, TP = rr×sl_atr×ATR от
// входа (те же rr/sl_atr, что у сканера: настройка «Базовый R/R» +
// config.BACKTEST_SL_ATR). Заполняются из /api/scan/grids (setLevelConfig).
let levelCfg = { rr: 2.0, slAt: 1.0 };

/* Синхронизировать множители уровней с настройкой сканера (rr = «базовый
   R/R», slAt = стоп ×ATR из /api/scan/grids). */
export function setLevelConfig(rr, slAt) {
  if (Number.isFinite(Number(rr))) levelCfg.rr = Number(rr);
  if (Number.isFinite(Number(slAt))) levelCfg.slAt = Number(slAt);
}

export class BacktestTradesRenderer {
  constructor(chart, series, container, candlesRef) {
    this.chart = chart;
    this.series = series;
    this.container = container;
    this.candlesRef = candlesRef || (() => []);
    this.trades = [];
    this.primitive = null;
  }

  render(trades) {
    this.clear();
    this.trades = Array.isArray(trades) ? trades : [];
    this._loggedOnce = false; // диагностика — раз за рендер
    if (this.trades.length === 0) return;

    this.primitive = new BacktestTradesPrimitive(this.trades, this);
    this.series.attachPrimitive(this.primitive);
  }

  clear() {
    if (this.primitive) {
      try { this.series.detachPrimitive(this.primitive); } catch (e) {}
      this.primitive = null;
    }
    this.trades = [];
  }

  // Вызывается primitive'ом при рендере.
  // timeToCoordinate возвращает null вне загруженных свечей — как в
  // DrawingsManager._timeToX, экстраполируем через barSpacing на основе
  // последней свечи из candlesRef.
  toPx(time, price) {
    const ts = this.chart.timeScale();
    let x = ts.timeToCoordinate(time);
    if (x == null) x = this._extrapolateX(time, ts);
    const y = this.series.priceToCoordinate(price);
    if (x == null || y == null) return null;
    return { x: x, y: y };
  }

  _extrapolateX(time, ts) {
    const arr = this.candlesRef();
    if (!arr || !arr.length) return null;
    const last = arr[arr.length - 1];
    const xLast = ts.timeToCoordinate(last.time);
    if (xLast == null) return null;
    const spacing = ts.options().barSpacing || 6;
    let step = 60;
    if (arr.length >= 2) step = last.time - arr[arr.length - 2].time;
    if (step <= 0) step = 60;
    return xLast + ((time - last.time) / step) * spacing;
  }
}

class BacktestTradesPrimitive {
  constructor(trades, manager) {
    this.trades = trades;
    this.manager = manager;
    this._requestUpdate = null;
  }

  attached(param) {
    if (param && typeof param.requestUpdate === 'function') {
      this._requestUpdate = param.requestUpdate;
    }
  }

  detached() { this._requestUpdate = null; }

  updateAllViews() {}

  paneViews() {
    return [{
      zOrder: () => 'top',
      renderer: () => new BacktestTradesRendererImpl(this.trades, this.manager),
    }];
  }
}

class BacktestTradesRendererImpl {
  constructor(trades, manager) {
    this.trades = trades;
    this.manager = manager;
  }

  draw(target) {
    target.useMediaCoordinateSpace((scope) => {
      const ctx = scope.context, size = scope.mediaSize;
      ctx.save();
      ctx.beginPath(); ctx.rect(0, 0, size.width, size.height); ctx.clip();
      // ОБХОДИМ ВСЕ сделки без среза/break: на графике должно быть столько же
      // позиций, сколько сделок в статистике. Сделки вне загруженных свечей
      // (toPx == null) не рисуются, но НЕ отбрасываются из счётчика — блоки
      // появятся при zoom-out (они просто вне текущего viewport).
      let drawn = 0, skipped = 0;
      for (let i = 0; i < this.trades.length; i++) {
        // Следующая сделка передаётся для запрета перекрытия: блок позиции
        // не имеет права залезть на блок соседней сделки.
        if (this._drawTrade(ctx, this.trades[i], this.trades[i + 1])) drawn++;
        else skipped++;
      }
      if (!this.manager._loggedOnce) {
        this.manager._loggedOnce = true;
        console.log(`[viz] drew ${drawn}/${this.trades.length} positions ` +
          `(no entry coords skipped: ${skipped})`);
      }
      ctx.restore();
    });
  }

  // Одна позиция = две зоны (профит/риск) + пунктиры входа и вход→выход
  // + подпись НАД позицией: чем закрылась (TP/SL) и результат в %.
  // Возвращает false только если нет пиксельной координаты входа — без неё
  // позицию рисовать не из чего. Всё, что попало за пределы видимой области,
  // отсекает canvas (lightweight-charts клипует сам).
  _drawTrade(ctx, t, nextTrade) {
    const mgr = this.manager;

    const pEntry = mgr.toPx(t.entry_time, t.entry_price);
    if (!pEntry) return false;

    // Ширина позиции: вход → выход. Выхода нет/вне экрана — MIN_WIDTH,
    // чтобы позиция не схлопывалась в вертикальную черту.
    const pExit = t.exit_time ? mgr.toPx(t.exit_time, t.exit_price) : null;
    let xEnd = pExit ? pExit.x : pEntry.x + MIN_WIDTH;
    if (xEnd - pEntry.x < MIN_WIDTH) xEnd = pEntry.x + MIN_WIDTH;

    // BLOCK-41: сделка НЕ МОЖЕТ быть внутри другой — правая граница блока
    // зажимается по входу следующей сделки (движок открывает новую позицию
    // только после закрытия предыдущей, но зажим страхует визуально).
    if (nextTrade) {
      const pNext = mgr.toPx(nextTrade.entry_time, nextTrade.entry_price);
      if (pNext && pNext.x > pEntry.x && xEnd > pNext.x) xEnd = pNext.x;
    }
    const width = xEnd - pEntry.x;

    // Y-уровни TP/SL: высота зон пропорциональна реальной волатильности.
    // Готовые tp_price/sl_price есть у каждой сделки run_backtest. Если уровня
    // нет — восстанавливаем из atr_at_entry (ATR на баре входа): TP = entry
    // ± 2×ATR, SL = entry ∓ 1×ATR. И только если нет ни уровней, ни atr
    // (легаси-данные) — последний костыль: фикс-смещения 40px/20px от входа,
    // чтобы зелёная зона визуально осталась ровно в 2 раза больше красной.
    const isLong = t.direction !== 'SELL';
    const dir = isLong ? 1 : -1;
    const atr = (t.atr_at_entry != null && isFinite(t.atr_at_entry))
      ? Number(t.atr_at_entry) : 0;
    const tpPrice = t.tp_price != null ? t.tp_price
      : (atr > 0 ? t.entry_price + dir * levelCfg.rr * levelCfg.slAt * atr : null);
    const slPrice = t.sl_price != null ? t.sl_price
      : (atr > 0 ? t.entry_price - dir * levelCfg.slAt * atr : null);
    const pTp = tpPrice != null ? mgr.toPx(t.entry_time, tpPrice) : null;
    const pSl = slPrice != null ? mgr.toPx(t.entry_time, slPrice) : null;
    const yProfit = pTp ? pTp.y : pEntry.y + (isLong ? -40 : 40);
    const yRisk = pSl ? pSl.y : pEntry.y + (isLong ? 20 : -20);

    // exit_reason: бэкенд отдаёт его для каждой сделки ('tp'/'sl'/'signal'/
    // 'end'). Если вдруг нет (легаси) — определяем по тому, какой уровень
    // (TP или SL) ближе к exit_price: чья линия встретилась первой, тот и виноват.
    let exitReason = t.exit_reason;
    if (!exitReason && t.exit_price != null && tpPrice != null && slPrice != null) {
      const dTp = Math.abs(t.exit_price - tpPrice);
      const dSl = Math.abs(t.exit_price - slPrice);
      exitReason = dTp < dSl ? 'tp' : (dSl < dTp ? 'sl' : '');
    }

    // 1. Зона профита: rect между entry и TP (для SHORT TP ниже entry —
    //    формула rect между двумя Y одинаковая).
    ctx.fillStyle = GREEN_FILL;
    ctx.fillRect(pEntry.x, Math.min(pEntry.y, yProfit), width,
      Math.abs(yProfit - pEntry.y));

    // 2. Зона риска: rect между entry и SL (ровно вдвое меньше зоны профита).
    ctx.fillStyle = RED_FILL;
    ctx.fillRect(pEntry.x, Math.min(pEntry.y, yRisk), width,
      Math.abs(yRisk - pEntry.y));

    // 3. Белая пунктирная вертикаль входа — левая граница позиции, от верха
    //    зелёной зоны до низа красной (как на макете).
    const yTop = Math.min(pEntry.y, yProfit, yRisk);
    const yBot = Math.max(pEntry.y, yProfit, yRisk);
    ctx.strokeStyle = ENTRY_LINE;
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(pEntry.x, yTop);
    ctx.lineTo(pEntry.x, yBot);
    ctx.stroke();

    // 4. Серый пунктир от точки входа к точке выхода (траектория результата).
    if (pExit) {
      ctx.strokeStyle = EXIT_LINE;
      ctx.beginPath();
      ctx.moveTo(pEntry.x, pEntry.y);
      ctx.lineTo(pExit.x, pExit.y);
      ctx.stroke();
    }

    // 5. Подпись НАД позицией: чем закрылась сделка (take/stop — а также
    //    SIG/END для сигнальных выходов) и результат в % со знаком.
    //    Слово — по ЛИНИИ, которую цена коснулась первой (exit_reason):
    //    стоп, задетый раньше тейка, всегда называется stop, а не take.
    const pct = t.pnl_pct != null ? Number(t.pnl_pct) : 0;
    const reasonText =
      exitReason === 'sl' ? 'stop'
      : exitReason === 'tp' ? 'take'
      : exitReason === 'signal' ? 'SIG'
      : exitReason === 'end' ? 'END' : '—';
    const label = reasonText + (pct >= 0 ? ' +' : ' ') + pct.toFixed(2) + '%';
    ctx.font = 'bold 11px "Segoe UI", Tahoma, sans-serif';
    ctx.fillStyle = t.pnl > 0 ? GREEN_TEXT
      : (t.pnl < 0 ? RED_TEXT : NEUTRAL_TEXT);
    ctx.fillText(label, pEntry.x + 2, yTop - 5);

    ctx.setLineDash([]);
    return true;
  }
}
