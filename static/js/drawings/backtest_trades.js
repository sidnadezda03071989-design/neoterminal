// BacktestTradesRenderer — primitive для отрисовки сделок бэктеста на графике.
// Паттерн (paneViews/attached/detach/attachPrimitive) повторяет DrawingsManager.
export class BacktestTradesRenderer {
  constructor(chart, series, container, candlesRef) {
    this.chart = chart;
    this.series = series;
    this.container = container;
    this.candlesRef = candlesRef || (() => []);
    this.trades = [];
    this.primitive = null;
    this._warnedSkipped = false;
  }

  render(trades) {
    this.clear();
    this.trades = Array.isArray(trades) ? trades : [];
    this._warnedSkipped = false;
    this._loggedOnce = false; // BLOCK-36: диагностика — раз за рендер
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
      // блоков, сколько сделок в статистике. Сделки вне загруженных свечей
      // (toPx == null) не рисуются, но НЕ отбрасываются из счётчика — блоки
      // появятся при zoom-out (они просто вне текущего viewport).
      // BLOCK-36: диагностика — сколько блоков реально нарисовано/пропущено.
      // Флаг на manager, т.к. impl-объект создаётся заново на каждый кадр.
      let drawn = 0, skipped = 0;
      for (const t of this.trades) {
        const pEntry = this.manager.toPx(t.entry_time, t.entry_price);
        if (!pEntry) { skipped++; continue; }
        this._drawTrade(ctx, size, t, pEntry);
        drawn++;
      }
      if (!this.manager._loggedOnce) {
        this.manager._loggedOnce = true;
        console.log(`[viz] drew ${drawn}/${this.trades.length} trades ` +
          `(null coords: ${skipped})`);
      }
      ctx.restore();
    });
  }

  _drawTrade(ctx, mediaSize, t, pEntry) {
    const mgr = this.manager;

    // Точки входа и выхода
    const pExit = t.exit_time ? mgr.toPx(t.exit_time, t.exit_price) : null;
    const pTp = t.tp_time ? mgr.toPx(t.tp_time, t.tp_price) : null;
    const pSl = t.sl_time ? mgr.toPx(t.sl_time, t.sl_price) : null;
    const pTpLine = mgr.toPx(t.entry_time, t.tp_price);   // для линии TP на всю ширину сделки
    const pSlLine = mgr.toPx(t.entry_time, t.sl_price);

    const xEntry = pEntry.x;
    // BLOCK-36-fix5: ширина блока — минимум 20px, иначе на плоских сделках
    // видна вертикальная черта вместо блока.
    let xExit = pExit ? pExit.x : xEntry + 20;
    if (xExit - xEntry < 20) xExit = xEntry + 20;
    const width = xExit - xEntry;
    const isWin = t.pnl >= 0;
    const MIN_HEIGHT = 30; // BLOCK-36-fix4/fix5: минимальная высота блока, px

    // 1. Прямоугольник сделки. Каскад источников Y-координат:
    //    TP/SL → entry/exit → fallback 30px вокруг entry (BLOCK-36-fix5:
    //    раньше при отсутствии TP/SL и exit-координат блок вообще не
    //    рисовался — на графике оставался только треугольник входа).
    let yTop = null, yBot = null;
    if (pTpLine && pSlLine) {
      // Вариант 1: есть TP/SL — прямоугольник от TP до SL
      yTop = Math.min(pTpLine.y, pSlLine.y);
      yBot = Math.max(pTpLine.y, pSlLine.y);
    } else if (pExit) {
      // Вариант 2: нет TP/SL — блок от entry до exit
      yTop = Math.min(pEntry.y, pExit.y);
      yBot = Math.max(pEntry.y, pExit.y);
    } else {
      // Вариант 3: НЕТ ни TP/SL, ни exit → блок 30px вокруг entry
      yTop = pEntry.y - MIN_HEIGHT / 2;
      yBot = pEntry.y + MIN_HEIGHT / 2;
    }

    const centerY = (yTop + yBot) / 2;
    if (yBot - yTop < MIN_HEIGHT) {
      yTop = centerY - MIN_HEIGHT / 2;
      yBot = centerY + MIN_HEIGHT / 2;
    }

    // Заливка + обводка блока (обводка сплошным цветом — контрастнее,
    // чем rgba-граница из fix4).
    ctx.fillStyle = isWin ? 'rgba(63, 185, 80, 0.25)' : 'rgba(248, 81, 73, 0.25)';
    ctx.fillRect(xEntry, yTop, width, yBot - yTop);
    ctx.strokeStyle = isWin ? '#3fb950' : '#f85149';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(xEntry, yTop, width, yBot - yTop);

    // 2. Линия entry (горизонтальная пунктирная)
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(xEntry, pEntry.y);
    ctx.lineTo(xExit, pEntry.y);
    ctx.stroke();
    ctx.setLineDash([]);

    // 3. Точка входа (треугольник вверх/вниз, 12px высота / 14px ширина,
    //    белая обводка 1px — читается на любом фоне)
    ctx.fillStyle = t.direction === 'BUY' ? '#3fb950' : '#f85149';
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 1;
    ctx.beginPath();
    if (t.direction === 'BUY') {
      ctx.moveTo(xEntry, pEntry.y + 12);
      ctx.lineTo(xEntry - 7, pEntry.y + 24);
      ctx.lineTo(xEntry + 7, pEntry.y + 24);
    } else {
      ctx.moveTo(xEntry, pEntry.y - 12);
      ctx.lineTo(xEntry - 7, pEntry.y - 24);
      ctx.lineTo(xEntry + 7, pEntry.y - 24);
    }
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // 4. Точка выхода
    if (pExit) {
      ctx.fillStyle = isWin ? '#3fb950' : '#f85149';
      ctx.beginPath();
      ctx.arc(pExit.x, pExit.y, 3, 0, Math.PI * 2);
      ctx.fill();
    }

    // 5. Метка с PnL — в центре блока
    const labelY = yTop + (yBot - yTop) / 2;
    const pct = t.pnl_pct != null ? t.pnl_pct : 0;
    const label = (isWin ? '+' : '') + pct.toFixed(2) + '%';
    ctx.font = 'bold 11px "Segoe UI", Tahoma, sans-serif';
    ctx.fillStyle = isWin ? '#3fb950' : '#f85149';
    const tw = ctx.measureText(label).width;
    ctx.fillText(label, xEntry + width / 2 - tw / 2, labelY + 4);

    // 6. Маркер TP (маленький зелёный кружок) если сработал
    if (t.exit_reason === 'tp' && pTp) {
      ctx.fillStyle = 'rgba(63, 185, 80, 0.9)';
      ctx.beginPath();
      ctx.arc(pTp.x, pTp.y, 2, 0, Math.PI * 2);
      ctx.fill();
    }

    // 7. Маркер SL (маленький красный кружок) если сработал
    if (t.exit_reason === 'sl' && pSl) {
      ctx.fillStyle = 'rgba(248, 81, 73, 0.9)';
      ctx.beginPath();
      ctx.arc(pSl.x, pSl.y, 2, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}