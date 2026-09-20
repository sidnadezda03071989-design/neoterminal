// BacktestTradesRenderer — primitive для отрисовки сделок бэктеста на графике.
// Паттерн (paneViews/attached/detach/attachPrimitive) повторяет DrawingsManager.

// BLOCK-36-fix6: валидная пиксельная координата — конечное число, не левее
// нуля по X и внутри canvas по Y. Хелпер нужен потому, что toPx(time, null)
// НЕ возвращает null: lightweight-charts считает y без проверки цены
// (null в арифметике → 0 → конечная координата далеко за canvas,
// undefined → NaN → fillRect молча ничего не рисует). Отсюда симптом
// "drew 97/97", при котором блоков на графике не видно.
function _validPx(p, canvasH) {
  return !!p && typeof p.x === 'number' && isFinite(p.x) && p.x >= 0
    && typeof p.y === 'number' && isFinite(p.y)
    && p.y >= 0 && p.y <= canvasH;
}

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
    this._firstTraceDone = false; // BLOCK-36-fix6: трассировка первой сделки
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
        // BLOCK-36-fix6: валидность координат проверяет сам _drawTrade
        // (невалидная точка входа → false, без рисования).
        if (this._drawTrade(ctx, size, t)) drawn++; else skipped++;
      }
      if (!this.manager._loggedOnce) {
        this.manager._loggedOnce = true;
        console.log(`[viz] drew ${drawn}/${this.trades.length} trades ` +
          `(invalid coords skipped: ${skipped})`);
      }
      ctx.restore();
    });
  }

  // Возвращает true, если сделка реально нарисована (BLOCK-36-fix6).
  _drawTrade(ctx, mediaSize, t) {
    const mgr = this.manager;
    const h = mediaSize.height;

    // BLOCK-36-fix6: диагностика — первая сделка один раз за рендер.
    if (!mgr._firstTraceDone && this.trades.length > 0) {
      const t0 = this.trades[0];
      const p0 = mgr.toPx(t0.entry_time, t0.entry_price);
      const pTp0 = mgr.toPx(t0.entry_time, t0.tp_price);
      const pSl0 = mgr.toPx(t0.entry_time, t0.sl_price);
      const pEx0 = t0.exit_time
        ? mgr.toPx(t0.exit_time, t0.exit_price) : null;
      console.log('[viz-trace] trade[0]:', {
        entry_price: t0.entry_price,
        exit_price: t0.exit_price,
        tp_price: t0.tp_price,
        sl_price: t0.sl_price,
        pEntry: p0, pExit: pEx0, pTp: pTp0, pSl: pSl0,
        canvasH: mgr.container?.clientHeight,
      });
      mgr._firstTraceDone = true;
    }

    // Точки входа и выхода. Валидной считается только координата-число
    // внутри canvas: null/отсутствующие цены lightweight-charts превращает
    // в y далеко за canvas или NaN (см. _validPx).
    const pEntry = _validPx(mgr.toPx(t.entry_time, t.entry_price), h)
      ? mgr.toPx(t.entry_time, t.entry_price) : null;
    // Без валидной точки входа рисовать нечего: entry — база для блока,
    // линии входа, треугольника и метки PnL.
    if (!pEntry) return false;

    const pExit = t.exit_time && t.exit_price != null
      && _validPx(mgr.toPx(t.exit_time, t.exit_price), h)
      ? mgr.toPx(t.exit_time, t.exit_price) : null;
    const pTp = t.tp_time && t.tp_price != null
      && _validPx(mgr.toPx(t.tp_time, t.tp_price), h)
      ? mgr.toPx(t.tp_time, t.tp_price) : null;
    const pSl = t.sl_time && t.sl_price != null
      && _validPx(mgr.toPx(t.sl_time, t.sl_price), h)
      ? mgr.toPx(t.sl_time, t.sl_price) : null;
    // Линии TP/SL на всю ширину сделки: цена null (signal/end-выход, скан
    // без TP/SL) или координата вне canvas → null, каскад возьмёт entry/exit.
    const pTpLine = t.tp_price != null
      && _validPx(mgr.toPx(t.entry_time, t.tp_price), h)
      ? mgr.toPx(t.entry_time, t.tp_price) : null;
    const pSlLine = t.sl_price != null
      && _validPx(mgr.toPx(t.entry_time, t.sl_price), h)
      ? mgr.toPx(t.entry_time, t.sl_price) : null;

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

    // BLOCK-36-fix6: после расширения до MIN_HEIGHT зажимаем блок в пределы
    // canvas — у сделки у края диапазона прямоугольник иначе уезжает за
    // границу и остаётся невидимым.
    yTop = Math.max(0, yTop);
    yBot = Math.min(h, yBot);

    // Заливка + обводка блока (BLOCK-36-fix7: заливка 0.12 вместо 0.25 —
    // блок читается как подложка, поверх неё яркие линии уровней).
    ctx.fillStyle = isWin ? 'rgba(63, 185, 80, 0.12)' : 'rgba(248, 81, 73, 0.12)';
    ctx.fillRect(xEntry, yTop, width, yBot - yTop);
    ctx.strokeStyle = isWin ? 'rgba(63, 185, 80, 0.8)' : 'rgba(248, 81, 73, 0.8)';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(xEntry, yTop, width, yBot - yTop);

    // 2. Линии уровней сделки (BLOCK-36-fix7). Метки IN/OUT/TP/SL — у правого
    //    края линии; шрифт задаётся в блоке ENTRY и дальше переиспользуется.
    //    Линия ENTRY — сплошная голубая (заменила белую пунктирную из fix6).
    ctx.strokeStyle = '#58a6ff';
    ctx.lineWidth = 2;
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(xEntry, pEntry.y);
    ctx.lineTo(xEntry + width, pEntry.y);
    ctx.stroke();
    ctx.font = 'bold 10px "Segoe UI", Tahoma, sans-serif';
    ctx.fillStyle = '#58a6ff';
    ctx.textAlign = 'left';
    ctx.fillText('IN', xEntry + width + 3, pEntry.y + 3);

    // Линия EXIT — сплошная, цвет по результату сделки.
    if (pExit) {
      const exitColor = isWin ? '#3fb950' : '#f85149';
      ctx.strokeStyle = exitColor;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(xEntry, pExit.y);
      ctx.lineTo(xEntry + width, pExit.y);
      ctx.stroke();
      ctx.fillStyle = exitColor;
      ctx.fillText('OUT', xEntry + width + 3, pExit.y + 3);
    }

    // Линия TP — пунктирная зелёная (только если у сделки есть tp_price).
    if (pTpLine) {
      ctx.strokeStyle = '#3fb950';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([5, 3]);
      ctx.beginPath();
      ctx.moveTo(xEntry, pTpLine.y);
      ctx.lineTo(xEntry + width, pTpLine.y);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#3fb950';
      ctx.fillText('TP', xEntry + width + 3, pTpLine.y + 3);
    }

    // Линия SL — пунктирная красная (только если у сделки есть sl_price).
    if (pSlLine) {
      ctx.strokeStyle = '#f85149';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([5, 3]);
      ctx.beginPath();
      ctx.moveTo(xEntry, pSlLine.y);
      ctx.lineTo(xEntry + width, pSlLine.y);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#f85149';
      ctx.fillText('SL', xEntry + width + 3, pSlLine.y + 3);
    }

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

    return true; // BLOCK-36-fix6: сделка нарисована
  }
}