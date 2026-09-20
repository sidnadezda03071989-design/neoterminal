// BacktestTradesRenderer — primitive для отрисовки сделок бэктеста на графике.
// Паттерн (paneViews/attached/detach/attachPrimitive) повторяет DrawingsManager.

// BLOCK-36-fix9: ручная валидация пиксельных координат (_validPx из fix6)
// убрана — она отбрасывала сделки с y вне canvas ("[viz] drew 1/97 trades"),
// хотя lightweight-charts/canvas сами клипуют всё за пределами области.

// BLOCK-39: подпись уровня позиции. Без цены и % от входа линия стопа ничего
// не сообщает, а у сделок с выходом по стопу уровень SL совпадает с ценой
// выхода — раньше подписи "OUT" и "SL" печатались в одну точку, и стоп
// визуально пропадал.
function _fmtPrice(v) {
  const n = Number(v);
  if (!isFinite(n)) return '—';
  return Math.abs(n) >= 10 ? n.toFixed(2) : n.toFixed(4);
}

function _levelPct(entry, level) {
  const e = Number(entry), l = Number(level);
  if (!isFinite(e) || !isFinite(l) || e === 0) return '';
  const pct = ((l - e) / e) * 100;
  return ' (' + (pct >= 0 ? '+' : '') + pct.toFixed(2) + '%)';
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
        // BLOCK-36-fix9: _drawTrade возвращает false ТОЛЬКО когда нет точки
        // входа; координаты за пределами области рисования не отбраковываются.
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

  // Возвращает true, если сделка нарисована; false — только если нет
  // пиксельной координаты входа (BLOCK-36-fix9). Ручная проверка "y внутри
  // canvas" убрана: lightweight-charts клипует всё за пределами области сам.
  _drawTrade(ctx, mediaSize, t) {
    const mgr = this.manager;
    const h = mediaSize.height;

    // BLOCK-36-fix6: диагностика — первая сделка один раз за рендер.
    if (!mgr._firstTraceDone && this.trades.length > 0) {
      const t0 = this.trades[0];
      const p0 = mgr.toPx(t0.entry_time, t0.entry_price);
      const pTp0 = t0.tp_price != null
        ? mgr.toPx(t0.entry_time, t0.tp_price) : null;
      const pSl0 = t0.sl_price != null
        ? mgr.toPx(t0.entry_time, t0.sl_price) : null;
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

    // Точки сделки (BLOCK-36-fix9 — простой расчёт без проверки координат).
    // Проверок ровно две: есть ли цена (null-цены: сделка без TP/SL из
    // сканера, ещё не сработавший TP/SL) и что toPx вернул координату.
    // Всё, что попало за пределы видимой области, отсекает canvas.
    const pEntry = mgr.toPx(t.entry_time, t.entry_price);
    // Без точки входа рисовать нечего: entry — база для блока, линии входа,
    // треугольника и метки PnL.
    if (!pEntry) return;

    const pExit = t.exit_time ? mgr.toPx(t.exit_time, t.exit_price) : null;
    const pTp = t.tp_time ? mgr.toPx(t.tp_time, t.tp_price) : null;
    const pSl = t.sl_time ? mgr.toPx(t.sl_time, t.sl_price) : null;
    // Линии TP/SL на всю ширину сделки: цена null (сделка без TP/SL) → null,
    // каскад ниже возьмёт entry/exit.
    const pTpLine = t.tp_price != null
      ? mgr.toPx(t.entry_time, t.tp_price) : null;
    const pSlLine = t.sl_price != null
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

    // После расширения до MIN_HEIGHT зажимаем ГЕОМЕТРИЮ блока в пределы canvas
    // (BLOCK-36-fix9: это не отбраковка сделки — сама сделка рисуется всегда,
    // зажим лишь не даёт прямоугольнику уехать за верх/низ области).
    yTop = Math.max(0, yTop);
    yBot = Math.min(h, yBot);

    // Заливка + обводка блока (BLOCK-36-fix7: заливка 0.12 вместо 0.25 —
    // блок читается как подложка, поверх неё яркие линии уровней).
    ctx.fillStyle = isWin ? 'rgba(63, 185, 80, 0.12)' : 'rgba(248, 81, 73, 0.12)';
    ctx.fillRect(xEntry, yTop, width, yBot - yTop);
    ctx.strokeStyle = isWin ? 'rgba(63, 185, 80, 0.8)' : 'rgba(248, 81, 73, 0.8)';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(xEntry, yTop, width, yBot - yTop);

    // 1b. Зоны позиции (BLOCK-39): риск entry↔SL (красная) и профит entry↔TP
    //     (зелёная). Именно красная зона отвечает на вопрос "где стоп": у
    //     sl-выходов пунктир SL ложился ровно на сплошную линию OUT и
    //     пропадал, а зона видна всегда.
    if (pSlLine) {
      ctx.fillStyle = 'rgba(248, 81, 73, 0.22)';
      const yRisk = Math.min(pEntry.y, pSlLine.y);
      ctx.fillRect(xEntry, yRisk, width, Math.abs(pSlLine.y - pEntry.y));
    }
    if (pTpLine) {
      ctx.fillStyle = 'rgba(63, 185, 80, 0.22)';
      const yReward = Math.min(pEntry.y, pTpLine.y);
      ctx.fillRect(xEntry, yReward, width, Math.abs(pTpLine.y - pEntry.y));
    }

    // 2. Линии уровней сделки (BLOCK-36-fix7, метки — BLOCK-39). Метки IN/TP/SL
    //    — у правого края линии, метка OUT — у левого (справа её накрывали
    //    метки уровней, когда выход совпал с TP/SL); шрифт задаётся в блоке
    //    ENTRY и дальше переиспользуется.
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
      // BLOCK-39: метка OUT — слева от блока (справа её перекрывали метки
      // TP/SL, когда выход произошёл ровно по уровню).
      ctx.textAlign = 'right';
      ctx.fillText('OUT', xEntry - 4, pExit.y + 3);
      ctx.textAlign = 'left';
    }

    // Линия TP — пунктирная зелёная + цена уровня и % от входа (BLOCK-39:
    // 2px и длинный пунктир — уровень читается поверх сплошных линий).
    if (pTpLine) {
      ctx.strokeStyle = '#3fb950';
      ctx.lineWidth = 2;
      ctx.setLineDash([7, 4]);
      ctx.beginPath();
      ctx.moveTo(xEntry, pTpLine.y);
      ctx.lineTo(xEntry + width, pTpLine.y);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#3fb950';
      ctx.fillText('TP ' + _fmtPrice(t.tp_price)
        + _levelPct(t.entry_price, t.tp_price),
        xEntry + width + 3, pTpLine.y + 3);
    }

    // Линия SL — пунктирная красная + цена стопа и % от входа. Рисуется
    // последней из уровней: для sl-выходов она совпадает с линией OUT, и
    // раньше уходила под неё (BLOCK-39).
    if (pSlLine) {
      ctx.strokeStyle = '#f85149';
      ctx.lineWidth = 2;
      ctx.setLineDash([7, 4]);
      ctx.beginPath();
      ctx.moveTo(xEntry, pSlLine.y);
      ctx.lineTo(xEntry + width, pSlLine.y);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#f85149';
      ctx.fillText('SL ' + _fmtPrice(t.sl_price)
        + _levelPct(t.entry_price, t.sl_price),
        xEntry + width + 3, pSlLine.y + 3);
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

    return true; // BLOCK-36-fix9: сделка нарисована
  }
}