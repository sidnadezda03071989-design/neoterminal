// AIProbZonesRenderer — forward-зоны вероятностей AI Backtest («Псевдо-Харон»).
// Паттерн (paneViews/attached/detach) — как BacktestTradesRenderer: рисуем в
// media-координатах canvas.
//
// Показывает ВСЕ уровни последнего расчёта: несколько уровней UP (выше цены,
// зелёные) и DOWN (ниже цены, красные). Каждый уровень:
//   • тонкая горизонтальная линия на цене уровня;
//   • пилюля у правой кромки зоны «+{diff} {prob}%» (diff — расстояние от
//     текущей цены, prob — вероятность дойти);
//   • полупрозрачная полоса-зона между СОСЕДНИМИ уровнями (градиент ярче у
//     линии дальнего уровня) — так зоны не перекрывают друг друга, как на
//     макете: ближняя полоса самая насыщенная, дальние — бледнее.
//
// Показываются только уровни с вероятностью не ниже MIN_PROB (мелкий шум
// LLM не засоряет график).
//
// Зоны проецируются ПРАВЕЕ последней видимой свечи (якорь — срез реплея в
// replay, последний бар в live), до правой кромки графика. Никаких
// сделок/вход-выходов — только уровни.

const UP_FILL = 'rgba(38, 166, 154, 0.40)';      // зона вверх
const DOWN_FILL = 'rgba(239, 83, 80, 0.40)';     // зона вниз
const LEVEL_LINE = 'rgba(255, 255, 255, 0.55)';  // линия уровня
const UP_PILL = '#26a69a';
const DOWN_PILL = '#ef5350';
const PILL_TEXT = '#ffffff';

const MIN_WIDTH = 20;    // мин ширина зоны, px
const MIN_PROB = 0.10;   // уровни с вероятностью ниже — не рисуем

// Прозрачность полосы уровня: ближний (самый вероятный) — насыщеннее.
// baseAlpha × (1 - 0.55 × index/maxIndex).
function _bandAlpha(baseAlpha, index, count) {
  if (count <= 1) return baseAlpha;
  return baseAlpha * (1 - 0.55 * (index / (count - 1)));
}

export class AIProbZonesRenderer {
  constructor(chart, series, container, candlesRef) {
    this.chart = chart;
    this.series = series;
    this.container = container;
    this.candlesRef = candlesRef || (() => []);
    this.levels = [];
    this.anchor = null;
    this.primitive = null;
  }

  /* render(result | levels, anchorOverride)
     result — ответ /api/ai-backtest ({levels, price, upto_sec, ...}) либо
     просто массив уровней. anchorOverride — {time, price} среза (для replay:
     барьер), иначе якорь берётся из полного ряда свечей. */
  render(result, anchorOverride) {
    this.clear();
    const levels = this._extractLevels(result);
    if (!levels.length) return;
    this.levels = levels;
    this.anchor = anchorOverride || this._anchor();
    if (!this.anchor) return;
    this.primitive = new AIProbZonesPrimitive(this.levels, this.anchor, this);
    this.series.attachPrimitive(this.primitive);
  }

  _extractLevels(result) {
    const raw = Array.isArray(result) ? result
      : (result && Array.isArray(result.levels)) ? result.levels : [];
    const out = [];
    for (const lv of raw) {
      if (!lv || !Number.isFinite(Number(lv.price))) continue;
      const prob = Number(lv.probability);
      if (!Number.isFinite(prob) || prob < MIN_PROB) continue;
      out.push({
        side: String(lv.side || '').toUpperCase() === 'DOWN' ? 'DOWN' : 'UP',
        price: Number(lv.price),
        probability: prob,
      });
    }
    return out;
  }

  clear() {
    if (this.primitive) {
      try { this.series.detachPrimitive(this.primitive); } catch (e) {}
      this.primitive = null;
    }
    this.levels = [];
    this.anchor = null;
  }

  /* Якорь зоны: последняя свеча ПЕРЕДАННОГО якоря (replay-барьер ставится
     вызывающим кодом), иначе последний бар полного ряда state.candles —
     не зависит от режима live/replay. */
  _anchor() {
    const arr = this.candlesRef();
    if (arr && arr.length) {
      const c = arr[arr.length - 1];
      return { time: Number(c.time), price: Number(c.close) };
    }
    return null;
  }

  // timeToCoordinate возвращает null вне загруженных свечей — экстраполируем
  // через barSpacing на основе последней свечи (как в BacktestTradesRenderer).
  toPx(time, price) {
    if (time == null || price == null || !Number.isFinite(Number(price))) return null;
    const ts = this.chart.timeScale();
    let x = ts.timeToCoordinate(Number(time));
    if (x == null) x = this._extrapolateX(Number(time), ts);
    const y = this.series.priceToCoordinate(Number(price));
    if (x == null || y == null) return null;
    return { x: x, y: y };
  }

  _extrapolateX(time, ts) {
    const arr = this.candlesRef();
    if (!arr || !arr.length) return null;
    const last = arr[arr.length - 1];
    const xLast = ts.timeToCoordinate(Number(last.time));
    if (xLast == null) return null;
    const spacing = ts.options().barSpacing || 6;
    let step = 60;
    if (arr.length >= 2) step = Number(last.time) - Number(arr[arr.length - 2].time);
    if (step <= 0) step = 60;
    return xLast + ((time - Number(last.time)) / step) * spacing;
  }
}

class AIProbZonesPrimitive {
  constructor(levels, anchor, manager) {
    this.levels = levels;
    this.anchor = anchor;
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
      renderer: () => new AIProbZonesRendererImpl(
        this.levels, this.anchor, this.manager),
    }];
  }
}

class AIProbZonesRendererImpl {
  constructor(levels, anchor, manager) {
    this.levels = levels;
    this.anchor = anchor;
    this.manager = manager;
  }

  draw(target) {
    target.useMediaCoordinateSpace((scope) => {
      const ctx = scope.context, size = scope.mediaSize;
      ctx.save();
      ctx.beginPath(); ctx.rect(0, 0, size.width, size.height); ctx.clip();
      this._drawForward(ctx, size);
      ctx.restore();
    });
  }

  _drawForward(ctx, size) {
    const mgr = this.manager;
    const anchor = this.anchor;
    if (!anchor) return;
    const series = mgr.series;
    const ts = mgr.chart.timeScale();

    // X-якорь: координата стартового бара среза; если она вне экрана
    // (replay-барьер слева) — зоны всё равно рисуются от правой кромки влево.
    const spacing = ts.options().barSpacing || 6;
    const xLast = ts.timeToCoordinate(anchor.time);
    let x0 = xLast != null ? xLast + spacing
                           : Math.max(MIN_WIDTH, size.width - MIN_WIDTH);
    let x1 = size.width;
    if (x1 - x0 < MIN_WIDTH) x1 = x0 + MIN_WIDTH;

    // Y-якорь: цена среза, от неё строятся все полосы.
    const yPrice = series.priceToCoordinate(anchor.price);
    if (yPrice == null) return;

    const ups = this.levels.filter((l) => l.side === 'UP')
      .sort((a, b) => a.price - b.price);          // ближний → дальний
    const downs = this.levels.filter((l) => l.side === 'DOWN')
      .sort((a, b) => b.price - a.price);

    this._drawSide(ctx, x0, x1, size, yPrice, anchor.price, ups, true);
    this._drawSide(ctx, x0, x1, size, yPrice, anchor.price, downs, false);
  }

  /* Полосы одной стороны: от текущей цены до ПЕРВОГО уровня, затем между
     соседними уровнями. index 0 — ближайший уровень (самая насыщенная
     полоса), последний — самый бледный. */
  _drawSide(ctx, x0, x1, size, yPrice, anchorPrice, levels, isUp) {
    const series = this.manager.series;
    for (let i = 0; i < levels.length; i++) {
      const lv = levels[i];
      const yLevel = series.priceToCoordinate(lv.price);
      if (yLevel == null) continue;
      const yPrev = i === 0 ? yPrice
        : series.priceToCoordinate(levels[i - 1].price);
      const yFrom = yPrev == null ? yPrice : yPrev;
      const top = Math.min(yFrom, yLevel);
      const bot = Math.max(yFrom, yLevel);
      const alpha = _bandAlpha(isUp ? 0.40 : 0.40, i, levels.length);
      const rgb = isUp ? '38, 166, 154' : '239, 83, 80';
      const grad = ctx.createLinearGradient(0, top, 0, bot);
      // Градиент ярче у ЛИНИИ уровня (внешняя граница полосы): у UP — сверху
      // (дальний уровень выше), у DOWN — снизу.
      if (isUp) {
        grad.addColorStop(0, `rgba(${rgb}, ${alpha})`);
        grad.addColorStop(1, `rgba(${rgb}, 0)`);
      } else {
        grad.addColorStop(0, `rgba(${rgb}, 0)`);
        grad.addColorStop(1, `rgba(${rgb}, ${alpha})`);
      }
      ctx.fillStyle = grad;
      ctx.fillRect(x0, top, x1 - x0, Math.max(1, bot - top));

      this._drawLevelLine(ctx, x0, x1, yLevel);
      this._drawPill(ctx, x1, lv.price - anchorPrice, lv.probability,
        isUp ? UP_PILL : DOWN_PILL, yLevel, top, bot, size);
    }
  }

  // Тонкая горизонтальная линия на цене уровня.
  _drawLevelLine(ctx, x0, x1, y) {
    ctx.strokeStyle = LEVEL_LINE;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x0, y);
    ctx.lineTo(x1, y);
    ctx.stroke();
  }

  // Пилюля у правой кромки зоны: скруглённый прямоугольник с текстом
  // «+{diff} {prob}%» (diff — расстояние уровня от текущей цены).
  _drawPill(ctx, xRight, diff, prob, bg, yLevel, top, bot, size) {
    const label = (diff >= 0 ? '+' : '−') + this._fmtPrice(Math.abs(diff)) +
      ' ' + this._fmtProb(prob) + '%';
    ctx.font = 'bold 10px "Segoe UI", Tahoma, sans-serif';
    const w = ctx.measureText(label).width;
    const padX = 5;
    const pillW = w + padX * 2;
    const pillH = 16;
    let y = yLevel - pillH / 2;
    y = Math.max(top - pillH, Math.min(bot, y));
    // Не уводить пилюлю за нижнюю кромку холста.
    y = Math.min(y, size.height - pillH - 2);
    const x = xRight - pillW;
    if (x < 0) return;

    ctx.fillStyle = bg;
    this._roundRect(ctx, x, y, pillW, pillH, 4);
    ctx.fill();
    ctx.fillStyle = PILL_TEXT;
    ctx.textBaseline = 'middle';
    ctx.fillText(label, x + padX, y + pillH / 2 + 0.5);
    ctx.textBaseline = 'alphabetic';
  }

  _roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  _fmtPrice(v) {
    if (v == null || !Number.isFinite(v)) return '?';
    const av = Math.abs(v);
    const dg = av >= 1000 ? 0 : av >= 1 ? 1 : av >= 0.01 ? 3 : 6;
    return Number(v).toFixed(dg);
  }

  _fmtProb(v) {
    const n = Number(v);
    if (!Number.isFinite(n)) return '?';
    const pct = n * 100;
    const dg = pct >= 100 || pct === Math.round(pct) ? 0 : 1;
    return pct.toFixed(dg);
  }
}
