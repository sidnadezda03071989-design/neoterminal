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
export const MIN_PROB = 0.10;   // уровни с вероятностью ниже — не рисуем
export const PER_SIDE_MIN = 2;  // минимум уровней на сторону: UP и DOWN

// Прозрачность полосы уровня: ближний (самый вероятный) — насыщеннее.
// baseAlpha × (1 - 0.55 × index/maxIndex).
function _bandAlpha(baseAlpha, index, count) {
  if (count <= 1) return baseAlpha;
  return baseAlpha * (1 - 0.55 * (index / (count - 1)));
}

/* Уровни для отображения (общий для панели и графика, чтобы счётчики
   совпадали): валидация price/prob + гарантия минимума PER_SIDE_MIN уровней
   на сторону. Сначала — уровни prob >= MIN_PROB (шум LLM отсекаем), потом,
   если на стороне осталось меньше PER_SIDE_MIN, добираем недостающие из
   остальных по убыванию вероятности — порог не «съедает» последний уровень.
   Если на стороне физически нет нужного числа уровней (например, LLM дала
   3 сверху и 1 снизу), такую сторону добираем резервным синтетическим
   уровнем рядом с якорем (anchorPrice) — гарантия «минимум 2 снизу и сверху». */
export function filterLevelsForDisplay(levels, anchorPrice) {
  const valid = [];
  for (const lv of (Array.isArray(levels) ? levels : [])) {
    if (!lv || !Number.isFinite(Number(lv.price))) continue;
    const prob = Number(lv.probability);
    if (!Number.isFinite(prob)) continue;
    valid.push({
      ...lv,
      side: String(lv.side || '').toUpperCase() === 'DOWN' ? 'DOWN' : 'UP',
      price: Number(lv.price),
      probability: prob,
    });
  }
  const anchor = Number.isFinite(Number(anchorPrice)) ? Number(anchorPrice)
    : _refPrice(valid);
  const ups = valid.filter((l) => l.side === 'UP');
  const downs = valid.filter((l) => l.side === 'DOWN');
  return _capTo100(
    _ensureSideMin(ups, anchor, 'UP').concat(_ensureSideMin(downs, anchor, 'DOWN'))
  );
}

/* Средняя цена уровней — fallback-якорь для резервных уровней (панель
   вызывается без якоря, в отличие от графика). */
function _refPrice(levels) {
  const ps = levels.map((l) => l.price).filter(Number.isFinite);
  if (!ps.length) return null;
  return ps.reduce((a, b) => a + b, 0) / ps.length;
}

function _synthLevel(side, ref) {
  const price = side === 'UP' ? ref * 1.01 : ref * 0.99;
  return {
    id: 'filler_' + side + '_' + Date.now().toString(36),
    price,
    probability: 0.02,
    side,
  };
}

function _ensureSideMin(list, anchor, side) {
  const good = list.filter((l) => l.probability >= MIN_PROB);
  if (good.length >= PER_SIDE_MIN) return good;
  const rest = list
    .filter((l) => l.probability < MIN_PROB)
    .sort((a, b) => b.probability - a.probability);
  const out = good.concat(rest.slice(0, PER_SIDE_MIN - good.length));
  if (out.length < PER_SIDE_MIN) {
    // Синтетический добор до гарантированного минимума на сторону.
    const sidePrices = list.map((l) => l.price).filter(Number.isFinite);
    const ref = anchor != null ? anchor
      : (side === 'UP' ? Math.max(...sidePrices) : Math.min(...sidePrices));
    if (ref != null && Number.isFinite(ref)) {
      for (let i = out.length; i < PER_SIDE_MIN; i++) {
        out.push(_synthLevel(side, ref));
      }
    }
  }
  return out;
}

/* Сумма показанных вероятностей не должна выходить за 100%: каждая
   вероятность — независимая, но панель/пилюли читаются как распределение,
   поэтому при сумме > 1 нормируем отображаемый набор на 100%. */
function _capTo100(list) {
  const total = list.reduce((sum, l) => sum + l.probability, 0);
  if (!(total > 1.0)) return list;
  const k = 1.0 / total;
  return list.map((l) => ({
    ...l,
    probability: Math.round(l.probability * k * 10000) / 10000,
  }));
}

/* ATR(period) в ценах актива по свечам ДО uptoTime включительно (Wilder).
   При нехватке баров или нечисловых данных — null. Нужен, чтобы потенциальные
   TP/SL не липли к цене: позиция не должна быть «мелочью» в 0.1·ATR. */
export function atrOf(candles, uptoTime, period) {
  period = period || 14;
  const hs = [], ls = [], cs = [];
  for (const c of (Array.isArray(candles) ? candles : [])) {
    if (uptoTime != null && Number(c.time) > uptoTime) break;
    hs.push(Number(c.high));
    ls.push(Number(c.low));
    cs.push(Number(c.close));
  }
  const n = hs.length;
  if (n < period + 1) return null;
  let atr = null;
  let sum = 0;
  let prevC = null;
  let count = 0;
  for (let i = 0; i < n; i++) {
    const h = hs[i], l = ls[i], c = cs[i];
    if (![h, l, c].every((v) => Number.isFinite(v))) { count = 0; prevC = null; continue; }
    if (prevC == null) { prevC = c; continue; }
    const tr = Math.max(h - l, Math.abs(h - prevC), Math.abs(l - prevC));
    prevC = c;
    if (count < period) {
      sum += tr;
      count += 1;
      if (count === period) atr = sum / period;
    } else if (atr != null) {
      atr = (atr * (period - 1) + tr) / period;
    }
  }
  return atr;
}

/* Минимальный разрыв между соседними линиями одного направления в долях ATR:
   вторая линия (TP/SL) не должна липнуть к первой. */
const MIN_GAP_ATR = 0.4;

/* Дистанция «фиксированного» ближнего уровня от цены в долях ATR. */
const NEAR_ATR = 1.0;

/* Фиксация ближних уровней и гарантия разрывов. Проблема: первые (самые
   вероятные) уровни из структуры стоят «почти вплотную» к цене / друг к
   другу, и самая насыщенная полоса вырождается. Шаги на каждую сторону:
   1. ближайший уровень (первая линия) СТРОГО ставится ровно на NEAR_ATR·ATR
      от цены (выше/ниже) — независимо от того, где реальная структура;
   2. остальные линии жадным проходом: каждая следующая не ближе
      MIN_GAP_ATR·ATR к предыдущей, иначе выносится точно на этот разрыв
      (упорядоченность стороны и «не меньше» гарантируются для всех пар).
   Идемпотентно: повторный проход уже выровненного списка ничего не меняет. */
export function pinNearLevels(levels, refPrice, atr) {
  if (!Array.isArray(levels) || !levels.length) return levels;
  if (!(atr > 0)) return levels;
  const ref = Number.isFinite(Number(refPrice)) ? Number(refPrice) : null;
  if (ref == null) return levels;
  const near = atr * NEAR_ATR;
  const gap = atr * MIN_GAP_ATR;
  const ups = levels.filter((l) => l.side === 'UP')
    .sort((a, b) => a.price - b.price);          // ближний → дальний
  const downs = levels.filter((l) => l.side === 'DOWN')
    .sort((a, b) => b.price - a.price);
  _pinSide(ups, ref, near, gap, 1);     // UP: следующие выше
  _pinSide(downs, ref, near, gap, -1);  // DOWN: следующие ниже
  return levels;
}

function _pinSide(side, ref, near, gap, dir) {
  const n = side.length;
  if (!n) return;
  const first = side[0];
  // Строго: первая линия ровно на NEAR_ATR·ATR от цены (выше/ниже), всегда.
  first.price = ref + dir * near;
  first.diff = first.price - ref;
  let prev = first.price;
  for (let i = 1; i < n; i++) {
    const want = prev + dir * gap;
    if ((dir > 0 && side[i].price < want) || (dir < 0 && side[i].price > want)) {
      side[i].price = want;
      side[i].diff = side[i].price - ref;
    }
    prev = side[i].price;
  }
}

/* Потенциальные TP/SL с выбором направления по вероятности:
   • сравниваются вероятности ПЕРВЫХ линий (обе закреплены на 1·ATR): куда
     выше вероятность дойти — туда и цель;
   • ЛОНГ (UP вероятнее): TP = вторая линия UP, SL = вторая линия DOWN;
   • ШОРТ (DOWN вероятнее): TP = вторая линия DOWN, SL = вторая линия UP.
   После pinNearLevels вторая линия отнесена от первой минимум на
   MIN_GAP_ATR·ATR, дальше — сколько реально, главное не ближе.
   При нехватке линий — первая; null, если направлений нет. */
export function pickTpsl(levels, refPrice, atr) {
  const ups = levels.filter((l) => l.side === 'UP')
    .sort((a, b) => a.price - b.price);          // ближний → дальний
  const downs = levels.filter((l) => l.side === 'DOWN')
    .sort((a, b) => b.price - a.price);
  const upLine = ups[0] || null;
  const dnLine = downs[0] || null;
  const upProb = upLine ? Number(upLine.probability) : -Infinity;
  const dnProb = dnLine ? Number(dnLine.probability) : -Infinity;
  let side, tp, sl;
  if (upProb >= dnProb) {
    side = 'long';
    tp = ups[1] || upLine;
    sl = downs[1] || dnLine;
  } else {
    side = 'short';
    tp = downs[1] || dnLine;
    sl = ups[1] || upLine;
  }
  if (!tp && !sl) return null;
  return { side, tp, sl };
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
    this.tpsl = null;
    this.enableTpsl = false;
  }

  /* Вкл/выкл TP/SL поверх уровней: перерисовывает текущие зоны без нового
     запроса (настройка в панели AI Backtest). */
  setTpsl(enabled) {
    this.enableTpsl = !!enabled;
    if (this.levels.length && this.anchor) {
      const levels = [...this.levels];
      const anchor = this.anchor;
      this.render(levels, anchor);
    }
  }

  /* render(result | levels, anchorOverride)
     result — ответ /api/ai-backtest ({levels, price, upto_sec, ...}) либо
     просто массив уровней. anchorOverride — {time, price} среза (для replay:
     барьер), иначе якорь берётся из полного ряда свечей. */
  render(result, anchorOverride) {
    this.clear();
    const anchor = anchorOverride || this._anchor();
    if (!anchor) return;
    const atr = atrOf(this.candlesRef(), anchor.time);
    const levels = pinNearLevels(
      this._extractLevels(result, anchor), anchor.price, atr);
    if (!levels.length) return;
    this.levels = levels;
    this.anchor = anchor;
    this.tpsl = this.enableTpsl ? pickTpsl(levels, anchor.price, atr) : null;
    this.primitive = new AIProbZonesPrimitive(
      this.levels, this.anchor, this.tpsl, this);
    this.series.attachPrimitive(this.primitive);
  }

  _extractLevels(result, anchor) {
    const raw = Array.isArray(result) ? result
      : (result && Array.isArray(result.levels)) ? result.levels : [];
    return filterLevelsForDisplay(raw, anchor && Number.isFinite(Number(anchor.price))
      ? Number(anchor.price) : null);
  }

  clear() {
    if (this.primitive) {
      try { this.series.detachPrimitive(this.primitive); } catch (e) {}
      this.primitive = null;
    }
    this.levels = [];
    this.anchor = null;
    this.tpsl = null;
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
  constructor(levels, anchor, tpsl, manager) {
    this.levels = levels;
    this.anchor = anchor;
    this.tpsl = tpsl || null;
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
        this.levels, this.anchor, this.tpsl, this.manager),
    }];
  }
}

class AIProbZonesRendererImpl {
  constructor(levels, anchor, tpsl, manager) {
    this.levels = levels;
    this.anchor = anchor;
    this.tpsl = tpsl || null;
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
    this._drawTpsl(ctx, x0, x1, size);
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

  /* TP/SL по уровням (сверх зон): цель = ближайший UP, стоп = ближайший
     DOWN. Пунктирные линии по всей зоне + метка у левой кромки, чтобы не
     путать с пилюлями вероятностей у правой кромки. */
  _drawTpsl(ctx, x0, x1, size) {
    const tpsl = this.tpsl;
    if (!tpsl) return;
    const series = this.manager.series;
    // Метка направления у левой кромки: цель на стороне большей вероятности.
    const dir = tpsl.side === 'short' ? 'ШОРТ' : 'ЛОНГ';
    ctx.font = 'bold 10px "Segoe UI", Tahoma, sans-serif';
    const dw = ctx.measureText(dir).width + 10;
    const dh = 15;
    const dx = Math.max(1, Math.min(x0 + 4, size.width - dw - 2));
    const dy = Math.max(2, Math.min(size.height - dh - 2, 2));
    ctx.fillStyle = tpsl.side === 'short' ? DOWN_PILL : UP_PILL;
    this._roundRect(ctx, dx, dy, dw, dh, 4);
    ctx.fill();
    ctx.fillStyle = PILL_TEXT;
    ctx.textBaseline = 'middle';
    ctx.fillText(dir, dx + 5, dy + dh / 2 + 0.5);
    ctx.textBaseline = 'alphabetic';

    const pairs = [['TP', tpsl.tp], ['SL', tpsl.sl]];
    for (const [kind, lv] of pairs) {
      if (!lv || !Number.isFinite(Number(lv.price))) continue;
      const y = series.priceToCoordinate(Number(lv.price));
      if (y == null) continue;
      const color = kind === 'TP' ? UP_PILL : DOWN_PILL;
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.6;
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      ctx.moveTo(x0, y);
      ctx.lineTo(x1, y);
      ctx.stroke();
      ctx.setLineDash([]);

      const label = kind + ' ' + this._fmtPrice(lv.price);
      ctx.font = 'bold 10px "Segoe UI", Tahoma, sans-serif';
      const w = ctx.measureText(label).width + 10;
      const h = 15;
      const lx = Math.max(1, Math.min(x0 + 4, size.width - w - 2));
      let ly = y - h / 2;
      ly = Math.max(2, Math.min(size.height - h - 2, ly));
      ctx.fillStyle = color;
      this._roundRect(ctx, lx, ly, w, h, 4);
      ctx.fill();
      ctx.fillStyle = PILL_TEXT;
      ctx.textBaseline = 'middle';
      ctx.fillText(label, lx + 5, ly + h / 2 + 0.5);
      ctx.textBaseline = 'alphabetic';
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
