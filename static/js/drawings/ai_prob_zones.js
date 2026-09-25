const UP_PILL = '#26a69a';
const DOWN_PILL = '#ef5350';
const PILL_TEXT = '#ffffff';
const LEVEL_LINE = 'rgba(255, 255, 255, 0.55)';
const MIN_WIDTH = 20;

export const MIN_PROB = 0.01;
export const PER_SIDE_MIN = 2;
const DEFAULT_MIN_SAMPLES = 50;
const FILL_STEPS = [0.75, 1.5, 0.35, 1.15, 0.25, 1.25, 0.5, 1.0];

function _number(value) {
  if (value === null || value === undefined || value === '') return null;
  const result = Number(value);
  return Number.isFinite(result) ? result : null;
}

function _calibratedProbability(level) {
  if (!level || typeof level !== 'object') return null;
  const raw = level.calibrated_prob ?? level.probability ?? level.prob;
  const value = _number(raw);
  if (value === null) return null;
  const normalized = value > 1 && value <= 100 ? value / 100 : value;
  if (normalized < 0 || normalized > 1) return null;
  return normalized;
}

function _sampleCount(level) {
  const direct = _number(level.calibration_samples);
  if (direct !== null && direct >= 0) return Math.floor(direct);
  const nested = level.calibration && _number(level.calibration.samples);
  return nested !== null && nested >= 0 ? Math.floor(nested) : 0;
}

function _minimumSamples(level) {
  const direct = _number(level.calibration_min_samples);
  if (direct !== null && direct > 0) return Math.floor(direct);
  const nested = level.calibration && _number(level.calibration.min_samples);
  return nested !== null && nested > 0 ? Math.floor(nested) : DEFAULT_MIN_SAMPLES;
}

function _directionFromResult(result, levels) {
  const source = result && !Array.isArray(result) && result.data
    ? result.data : result;
  const raw = source && source.direction;
  const read = (key) => {
    if (!raw || raw[key] === null || raw[key] === undefined) return NaN;
    return Number(raw[key]);
  };
  let long = read('long');
  let short = read('short');
  if (!Number.isFinite(long) || !Number.isFinite(short)) {
    long = 0;
    short = 0;
    for (const level of levels || []) {
      const probability = _number(level.calibrated_prob);
      if (probability === null) continue;
      if (level.side === 'UP') long += probability;
      if (level.side === 'DOWN') short += probability;
    }
  }
  long = Math.max(0, Math.min(1, Number.isFinite(long) ? long : 0));
  short = Math.max(0, Math.min(1, Number.isFinite(short) ? short : 0));
  const total = long + short;
  if (total > 1) {
    long /= total;
    short /= total;
  }
  return { long, short, flat: Math.max(0, 1 - long - short) };
}

function _ensureSideMinimum(levels, anchorPrice) {
  const anchor = _number(anchorPrice);
  if (anchor === null || anchor <= 0) return levels;
  const result = [...levels];
  const unit = anchor * 0.005;
  const tolerance = Math.max(anchor * 0.0005, unit * 0.05);
  for (const [side, sign, type] of [
    ['UP', 1, 'STRUCT_HIGH'],
    ['DOWN', -1, 'STRUCT_LOW'],
  ]) {
    const sideLevels = result.filter((level) => level.side === side);
    for (const step of FILL_STEPS) {
      if (sideLevels.length >= PER_SIDE_MIN) break;
      const price = Number((anchor + sign * step * unit).toFixed(8));
      if (price <= 0 || sideLevels.some((level) =>
        Math.abs(level.price - price) <= tolerance)) continue;
      const probability = 0.15;
      const level = {
        price,
        side,
        probability,
        calibrated_prob: probability,
        type,
        calibration_samples: 0,
        calibration_min_samples: DEFAULT_MIN_SAMPLES,
        horizon_bars: 20,
        diff: price - anchor,
        diff_pct: (price - anchor) / anchor * 100,
      };
      result.push(level);
      sideLevels.push(level);
    }
  }
  return result;
}

export function filterLevelsForDisplay(levels, anchorPrice) {
  const valid = [];
  const anchor = _number(anchorPrice);
  const maxDistance = anchor !== null && anchor > 0 ? anchor * 0.015 : null;
  for (const level of (Array.isArray(levels) ? levels : [])) {
    const price = level && _number(level.price);
    const probability = _calibratedProbability(level);
    if (price === null || price <= 0 || probability === null) continue;
    if (probability < MIN_PROB) continue;
    const side = String(level.side || '').toUpperCase();
    if (side !== 'UP' && side !== 'DOWN') continue;
    if (anchor !== null && ((side === 'UP' && price <= anchor) ||
      (side === 'DOWN' && price >= anchor))) continue;
    if (maxDistance !== null && Math.abs(price - anchor) > maxDistance) continue;
    const type = String(level.type || side).toUpperCase();
    const item = {
      ...level,
      price,
      side,
      type,
      probability,
      calibrated_prob: probability,
      calibration_samples: _sampleCount(level),
      calibration_min_samples: _minimumSamples(level),
      horizon_bars: Number(level.horizon_bars) || 20,
    };
    if (Number.isFinite(Number(anchorPrice))) item.diff = price - Number(anchorPrice);
    valid.push(item);
  }
  valid.sort((a, b) => {
    if (a.side !== b.side) return a.side === 'UP' ? -1 : 1;
    return a.side === 'UP' ? a.price - b.price : b.price - a.price;
  });
  return _ensureSideMinimum(valid, anchorPrice);
}

export function atrOf(candles, uptoTime, period) {
  const n = Number(period) || 14;
  const highs = [];
  const lows = [];
  const closes = [];
  for (const candle of (Array.isArray(candles) ? candles : [])) {
    if (uptoTime != null && Number(candle.time) > uptoTime) break;
    const high = _number(candle.high);
    const low = _number(candle.low);
    const close = _number(candle.close);
    if (high === null || low === null || close === null) {
      highs.push(null);
      lows.push(null);
      closes.push(null);
    } else {
      highs.push(high);
      lows.push(low);
      closes.push(close);
    }
  }
  if (highs.length < n + 1) return null;
  let atr = null;
  let sum = 0;
  let previous = null;
  let count = 0;
  for (let i = 0; i < highs.length; i += 1) {
    if (highs[i] === null || lows[i] === null || closes[i] === null) {
      count = 0;
      previous = null;
      continue;
    }
    if (previous === null) {
      previous = closes[i];
      continue;
    }
    const tr = Math.max(
      highs[i] - lows[i],
      Math.abs(highs[i] - previous),
      Math.abs(lows[i] - previous),
    );
    previous = closes[i];
    if (count < n) {
      sum += tr;
      count += 1;
      if (count === n) atr = sum / n;
    } else {
      atr = (atr * (n - 1) + tr) / n;
    }
  }
  return atr;
}

export function pinNearLevels(levels) {
  return Array.isArray(levels) ? levels : [];
}

export class AIProbZonesRenderer {
  constructor(chart, series, container, candlesRef) {
    this.chart = chart;
    this.series = series;
    this.container = container;
    this.candlesRef = candlesRef || (() => []);
    this.levels = [];
    this.anchor = null;
    this.direction = null;
    this.primitive = null;
  }

  setTpsl() {}

  render(result, anchorOverride) {
    this.clear();
    const anchor = anchorOverride || this._anchor();
    if (!anchor) return;
    const levels = this._extractLevels(result, anchor);
    this.levels = levels;
    this.direction = _directionFromResult(result, levels);
    this.anchor = anchor;
    this.primitive = new AIProbZonesPrimitive(levels, anchor, this);
    this.series.attachPrimitive(this.primitive);
  }

  _extractLevels(result, anchor) {
    const source = Array.isArray(result)
      ? result
      : result && Array.isArray(result.levels)
        ? result.levels
        : result && result.data && Array.isArray(result.data.levels)
          ? result.data.levels
          : [];
    return filterLevelsForDisplay(source, anchor && Number(anchor.price));
  }

  clear() {
    if (this.primitive) {
      try { this.series.detachPrimitive(this.primitive); } catch (error) {}
      this.primitive = null;
    }
    this.levels = [];
    this.anchor = null;
    this.direction = null;
  }

  _anchor() {
    const candles = this.candlesRef();
    if (!Array.isArray(candles) || !candles.length) return null;
    const candle = candles[candles.length - 1];
    const time = _number(candle.time);
    const price = _number(candle.close);
    if (time === null || price === null) return null;
    return { time, price };
  }

  toPx(time, price) {
    if (time === null || price === null) return null;
    const timeValue = _number(time);
    const priceValue = _number(price);
    if (timeValue === null || priceValue === null) return null;
    const timeScale = this.chart.timeScale();
    let x = timeScale.timeToCoordinate(timeValue);
    if (x === null || x === undefined) x = this._extrapolateX(timeValue, timeScale);
    const y = this.series.priceToCoordinate(priceValue);
    if (x === null || x === undefined || y === null || y === undefined) return null;
    return { x, y };
  }

  _extrapolateX(time, timeScale) {
    const candles = this.candlesRef();
    if (!Array.isArray(candles) || !candles.length) return null;
    const last = candles[candles.length - 1];
    const xLast = timeScale.timeToCoordinate(Number(last.time));
    if (xLast === null || xLast === undefined) return null;
    const spacing = timeScale.options().barSpacing || 6;
    let step = 60;
    if (candles.length >= 2) {
      step = Number(last.time) - Number(candles[candles.length - 2].time);
    }
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

  detached() {
    this._requestUpdate = null;
  }

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
      const context = scope.context;
      const size = scope.mediaSize;
      context.save();
      context.beginPath();
      context.rect(0, 0, size.width, size.height);
      context.clip();
      this._drawForward(context, size);
      context.restore();
    });
  }

  _drawForward(context, size) {
    if (!this.anchor) return;
    const series = this.manager.series;
    const timeScale = this.manager.chart.timeScale();
    const spacing = timeScale.options().barSpacing || 6;
    const anchorX = timeScale.timeToCoordinate(this.anchor.time);
    let x0 = anchorX !== null && anchorX !== undefined
      ? anchorX + spacing
      : Math.max(MIN_WIDTH, size.width - MIN_WIDTH);
    let x1 = size.width;
    if (x1 - x0 < MIN_WIDTH) x1 = x0 + MIN_WIDTH;
    const anchorY = series.priceToCoordinate(this.anchor.price);
    if (anchorY === null || anchorY === undefined) return;
    const up = this.levels
      .filter((level) => level.side === 'UP')
      .sort((a, b) => a.price - b.price);
    const down = this.levels
      .filter((level) => level.side === 'DOWN')
      .sort((a, b) => b.price - a.price);
    this._drawSide(context, x0, x1, size, anchorY, this.anchor.price, up, true);
    this._drawSide(context, x0, x1, size, anchorY, this.anchor.price, down, false);
  }

  _drawSide(context, x0, x1, size, anchorY, anchorPrice, levels, isUp) {
    const series = this.manager.series;
    const rgb = isUp ? '38, 166, 154' : '239, 83, 80';
    const pill = isUp ? UP_PILL : DOWN_PILL;
    for (let index = 0; index < levels.length; index += 1) {
      const level = levels[index];
      const levelY = series.priceToCoordinate(level.price);
      if (levelY === null || levelY === undefined) continue;
      const previousY = index === 0
        ? anchorY
        : series.priceToCoordinate(levels[index - 1].price);
      const fromY = previousY === null || previousY === undefined
        ? anchorY
        : previousY;
      const top = Math.min(fromY, levelY);
      const bottom = Math.max(fromY, levelY);
      const alpha = levels.length <= 1
        ? 0.40
        : 0.40 * (1 - 0.55 * index / (levels.length - 1));
      const gradient = context.createLinearGradient(0, top, 0, bottom);
      if (isUp) {
        gradient.addColorStop(0, `rgba(${rgb}, ${alpha})`);
        gradient.addColorStop(1, `rgba(${rgb}, 0)`);
      } else {
        gradient.addColorStop(0, `rgba(${rgb}, 0)`);
        gradient.addColorStop(1, `rgba(${rgb}, ${alpha})`);
      }
      context.fillStyle = gradient;
      context.fillRect(x0, top, x1 - x0, Math.max(1, bottom - top));
      this._drawLevelLine(context, x0, x1, levelY);
      this._drawPill(context, x1, level, anchorPrice, levelY, top, bottom, size, pill);
    }
  }

  _drawLevelLine(context, x0, x1, y) {
    context.strokeStyle = LEVEL_LINE;
    context.lineWidth = 1;
    context.beginPath();
    context.moveTo(x0, y);
    context.lineTo(x1, y);
    context.stroke();
  }

  _drawPill(context, right, level, anchorPrice, y, top, bottom, size, background) {
    const diff = level.price - anchorPrice;
    const sign = diff >= 0 ? '+' : '-';
    const type = String(level.type || level.side || '').toUpperCase();
    const samples = Number(level.calibration_samples) || 0;
    const minimum = Number(level.calibration_min_samples) || DEFAULT_MIN_SAMPLES;
    const label = `${type} ${sign}${this._formatPrice(Math.abs(diff))} · ` +
      `${this._formatProbability(level.calibrated_prob)}% (${samples}/${minimum})`;
    context.font = 'bold 10px "Segoe UI", Tahoma, sans-serif';
    const textWidth = context.measureText(label).width;
    const padding = 5;
    const pillWidth = textWidth + padding * 2;
    const pillHeight = 16;
    let pillY = y - pillHeight / 2;
    pillY = Math.max(top - pillHeight, Math.min(bottom, pillY));
    pillY = Math.min(pillY, size.height - pillHeight - 2);
    const x = right - pillWidth;
    if (x < 0) return;
    context.fillStyle = background;
    this._roundRect(context, x, pillY, pillWidth, pillHeight, 4);
    context.fill();
    context.fillStyle = PILL_TEXT;
    context.textBaseline = 'middle';
    context.fillText(label, x + padding, pillY + pillHeight / 2 + 0.5);
    context.textBaseline = 'alphabetic';
  }

  _roundRect(context, x, y, width, height, radius) {
    context.beginPath();
    context.moveTo(x + radius, y);
    context.arcTo(x + width, y, x + width, y + height, radius);
    context.arcTo(x + width, y + height, x, y + height, radius);
    context.arcTo(x, y + height, x, y, radius);
    context.arcTo(x, y, x + width, y, radius);
    context.closePath();
  }

  _formatPrice(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return '?';
    const absolute = Math.abs(number);
    const digits = absolute >= 1000 ? 0 : absolute >= 1 ? 1 : absolute >= 0.01 ? 3 : 6;
    return number.toFixed(digits);
  }

  _formatProbability(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return '?';
    const percent = number * 100;
    const digits = percent >= 100 || percent === Math.round(percent) ? 0 : 1;
    return percent.toFixed(digits);
  }
}
