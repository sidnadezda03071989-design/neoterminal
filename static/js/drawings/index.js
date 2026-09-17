// DrawingsManager — движок рисования на lightweight-charts v5.
import { state } from '../state.js';

const USER_COLOR = '#2962ff';
const AI_COLOR   = '#ff9800';

function hexToRgba(hex, alpha) {
  let h = String(hex || '#ffffff').replace('#', '');
  if (h.length === 3) h = h.split('').map((c) => c + c).join('');
  const n = parseInt(h.length >= 6 ? h.slice(0, 6) : 'ffffff', 16);
  return 'rgba(' + ((n >> 16) & 255) + ',' + ((n >> 8) & 255) + ',' + (n & 255) + ',' + alpha + ')';
}

function distToSegment(px, py, x1, y1, x2, y2) {
  const dx = x2 - x1, dy = y2 - y1;
  if (dx === 0 && dy === 0) return Math.hypot(px - x1, py - y1);
  const t = Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)));
  return Math.hypot(px - (x1 + t * dx), py - (y1 + t * dy));
}

function fmtPrice(v) {
  if (v == null || !isFinite(v)) return '—';
  const av = Math.abs(v);
  const d = av >= 1000 ? 1 : av >= 1 ? 2 : av >= 0.01 ? 4 : 6;
  return Number(v).toFixed(d);
}

function tfSeconds(tf) {
  const m = /^(\d+)\s*([smhdwSMHDW])$/.exec(String(tf || '').trim());
  if (!m) return 3600;
  const n = parseInt(m[1], 10) || 1;
  const mult = { s: 1, m: 60, H: 3600, h: 3600, D: 86400, d: 86400, W: 604800, w: 604800 }[m[2]];
  return n * (mult || 3600);
}

function fmtDuration(sec) {
  sec = Math.max(0, Math.round(sec));
  const d = Math.floor(sec / 86400);
  const h = Math.floor((sec % 86400) / 3600);
  const m = Math.round((sec % 3600) / 60);
  if (d > 0) return d + 'd ' + h + 'h';
  if (h > 0) return h + 'h' + (m ? ' ' + m + 'm' : '');
  return m + 'm';
}

function strokeMain(ctx, x1, y1, x2, y2, color, selected, dashed, width) {
  if (selected) {
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2);
    ctx.strokeStyle = 'rgba(255,255,255,0.75)';
    ctx.lineWidth = (width || 2) + 3;
    ctx.setLineDash([]); ctx.stroke();
  }
  ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2);
  ctx.strokeStyle = color;
  ctx.lineWidth = width || 2;
  ctx.setLineDash(dashed ? [6, 4] : []);
  ctx.stroke();
  ctx.setLineDash([]);
}

function strokePoly(ctx, pts, color, selected) {
  if (pts.length < 2) return;
  if (selected) {
    ctx.beginPath();
    pts.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
    ctx.strokeStyle = 'rgba(255,255,255,0.75)';
    ctx.lineWidth = 5; ctx.stroke();
  }
  ctx.beginPath();
  pts.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
  ctx.strokeStyle = color;
  ctx.lineWidth = 2; ctx.stroke();
}

function drawTag(ctx, text, x, y, color, alignRight) {
  ctx.font = '10px "Segoe UI", Tahoma, sans-serif';
  const w = ctx.measureText(text).width + 8;
  const h = 16;
  let lx = alignRight ? x - w - 6 : x + 4;
  let ty = y - h - 5;
  if (lx < 2) lx = 2;
  if (ty < 2) ty = y + 5;
  ctx.fillStyle = 'rgba(19,23,34,0.88)';
  ctx.strokeStyle = color;
  ctx.fillRect(lx, ty, w, h); ctx.strokeRect(lx, ty, w, h);
  ctx.fillStyle = color;
  ctx.fillText(text, lx + 4, ty + 12);
}

class Renderer {
  constructor(dm, d) { this.dm = dm; this.d = d; }
  draw(target) {
    const dm = this.dm, d = this.d;
    target.useMediaCoordinateSpace((scope) => {
      const ctx = scope.context, size = scope.mediaSize;
      const pts = (d.points || []).map((p) => dm.toPx(p.time, p.price)).filter(Boolean);
      if (!pts.length) return;
      ctx.save();
      ctx.beginPath(); ctx.rect(0, 0, size.width, size.height); ctx.clip();
      this.paint(ctx, size, d, pts);
      ctx.restore();
    });
  }
  paint(ctx, size, d, pts) {
    const color = d.color || (d.created_by === 'ai' ? AI_COLOR : USER_COLOR);
    const selected = this.dm.selectedId === d.id;
    switch (d.type) {
      case 'h_line': {
        const y = pts[0].y;
        strokeMain(ctx, 0, y, size.width, y, color, selected);
        drawTag(ctx, fmtPrice(d.points[0].price), size.width - 4, y, color, true);
        if (selected) {
          ctx.beginPath(); ctx.arc(pts[0].x, y, 4, 0, Math.PI * 2);
          ctx.fillStyle = '#fff'; ctx.fill();
        }
        break;
      }
      case 'trendline': {
        if (pts.length < 2) return;
        strokeMain(ctx, pts[0].x, pts[0].y, pts[1].x, pts[1].y, color, selected);
        if (selected) this.knots(ctx, pts.slice(0, 2));
        break;
      }
      case 'ray': {
        if (pts.length < 2) return;
        const a = pts[0], b = pts[1];
        const tx = b.x > a.x ? size.width : 0;
        if (b.x === a.x) {
          strokeMain(ctx, a.x, 0, a.x, size.height, color, selected);
        } else {
          const slope = (b.y - a.y) / (b.x - a.x);
          strokeMain(ctx, a.x, a.y, tx, a.y + (tx - a.x) * slope, color, selected);
        }
        if (selected) this.knots(ctx, [a, b]);
        break;
      }
      case 'rectangle': {
        if (pts.length < 2) return;
        const x1 = Math.min(pts[0].x, pts[1].x), x2 = Math.max(pts[0].x, pts[1].x);
        const y1 = Math.min(pts[0].y, pts[1].y), y2 = Math.max(pts[0].y, pts[1].y);
        ctx.fillStyle = hexToRgba(color, selected ? 0.12 : 0.06);
        ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
        ctx.strokeStyle = color; ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        if (selected) this.knots(ctx, pts);
        break;
      }
      case 'fib': {
        if (pts.length < 2) return;
        this.paintFib(ctx, d, pts, selected, color);
        break;
      }
      case 'pen': {
        strokePoly(ctx, pts, color, selected);
        break;
      }
      case 'text': {
        ctx.font = '12px "Segoe UI", Tahoma, sans-serif';
        ctx.fillStyle = color;
        ctx.fillText(d.label || 'Text', pts[0].x, pts[0].y);
        if (selected) this.knots(ctx, [pts[0]]);
        break;
      }
    }
  }
  knots(ctx, pts) {
    pts.forEach((p) => {
      ctx.beginPath(); ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
      ctx.fillStyle = '#fff'; ctx.fill();
      ctx.strokeStyle = '#131722'; ctx.lineWidth = 1.5; ctx.stroke();
    });
  }
  paintFib(ctx, d, pts, selected, color) {
    const p0 = d.points[0], p1 = d.points[1];
    const priceTop = Math.max(p0.price, p1.price), priceBot = Math.min(p0.price, p1.price);
    const yTop = Math.min(pts[0].y, pts[1].y), yBot = Math.max(pts[0].y, pts[1].y);
    const xa = Math.min(pts[0].x, pts[1].x), xb = Math.max(pts[0].x, pts[1].x);
    const levels = [[0, '#2962ff'], [0.236, '#26a69a'], [0.382, '#e91e63'],
                    [0.5, '#ff9800'], [0.618, '#ab47bc'], [0.786, '#d32f2f'], [1, '#787b86']];
    strokeMain(ctx, xa, yTop, xa, yBot, hexToRgba(color, 0.6), selected, true, 1);
    for (const lv of levels) {
      const r = lv[0], lc = lv[1];
      const y = yTop + r * (yBot - yTop);
      const price = priceBot + (1 - r) * (priceTop - priceBot);
      strokeMain(ctx, xa, y, xb, y, selected ? color : lc, selected, false, 1.5);
      drawTag(ctx, (r * 100).toFixed(1) + '% ' + fmtPrice(price), xa, y, selected ? color : lc, false);
    }
  }
}

class Primitive {
  constructor(dm, d) { this.dm = dm; this.d = d; this._requestUpdate = null; }
  attached(param) {
    if (param && typeof param.requestUpdate === 'function') {
      this._requestUpdate = param.requestUpdate;
      this.dm._requestUpdate = param.requestUpdate;
    }
  }
  detached() { this._requestUpdate = null; }
  updateAllViews() {}
  paneViews() {
    return [{ zOrder: function() { return 'top'; }, renderer: () => new Renderer(this.dm, this.d) }];
  }
}

// --- Measure (📏): временный измеритель, НЕ сохраняется в БД ---
function paintMeasure(ctx, size, dm) {
  const m = dm.measurement;
  if (!m || !m.p1 || !m.p2) return;
  const a = dm.toPx(m.p1.time, m.p1.price);
  const b = dm.toPx(m.p2.time, m.p2.price);
  if (!a || !b) return;
  const x1 = Math.min(a.x, b.x), x2 = Math.max(a.x, b.x);
  const y1 = Math.min(a.y, b.y), y2 = Math.max(a.y, b.y);

  // полупрозрачный прямоугольник + пунктирная синяя рамка
  ctx.fillStyle = 'rgba(41,98,255,0.12)';
  ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
  ctx.strokeStyle = '#2962ff';
  ctx.lineWidth = 1;
  ctx.setLineDash([5, 4]);
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  ctx.setLineDash([]);

  // линия от первой точки ко второй
  ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y);
  ctx.strokeStyle = '#2962ff'; ctx.lineWidth = 1.5; ctx.stroke();
  ctx.setLineDash([]);

  // метрики
  const tf = tfSeconds(dm.timeframe);
  const arr = dm._times();
  let bars = Math.abs(m.p2.time - m.p1.time) / tf;
  if (arr && arr.length) {
    const i1 = dm._indexOfTime(m.p1.time);
    const i2 = dm._indexOfTime(m.p2.time);
    if (i1 >= 0 && i2 >= 0) bars = Math.abs(i2 - i1);
  }
  const dp = m.p2.price - m.p1.price;
  const pct = m.p1.price ? (dp / m.p1.price) * 100 : 0;
  const sign = dp > 0 ? '+' : '';
  const color = dp > 0 ? '#26a69a' : (dp < 0 ? '#ef5350' : '#787b86');
  const lines = [
    'Δ цены: ' + sign + fmtPrice(dp) + ' (' + sign + pct.toFixed(2) + '%)',
    'Δ времени: ' + Math.round(bars) + ' bars / ' + fmtDuration(bars * tf),
    'Скорость: ' + (bars > 0.5 ? sign + fmtPrice(dp / bars) : '—') + '/bar',
  ];

  ctx.font = '12px "Segoe UI", Tahoma, sans-serif';
  let w = 0;
  for (const t of lines) w = Math.max(w, ctx.measureText(t).width);
  w += 16;
  const h = lines.length * 16 + 10;
  let bx = (x1 + x2) / 2 - w / 2;
  let by = (y1 + y2) / 2 - h / 2;
  bx = Math.max(2, Math.min(bx, size.width - w - 2));
  by = Math.max(2, Math.min(by, size.height - h - 2));
  ctx.fillStyle = 'rgba(19,23,34,0.85)';
  ctx.fillRect(bx, by, w, h);
  ctx.strokeStyle = color; ctx.lineWidth = 1;
  ctx.strokeRect(bx, by, w, h);
  ctx.fillStyle = color;
  for (let i = 0; i < lines.length; i++) {
    ctx.fillText(lines[i], bx + 8, by + 18 + i * 16);
  }
}

class MeasureRenderer {
  constructor(dm) { this.dm = dm; }
  draw(target) {
    const dm = this.dm;
    if (!dm.measurement) return;
    target.useMediaCoordinateSpace((scope) => {
      const ctx = scope.context, size = scope.mediaSize;
      ctx.save();
      ctx.beginPath(); ctx.rect(0, 0, size.width, size.height); ctx.clip();
      paintMeasure(ctx, size, dm);
      ctx.restore();
    });
  }
}

class MeasurePrimitive {
  constructor(dm) { this.dm = dm; this._requestUpdate = null; }
  attached(param) {
    if (param && typeof param.requestUpdate === 'function') {
      this._requestUpdate = param.requestUpdate;
      this.dm._requestUpdate = param.requestUpdate;
    }
  }
  detached() { this._requestUpdate = null; }
  updateAllViews() {}
  paneViews() {
    return [{ zOrder: function () { return 'top'; }, renderer: () => new MeasureRenderer(this.dm) }];
  }
}

export class DrawingsManager {
  constructor(cfg) {
    this.chart = cfg.chart;
    this.series = cfg.series;
    this.container = cfg.container;
    this.candlesRef = cfg.candlesRef;
    this.onChanged = cfg.onChanged || (() => {});
    this.symbol = cfg.symbol || 'BTCUSDT';
    this.timeframe = cfg.timeframe || '1H';
    this.drawings = [];
    this._prims = new Map();
    this.tool = 'cursor';
    this.selectedId = null;
    this.color = USER_COLOR;
    this._requestUpdate = null;
    this.drag = null;
    this._draftPrim = null;
    this.magnet = false;
    this.measurement = null; // { p1, p2 } — временный рисунок, в БД не пишется
    this.series.attachPrimitive(new MeasurePrimitive(this));
    this._bind();
  }

  _times() {
    const arr = this.candlesRef();
    return arr && arr.length ? arr : null;
  }
  _nearestTime(t) {
    const arr = this._times();
    if (!arr || t == null) return t;
    if (t <= arr[0].time) return t;
    if (t >= arr[arr.length - 1].time) return t;
    let lo = 0, hi = arr.length - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (arr[mid].time <= t) lo = mid; else hi = mid;
    }
    return (t - arr[lo].time) <= (arr[hi].time - t) ? arr[lo].time : arr[hi].time;
  }
  _timeToX(time) {
    const ts = this.chart.timeScale();
    const x = ts.timeToCoordinate(time);
    if (x != null) return x;
    const arr = this._times();
    if (!arr || !arr.length) return null;
    const last = arr[arr.length - 1];
    const xLast = ts.timeToCoordinate(last.time);
    if (xLast == null) return null;
    const spacing = (ts.options().barSpacing) || 6;
    let step = 60;
    if (arr.length >= 2) step = last.time - arr[arr.length - 2].time;
    if (step <= 0) step = 60;
    return xLast + ((time - last.time) / step) * spacing;
  }

  _xToTime(x) {
    const ts = this.chart.timeScale();
    const t = ts.coordinateToTime(x);
    if (t != null) return t;
    const arr = this._times();
    if (!arr || !arr.length) return null;
    const last = arr[arr.length - 1];
    const xLast = ts.timeToCoordinate(last.time);
    if (xLast == null) return null;
    const spacing = (ts.options().barSpacing) || 6;
    let step = 60;
    if (arr.length >= 2) step = last.time - arr[arr.length - 2].time;
    if (step <= 0) step = 60;
    return Math.round(last.time + ((x - xLast) / spacing) * step);
  }

  toPx(time, price) {
    const x = this._timeToX(time);
    const y = this.series.priceToCoordinate(price);
    if (x == null || y == null) return null;
    return { x: x, y: y };
  }
  _indexOfTime(time) {
    const arr = this._times();
    if (!arr || !arr.length || time == null) return -1;
    let lo = 0, hi = arr.length - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (arr[mid].time === time) return mid;
      if (arr[mid].time < time) lo = mid + 1; else hi = mid - 1;
    }
    return -1;
  }

  _snap(px, py) {
    const t = this._xToTime(px);
    const p = this.series.coordinateToPrice(py);
    if (t == null || p == null) return null;
    const nt = this._nearestTime(t);
    if (this.magnet) {
      // магнит: прилипаем к ближайшей свече по X и к ближайшей из OHLC по Y
      const c = this._times() ? this._times()[this._indexOfTime(nt)] : null;
      if (c) {
        const cands = [c.open, c.high, c.low, c.close];
        let best = cands[0], bd = Infinity;
        for (let i = 0; i < cands.length; i++) {
          const d = Math.abs(cands[i] - p);
          if (d < bd) { bd = d; best = cands[i]; }
        }
        return { time: nt, price: best };
      }
    }
    return { time: nt, price: p };
  }
  _point(e) {
    const r = this.container.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  }

  _attach(d) {
    const prim = new Primitive(this, d);
    this.series.attachPrimitive(prim);
    this._prims.set(d.id, prim);
  }
  _detach(id) {
    const p = this._prims.get(id);
    if (p) {
      try { this.series.detachPrimitive(p); } catch (e) {}
      this._prims.delete(id);
    }
  }
  _teardown() {
    this._prims.forEach((p) => { try { this.series.detachPrimitive(p); } catch (e) {} });
    this._prims.clear();
  }
  _fire() { if (this._requestUpdate) this._requestUpdate(); }

  refresh(list) {
    this._teardown();
    this.drawings = Array.isArray(list) ? list : [];
    for (const d of this.drawings) this._attach(d);
    if (this.selectedId && !this.drawings.some((d) => d.id === this.selectedId)) {
      this.selectedId = null;
    }
    this.onChanged();
    this._fire();
  }

  loadFromServer(symbol, timeframe) {
    this.symbol = symbol || this.symbol;
    this.timeframe = timeframe || this.timeframe;
    const qs = new URLSearchParams();
    if (symbol) qs.set('symbol', symbol);
    if (timeframe) qs.set('timeframe', timeframe);
    fetch('/api/drawings' + (qs.toString() ? '?' + qs : ''))
      .then((r) => r.json())
      .then((data) => this.refresh(data.drawings || []))
      .catch(() => this.refresh([]));
  }

  setCurrentContext(symbol, timeframe) {
    if (symbol === this.symbol && timeframe === this.timeframe) return;
    this.symbol = symbol;
    this.timeframe = timeframe;
    this.measurement = null; // смена символа/ТФ — сбросить measure
    this._fire();
    this.loadFromServer(symbol, timeframe);
  }

  setTool(tool) {
    this.tool = tool;
    if (tool !== 'cursor') this.selectedId = null;
    this.drag = null;
    this._discardGhost();
    this._fire();
  }

  setMagnet(v) {
    this.magnet = !!v;
    this._fire();
  }

  _minPoints(type) {
    if (type === 'h_line' || type === 'text') return 1;
    return 2;
  }

  _bind() {
    const self = this;
    this.container.addEventListener('mousedown', function(e) { self._onDown(e); }, true);
    document.addEventListener('mousemove', function(e) { self._onMove(e); });
    document.addEventListener('mouseup', function(e) { self._onUp(e); });
    document.addEventListener('keydown', function(e) {
      if ((e.key === 'Delete' || e.key === 'Backspace') && self.selectedId) {
        e.preventDefault();
        self.deleteSelected();
      }
      if (e.key === 'Escape') {
        if (self.measurement || (self.drag && self.drag.mode === 'measure')) {
          self.measurement = null; // Esc — отмена measure
          self.drag = null;
        }
        self.selectedId = null; self._fire();
      }
    });
  }

  _onDown(e) {
    if (e.button !== 0) return;
    const p = this._point(e);
    if (p.x < 0 || p.y < 0 || p.x > this.container.clientWidth || p.y > this.container.clientHeight) return;

    if (this.tool === 'cursor') {
      const hit = this.hitTest(p.x, p.y);
      if (hit) {
        e.stopPropagation(); e.preventDefault();
        this.selectedId = hit.id;
        const pi = this._nearestKnot(hit, p.x, p.y);
        this.drag = {
          mode: pi >= 0 ? 'point' : 'move',
          drawing: hit, pointIndex: pi,
          startX: p.x, startY: p.y,
          orig: hit.points.map((pt) => ({ time: pt.time, price: pt.price })),
        };
        this._fire(); this.onChanged();
      } else if (this.selectedId) {
        this.selectedId = null; this._fire(); this.onChanged();
      }
      return;
    }

    if (this.tool === 'measure') {
      e.stopPropagation(); e.preventDefault();
      const snap = this._snap(p.x, p.y);
      if (!snap) return;
      this.measurement = { p1: snap, p2: snap }; // новый клик — сброс и новое измерение
      this.drag = { mode: 'measure' };
      this._fire();
      return;
    }

    e.stopPropagation(); e.preventDefault();
    const snap = this._snap(p.x, p.y);
    if (!snap) return;
    const draft = {
      id: 'draft_' + Date.now().toString(36),
      type: this.tool,
      points: [snap],
      color: this.color,
      label: '',
      created_by: 'user',
    };
    this.drag = { mode: 'create', drawing: draft };
    this._draftPrim = new Primitive(this, draft);
    this.series.attachPrimitive(this._draftPrim);
  }

  _onMove(e) {
    const dg = this.drag;
    if (!dg) return;
    const p = this._point(e);

    if (dg.mode === 'measure') {
      const snap = this._snap(p.x, p.y);
      if (snap && this.measurement) { this.measurement.p2 = snap; this._fire(); }
      return;
    }

    if (dg.mode === 'create') {
      const d = dg.drawing;
      const snap = this._snap(p.x, p.y);
      if (!snap) return;
      if (d.type === 'pen') {
        const last = d.points[d.points.length - 1];
        const q = last ? this.toPx(last.time, last.price) : null;
        if (!q || Math.hypot(q.x - p.x, q.y - p.y) > 2) d.points.push(snap);
      } else {
        if (d.points.length < 2) d.points.push(snap);
        else d.points[1] = snap;
      }
      this._fire();
      return;
    }

    if (dg.mode === 'point') {
      const snap = this._snap(p.x, p.y);
      if (snap) { dg.drawing.points[dg.pointIndex] = snap; this._fire(); }
      return;
    }

    if (dg.mode === 'move') {
      const arr = this._times();
      if (!arr) return;
      const t0 = arr[0].time;
      const t1 = arr[arr.length - 1].time;
      const tspan = Math.max(1, t1 - t0);
      const xspan = Math.max(1, this.container.clientWidth);
      const dxT = ((p.x - dg.startX) / xspan) * tspan;
      const p0 = this.series.coordinateToPrice(dg.startY);
      const p1 = this.series.coordinateToPrice(p.y);
      const dyPrice = (p1 != null && p0 != null) ? (p1 - p0) : 0;
      dg.drawing.points = dg.orig.map((pt) => ({
        time: this._nearestTime(pt.time + dxT),
        price: pt.price + dyPrice,
      }));
      this._fire();
    }
  }

  _onUp() {
    const dg = this.drag;
    if (!dg) return;
    this.drag = null;

    if (dg.mode === 'measure') {
      this._fire(); // измеритель зафиксирован, но не сохраняется в БД
      return;
    }

    if (dg.mode === 'create') {
      const d = dg.drawing;
      this._discardGhost();
      if (d.type === 'pen' && d.points.length < 2) return;
      if (d.points.length < this._minPoints(d.type)) return;
      delete d.id;
      d.id = 'c_' + Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
      d.symbol = this.symbol;
      d.timeframe = this.timeframe;
      this.drawings.push(d);
      this._attach(d);
      this._fire(); this.onChanged();
      this._save(d);
      this.setTool('cursor');
      document.querySelectorAll('.tool-btn').forEach((b) => {
        b.classList.toggle('active', b.dataset.tool === 'cursor');
      });
      return;
    }

    if (dg.mode === 'point' || dg.mode === 'move') {
      this._update(dg.drawing);
    }
  }

  _discardGhost() {
    if (this._draftPrim) {
      try { this.series.detachPrimitive(this._draftPrim); } catch (e) {}
      this._draftPrim = null;
    }
  }

  _nearestKnot(d, px, py) {
    let bi = -1, bd = 10;
    (d.points || []).forEach((p, i) => {
      const q = this.toPx(p.time, p.price);
      if (!q) return;
      const dd = Math.hypot(q.x - px, q.y - py);
      if (dd < bd) { bd = dd; bi = i; }
    });
    return bi;
  }

  hitTest(px, py) {
    let best = null, bestD = 8;
    for (const d of this.drawings) {
      const pts = (d.points || []).map((p) => this.toPx(p.time, p.price)).filter(Boolean);
      if (!pts.length) continue;
      const dist = this._distToDrawing(d, pts, px, py);
      if (dist != null && dist < bestD) { bestD = dist; best = d; }
    }
    return best;
  }

  _distToDrawing(d, pts, px, py) {
    switch (d.type) {
      case 'h_line': return Math.abs(py - pts[0].y);
      case 'trendline': return pts.length >= 2 ? distToSegment(px, py, pts[0].x, pts[0].y, pts[1].x, pts[1].y) : Infinity;
      case 'ray': {
        if (pts.length < 2) return Infinity;
        const a = pts[0], b = pts[1];
        const tx = b.x > a.x ? this.container.clientWidth : 0;
        const slope = (b.y - a.y) / (b.x - a.x || 1e-9);
        return distToSegment(px, py, a.x, a.y, tx, a.y + (tx - a.x) * slope);
      }
      case 'rectangle': {
        if (pts.length < 2) return Infinity;
        const x1 = Math.min(pts[0].x, pts[1].x), x2 = Math.max(pts[0].x, pts[1].x);
        const y1 = Math.min(pts[0].y, pts[1].y), y2 = Math.max(pts[0].y, pts[1].y);
        if (px >= x1 && px <= x2 && py >= y1 && py <= y2) return 0;
        return Math.min(Math.abs(py - y1), Math.abs(py - y2), Math.abs(px - x1), Math.abs(px - x2));
      }
      case 'fib': {
        if (pts.length < 2) return Infinity;
        const yTop = Math.min(pts[0].y, pts[1].y), yBot = Math.max(pts[0].y, pts[1].y);
        let best = Infinity;
        for (let r = 0; r <= 1.0001; r += 0.236) {
          best = Math.min(best, Math.abs(py - (yTop + r * (yBot - yTop))));
        }
        return best;
      }
      case 'pen': {
        let best = Infinity;
        for (let i = 0; i < pts.length - 1; i++) {
          best = Math.min(best, distToSegment(px, py, pts[i].x, pts[i].y, pts[i + 1].x, pts[i + 1].y));
        }
        return best;
      }
      case 'text': return Math.hypot(px - pts[0].x, py - pts[0].y);
      default: return Infinity;
    }
  }

  _save(d) {
    const self = this;
    fetch('/api/drawings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ drawing: d }),
    }).then(function(r) { return r.json(); }).then(function(data) {
      if (data.drawings) self.refresh(data.drawings);
    }).catch(function() {});
  }
  _update(d) {
    fetch('/api/drawings/update', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ drawing: d }),
    }).catch(function() {});
  }

  deleteSelected() {
    const self = this;
    const d = this.drawings.find(function(x) { return x.id === self.selectedId; });
    if (!d) return;
    fetch('/api/drawings/' + encodeURIComponent(d.id), { method: 'DELETE' })
      .then(function() {
        self._detach(d.id);
        self.drawings = self.drawings.filter(function(x) { return x.id !== d.id; });
        self.selectedId = null;
        self._fire(); self.onChanged();
      }).catch(function() {});
  }

  clearAll() {
    const self = this;
    fetch('/api/drawings', { method: 'DELETE' })
      .then(function() { self.selectedId = null; self.refresh([]); })
      .catch(function() {});
  }
  eraseAI() {
    const self = this;
    fetch('/api/drawings?created_by=ai', { method: 'DELETE' })
      .then(function(r) { return r.json(); })
      .then(function(data) { self.refresh(data.drawings || []); })
      .catch(function() { self.loadFromServer(self.symbol, self.timeframe); });
  }
  exportJSON() {
    const blob = new Blob([JSON.stringify({ version: 1, drawings: this.drawings }, null, 2)],
      { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'drawings-' + new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-') + '.json';
    a.click();
    URL.revokeObjectURL(a.href);
  }
  importJSON(file) {
    const self = this;
    const rd = new FileReader();
    rd.onload = function() {
      try {
        const data = JSON.parse(String(rd.result));
        const list = Array.isArray(data) ? data : (data.drawings || []);
        fetch('/api/drawings/import', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ drawings: list }),
        }).then(function(r) { return r.json(); })
          .then(function(resp) { self.refresh(resp.drawings || []); })
          .catch(function() { self.loadFromServer(self.symbol, self.timeframe); });
      } catch (e) {
        alert('Не удалось разобрать JSON');
      }
    };
    rd.readAsText(file);
  }
  aiDrawings() {
    return this.drawings.filter(function(d) { return d.created_by === 'ai'; });
  }
}