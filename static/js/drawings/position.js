// Long/Short Position tool (как в TradingView).
// 3 клика: entry → target → SL. Тип рисунка задаёт направление явно:
//   long_position  — target ВЫШЕ entry (зелёный блок сверху), SL ниже (красный);
//   short_position — target НИЖЕ entry (зелёный блок снизу), SL выше.
// Legacy-тип 'position' — направление выводится автоматически
// (target > entry → long).
// Рендер:
//   — зелёный прямоугольник entry→target, красный entry→SL;
//   — метка LONG/SHORT и R:R — слева вверху блока;
//   — лейблы «TP ±%» / «SL ±%» привязаны к своим линиям внутри блока,
//     поэтому при перетаскивании стопа/тейка процент движется вместе с линией;
//   — R:R = |target−entry| / |entry−SL| (та же формула, что rr_levels
//     в app_pkg/ai/backtest.py).

const MIN_WIDTH = 20;

function rectZone(ctx, x0, x1, ya, yb, fill) {
  const y1 = Math.min(ya, yb), y2 = Math.max(ya, yb);
  ctx.fillStyle = fill;
  ctx.fillRect(x0, y1, x1 - x0, y2 - y1);
}

function distToRect(px, py, x0, x1, y0, y1) {
  if (px >= x0 && px <= x1 && py >= y0 && py <= y1) return 0;
  const dx = Math.max(0, x0 - px, px - x1);
  const dy = Math.max(0, y0 - py, py - y1);
  return Math.hypot(dx, dy);
}

export function isLong(d) {
  if (d.type === 'long_position') return true;
  if (d.type === 'short_position') return false;
  const tp = d.points && d.points[1] && d.points[1].price;
  const ep = d.points && d.points[0] && d.points[0].price;
  return tp != null && ep != null ? tp >= ep : true;
}

/* Лейбл у линии: baseline 'top' — текст внутри блока под линией,
   baseline 'bottom' — внутри блока над линией. */
function sideLabel(ctx, text, x, y, baseline, color) {
  ctx.font = '10px "Segoe UI", Tahoma, sans-serif';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'alphabetic';
  const w = ctx.measureText(text).width;
  const ly = baseline === 'top' ? y + 4 : y - 4;
  ctx.fillStyle = 'rgba(19,23,34,0.72)';
  ctx.fillRect(x - w - 10, ly - 10, w + 8, 13);
  ctx.fillStyle = color;
  if (baseline === 'top') ctx.fillText(text, x - 6, ly - 2);
  else ctx.fillText(text, x - 6, ly + 1);
  ctx.textBaseline = 'alphabetic';
}

export function paint(opts) {
  const { ctx, d, pts, selected, ui, kn, preview } = opts;
  if (!pts.length) return;
  const E = pts[0];
  const T = pts[1] || preview;   // target (или превью 2-й точки)
  if (!T) { kn([E], selected); return; }
  const Sl = pts[2] || null;     // SL — только реальная 3-я точка
  const long = isLong(d);

  const x0 = Math.min(E.x, T.x, Sl ? Sl.x : T.x);
  let x1 = Math.max(E.x, T.x, Sl ? Sl.x : T.x);
  if (x1 - x0 < MIN_WIDTH) x1 = x0 + MIN_WIDTH;

  const green = ui.cssVar('--up', '#3fb950');
  const red = ui.cssVar('--down', '#f85149');

  rectZone(ctx, x0, x1, E.y, T.y, ui.hexToRgba(green, 0.30));
  if (Sl) rectZone(ctx, x0, x1, E.y, Sl.y, ui.hexToRgba(red, 0.22));

  const yTop = Math.min(E.y, T.y, Sl ? Sl.y : T.y);
  const yBot = Math.max(E.y, T.y, Sl ? Sl.y : T.y);

  // Пунктирная линия входа от верхней границы до нижней.
  ctx.strokeStyle = ui.hexToRgba('#ffffff', 0.45);
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 3]);
  ctx.beginPath(); ctx.moveTo(E.x, yTop);
  ctx.lineTo(E.x, yBot);
  ctx.stroke();
  ctx.setLineDash([]);

  // Метка направления + R:R слева вверху блока.
  const pctTp = pricePct(d, 1);
  const pctSl = pricePct(d, 2);
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  ctx.font = 'bold 10px "Segoe UI", Tahoma, sans-serif';
  ctx.fillStyle = long ? green : red;
  ctx.fillText(long ? 'LONG' : 'SHORT', x0 + 4, yTop + 3);
  if (pctTp != null && pctSl != null) {
    const risk = Math.abs(epPrice(d) - slPrice(d));
    const reward = Math.abs(tpPrice(d) - epPrice(d));
    const rr = risk > 1e-12 ? reward / risk : Infinity;
    ctx.font = '10px "Segoe UI", Tahoma, sans-serif';
    ctx.fillStyle = ui.cssVar('--text', '#e6edf3');
    ctx.fillText('R:R ' + (isFinite(rr) ? rr.toFixed(2) : '∞'), x0 + 4, yTop + 17);
  }
  ctx.textBaseline = 'alphabetic';

  if (selected) {
    ctx.strokeStyle = 'rgba(255,255,255,0.75)';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(x0, yTop, x1 - x0, yBot - yTop);
    kn([E, Sl || T], selected);
  }

  // Лейблы TP/SL, привязанные к своим линиям внутри блока.
  if (T && pctTp != null) {
    const insideTop = long; // для long линия TP — верхняя граница блока
    sideLabel(ctx, 'TP ' + (pctTp >= 0 ? '+' : '') + pctTp.toFixed(2) + '%',
      x1, T.y, insideTop ? 'top' : 'bottom', green);
  }
  if (Sl && pctSl != null) {
    sideLabel(ctx, 'SL ' + (pctSl >= 0 ? '+' : '') + pctSl.toFixed(2) + '%',
      x1, Sl.y, long ? 'bottom' : 'top', red);
  }
}

function fp(d, i) {
  const p = d.points && d.points[i];
  return p && Number.isFinite(Number(p.price)) ? Number(p.price) : null;
}

function epPrice(d) { return fp(d, 0); }
function tpPrice(d) { return fp(d, 1); }
function slPrice(d) { return fp(d, 2); }

function pricePct(d, i) {
  const ep = epPrice(d), v = fp(d, i);
  if (ep == null || v == null || ep === 0) return null;
  return (v - ep) / Math.abs(ep) * 100;
}

export function hitDist(pts, px, py) {
  if (!pts.length) return Infinity;
  const E = pts[0];
  const T = pts[1] || E;
  const S = pts[2] || T;
  const x0 = Math.min(E.x, T.x, S.x);
  let x1 = Math.max(E.x, T.x, S.x);
  if (x1 - x0 < MIN_WIDTH) x1 = x0 + MIN_WIDTH;
  const y0 = Math.min(E.y, T.y, S.y);
  const y1 = Math.max(E.y, T.y, S.y);
  return distToRect(px, py, x0, x1, y0, y1);
}