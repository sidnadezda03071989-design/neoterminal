// Parallel Channel: 2 параллельных линии + полупрозрачная заливка между ними.
// 3 клика: p1 — начало 1-й линии, p2 — конец 1-й линии, p3 — ширина (2-я линия).
// Рендер в пикселях; направление канала — вектор (p2 - p1), 2-я линия
// проходит через p3. Заливка — CSS-переменная --drawing-fill.

function projectY(ax, ay, dx, dy, x) {
  if (Math.abs(dx) < 1e-6) return null;
  return ay + ((x - ax) / dx) * dy;
}

function lineSpan(ctx, ax, ay, dx, dy, x0, x1, stroke) {
  if (Math.abs(dx) < 1e-6) {
    ctx.beginPath(); ctx.moveTo(ax, -100000); ctx.lineTo(ax, 100000);
    stroke();
    return;
  }
  const y0 = projectY(ax, ay, dx, dy, x0);
  const y1 = projectY(ax, ay, dx, dy, x1);
  ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1);
  stroke();
}

function pointInPoly(px, py, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const a = poly[i], b = poly[j];
    if ((a.y > py) !== (b.y > py) &&
        px < ((b.x - a.x) * (py - a.y)) / (b.y - a.y) + a.x) inside = !inside;
  }
  return inside;
}

function distToSeg(px, py, x1, y1, x2, y2) {
  const dx = x2 - x1, dy = y2 - y1;
  if (dx === 0 && dy === 0) return Math.hypot(px - x1, py - y1);
  const t = Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)));
  return Math.hypot(px - (x1 + t * dx), py - (y1 + t * dy));
}

export function paint(opts) {
  const { ctx, d, pts, color, selected, ui, kn, preview } = opts;
  if (pts.length < 2) return;
  const A = pts[0], B = pts[1];
  const C = d.points[2] ? pts[2] : (preview || null);

  if (!C) {
    ui.strokeMain(ctx, A.x, A.y, B.x, B.y, color, selected);
    kn(pts, selected);
    return;
  }

  const dx = B.x - A.x, dy = B.y - A.y;
  const x0 = Math.min(A.x, B.x, C.x);
  const x1 = Math.max(A.x, B.x, C.x);
  const D = { x: C.x + dx, y: C.y + dy };
  const quad = [A, B, D, C];

  // Заливка между линиями (параллелограмм A,B,D,C).
  ctx.beginPath();
  quad.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
  ctx.closePath();
  ctx.fillStyle = ui.cssVar('--drawing-fill', 'rgba(88,166,255,0.18)');
  ctx.fill();

  // Две параллельные линии, протянутые на общий x-диапазон всех точек.
  const stroke = () => {
    ctx.strokeStyle = color;
    ctx.lineWidth = selected ? 3 : 2;
    ctx.stroke();
  };
  lineSpan(ctx, A.x, A.y, dx, dy, x0, x1, stroke);
  lineSpan(ctx, C.x, C.y, dx, dy, x0, x1, stroke);
  if (selected) {
    const slide = () => {
      ctx.strokeStyle = 'rgba(255,255,255,0.35)';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.stroke();
      ctx.setLineDash([]);
    };
    lineSpan(ctx, A.x, A.y, dx, dy, x0, x1, slide);
    lineSpan(ctx, C.x, C.y, dx, dy, x0, x1, slide);
  }
  kn(pts, selected);
}

export function hitDist(pts, px, py) {
  if (pts.length < 2) return Infinity;
  const A = pts[0], B = pts[1];
  if (pts.length < 3) return distToSeg(px, py, A.x, A.y, B.x, B.y);
  const C = pts[2];
  const dx = B.x - A.x, dy = B.y - A.y;
  const D = { x: C.x + dx, y: C.y + dy };
  const quad = [A, B, D, C];
  if (pointInPoly(px, py, quad)) return 0;
  return Math.min(
    distToSeg(px, py, A.x, A.y, B.x, B.y),
    distToSeg(px, py, C.x, C.y, D.x, D.y),
  );
}