// Horizontal Ray: горизонтальная линия от точки вправо в бесконечность.
// Хранится одна точка { time, price }; правый край viewport = условный +∞
// (тот же приём, что у ray.js: линия тянется до границы поля).

export function paint(opts) {
  const { ctx, size, d, pts, color, selected, ui, kn } = opts;
  if (!pts.length) return;
  const p = pts[0];
  ui.strokeMain(ctx, p.x, p.y, size.width, p.y, color, selected);
  ui.drawTag(ctx, ui.fmtPrice(d.points[0].price), size.width - 4, p.y, color, true);
  kn([p], selected);
}

// Hit только справа от точки (левая половина невидима).
export function hitDist(pts, px, py) {
  if (!pts.length) return Infinity;
  const p = pts[0];
  if (px < p.x - 4) return Infinity;
  return Math.abs(py - p.y);
}