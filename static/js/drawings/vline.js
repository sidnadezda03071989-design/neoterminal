// Vertical line: вертикальная линия на конкретном баре (разметка событий).
// Одна точка { time, price } → линия от верха до низа графика на x бара.
// Рисуется тем же CustomPrimitive, что и остальные рисунки.

export function paint(opts) {
  const { ctx, size, d, pts, color, selected, ui, kn } = opts;
  if (!pts.length) return;
  const p = pts[0];
  ui.strokeMain(ctx, p.x, 0, p.x, size.height, color, selected);
  ui.drawTag(ctx, ui.fmtTime(d.points[0].time), p.x, 4, color, false);
  kn([p], selected);
}

export function hitDist(pts, px, py) {
  if (!pts.length) return Infinity;
  return Math.abs(px - pts[0].x);
}