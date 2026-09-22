// Стрелки/маркеры на баре. Рисуются через createSeriesMarkers
// (lightweight-charts нативно рисует arrowUp/arrowDown).
// Одна точка { time, price }; направление — по положению клика относительно
// свечи: клик ниже свечи → 'up', выше → 'down'. Точное место клика
// сохраняется (позиция маркера atPriceMiddle), чтобы не терять price.

// d.points[0].dir — направление, вычисленное при создании/перетаскивании.
// Если свеча не загружена — используем сохранённое, иначе 'up'.
export function dirFor(d, candle) {
  const p = d.points && d.points[0];
  if (!p) return 'up';
  if (candle && isFinite(candle.close)) {
    if (p.price < candle.close) return 'up';
    if (p.price > candle.close) return 'down';
  }
  return p.dir || 'up';
}

export function buildMarker(d, defaultColor) {
  const p = d.points && d.points[0];
  if (!p) return null;
  return {
    time: p.time,
    price: p.price,
    position: 'atPriceMiddle',
    shape: dirFor(d, null) === 'down' ? 'arrowDown' : 'arrowUp',
    color: d.color || defaultColor,
    id: 'd_' + d.id,
    size: 1,
  };
}

export function hitDist(pts, px, py) {
  if (!pts.length) return Infinity;
  return Math.hypot(px - pts[0].x, py - pts[0].y);
}