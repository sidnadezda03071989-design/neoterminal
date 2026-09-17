export function applyLayout(layout) {
  const wrap = document.querySelector('.chart-wrap');
  if (!wrap) return;
  wrap.classList.remove('chart-layout-1','chart-layout-2h','chart-layout-2v','chart-layout-4');
  wrap.classList.add('chart-layout-' + layout);
}
