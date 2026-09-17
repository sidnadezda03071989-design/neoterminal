export function showLoading() {
  let ov = document.getElementById('loading-overlay');
  if (!ov) {
    ov = document.createElement('div');
    ov.id = 'loading-overlay';
    ov.style.cssText = 'position:fixed;inset:0;background:rgba(19,23,34,0.7);' +
      'z-index:9998;display:flex;align-items:center;justify-content:center;color:#d1d4dc';
    ov.textContent = 'Загрузка…';
    document.body.appendChild(ov);
  }
  ov.style.display = 'flex';
}

export function hideLoading() {
  const ov = document.getElementById('loading-overlay');
  if (ov) ov.style.display = 'none';
}
