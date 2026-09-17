import { state } from '../state.js';

const $ = (id) => document.getElementById(id);
let _switchSymbol = null;

export function setSwitchSymbolHandler(fn) { _switchSymbol = fn; }

export function highlightActiveWatchlist(symbol) {
  document.querySelectorAll('#watchlist-items .watchlist-item').forEach((el) => {
    const sym = el.querySelector('.wl-symbol');
    if (sym) el.classList.toggle('active', sym.textContent === symbol);
  });
}

export function loadWatchlist() {
  const container = $('watchlist-items');
  if (!container) return;
  fetch('/api/watchlist')
    .then(r => r.json())
    .then(data => {
      const items = data.watchlist || [];
      container.innerHTML = '';
      for (const item of items) {
        const div = document.createElement('div');
        const active = item.symbol === state.symbol ? ' active' : '';
        const chg = item.change != null ? item.change : 0;
        const cls = chg > 0 ? 'up' : chg < 0 ? 'down' : '';
        div.className = 'watchlist-item' + active;
        div.innerHTML =
          '<div class="wl-symbol">' + item.symbol + '</div>' +
          '<div class="wl-price">' + (item.price != null ? Number(item.price).toFixed(item.price >= 1000 ? 2 : item.price >= 1 ? 4 : 5) : '—') + '</div>' +
          '<div class="wl-change ' + cls + '">' +
          (chg > 0 ? '+' : '') + chg.toFixed(2) + '%</div>';
        div.addEventListener('click', () => { if (_switchSymbol) _switchSymbol(item.symbol); });
        container.appendChild(div);
      }
    })
    .catch(e => console.error('watchlist:', e));
}
