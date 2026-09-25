import { state } from '../state.js';

const $ = (id) => document.getElementById(id);
const REFRESH_INTERVAL_MS = 30000;
const REQUEST_TIMEOUT_MS = 12000;
let _switchSymbol = null;
let _refreshTimer = null;
let _loadPromise = null;

export function setSwitchSymbolHandler(fn) { _switchSymbol = fn; }

function _setStatus(text, stateName = '') {
  const status = $('watchlist-status');
  const label = $('watchlist-status-text') || status;
  if (label) label.textContent = text;
  if (status) status.dataset.state = stateName;
}

function _formatPrice(value) {
  if (value == null || !Number.isFinite(Number(value))) return '—';
  const price = Number(value);
  const digits = price >= 1000 ? 2 : price >= 1 ? 4 : 5;
  return price.toLocaleString('ru-RU', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function _renderWatchlist(items) {
  const container = $('watchlist-items');
  if (!container) return;
  container.innerHTML = '';
  for (const item of items) {
    if (!item || !item.symbol) continue;
    const row = document.createElement('div');
    row.className = 'watchlist-item' + (item.symbol === state.symbol ? ' active' : '');
    row.dataset.symbol = item.symbol;

    const symbol = document.createElement('div');
    symbol.className = 'wl-symbol';
    symbol.textContent = item.symbol;
    const name = document.createElement('div');
    name.className = 'wl-name';
    name.textContent = item.name || item.symbol;

    const price = document.createElement('div');
    price.className = 'wl-price';
    price.textContent = _formatPrice(item.price);

    const hasChange = item.change != null && Number.isFinite(Number(item.change));
    const changeValue = Number(item.change);
    const change = document.createElement('div');
    change.className = 'wl-change ' + (!hasChange ? '' : changeValue > 0 ? 'up' : changeValue < 0 ? 'down' : '');
    change.textContent = hasChange
      ? (changeValue > 0 ? '+' : '') + changeValue.toFixed(2) + '%'
      : '—';

    row.append(symbol, name, price, change);
    row.addEventListener('click', () => {
      if (_switchSymbol) _switchSymbol(item.symbol);
    });
    container.appendChild(row);
  }
}

export function highlightActiveWatchlist(symbol) {
  document.querySelectorAll('#watchlist-items .watchlist-item').forEach((el) => {
    el.classList.toggle('active', el.dataset.symbol === symbol);
  });
}

export function loadWatchlist() {
  // main.js прогревает список при старте, а открытие панели может вызвать
  // повторный load в том же тике. Переиспользуем один сетевой запрос.
  if (_loadPromise) return _loadPromise;

  _setStatus('Загрузка котировок…', 'loading');
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  const request = fetch('/api/watchlist', {
    cache: 'no-store',
    signal: controller.signal,
  })
    .then((r) => {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.json();
    })
    .then((data) => {
      const items = Array.isArray(data.watchlist) ? data.watchlist : [];
      _renderWatchlist(items);
      if (!items.length) {
        _setStatus('Список пуст — повторите ⟳', 'error');
      } else if (items.some((item) => item.price == null)) {
        _setStatus('Изменение за сегодня · часть данных недоступна', 'partial');
      } else {
        _setStatus('Изменение за сегодня', 'ready');
      }
      return items;
    })
    .catch((e) => {
      console.error('watchlist:', e);
      _setStatus('Не удалось загрузить — нажмите ⟳', 'error');
      return [];
    })
    .finally(() => {
      clearTimeout(timeout);
      if (_loadPromise === request) _loadPromise = null;
    });

  _loadPromise = request;
  return request;
}

export function startWatchlistRefresh() {
  stopWatchlistRefresh();
  void loadWatchlist();
  _refreshTimer = setInterval(() => void loadWatchlist(), REFRESH_INTERVAL_MS);
}

export function stopWatchlistRefresh() {
  if (_refreshTimer) {
    clearInterval(_refreshTimer);
    _refreshTimer = null;
  }
}
