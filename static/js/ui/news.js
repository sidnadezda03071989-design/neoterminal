// Новости по активу: лента Finnhub (+CoinGecko, если доступна) в нижней
// части правого dock со списком котировок; перезагружается при смене актива
// и по кнопке ⟳ (bypass серверного кеша, ?refresh=1).
// Под лентой — карточка «Статистика стратегий (Scanner Edge)»: лучшая
// комбинация сканера для текущих symbol+timeframe (GET /api/scanner/stats),
// с цветовой оценкой Sharpe и текстовым пояснением (надёжная / убыточная /
// возможное переобучение) — для трейдера и для копирования в промпт ИИ.

import { state } from '../state.js';

let _symbol = null;      // символ, для которого сейчас показаны новости
let _loading = false;
let _edgeSymbol = null;  // символ, для которого показана карточка Scanner Edge

const panel = () => document.getElementById('news-panel');
const body = () => document.getElementById('news-body');
const label = () => document.getElementById('news-symbol-label');
const scanEdgeCard = () => document.getElementById('scan-edge-card');
const newsList = () => document.getElementById('news-list');

function _isVisible() {
  const p = panel();
  if (!p || p.style.display === 'none') return false;
  const host = p.closest('#watchlist-panel');
  return !host || host.style.display !== 'none';
}

function _fmtTime(epoch) {
  if (!epoch) return '';
  try {
    return new Date(epoch * 1000).toLocaleString('ru-RU', {
      day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit',
    });
  } catch { return ''; }
}

function _esc(s) {
  const d = document.createElement('div');
  d.textContent = String(s == null ? '' : s);
  return d.innerHTML;
}

function _render(data) {
  const b = newsList() || body();
  if (!b) return;
  const items = (data && data.news) || [];
  if (!items.length) {
    b.innerHTML = '<div class="news-empty">Новости получить не удалось.<br>' +
      'Проверьте FINNHUB_API_KEY в .env и подключение к сети.</div>';
    return;
  }
  b.innerHTML = items.map((n) => {
    const img = n.image
      ? `<img class="news-thumb" src="${_esc(n.image)}" alt="" loading="lazy"`
        + ` onerror="this.style.display='none'">`
      : '';
    const summary = n.summary
      ? `<div class="news-summary">${_esc(n.summary)}</div>` : '';
    return `<a class="news-item" href="${_esc(n.url)}" target="_blank"`
      + ` rel="noopener noreferrer">${img}<div class="news-item-main">`
      + `<div class="news-title">${_esc(n.title)}</div>`
      + `<div class="news-meta"><span class="src">${_esc(n.source)}</span>`
      + `<span>${_fmtTime(n.datetime)}</span></div>${summary}</div></a>`;
  }).join('');
}

/* ---------- Статистика стратегий (Scanner Edge) ---------- */

function _fmtSharpe(v) {
  const n = Number(v);
  return (v == null || !isFinite(n)) ? '—' : n.toFixed(2);
}

function _fmtPct(v) {
  const n = Number(v);
  return (v == null || !isFinite(n)) ? '—' : `${(n * 100).toFixed(1)}%`;
}

function _sharpeClass(v) {
  const n = Number(v);
  if (v == null || !isFinite(n)) return '';
  if (n < 0) return 'scan-edge-bad';
  if (n > 1.5) return 'scan-edge-good';
  return 'scan-edge-mid';
}

/* Пояснение по карточке: (текст снизу, класс цвета). Приоритет — от грубого
   (убыток на истории) к тонкому (train >> test = переобучение). */
function _scanEdgeExplain(d) {
  const train = Number(d.train_sharpe);
  const test = Number(d.test_sharpe);
  if (Number.isFinite(test) && test < 0) {
    return { text: '❌ Стратегия убыточна на истории. Высокий риск.',
             cls: 'scan-edge-bad' };
  }
  if (Number.isFinite(train) && Number.isFinite(test)
      && test < train * 0.6 && train - test > 0.4) {
    return { text: '⚠️ Возможное переобучение (Overfitting): на новых данных'
             + ' результат хуже, чем на обучении.', cls: 'scan-edge-mid' };
  }
  if ((Number(d.combined_sharpe) || 0) > 3) {
    return { text: '✅ Исторически очень надежная стратегия на этом активе.',
             cls: 'scan-edge-good' };
  }
  return { text: `ℹ️ ${d.verdict_text || 'Исторический edge умеренный —'
             + ' подтверждай сигнал структурой рынка.'}`,
           cls: 'scan-edge-mid' };
}

function _renderScanEdge(d) {
  const card = scanEdgeCard();
  if (!card) return;
  if (!d || !d.strategy) {
    card.style.display = 'none';
    card.innerHTML = '';
    return;
  }
  const sc = _sharpeClass(d.combined_sharpe);
  const expl = _scanEdgeExplain(d);
  const params = Object.keys(d.params || {}).length
    ? ` (${Object.entries(d.params).map(([k, v]) => `${k}=${v}`).join(', ')})`
    : '';
  card.innerHTML =
    '<div class="scan-edge-head">📊 Статистика стратегий (Scanner Edge)</div>'
    + `<div class="scan-edge-strategy">${_esc(d.strategy_label || d.strategy)}`
    + `${_esc(params)}</div>`
    + '<div class="scan-edge-metrics">'
    + `<div class="scan-edge-metric"><span class="scan-edge-label">Sharpe</span>`
    + `<span class="scan-edge-value ${sc}"`
    + ` title="combined_sharpe = min(train, test) — консервативная оценка">${_fmtSharpe(d.combined_sharpe)}</span></div>`
    + `<div class="scan-edge-metric"><span class="scan-edge-label">Winrate</span>`
    + `<span class="scan-edge-value">${_fmtPct(d.winrate)}</span></div>`
    + `<div class="scan-edge-metric"><span class="scan-edge-label">Max DD</span>`
    + `<span class="scan-edge-value">${_fmtPct(d.max_dd)}</span></div>`
    + '</div>'
    + `<div class="scan-edge-explain ${expl.cls}">${_esc(expl.text)}</div>`
    + '<div class="scan-edge-note">Данные последнего прогона сканера'
    + ' (out-of-sample winrate/max_dd). Прогоните 🔍 Сканер, чтобы обновить.'
    + '</div>';
  card.style.display = 'block';
}

/* Карточка Scanner Edge для текущих symbol+timeframe. Панель закрыта —
   только запоминаем символ (карточка подхватится при открытии). */
export async function loadScanEdge(force = false) {
  const card = scanEdgeCard();
  if (!card) return;
  const sym = state.symbol;
  if (!_isVisible() && !force) {
    _edgeSymbol = null;
    return;
  }
  if (force) card.innerHTML = '<div class="news-status">Загружаю статистику…</div>';
  try {
    const tf = encodeURIComponent(state.timeframe);
    const resp = await fetch(
      `/api/scanner/stats?symbol=${encodeURIComponent(sym)}&timeframe=${tf}`);
    if (!resp.ok) return;
    const data = await resp.json();
    if (state.symbol !== sym) return;   // символ сменился во время запроса
    _edgeSymbol = sym;
    _renderScanEdge((data && data.strategy) ? data : null);
  } catch { /* карточка не критична — молча оставляем как есть */ }
}

/* Загрузка ленты для текущего символа. force=true — игнорировать серверный
   кеш (?refresh=1, кнопка ⟳). Пока панель закрыта — только запоминаем,
   какой символ требуется (load при открытии подхватит актуальный). */
export async function loadNews(force = false) {
  const sym = state.symbol;
  const p = panel();
  if (!p) return;
  if (!_isVisible()) { _symbol = null; return; }
  if (_loading) return;
  _loading = true;
  const b = newsList() || body();
  if (b && !b.querySelector('.news-item')) {
    b.innerHTML = '<div class="news-status">Загружаю новости…</div>';
  }
  try {
    const resp = await fetch(
      `/api/news/${encodeURIComponent(sym)}${force ? '?refresh=1' : ''}`);
    const data = await resp.json();
    if (state.symbol !== sym) return;  // символ сменился во время запроса
    _symbol = sym;
    if (label()) label().textContent = `📰 Новости · ${sym}`;
    _render(data);
  } catch {
    if (b && state.symbol === sym) {
      b.innerHTML = '<div class="news-status err">Не удалось загрузить новости</div>';
    }
  } finally {
    _loading = false;
  }
}

export function openNews(symbol) {
  const p = panel();
  if (!p) return;
  p.style.display = 'flex';
  const sym = symbol || state.symbol;
  if (label()) label().textContent = `📰 Новости · ${sym}`;
  if (sym !== _symbol || !(newsList() || body()).querySelector('.news-item')) {
    loadNews();
  }
  // тот же символ с уже загруженным списком — показываем как есть
  const card = scanEdgeCard();
  if (sym !== _edgeSymbol
      || !card || !card.querySelector('.scan-edge-head')) loadScanEdge();
}

export function onSymbolChanged(newSym) {
  if (!_isVisible()) return;
  if (newSym === _symbol) return;
  loadNews();
  loadScanEdge();
}

export function initNewsUI() {
  const refresh = document.getElementById('news-refresh-btn');
  if (refresh) {
    refresh.addEventListener('click', () => { loadNews(true); loadScanEdge(true); });
  }
}
