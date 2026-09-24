import { showChartLoading } from './chart/series.js';
import { state } from './state.js';
import { loadLive } from './data/live.js';
import { loadReplay, playReplay, pauseReplay, resetReplay,
  replayStepBack, replayStepForward, applyReplayPreset,
  stopReplay, replaySetData, refreshIndicatorsUpto,
  initReplayBarrierDrag } from './data/replay.js';
import { runAIAnalysis, updateAiDrawingsUI } from './ai/analysis.js';
import { openChat, closeChat, sendChatMessage } from './ai/chat.js';
import { setTool, toggleMagnet, syncToolbarUI, setSymbolChangeHandler,
  switchSymbol, confirmedDanger } from './ui/toolbar.js';
import { loadWatchlist, highlightActiveWatchlist, setSwitchSymbolHandler } from './ui/watchlist.js';
import { toggleCommandPalette } from './ui/command_palette.js';
import { bindHotkeys } from './ui/hotkeys.js';
import { indMap, toggleIndicator } from './ui/indicators.js';
import { openAlerts, closeAlerts, showAlertToast, createAlert } from './ui/alerts.js';
import { toggleNotes, openNotes, closeNotes, initNotesUI, onSymbolChanged } from './ui/notes.js';
import { openNews, closeNews, loadNews, initNewsUI,
  onSymbolChanged as onNewsSymbolChanged } from './ui/news.js';
import { openJournal, closeJournal, initJournalUI } from './ui/journal.js';
import { closeBacktest, openBacktest, runBacktest } from './backtest.js';
import { openAiBacktest, closeAiBacktest, runAiBacktest, updateAiBacktestProgress,
  initAiBacktestSymbols, hideAiProbZones, setAiProbZonesRenderer,
  resetAiProbZones }
  from './ui/ai_backtest.js';
import { openScanner, closeScanner, runScan, cancelScan, loadResults, exportCsv,
  updateScanProgress, toggleScanConfig, resetScanConfig, toggleAllStrategies,
  copySummaryReport } from './scanner.js';
import { openAiDataPanel, closeAiDataPanel, onSymbolTfChanged,
  initAiDataUI } from './ui/ai_data_panel.js';
import { initCharonUI } from './charon_panel.js';
import { chart, candleSeries, container, setReplayBarrier } from './chart/setup.js';
import { DrawingsManager } from './drawings/index.js';
import { BacktestTradesRenderer } from './drawings/backtest_trades.js';
import { TZ_LIST, tzOptionLabel, getActiveTzId, setActiveTzId,
  initTimezone, activeOffsetSeconds } from './ui/timezone.js';
import { startCandleTimer } from './ui/candle_timer.js';
import { updateLegend } from './ui/legend.js';

const $ = (id) => document.getElementById(id);

/* ————— Часовые пояса: форматтеры оси времени и crosshair —————
   offsetApply() — ленивая подстановка активного смещения в каждый вызов,
   чтобы при переключении селекта ось перерисовалась сразу же. */
function _fmtTick(t, tickType, isUtc) {
  const off = activeOffsetSeconds();
  const d = new Date((t + off) * 1000);
  const pad = (n) => String(n).padStart(2, '0');
  if (tickType === 3 /* Year */) return String(d.getUTCFullYear());
  if (tickType === 0 /* Month */) return pad(d.getUTCMonth() + 1) + '/' + pad(d.getUTCDate());
  if (tickType === 1 /* DayOfMonth */) return pad(d.getUTCDate());
  if (tickType === 4 /* Time */) return pad(d.getUTCHours()) + ':' + pad(d.getUTCMinutes());
  return pad(d.getUTCHours()) + ':' + pad(d.getUTCMinutes());
}

function _fmtTime(t, isUtc) {
  const off = activeOffsetSeconds();
  const d = new Date((t + off) * 1000);
  const pad = (n) => String(n).padStart(2, '0');
  return pad(d.getUTCDate()) + '.' + pad(d.getUTCMonth() + 1) + '.' + d.getUTCFullYear()
    + ' ' + pad(d.getUTCHours()) + ':' + pad(d.getUTCMinutes());
}

/* Селект пояса + навешивание форматтеров на уже созданный chart. */
function initTimezoneUI() {
  const sel = $('timezone-select');
  if (!sel) return;
  initTimezone();
  sel.innerHTML = '';
  const local = new Option('Локальный (по системе)', 'local');
  sel.appendChild(local);
  for (const tz of TZ_LIST) {
    sel.appendChild(new Option(tzOptionLabel(tz), tz.id));
  }
  sel.value = getActiveTzId();
  sel.addEventListener('change', () => {
    setActiveTzId(sel.value);
    applyChartFormatters();
    updateLegend(null);
  });
  applyChartFormatters();
}

function applyChartFormatters() {
  if (!chart) return;
  try {
    chart.applyOptions({
      timeScale: { tickMarkFormatter: _fmtTick },
      localization: { timeFormatter: _fmtTime, locale: 'ru-RU' },
    });
  } catch (e) { console.warn('[tz] applyFormatters:', e); }
}

/* Подсветка активного пресета скорости реплея (1x/2x/5x). */
function _syncSpeedBtns(v) {
  document.querySelectorAll('.speed-btn').forEach((btn) => {
    const bv = parseFloat(btn.dataset.speed) || 0;
    btn.classList.toggle('active', Math.abs(bv - v) < 0.01);
  });
}

/* ---------- Floating-панели (ЧАСТЬ C/D): открыта одна правая панель ---------- */
function _isOpen(id) {
  const p = $(id);
  return !!p && p.style.display !== 'none';
}

/* Синхронизация active-состояния toggle-кнопок с состоянием панелей. */
function _syncPanelButtons() {
  const map = [
    ['watchlist-toggle-btn', 'watchlist-panel'],
    ['ai-toggle-btn', 'ai-panel'],
    ['chat-toggle-btn', 'chat-panel'],
    ['alerts-toggle-btn', 'alerts-panel'],
    ['backtest-btn', 'backtest-panel'],
    ['ai-backtest-btn', 'ai-backtest-panel'],
    ['scanner-btn', 'scanner-panel'],
    ['notes-btn', 'notes-panel'],
    ['news-btn', 'news-panel'],
    ['journal-btn', 'journal-panel'],
    ['ai-data-toggle-btn', 'ai-data-panel'],
  ];
  for (const [btnId, panelId] of map) {
    const btn = $(btnId);
    if (btn) btn.classList.toggle('panel-active', _isOpen(panelId));
  }
}

/* Закрыть AI, Chat, Alerts, Backtest — открыта только одна правая панель. */
function _closeAllRightPanels() {
  const ai = $('ai-panel');
  if (ai) ai.style.display = 'none';
  closeChat();
  closeAlerts();
  closeBacktest();
  closeAiBacktest();
  closeScanner();
  closeNotes();
  closeNews();
  closeJournal();
  closeAiDataPanel();
}

function toggleWatchlistPanel() {
  const panel = $('watchlist-panel');
  if (!panel) return;
  panel.style.display = _isOpen('watchlist-panel') ? 'none' : 'block';
}

function toggleAIPanel() {
  if (_isOpen('ai-panel')) {
    $('ai-panel').style.display = 'none';
    return;
  }
  _closeAllRightPanels();
  const panel = $('ai-panel');
  if (panel) panel.style.display = 'flex';
}

function toggleChatPanel() {
  if (_isOpen('chat-panel')) { closeChat(); return; }
  _closeAllRightPanels();
  openChat();
}

function toggleAlertsPanel() {
  if (_isOpen('alerts-panel')) { closeAlerts(); return; }
  _closeAllRightPanels();
  openAlerts();
}

function toggleBacktestPanel() {
  if (_isOpen('backtest-panel')) { closeBacktest(); return; }
  _closeAllRightPanels();
  openBacktest();
}

function toggleAiBacktestPanel() {
  if (_isOpen('ai-backtest-panel')) { closeAiBacktest(); return; }
  _closeAllRightPanels();
  openAiBacktest();
}

function toggleScannerPanel() {
  if (_isOpen('scanner-panel')) { closeScanner(); return; }
  _closeAllRightPanels();
  openScanner();
}

function toggleNotesPanel() {
  if (_isOpen('notes-panel')) { closeNotes(); return; }
  _closeAllRightPanels();
  openNotes();
}

function toggleNewsPanel() {
  if (_isOpen('news-panel')) { closeNews(); return; }
  _closeAllRightPanels();
  openNews();
}

function toggleJournalPanel() {
  if (_isOpen('journal-panel')) { closeJournal(); return; }
  _closeAllRightPanels();
  openJournal();
}

function toggleAiDataPanel() {
  if (aiDataPanelIsOpen()) { closeAiDataPanel(); return; }
  _closeAllRightPanels();
  openAiDataPanel();
}

/* Панель «Данные для ИИ» использует классы .floating-panel/.hidden, поэтому
   _isOpen (style.display) для неё не работает — проверяем класс напрямую. */
function aiDataPanelIsOpen() {
  const el = $('ai-data-panel');
  return !!el && !el.classList.contains('hidden');
}

export function initDrawingsManager() {
  const dm = new DrawingsManager({
    chart, series: candleSeries, container,
    candlesRef: () => state.activeCandles,
    onChanged: updateAiDrawingsUI,
    onHistoryChange: syncHistoryButtons,
    symbol: state.symbol, timeframe: state.timeframe,
  });
  state.dm = dm;
  syncHistoryButtons();
}

export function syncHistoryButtons() {
  const u = $('undo-btn');
  const r = $('redo-btn');
  if (u) u.disabled = !(state.dm && state.dm.canUndo());
  if (r) r.disabled = !(state.dm && state.dm.canRedo());
}

export function initBacktestRenderer() {
  if (!chart || !candleSeries) return null;
  const renderer = new BacktestTradesRenderer(
    chart, candleSeries, container, () => state.activeCandles,
  );
  state.backtestRenderer = renderer;
  return renderer;
}

export function initAiProbZonesRenderer() {
  if (!chart || !candleSeries) return null;
  setAiProbZonesRenderer(chart, candleSeries, container);
  return state.aiProbRenderer;
}

export function onSymbolOrTfChange() {
  const symSel = $('symbol-select');
  const tfSel = $('timeframe-select');
  const newSym = symSel ? symSel.value : 'BTCUSDT';
  const newTf = tfSel ? tfSel.value : '15m';

  /* BLOCK-36: symbol/tf НЕ изменились — НЕ перезагружать график. Раньше
     любой вызов (в т.ч. ложный из switchSymbol/setTimeframe) запускал
     loadLive(true) → progressive → fitChartToData, который перебивал
     setVisibleRange сделок бэктеста (97 → 2 блока). */
  if (state.symbol === newSym && state.timeframe === newTf) {
    console.log('[init] onSymbolOrTfChange: no change, skip');
    return;
  }

  stopReplay();
  pauseReplay();
  showChartLoading();
  state.symbol = newSym;
  state.timeframe = newTf;
  highlightActiveWatchlist(state.symbol);
  state.candles = [];
  state.ind = null;
  resetAiProbZones();
  onSymbolChanged(newSym);
  onNewsSymbolChanged(newSym);
  onSymbolTfChanged();
  if (state.dm && state.dm.setCurrentContext) {
    state.dm.setCurrentContext(state.symbol, state.timeframe);
  }
  // BLOCK-34: смена symbol/tf — разовый fit после загрузки, дальше график
  // реагирует только на действия пользователя.
  if (state.mode === 'live') { setReplayBarrier(null); state.chartNeedsFit = true; loadLive(true); }
  else { state.chartNeedsFit = true; loadReplay(true); }  // preserveTime: барьер остаётся на прежнем времени
}

export function initUI() {
  /* SSE-прогресс сканера: integration/sse.js дергает window.__scanProgress
     (формат события scan_progress: {run_id, done, total, current}). */
  window.__scanProgress = updateScanProgress;
  window.__aiBacktestProgress = updateAiBacktestProgress;
  // Пока данные графика догружаются фоном, кнопка «🔍 Сканер → Запустить»
  // заблокирована (state.progressiveLoaded ставится в live.js).
  setInterval(() => {
    const btn = document.getElementById('scan-run-btn');
    if (!btn) return;
    btn.disabled = !state.progressiveLoaded;
    btn.title = state.progressiveLoaded ? '' : 'Данные загружаются…';
  }, 500);
  document.querySelectorAll('.tool-btn[data-tool]').forEach((btn) => {
    btn.addEventListener('click', () => setTool(btn.dataset.tool));
  });
  const meaBtn = $('measure-btn');
  if (meaBtn) meaBtn.addEventListener('click', () => setTool('measure'));
  const magBtn = $('magnet-btn');
  if (magBtn) magBtn.addEventListener('click', toggleMagnet);
  const modeToggle = $('mode-toggle');
  if (modeToggle) {
    modeToggle.querySelectorAll('button').forEach((b) => {
      b.addEventListener('click', () => {
        state.mode = b.dataset.mode;
        syncToolbarUI();
        resetAiProbZones();  // срез сменился (live <-> replay) — зоны устарели
        if (state.mode === 'live') { setReplayBarrier(null); loadLive(true); }
        else { stopReplay(); loadReplay(); }
      });
    });
  }
  const symSel = $('symbol-select');
  if (symSel) symSel.addEventListener('change', onSymbolOrTfChange);
  const tfSel = $('timeframe-select');
  if (tfSel) tfSel.addEventListener('change', onSymbolOrTfChange);
  const lb = $('load-replay-btn'); if (lb) lb.addEventListener('click', loadReplay);
  const pb = $('play-btn'); if (pb) pb.addEventListener('click', playReplay);
  const pa = $('pause-btn'); if (pa) pa.addEventListener('click', pauseReplay);
  const rb = $('reset-btn'); if (rb) rb.addEventListener('click', resetReplay);
  const sb = $('step-back-btn'); if (sb) sb.addEventListener('click', replayStepBack);
  const sf = $('step-forward-btn'); if (sf) sf.addEventListener('click', replayStepForward);
  document.querySelectorAll('.preset-btn').forEach((btn) => {
    btn.addEventListener('click', () => applyReplayPreset(btn.dataset.preset));
  });
  const slider = $('progress-slider');
  if (slider) slider.addEventListener('input', (e) => {
    stopReplay();
    state.replay.index = Number(e.target.value);
    replaySetData(state.replay.index);
    refreshIndicatorsUpto();
  });
  const speed = $('speed-control');
  if (speed) speed.addEventListener('input', (e) => {
    const wasPlaying = state.replay.playing;
    if (wasPlaying) pauseReplay();  // перезапуск таймера уже с новой скоростью
    state.replay.speed = parseFloat(e.target.value) || 2;
    _syncSpeedBtns(state.replay.speed);
    if (wasPlaying) playReplay();
  });
  // Пресеты скорости 1x/2x/5x — дублируют слайдер speed-control.
  document.querySelectorAll('.speed-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const v = parseFloat(btn.dataset.speed) || 1;
      const sp = $('speed-control');
      if (sp) sp.value = String(v);
      const wasPlaying = state.replay.playing;
      if (wasPlaying) pauseReplay();
      state.replay.speed = v;
      _syncSpeedBtns(v);
      if (wasPlaying) playReplay();
    });
  });
  _syncSpeedBtns(state.replay.speed);
  const indBtn = $('indicators-btn');
  const indDd = $('indicators-dropdown');
  if (indBtn && indDd) {
    indBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      indDd.classList.toggle('open');
    });
    document.addEventListener('click', () => indDd.classList.remove('open'));
    const menu = $('indicators-menu');
    if (menu) menu.addEventListener('click', (e) => e.stopPropagation());
  }
  for (const key of Object.keys(indMap)) {
    const cb = $(key);
    if (cb) cb.addEventListener('change', (e) => toggleIndicator(indMap[key], e.target.checked));
  }
  const aiBtn = $('ai-analysis-btn'); if (aiBtn) aiBtn.addEventListener('click', runAIAnalysis);
  const aiRef = $('ai-refresh-btn'); if (aiRef) aiRef.addEventListener('click', runAIAnalysis);
  const chatBtn = $('chat-toggle-btn');
  if (chatBtn) chatBtn.addEventListener('click', toggleChatPanel);
  const chatClose = $('chat-close-btn'); if (chatClose) chatClose.addEventListener('click', closeChat);
  const chatSend = $('chat-send-btn'); if (chatSend) chatSend.addEventListener('click', sendChatMessage);
  const chatInp = $('chat-input');
  if (chatInp) chatInp.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChatMessage(); }
  });
  const alBtn = $('alerts-toggle-btn');
  if (alBtn) alBtn.addEventListener('click', toggleAlertsPanel);
  const alClose = $('alerts-close-btn');
  if (alClose) alClose.addEventListener('click', closeAlerts);
  const alChan = $('alert-channel');
  if (alChan) alChan.addEventListener('change', () => {
    const dest = $('alert-destination');
    if (dest) dest.style.display = alChan.value === 'browser' ? 'none' : 'block';
  });
  const alCreate = $('alert-create-btn');
  if (alCreate) alCreate.addEventListener('click', createAlert);
  const btBtn = $('backtest-btn');
  if (btBtn) btBtn.addEventListener('click', toggleBacktestPanel);
  const btClose = $('bt-close-btn');
  if (btClose) btClose.addEventListener('click', closeBacktest);
  const btRun = $('bt-run-btn');
  if (btRun) btRun.addEventListener('click', runBacktest);
  initAiBacktestSymbols();
  const aibtBtn = $('ai-backtest-btn');
  if (aibtBtn) aibtBtn.addEventListener('click', toggleAiBacktestPanel);
  const aibtClose = $('aibt-close-btn');
  if (aibtClose) aibtClose.addEventListener('click', closeAiBacktest);
  const aibtRun = $('aibt-run-btn');
  if (aibtRun) aibtRun.addEventListener('click', runAiBacktest);
  const aibtHide = $('aibt-hide-zones-btn');
  if (aibtHide) aibtHide.addEventListener('click', hideAiProbZones);
  const scBtn = $('scanner-btn');
  if (scBtn) scBtn.addEventListener('click', toggleScannerPanel);
  const scClose = $('scanner-close-btn');
  if (scClose) scClose.addEventListener('click', closeScanner);
  const scRun = $('scan-run-btn');
  if (scRun) scRun.addEventListener('click', runScan);
  const scCancel = $('scan-cancel-btn');
  if (scCancel) scCancel.addEventListener('click', cancelScan);
  const scExport = $('scan-export-btn');
  if (scExport) scExport.addEventListener('click', exportCsv);
  const scMin = $('scan-min-sharpe');
  if (scMin) scMin.addEventListener('change', loadResults);
  const scCfg = $('scan-config-toggle');
  if (scCfg) scCfg.addEventListener('click', toggleScanConfig);
  const scCfgReset = $('scan-config-reset');
  if (scCfgReset) scCfgReset.addEventListener('click', resetScanConfig);
  // Чекбокс «Все стратегии»: отметить/снять все 12 (counter и оценка
  // пересчитываются в scanner.js::_onSelectionChange).
  const scAll = $('scan-all-strategies');
  if (scAll) scAll.addEventListener('change', toggleAllStrategies);
  const scCopy = $('scan-copy-btn');
  if (scCopy) scCopy.addEventListener('click', copySummaryReport);
  const ntBtn = $('notes-btn');
  if (ntBtn) ntBtn.addEventListener('click', toggleNotesPanel);
  const ntClose = $('notes-close-btn');
  if (ntClose) ntClose.addEventListener('click', closeNotes);
  const nwBtn = $('news-btn');
  if (nwBtn) nwBtn.addEventListener('click', toggleNewsPanel);
  const nwClose = $('news-close-btn');
  if (nwClose) nwClose.addEventListener('click', closeNews);
  const nwRefresh = $('news-refresh-btn');
  if (nwRefresh) nwRefresh.addEventListener('click', () => loadNews(true));
  const aiddBtn = $('ai-data-toggle-btn');
  if (aiddBtn) aiddBtn.addEventListener('click', toggleAiDataPanel);
  const jrClose = $('journal-close-btn');
  if (jrClose) jrClose.addEventListener('click', closeJournal);
  const jrBtn = $('journal-btn');
  if (jrBtn) jrBtn.addEventListener('click', toggleJournalPanel);
  const clr = $('clear-btn');
  if (clr) clr.addEventListener('click', () => {
    if (state.dm && confirmedDanger('Удалить все рисунки?')) state.dm.clearAll();
  });
  const uBtn = $('undo-btn');
  const rBtn = $('redo-btn');
  if (uBtn) uBtn.addEventListener('click', () => { if (state.dm) state.dm.undo(); });
  if (rBtn) rBtn.addEventListener('click', () => { if (state.dm) state.dm.redo(); });
  const eraseAi = $('erase-ai-btn');
  if (eraseAi) eraseAi.addEventListener('click', () => {
    if (state.dm && confirmedDanger('Стереть рисунки AI?')) state.dm.eraseAI();
  });
  const exp = $('export-btn');
  if (exp) exp.addEventListener('click', () => { if (state.dm) state.dm.exportJSON(); });
  const imp = $('import-btn');
  const impFile = $('import-file');
  if (imp && impFile) {
    imp.addEventListener('click', () => impFile.click());
    impFile.addEventListener('change', (e) => {
      const f = e.target.files && e.target.files[0];
      if (f && state.dm) state.dm.importJSON(f);
      e.target.value = '';
    });
  }
  const cmdBtn = $('cmd-palette-btn');
  if (cmdBtn) cmdBtn.addEventListener('click', toggleCommandPalette);
  const wlBtn = $('watchlist-toggle-btn');
  if (wlBtn) wlBtn.addEventListener('click', toggleWatchlistPanel);
  const aiPnlBtn = $('ai-toggle-btn');
  if (aiPnlBtn) aiPnlBtn.addEventListener('click', toggleAIPanel);
  /* Кнопки panel-active синхронизируются с любым изменением style панелей
     (покрывает и command palette, и ✕-кнопки внутри панелей). */
  const panelObserver = new MutationObserver(_syncPanelButtons);
  for (const id of ['watchlist-panel', 'ai-panel', 'chat-panel',
                    'alerts-panel', 'backtest-panel', 'ai-backtest-panel',
                    'scanner-panel', 'notes-panel', 'news-panel',
                    'journal-panel']) {
    const p = $(id);
    if (p) panelObserver.observe(p, { attributes: true, attributeFilter: ['style'] });
  }
  _syncPanelButtons();
  setSymbolChangeHandler(onSymbolOrTfChange);
  setSwitchSymbolHandler(switchSymbol);
  bindHotkeys();
  initNotesUI();
  initNewsUI();
  initJournalUI();
  initAiDataUI();
  initCharonUI();
  initReplayBarrierDrag();
  initTimezoneUI();
  startCandleTimer();
  setTool('cursor');
  syncToolbarUI();
}
