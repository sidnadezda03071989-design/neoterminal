import { showChartLoading } from './chart/series.js';
import { state } from './state.js';
import { loadLive } from './data/live.js';
import { loadReplay, playReplay, pauseReplay, resetReplay,
  replayStepBack, replayStepForward, applyReplayPreset,
  stopReplay, replaySetData } from './data/replay.js';
import { runAIAnalysis, updateAiDrawingsUI } from './ai/analysis.js';
import { openChat, closeChat, sendChatMessage } from './ai/chat.js';
import { setTool, toggleMagnet, syncToolbarUI, setSymbolChangeHandler,
  switchSymbol, confirmedDanger } from './ui/toolbar.js';
import { loadWatchlist, highlightActiveWatchlist, setSwitchSymbolHandler } from './ui/watchlist.js';
import { toggleCommandPalette } from './ui/command_palette.js';
import { bindHotkeys } from './ui/hotkeys.js';
import { indMap, toggleIndicator } from './ui/indicators.js';
import { openAlerts, closeAlerts, showAlertToast, createAlert } from './ui/alerts.js';
import { closeBacktest, openBacktest, runBacktest } from './backtest.js';
import { openScanner, closeScanner, runScan, loadResults, exportCsv,
  updateScanProgress, toggleScanConfig, resetScanConfig, toggleAllStrategies,
  copySummaryReport } from './scanner.js';
import { chart, candleSeries, container } from './chart/setup.js';
import { DrawingsManager } from './drawings/index.js';

const $ = (id) => document.getElementById(id);

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
    ['scanner-btn', 'scanner-panel'],
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
  closeScanner();
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

function toggleScannerPanel() {
  if (_isOpen('scanner-panel')) { closeScanner(); return; }
  _closeAllRightPanels();
  openScanner();
}

export function initDrawingsManager() {
  const dm = new DrawingsManager({
    chart, series: candleSeries, container,
    candlesRef: () => state.activeCandles,
    onChanged: updateAiDrawingsUI,
    symbol: state.symbol, timeframe: state.timeframe,
  });
  state.dm = dm;
}

export function onSymbolOrTfChange() {
  stopReplay();
  pauseReplay();
  showChartLoading();
  const symSel = $('symbol-select');
  const tfSel = $('timeframe-select');
  state.symbol = symSel ? symSel.value : 'BTCUSDT';
  state.timeframe = tfSel ? tfSel.value : '1H';
  highlightActiveWatchlist(state.symbol);
  state.candles = [];
  state.ind = null;
  if (state.dm && state.dm.setCurrentContext) {
    state.dm.setCurrentContext(state.symbol, state.timeframe);
  }
  if (state.mode === 'live') loadLive(true);
  else loadReplay();
}

export function initUI() {
  /* SSE-прогресс сканера: integration/sse.js дергает window.__scanProgress
     (формат события scan_progress: {run_id, done, total, current}). */
  window.__scanProgress = updateScanProgress;
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
        if (state.mode === 'live') { loadLive(true); }
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
  });
  const speed = $('speed-control');
  if (speed) speed.addEventListener('input', (e) => {
    const wasPlaying = state.replay.playing;
    if (wasPlaying) pauseReplay();  // перезапуск таймера уже с новой скоростью
    state.replay.speed = parseFloat(e.target.value) || 2;
    if (wasPlaying) playReplay();
  });
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
  const scBtn = $('scanner-btn');
  if (scBtn) scBtn.addEventListener('click', toggleScannerPanel);
  const scClose = $('scanner-close-btn');
  if (scClose) scClose.addEventListener('click', closeScanner);
  const scRun = $('scan-run-btn');
  if (scRun) scRun.addEventListener('click', runScan);
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
  const clr = $('clear-btn');
  if (clr) clr.addEventListener('click', () => {
    if (state.dm && confirmedDanger('Удалить все рисунки?')) state.dm.clearAll();
  });
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
                    'alerts-panel', 'backtest-panel', 'scanner-panel']) {
    const p = $(id);
    if (p) panelObserver.observe(p, { attributes: true, attributeFilter: ['style'] });
  }
  _syncPanelButtons();
  setSymbolChangeHandler(onSymbolOrTfChange);
  setSwitchSymbolHandler(switchSymbol);
  bindHotkeys();
  setTool('cursor');
  syncToolbarUI();
}
