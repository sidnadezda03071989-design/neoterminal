import { state } from './state.js';
import { loadAppState, saveAppState } from './integration/autosave.js';
import { loadLive, startLivePolling } from './data/live.js';
import { loadWatchlist } from './ui/watchlist.js';
import { toggleCommandPalette } from './ui/command_palette.js';
import { setTool } from './ui/toolbar.js';
import { chart } from './chart/setup.js';
import { initDrawingsManager, initUI } from './app_init.js';
import { showAlertToast } from './ui/alerts.js';

const $ = (id) => document.getElementById(id);

window.__setTool = setTool;
window.__toggleCmdPalette = toggleCommandPalette;
window.__alertToast = showAlertToast;
state.chart = chart;

document.addEventListener('DOMContentLoaded', () => {
  const saved = loadAppState();
  if (saved) {
    if (saved.symbol) state.symbol = saved.symbol;
    if (saved.timeframe) state.timeframe = saved.timeframe;
    if (saved.indicators) Object.assign(state.indicators, saved.indicators);
       const ss = $('symbol-select'); if (ss) ss.value = state.symbol;
       const ts = $('timeframe-select'); if (ts) ts.value = state.timeframe;
  }
  initDrawingsManager();
  initUI();
  loadLive(true).then(() => startLivePolling());
  loadWatchlist();
  if (state.dm && state.dm.loadFromServer) {
    state.dm.loadFromServer(state.symbol, state.timeframe);
  }
  document.addEventListener('change', () => saveAppState(state));
});
