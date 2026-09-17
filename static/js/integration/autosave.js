export function saveAppState(state) {
  try {
    const data = {
      symbol: state.symbol,
      timeframe: state.timeframe,
      mode: state.mode,
      indicators: state.indicators,
    };
    localStorage.setItem('neoterminal_state', JSON.stringify(data));
  } catch (e) { /* noop */ }
}

export function loadAppState() {
  try {
    const raw = localStorage.getItem('neoterminal_state');
    return raw ? JSON.parse(raw) : null;
  } catch (e) { return null; }
}
