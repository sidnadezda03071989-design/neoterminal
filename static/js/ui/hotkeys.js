export function bindHotkeys() {
  document.addEventListener('keydown', (e) => {
    const tag = (e.target && e.target.tagName) || '';
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
    if (e.ctrlKey || e.metaKey) {
      if (e.key === 'k' || e.key === 'K') {
        e.preventDefault();
        if (window.__toggleCmdPalette) window.__toggleCmdPalette();
      } else if (e.key.toLowerCase() === 'z') {
        e.preventDefault();
        if (e.shiftKey) { if (window.__redo) window.__redo(); }
        else if (window.__undo) window.__undo();
      } else if (e.key.toLowerCase() === 'y') {
        e.preventDefault();
        if (window.__redo) window.__redo();
      }
      return;
    }
    switch (e.key.toLowerCase()) {
      case 't': if (window.__setTool) window.__setTool('trendline'); break;
      case 'h': if (window.__setTool) window.__setTool('h_line'); break;
      case 'r': if (window.__setTool) window.__setTool('rectangle'); break;
      case 'f': if (window.__setTool) window.__setTool('fib'); break;
      case 'p': if (window.__setTool) window.__setTool('pen'); break;
      case 'v':
        if (e.shiftKey) { if (window.__setTool) window.__setTool('vline'); }
        else if (window.__setTool) window.__setTool('cursor');
        break;
      case 'escape': if (window.__setTool) window.__setTool('cursor'); break;
    }
  });
}
