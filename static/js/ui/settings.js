import { state } from '../state.js';
import { COLORS } from '../config.js';
import { chart, candleSeries } from '../chart/setup.js';
import { updateLegend } from './legend.js';
import { TZ_LIST, setActiveTzId, tzOptionLabel } from './timezone.js';

const $ = (id) => document.getElementById(id);
const SETTINGS_KEY = 'neoterminal_ui_settings_v1';
const PANEL_KEY = 'neoterminal_settings_panel_v1';
const MAX_UPLOAD_BYTES = 1_500_000;

const THEMES = {
  gray: {
    bg: '#131722', chartBg: '#131722', elevated: '#1e222d', hover: '#2a2e39',
    border: '#2a2e39', borderSoft: '#363a45', grid: '#1e222d',
    text: '#d1d4dc', muted: '#787b86', subtle: '#50535e',
  },
  white: {
    bg: '#ffffff', chartBg: '#ffffff', elevated: '#ffffff', hover: '#f1f3f6',
    border: '#d5dae2', borderSoft: '#e7eaf0', grid: '#e6eaf0',
    text: '#1f2937', muted: '#667085', subtle: '#98a2b3',
  },
  dark: {
    bg: '#000000', chartBg: '#000000', elevated: '#050505', hover: '#151515',
    border: '#2b2b2b', borderSoft: '#1c1c1c', grid: '#171717',
    text: '#f5f5f5', muted: '#aaaaaa', subtle: '#777777',
  },
};

const DEFAULTS = {
  theme: 'gray',
  background: '',
  timezone: 'local',
  upColor: '#26a69a',
  downColor: '#ef5350',
  notifications: true,
  crosshairColor: '#aeb6c5',
  crosshairWidth: 1,
  crosshairDashed: true,
};

let settings = _readSettings();
let initialized = false;
let dragState = null;
let panelSaveTimer = null;

function _validColor(value, fallback) {
  return /^#[0-9a-f]{6}$/i.test(String(value || '')) ? value : fallback;
}

function _readSettings() {
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem(SETTINGS_KEY) || '{}') || {}; } catch (e) { /* defaults */ }
  const theme = THEMES[saved.theme] ? saved.theme : DEFAULTS.theme;
  return {
    ...DEFAULTS,
    ...saved,
    theme,
    background: typeof saved.background === 'string' ? saved.background : '',
    notifications: saved.notifications !== false,
    upColor: _validColor(saved.upColor, DEFAULTS.upColor),
    downColor: _validColor(saved.downColor, DEFAULTS.downColor),
    crosshairColor: _validColor(saved.crosshairColor, DEFAULTS.crosshairColor),
    crosshairWidth: Math.min(3, Math.max(1, Number(saved.crosshairWidth) || DEFAULTS.crosshairWidth)),
    crosshairDashed: saved.crosshairDashed !== false,
  };
}

function _saveSettings() {
  try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings)); } catch (e) { /* quota/private mode */ }
}

function _status(message, isError = false) {
  const el = $('settings-status');
  if (!el) return;
  el.textContent = message;
  el.classList.toggle('error', isError);
}

function _palette() {
  return THEMES[settings.theme] || THEMES.gray;
}

function _validTimezone(value) {
  return value === 'local' || value === 'auto'
    || TZ_LIST.some((tz) => tz.id === value) ? value : 'local';
}

function _backgroundValue() {
  const value = String(settings.background || '');
  return /^(https?:\/\/|data:image\/)/i.test(value) ? value : '';
}

function _applyBackground() {
  const value = _backgroundValue();
  const escaped = value.replace(/["\\]/g, '\\$&');
  document.body.style.setProperty(
    '--settings-bg-image', value ? `url("${escaped}")` : 'none',
  );
  document.body.classList.toggle('has-settings-background', !!value);
}

function _applyChartSettings() {
  const palette = _palette();
  const hasBackground = !!_backgroundValue();
  const chartBackground = hasBackground
    ? (settings.theme === 'white' ? 'rgba(255,255,255,0.82)'
      : settings.theme === 'dark' ? 'rgba(0,0,0,0.82)' : 'rgba(19,23,34,0.82)')
    : palette.chartBg;
  const crosshairStyle = settings.crosshairDashed ? 2 : 0;
  const crosshairMode = window.LightweightCharts?.CrosshairMode?.Normal ?? 0;

  Object.assign(COLORS, {
    bg: palette.chartBg,
    grid: palette.grid,
    border: palette.border,
    text: palette.text,
    up: settings.upColor,
    down: settings.downColor,
  });
  document.documentElement.style.setProperty('--up', settings.upColor);
  document.documentElement.style.setProperty('--down', settings.downColor);
  document.body.classList.remove('theme-gray', 'theme-white', 'theme-dark');
  document.body.classList.add(`theme-${settings.theme}`);

  try {
    chart.applyOptions({
      layout: {
        background: { type: 'solid', color: chartBackground },
        textColor: palette.text,
        attributionLogo: false,
        panes: {
          separatorColor: palette.border,
          separatorHoverColor: settings.theme === 'white' ? 'rgba(0,0,0,0.2)' : 'rgba(255,255,255,0.2)',
        },
      },
      grid: {
        vertLines: { color: palette.grid },
        horzLines: { color: palette.grid },
      },
      rightPriceScale: { borderColor: palette.border, autoScale: true },
      timeScale: { borderColor: palette.border },
      crosshair: {
        mode: crosshairMode,
        vertLine: {
          color: settings.crosshairColor,
          width: settings.crosshairWidth,
          style: crosshairStyle,
          labelBackgroundColor: palette.border,
        },
        horzLine: {
          color: settings.crosshairColor,
          width: settings.crosshairWidth,
          style: crosshairStyle,
          labelBackgroundColor: palette.border,
        },
      },
    });
  } catch (e) { /* chart can be unavailable during bootstrap */ }

  try {
    candleSeries.applyOptions({
      upColor: settings.upColor,
      downColor: settings.downColor,
      borderUpColor: settings.upColor,
      borderDownColor: settings.downColor,
      wickUpColor: settings.upColor,
      wickDownColor: settings.downColor,
    });
  } catch (e) { /* noop */ }
  _applyBackground();
  try { updateLegend(null); } catch (e) { /* noop */ }
}

function _populateTimezone(select) {
  if (!select) return;
  select.innerHTML = '';
  select.appendChild(new Option('Локальный (по системе)', 'local'));
  for (const tz of TZ_LIST) select.appendChild(new Option(tzOptionLabel(tz), tz.id));
  select.value = _validTimezone(settings.timezone);
}

function _setTimezone(value) {
  const next = _validTimezone(value);
  settings.timezone = next;
  state.timeZone = next;
  const clock = $('tz-clock-select');
  if (clock) {
    clock.value = next;
    clock.dispatchEvent(new Event('change', { bubbles: true }));
  }
  setActiveTzId(next);
  _saveSettings();
}

function _setBackground(value, message = '') {
  settings.background = String(value || '');
  _applyChartSettings();
  _saveSettings();
  if (message) _status(message);
}

function _bindBackground() {
  const url = $('settings-bg-url');
  const file = $('settings-bg-file');
  const clear = $('settings-bg-clear');
  if (url) {
    url.value = _backgroundValue().startsWith('http') ? _backgroundValue() : '';
    url.addEventListener('change', () => {
      const value = url.value.trim();
      if (value && !/^https?:\/\//i.test(value)) {
        _status('Укажите корректный URL изображения', true);
        return;
      }
      _setBackground(value, value ? 'Фон загружен' : 'Фон очищен');
    });
  }
  if (file) {
    file.addEventListener('change', () => {
      const selected = file.files && file.files[0];
      if (!selected) return;
      if (!selected.type.startsWith('image/')) {
        _status('Нужна картинка', true);
        return;
      }
      if (selected.size > MAX_UPLOAD_BYTES) {
        _status('Картинка должна быть меньше 1.5 МБ', true);
        return;
      }
      const reader = new FileReader();
      reader.onload = () => {
        _setBackground(String(reader.result || ''), 'Фон загружен');
        file.value = '';
      };
      reader.onerror = () => _status('Не удалось прочитать картинку', true);
      reader.readAsDataURL(selected);
    });
  }
  if (clear) clear.addEventListener('click', () => {
    if (url) url.value = '';
    _setBackground('', 'Фон очищен');
  });
}

function _bindSettings() {
  const theme = $('settings-theme');
  if (theme) theme.addEventListener('change', () => {
    settings.theme = THEMES[theme.value] ? theme.value : DEFAULTS.theme;
    _applyChartSettings();
    _saveSettings();
  });

  const timezone = $('settings-timezone');
  _populateTimezone(timezone);
  if (timezone) timezone.addEventListener('change', () => _setTimezone(timezone.value));

  const up = $('settings-up-color');
  const down = $('settings-down-color');
  if (up) up.addEventListener('input', () => {
    settings.upColor = _validColor(up.value, DEFAULTS.upColor);
    _applyChartSettings(); _saveSettings();
  });
  if (down) down.addEventListener('input', () => {
    settings.downColor = _validColor(down.value, DEFAULTS.downColor);
    _applyChartSettings(); _saveSettings();
  });

  const notifications = $('settings-notifications');
  if (notifications) notifications.addEventListener('change', () => {
    settings.notifications = notifications.checked;
    _saveSettings();
    _status(settings.notifications ? 'Системные уведомления включены' : 'Системные уведомления выключены');
  });
  const permission = $('settings-notification-permission');
  if (permission) permission.addEventListener('click', () => {
    if (!window.Notification) {
      _status('Браузер не поддерживает системные уведомления', true);
      return;
    }
    if (Notification.permission === 'granted') {
      _status('Разрешение на уведомления уже выдано');
    } else if (Notification.permission === 'denied') {
      _status('Уведомления заблокированы в браузере', true);
    } else {
      Notification.requestPermission().then((result) => {
        _status(result === 'granted' ? 'Уведомления разрешены' : 'Разрешение не выдано', result !== 'granted');
      }).catch(() => _status('Не удалось запросить разрешение', true));
    }
  });

  const crosshairColor = $('settings-crosshair-color');
  const crosshairWidth = $('settings-crosshair-width');
  const crosshairWidthOut = $('settings-crosshair-width-value');
  const crosshairDashed = $('settings-crosshair-dashed');
  if (crosshairColor) crosshairColor.addEventListener('input', () => {
    settings.crosshairColor = _validColor(crosshairColor.value, DEFAULTS.crosshairColor);
    _applyChartSettings(); _saveSettings();
  });
  if (crosshairWidth) crosshairWidth.addEventListener('input', () => {
    settings.crosshairWidth = Number(crosshairWidth.value) || DEFAULTS.crosshairWidth;
    if (crosshairWidthOut) crosshairWidthOut.textContent = String(settings.crosshairWidth);
    _applyChartSettings(); _saveSettings();
  });
  if (crosshairDashed) crosshairDashed.addEventListener('change', () => {
    settings.crosshairDashed = crosshairDashed.checked;
    _applyChartSettings(); _saveSettings();
  });

  _bindBackground();

  const clock = $('tz-clock-select');
  if (clock) clock.addEventListener('change', () => {
    const value = _validTimezone(clock.value);
    settings.timezone = value;
    state.timeZone = value;
    if (timezone) timezone.value = value;
    _saveSettings();
  });

  const values = {
    'settings-theme': settings.theme,
    'settings-up-color': settings.upColor,
    'settings-down-color': settings.downColor,
    'settings-crosshair-color': settings.crosshairColor,
    'settings-crosshair-width': settings.crosshairWidth,
    'settings-crosshair-width-value': settings.crosshairWidth,
    'settings-notifications': settings.notifications,
    'settings-crosshair-dashed': settings.crosshairDashed,
  };
  for (const [id, value] of Object.entries(values)) {
    const el = $(id);
    if (!el) continue;
    if (el.type === 'checkbox') el.checked = !!value;
    else el.value = value;
  }
  if (crosshairWidthOut) crosshairWidthOut.textContent = String(settings.crosshairWidth);
  if (timezone) timezone.value = _validTimezone(settings.timezone);
  _setTimezone(settings.timezone);
  _applyChartSettings();
}

function _readPanel() {
  try { return JSON.parse(localStorage.getItem(PANEL_KEY) || '{}') || {}; } catch (e) { return {}; }
}

function _savePanel(panel) {
  if (!panel) return;
  const rect = panel.getBoundingClientRect();
  try {
    localStorage.setItem(PANEL_KEY, JSON.stringify({
      left: rect.left, top: rect.top,
      width: parseFloat(panel.style.width) || rect.width,
      height: parseFloat(panel.style.height) || rect.height,
      minimized: panel.classList.contains('is-minimized'),
    }));
  } catch (e) { /* noop */ }
}

function _applyPanelGeometry(panel) {
  const saved = _readPanel();
  const width = Math.min(Math.max(320, Number(saved.width) || 520), window.innerWidth - 16);
  const height = Math.min(Math.max(220, Number(saved.height) || 390), window.innerHeight - 64);
  const maxLeft = Math.max(8, window.innerWidth - width - 8);
  const maxTop = Math.max(56, window.innerHeight - height - 8);
  panel.style.width = `${width}px`;
  panel.style.height = `${height}px`;
  panel.style.left = `${Math.min(Math.max(8, Number(saved.left) || ((window.innerWidth - width) / 2)), maxLeft)}px`;
  panel.style.top = `${Math.min(Math.max(56, Number(saved.top) || ((window.innerHeight - height) / 2)), maxTop)}px`;
  panel.classList.toggle('is-minimized', !!saved.minimized);
  const minimize = $('settings-minimize-btn');
  if (minimize) minimize.textContent = panel.classList.contains('is-minimized') ? '▣' : '—';
}

function _initPanelInteractions(panel) {
  const handle = $('settings-drag-handle');
  const launcher = $('settings-launcher');
  const close = $('settings-close-btn');
  const minimize = $('settings-minimize-btn');
  const toggle = () => {
    if (!panel.classList.contains('open') && panel.classList.contains('is-minimized')) {
      panel.classList.remove('is-minimized');
      if (minimize) minimize.textContent = '—';
    }
    const open = panel.classList.toggle('open');
    panel.style.display = open ? 'flex' : 'none';
    if (open) _status('');
    _savePanel(panel);
  };
  if (launcher) launcher.addEventListener('click', toggle);
  if (close) close.addEventListener('click', () => {
    panel.classList.remove('open'); panel.style.display = 'none'; _savePanel(panel);
  });
  if (minimize) minimize.addEventListener('click', () => {
    panel.classList.toggle('is-minimized');
    minimize.textContent = panel.classList.contains('is-minimized') ? '▣' : '—';
    _savePanel(panel);
  });

  handle?.addEventListener('pointerdown', (event) => {
    if (event.target.closest('button')) return;
    event.preventDefault();
    const rect = panel.getBoundingClientRect();
    dragState = { dx: event.clientX - rect.left, dy: event.clientY - rect.top };
    handle.setPointerCapture?.(event.pointerId);
  });
  handle?.addEventListener('pointermove', (event) => {
    if (!dragState) return;
    const rect = panel.getBoundingClientRect();
    const left = Math.min(Math.max(8, event.clientX - dragState.dx), window.innerWidth - rect.width - 8);
    const top = Math.min(Math.max(56, event.clientY - dragState.dy), window.innerHeight - rect.height - 8);
    panel.style.left = `${left}px`;
    panel.style.top = `${top}px`;
  });
  const endDrag = () => {
    if (!dragState) return;
    dragState = null;
    _savePanel(panel);
  };
  handle?.addEventListener('pointerup', endDrag);
  handle?.addEventListener('pointercancel', endDrag);
  window.addEventListener('blur', endDrag);

  if ('ResizeObserver' in window) {
    const observer = new ResizeObserver(() => {
      clearTimeout(panelSaveTimer);
      panelSaveTimer = setTimeout(() => _savePanel(panel), 150);
    });
    observer.observe(panel);
  }
  window.addEventListener('resize', () => _applyPanelGeometry(panel));
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && panel.classList.contains('open')) {
      panel.classList.remove('open'); panel.style.display = 'none'; _savePanel(panel);
    }
  });
  _applyPanelGeometry(panel);
}

export function notificationsEnabled() {
  return settings.notifications !== false;
}

export function initSettings() {
  if (initialized) return;
  initialized = true;
  _bindSettings();
  const panel = $('settings-panel');
  if (panel) _initPanelInteractions(panel);
}
