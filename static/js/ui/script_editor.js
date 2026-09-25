/* Monaco-based Python script editor.  No framework and no client-side eval. */
const $ = (id) => document.getElementById(id);
const api = async (url, options = {}) => {
  const response = await fetch(url, { headers: { 'Content-Type': 'application/json' }, ...options });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
};

let scripts = [];
let templates = [];
let currentId = null;
let editor = null;
let fallback = null;
let dirty = false;
let trainTimer = null;
let validationTimer = null;
let trainRunId = null;
let pendingCode = 'result = {"signal": "FLAT", "confidence": 0.0}';

function setCode(value) {
  dirty = true;
  pendingCode = value || '';
  if (editor) editor.setValue(pendingCode);
  else if (fallback) fallback.value = pendingCode;
  scheduleValidation();
}

function getCode() {
  return editor ? editor.getValue() : (fallback ? fallback.value : pendingCode);
}

function setStatus(message, error = false) {
  const node = $('script-status');
  node.textContent = message;
  node.classList.toggle('error', !!error);
}

function log(message, error = false) {
  const node = $('script-console');
  const line = `${new Date().toLocaleTimeString('ru-RU')}  ${message}`;
  node.textContent = `${node.textContent === 'Выберите или создайте скрипт.' ? '' : node.textContent + '\n'}${line}`;
  node.scrollTop = node.scrollHeight;
  if (error) setStatus(message, true);
}

function renderList() {
  const list = $('script-list');
  list.replaceChildren();
  scripts.forEach((item) => {
    const button = document.createElement('button');
    button.className = 'script-item' + (item.id === currentId ? ' active' : '');
    button.type = 'button';
    button.textContent = item.name || 'Untitled';
    const description = document.createElement('small');
    description.textContent = item.description || item.updated_at || '';
    button.appendChild(description);
    button.addEventListener('click', () => loadScript(item.id));
    list.appendChild(button);
  });
  if (!scripts.length) {
    const empty = document.createElement('div');
    empty.className = 'script-status';
    empty.textContent = 'Пока пусто';
    list.appendChild(empty);
  }
}

function renderTemplates() {
  const list = $('template-list');
  list.replaceChildren();
  templates.forEach((item) => {
    const button = document.createElement('button');
    button.className = 'script-item';
    button.type = 'button';
    button.textContent = item.name;
    const description = document.createElement('small');
    description.textContent = item.description || '';
    button.appendChild(description);
    button.addEventListener('click', () => {
      currentId = null;
      $('script-name').value = item.name;
      setCode(item.code);
      $('train-params').value = JSON.stringify(item.default_params || {}, null, 2);
      renderList();
      log(`Загружен шаблон: ${item.name}`);
    });
    list.appendChild(button);
  });
}

async function loadScript(id) {
  try {
    const payload = await api(`/api/scripts/${encodeURIComponent(id)}`);
    const item = payload.script;
    currentId = item.id;
    $('script-name').value = item.name || 'Untitled';
    $('train-params').value = JSON.stringify(item.params || {}, null, 2);
    setCode(item.code || '');
    dirty = false;
    renderList();
    setStatus(`Загружен: ${item.name}`);
  } catch (error) {
    log(error.message, true);
  }
}

async function refresh() {
  try {
    const payload = await api('/api/scripts');
    scripts = payload.scripts || [];
    templates = payload.templates || [];
    renderList();
    renderTemplates();
    if (!currentId && templates.length) {
      $('script-name').value = templates[0].name;
      $('train-params').value = JSON.stringify(templates[0].default_params || {}, null, 2);
      setCode(templates[0].code);
      dirty = false;
      setStatus('Готово. Можно изменить код и запустить.');
    } else if (!getCode()) {
      setCode('result = {"signal": "FLAT", "confidence": 0.0}');
      dirty = false;
      setStatus('Готово.');
    }
  } catch (error) {
    setStatus(error.message, true);
  }
}

async function validateNow() {
  const code = getCode();
  if (!code.trim()) return false;
  try {
    const payload = await api('/api/scripts/validate', { method: 'POST', body: JSON.stringify({ code }) });
    if (payload.valid) {
      setStatus('AST: допустимо');
      return true;
    }
    setStatus(`AST: ${payload.error}`, true);
    return false;
  } catch (error) {
    setStatus(error.message, true);
    return false;
  }
}

function scheduleValidation() {
  clearTimeout(validationTimer);
  validationTimer = setTimeout(validateNow, 450);
}

async function save() {
  if (!await validateNow()) return null;
  const body = {
    name: $('script-name').value.trim() || 'Untitled',
    code: getCode(),
    params: parseParams(),
  };
  try {
    const payload = currentId
      ? await api(`/api/scripts/${encodeURIComponent(currentId)}`, { method: 'PUT', body: JSON.stringify(body) })
      : await api('/api/scripts', { method: 'POST', body: JSON.stringify(body) });
    currentId = payload.script.id;
    dirty = false;
    await refresh();
    await loadScript(currentId);
    log(`Сохранено: ${payload.script.name}`);
    return currentId;
  } catch (error) {
    log(error.message, true);
    return null;
  }
}

function parseParams() {
  try {
    const value = JSON.parse($('train-params').value || '{}');
    return value && typeof value === 'object' && !Array.isArray(value) ? value : {};
  } catch {
    return {};
  }
}

async function run() {
  const id = currentId || await save();
  if (!id) return;
  setStatus('Запуск…');
  try {
    const payload = await api(`/api/scripts/${encodeURIComponent(id)}/run`, {
      method: 'POST',
      body: JSON.stringify({ symbol: $('script-symbol').value, timeframe: $('script-timeframe').value }),
    });
    const result = payload.result || {};
    log(`Run ${result.signal || '—'} · confidence ${result.confidence ?? 0}`);
    log(JSON.stringify(result, null, 2));
    drawPreview(result.preview || []);
    setStatus('Run завершён');
  } catch (error) {
    log(`Run error: ${error.message}`, true);
  }
}

function drawPreview(points) {
  const host = $('script-chart');
  host.replaceChildren();
  if (!window.LightweightCharts || !points.length) {
    host.textContent = points.length ? 'Preview недоступен без графика' : 'Запустите скрипт для preview';
    return;
  }
  const chart = LightweightCharts.createChart(host, {
    width: host.clientWidth || 340,
    height: host.clientHeight || 225,
    layout: { background: { color: '#1e222d' }, textColor: '#d1d4dc' },
    grid: { vertLines: { color: '#2a2e39' }, horzLines: { color: '#2a2e39' } },
    timeScale: { timeVisible: false },
  });
  const series = chart.addSeries(LightweightCharts.LineSeries, { color: '#2962ff', lineWidth: 2 });
  series.setData(points.map((point) => ({ time: Number(point.time), value: Number(point.value) })));
  window.setTimeout(() => chart.applyOptions({ width: host.clientWidth || 340 }), 50);
}

function openTrain() {
  if (!currentId) {
    log('Сначала сохраните скрипт', true);
    return;
  }
  $('train-modal').classList.add('open');
  $('train-status').textContent = 'Прогон будет без look-ahead bias.';
}

function closeTrain() {
  $('train-modal').classList.remove('open');
  if (trainTimer) clearTimeout(trainTimer);
  trainTimer = null;
}

function dateSeconds(value, endOfDay = false) {
  if (!value) return 0;
  const date = new Date(`${value}T${endOfDay ? '23:59:59' : '00:00:00'}Z`);
  return Math.floor(date.getTime() / 1000);
}

async function startTrain() {
  const params = parseParams();
  const start = dateSeconds($('train-start').value);
  const end = dateSeconds($('train-end').value, true);
  if (!start || !end || end <= start) {
    $('train-status').textContent = 'Укажите корректный диапазон дат.';
    return;
  }
  $('train-start-button').disabled = true;
  $('train-status').textContent = 'Запуск…';
  try {
    const payload = await api(`/api/scripts/${encodeURIComponent(currentId)}/train`, {
      method: 'POST',
      body: JSON.stringify({ symbol: $('script-symbol').value, timeframe: $('script-timeframe').value, start_sec: start, end_sec: end, params }),
    });
    trainRunId = payload.run_id;
    log(`Train запущен: ${trainRunId}`);
    pollTrain();
  } catch (error) {
    $('train-status').textContent = error.message;
    $('train-start-button').disabled = false;
  }
}

async function pollTrain() {
  if (!trainRunId) return;
  try {
    const payload = await api(`/api/scripts/train/${encodeURIComponent(trainRunId)}`);
    const progress = payload.progress || {};
    $('train-status').textContent = payload.status === 'done'
      ? 'Готово'
      : `${payload.status}: ${progress.done || 0}/${progress.total || 0}`;
    if (payload.status === 'done') {
      $('train-start-button').disabled = false;
      log('Train завершён');
      log(JSON.stringify(payload.result, null, 2));
      closeTrain();
      return;
    }
    if (payload.status === 'error') {
      $('train-start-button').disabled = false;
      $('train-status').textContent = `Ошибка: ${payload.error || 'training'}`;
      return;
    }
    trainTimer = setTimeout(pollTrain, 900);
  } catch (error) {
    $('train-start-button').disabled = false;
    $('train-status').textContent = error.message;
  }
}

function newScript() {
  currentId = null;
  $('script-name').value = 'Новый скрипт';
  $('train-params').value = '{}';
  setCode('result = {"signal": "FLAT", "confidence": 0.0}');
  renderList();
  setStatus('Новый скрипт');
}

async function removeScript() {
  if (!currentId || !window.confirm('Удалить выбранный скрипт?')) return;
  try {
    await api(`/api/scripts/${encodeURIComponent(currentId)}`, { method: 'DELETE' });
    newScript();
    await refresh();
    log('Скрипт удалён');
  } catch (error) { log(error.message, true); }
}

function initFallback() {
  fallback = $('script-editor-fallback');
  $('script-editor').style.display = 'none';
  fallback.style.display = 'block';
  fallback.addEventListener('input', () => { dirty = true; scheduleValidation(); });
  fallback.value = getCode() || pendingCode;
}

function initMonaco() {
  if (!window.require) { initFallback(); return; }
  try {
    window.require.config({ paths: { vs: 'https://cdn.jsdelivr.net/npm/monaco-editor@0.52.2/min/vs' } });
    window.require(['vs/editor/editor.main'], () => {
      editor = window.monaco.editor.create($('script-editor'), {
        value: getCode() || pendingCode, language: 'python', theme: 'vs-dark', automaticLayout: true,
        minimap: { enabled: false }, fontSize: 13, tabSize: 4, wordWrap: 'on',
        padding: { top: 10 }, scrollBeyondLastLine: false,
      });
      editor.onDidChangeModelContent(() => { dirty = true; scheduleValidation(); });
      $('script-editor-fallback').style.display = 'none';
      $('script-editor').style.display = 'block';
    }, initFallback);
  } catch { initFallback(); }
}

function init() {
  $('script-new').addEventListener('click', newScript);
  $('script-save').addEventListener('click', save);
  $('script-run').addEventListener('click', run);
  $('script-train').addEventListener('click', openTrain);
  $('script-delete').addEventListener('click', removeScript);
  $('train-cancel').addEventListener('click', closeTrain);
  $('train-start-button').addEventListener('click', startTrain);
  const today = new Date();
  const from = new Date(today.getTime() - 180 * 86400000);
  $('train-start').value = from.toISOString().slice(0, 10);
  $('train-end').value = today.toISOString().slice(0, 10);
  initMonaco();
  refresh();
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
else init();
