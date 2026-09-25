/* Часовые пояса: фиксированный список «город (UTC±X)»,
   у каждого — стабильный id и смещение в секундах (без учёта DST, как и
   uvOffset в LWC). Лениво резолвим IANA-имя через Intl, чтобы не тащить
   тяжёлую таблицу городов и не спорить с летним временем. */

export const TZ_LIST = [
  { id: 'utc',      label: 'UTC',              offset: 0 },
  { id: 'Europe/London',     label: 'London',     offset: null },
  { id: 'Europe/Berlin',     label: 'Berlin',     offset: null },
  { id: 'Europe/Moscow',     label: 'Moscow',     offset: null },
  { id: 'Europe/Istanbul',   label: 'Istanbul',   offset: null },
  { id: 'Asia/Dubai',        label: 'Dubai',      offset: null },
  { id: 'Asia/Kolkata',      label: 'Kolkata',    offset: null },
  { id: 'Asia/Singapore',    label: 'Singapore',  offset: null },
  { id: 'Asia/Tokyo',        label: 'Tokyo',      offset: null },
  { id: 'Australia/Sydney',  label: 'Sydney',     offset: null },
  { id: 'America/New_York',  label: 'New York',   offset: null },
  { id: 'America/Chicago',   label: 'Chicago',    offset: null },
  { id: 'America/Los_Angeles', label: 'Los Angeles', offset: null },
  { id: 'America/Sao_Paulo', label: 'Sao Paulo',  offset: null },
];

const LS_KEY = 'neoterminal_tz';
let _activeId = 'local';  // 'local' | 'auto' (синоним) | id из TZ_LIST
const _offsetCache = new Map();

/* offsetSeconds по IANA id на текущий момент (DST учитывает Intl). */
export function tzOffsetSeconds(iana) {
  if (iana === 'UTC' || iana === 'utc') return 0;
  if (_offsetCache.has(iana)) return _offsetCache.get(iana);
  let off = 0;
  try {
    const fmt = new Intl.DateTimeFormat('en-US', {
      timeZone: iana, hour12: false,
      year: 'numeric', month: '2-digit', day: '2-digit',
      hour: '2-digit', minute: '2-digit', second: '2-digit',
    });
    const parts = {};
    for (const p of fmt.formatToParts(new Date())) parts[p.type] = p.value;
    const asUtc = Date.UTC(+parts.year, +parts.month - 1, +parts.day,
      +parts.hour % 24, +parts.minute, +parts.second);
    off = Math.round((asUtc - Math.floor(Date.now() / 1000) * 1000) / 1000);
  } catch (e) { off = 0; }
  _offsetCache.set(iana, off);
  return off;
}

/* «UTC+03:00» / «UTC−05:30» / «UTC» для фиксированного смещения. */
export function fmtUtcOffset(seconds) {
  const s = Math.round(seconds || 0);
  if (s === 0) return 'UTC';
  const sign = s < 0 ? '−' : '+';
  const abs = Math.abs(s);
  const h = Math.floor(abs / 3600);
  const m = Math.floor((abs % 3600) / 60);
  return 'UTC' + sign + String(h).padStart(2, '0') + ':' + String(m).padStart(2, '0');
}

/* Полная подпись пункта: «Moscow (UTC+03:00)». */
export function tzOptionLabel(tz) {
  const off = tz.offset != null ? tz.offset : tzOffsetSeconds(tz.id);
  return tz.label + ' (' + fmtUtcOffset(off) + ')';
}

/* Активный пункт (или null для локального пояса браузера). */
export function getActiveTz() {
  if (_activeId === 'local' || _activeId === 'auto') return null;
  return TZ_LIST.find((t) => t.id === _activeId) || null;
}

export function getActiveTzId() { return _activeId; }

function _localTzLabel() {
  let name = 'Local';
  try {
    name = Intl.DateTimeFormat().resolvedOptions().timeZone
      .split('/').pop().replace(/_/g, ' ') || 'Local';
  } catch (e) { /* noop */ }
  const off = -new Date().getTimezoneOffset() * 60;
  return name + ' (' + fmtUtcOffset(off) + ')';
}

/* Смещение в секундах, которое сейчас применяется к графику. */
export function activeOffsetSeconds() {
  const tz = getActiveTz();
  if (tz) return tz.offset != null ? tz.offset : tzOffsetSeconds(tz.id);
  return -new Date().getTimezoneOffset() * 60;
}

/* Установить активный пояс по id; возвращает id. Используется и при
   восстановлении из localStorage (валидируем значение). */
export function setActiveTzId(id) {
  if (id && id !== 'local' && id !== 'auto'
      && TZ_LIST.some((t) => t.id === id)) {
    _activeId = id;
  } else {
    _activeId = 'local';
  }
  try { localStorage.setItem(LS_KEY, _activeId); } catch (e) { /* noop */ }
  return _activeId;
}

export function initTimezone() {
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (raw) setActiveTzId(raw);
  } catch (e) { /* noop */ }
  return _activeId;
}

/* Формат времени для легенды/реплея в АКТИВНОМ поясе. */
export function formatInTz(epochSec, opts) {
  const tz = getActiveTz();
  const timeOnly = !!opts && (
    opts.hour != null || opts.minute != null || opts.second != null
  ) && opts.year == null && opts.month == null && opts.day == null;
  const o = timeOnly
    ? Object.assign({
        hour: '2-digit', minute: '2-digit', second: '2-digit',
        hour12: false,
      }, opts)
    : Object.assign({
        year: 'numeric', month: '2-digit', day: '2-digit',
        hour: '2-digit', minute: '2-digit', second: '2-digit',
        hour12: false,
      }, opts || {});
  const d = new Date((epochSec || 0) * 1000);
  if (tz) {
    try { return new Intl.DateTimeFormat('ru-RU', Object.assign({ timeZone: tz.id }, o)).format(d); }
    catch (e) { /* noop */ }
  }
  try { return new Intl.DateTimeFormat('ru-RU', o).format(d); }
  catch (e) { return d.toLocaleString(); }
}
