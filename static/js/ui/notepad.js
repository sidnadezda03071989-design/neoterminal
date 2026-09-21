// Общий «блокнот»: текст + вложения (файлы и скриншоты через drag&drop,
// base64 dataURL) + автосохранение по blur. Используется заметками (по
// символу) и журналом (одна глобальная запись).

const $ = (id) => document.getElementById(id);

const MAX_ATTACH = 12;
const MAX_CHARS = 12_000_000; // ~9МБ в dataURL — ключ лимита на бэкенде

export function createNotepad(cfg) {
  // cfg: { panelId, textareaId, statusId, dropzoneId, fileInputId,
  //        screenshotsId, saveBtnId, labelId, endpoint }
  // endpoint: (ctx) => URL; ctx — символ (null для журнала).
  const np = {
    _symbol: null,     // текущий символ (null = журнал)
    _attachments: [],
    _dirty: false,
    _bound: false,
  };

  function el(id) { return $(id); }
  function panel() { return el(cfg.panelId); }

  function setStatus(msg, ok) {
    const s = el(cfg.statusId);
    if (!s) return;
    s.textContent = msg;
    s.className = 'notes-status ' + (ok ? 'ok' : 'err');
    setTimeout(() => { if (el(cfg.statusId)) el(cfg.statusId).textContent = ''; }, 2500);
  }

  async function load(ctx) {
    np._symbol = ctx;
    const ta = el(cfg.textareaId);
    try {
      const r = await fetch(cfg.endpoint(ctx));
      if (!r.ok) throw new Error('HTTP ' + r.status);
      const d = await r.json();
      if (ta) ta.value = d.content || '';
      np._attachments = Array.isArray(d.screenshots) ? d.screenshots : [];
      render();
      np._dirty = false;
      if (cfg.labelId) {
        const lbl = el(cfg.labelId);
        if (lbl) lbl.textContent = cfg.title(ctx);
      }
    } catch (e) {
      console.error('notepad load:', e);
      if (ta) ta.value = '';
      np._attachments = [];
      render();
      setStatus('⚠ Не удалось загрузить', false);
    }
  }

  async function save() {
    const ta = el(cfg.textareaId);
    const content = ta ? ta.value : '';
    try {
      const resp = await fetch(cfg.endpoint(np._symbol), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content, screenshots: np._attachments }),
      });
      const r = await resp.json();
      if (!resp.ok) throw new Error(r.error || ('HTTP ' + resp.status));
      np._dirty = false;
      setStatus('💾 Сохранено', true);
    } catch (e) {
      console.error('notepad save:', e);
      setStatus('⚠ ' + e.message, false);
    }
  }

  function render() {
    const cont = el(cfg.screenshotsId);
    if (!cont) return;
    cont.innerHTML = '';
    np._attachments.forEach((a, i) => {
      const isImg = String(a.type || '').startsWith('image/');
      const name = a.name || 'screenshot.png';
      const wrap = document.createElement('div');
      wrap.className = 'notes-thumb';
      if (isImg) {
        wrap.innerHTML = '<img src="' + a.data + '" alt="' + name + '">' +
          '<button type="button" class="notes-thumb-del" data-idx="' + i +
          '" title="Удалить">✕</button>';
      } else {
        wrap.classList.add('notes-file-chip');
        wrap.innerHTML =
          '<a class="notes-file-link" href="' + a.data + '" download="' + name +
          '" title="' + name + '">📄</a>' +
          '<span class="notes-file-name" title="' + name + '">' + name + '</span>' +
          '<button type="button" class="notes-thumb-del" data-idx="' + i +
          '" title="Удалить">✕</button>';
      }
      cont.appendChild(wrap);
    });
  }

  function addFiles(fileList) {
    const files = Array.from(fileList || []);
    if (!files.length) return;
    files.forEach((file) => {
      if (!file) return;
      if (np._attachments.length >= MAX_ATTACH) {
        setStatus('⚠ Максимум ' + MAX_ATTACH + ' вложений', false);
        return;
      }
      const reader = new FileReader();
      reader.onload = () => {
        if (typeof reader.result !== 'string' || reader.result.length > MAX_CHARS) {
          setStatus('⚠ Файл ' + file.name + ' слишком большой', false);
          return;
        }
        np._attachments.push({
          name: file.name,
          type: file.type || 'application/octet-stream',
          data: reader.result,
        });
        np._dirty = true;
        render();
        save();
      };
      reader.readAsDataURL(file);
    });
  }

  function bind() {
    if (np._bound) return;
    np._bound = true;

    const ta = el(cfg.textareaId);
    if (ta) {
      ta.addEventListener('input', () => { np._dirty = true; });
      ta.addEventListener('blur', () => { if (np._dirty) save(); });
    }
    const saveBtn = el(cfg.saveBtnId);
    if (saveBtn) {
      saveBtn.addEventListener('click', () => { np._dirty = true; save(); });
    }
    const dz = el(cfg.dropzoneId);
    const fileInp = el(cfg.fileInputId);
    if (dz) {
      dz.addEventListener('dragover', (e) => { e.preventDefault(); dz.classList.add('over'); });
      dz.addEventListener('dragleave', () => dz.classList.remove('over'));
      dz.addEventListener('drop', (e) => {
        e.preventDefault();
        dz.classList.remove('over');
        addFiles(e.dataTransfer && e.dataTransfer.files);
      });
      dz.addEventListener('click', () => { if (fileInp) fileInp.click(); });
    }
    if (fileInp) {
      fileInp.addEventListener('change', (e) => {
        addFiles(e.target.files);
        fileInp.value = '';
      });
    }
    const cont = el(cfg.screenshotsId);
    if (cont) {
      cont.addEventListener('click', (e) => {
        const del = e.target.closest('.notes-thumb-del');
        if (!del) return;
        const idx = Number(del.dataset.idx);
        if (!Number.isNaN(idx) && idx >= 0 && idx < np._attachments.length) {
          np._attachments.splice(idx, 1);
          np._dirty = true;
          render();
          save();
        }
      });
    }
  }

  return {
    toggle: () => {
      const p = panel();
      if (!p) return;
      if (p.style.display === 'none') np.open();
      else np.close();
    },
    open: (ctx) => {
      const p = panel();
      if (!p) return;
      p.style.display = 'block';
      bind();
      load(ctx);
    },
    close: () => {
      const p = panel();
      if (!p) return;
      if (np._dirty) save();
      p.style.display = 'none';
    },
    onSymbolChanged: (newSym) => {
      const p = panel();
      if (!p || p.style.display === 'none') return;
      if (newSym === np._symbol) return;
      if (np._dirty) save();
      load(newSym);
    },
    bind,
    isOpen: () => { const p = panel(); return !!p && p.style.display !== 'none'; },
  };
}