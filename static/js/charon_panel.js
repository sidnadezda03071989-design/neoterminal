// «Запросить сигнал» = та же функция, что и AI Backtest: детерминированный
// вердикт правил 1-20 (pu/pd/sig/fired) приходит в том же ответе /api/ai-backtest
// (поле signal) и показывается в панели AI Backtest. Отдельного эндпоинта и
// отдельной панели на графике нет — одна кнопка, один расчёт, одна вкладка.

import { openAiBacktest, runAiBacktest } from './ui/ai_backtest.js';

async function requestSignal() {
  const btn = document.getElementById('charon-refresh-btn');
  if (btn) btn.disabled = true;
  try {
    openAiBacktest();
    await runAiBacktest();
  } finally {
    if (btn) btn.disabled = false;
  }
}

export function initCharonUI() {
  const btn = document.getElementById('charon-refresh-btn');
  if (btn) btn.addEventListener('click', requestSignal);
}