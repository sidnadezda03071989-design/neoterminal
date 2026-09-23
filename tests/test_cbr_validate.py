"""Тесты валидации CBR-only (B): scripts/cbr_validate_B.py.

Проверяются чистые хелперы (комиссии, сплиты, статусы, вердикт) без прогона
тяжёлого бэктеста: пакет импортируется, но run_b_only/build_shuffled_db не
вызываются. 8 тестов по ТЗ.
"""

import itertools
import math

import numpy as np
import pytest

from scripts.cbr_validate_b import (
    FEE_PER_SIDE,
    SLIPPAGE,
    annualized_sharpe,
    assemble_verdict,
    fees_status,
    net_pnl_pct,
    oos_status,
    sample_status,
    shuffle_ratio,
    shuffle_status,
    split_into_periods,
    split_train_test,
    wf_status,
)


def _checks(**statuses):
    ids = ("sample", "fees", "walk_forward", "oos", "shuffle")
    out = []
    for cid in ids:
        out.append({"id": cid, "title": cid,
                    "status": statuses.get(cid, "ok"), "detail": ""})
    return out


def test_sharpe_after_fees():
    # известный gross -> правильный net (формула ТЗ) и net Sharpe.
    entry, exit_price, gross = 100.0, 120.0, 2.0
    cost = (FEE_PER_SIDE + SLIPPAGE) * (entry + exit_price) / entry * 100.0
    net = net_pnl_pct(gross, entry, exit_price)
    assert net == pytest.approx(gross - cost, abs=1e-9)

    pnls = [net, net + 1.0, net - 0.5, net + 0.25]
    arr = np.asarray(pnls, dtype="float64")
    sharpe = annualized_sharpe(pnls)
    expected = float(arr.mean()) / float(arr.std()) * math.sqrt(252.0)
    assert sharpe == pytest.approx(expected, abs=1e-9)
    # gross > net: издержки всегда уменьшают результат
    assert annualized_sharpe([gross] * 10) >= annualized_sharpe([net] * 10)


def test_walk_forward_split():
    ts_min, ts_max = 0, 700 * 86400  # 700 дней в секундах
    periods = split_into_periods(ts_min, ts_max, 3)
    assert len(periods) == 3
    width = (ts_max - ts_min) / 3.0
    for i, (s, e) in enumerate(periods):
        assert s == pytest.approx(i * width, abs=1)
        assert e == pytest.approx(min((i + 1) * width, ts_max), abs=1)
    assert periods[0][0] == ts_min
    assert periods[-1][1] == ts_max
    # непрерывность: e[i] == s[i+1]
    for (_, e), (s, _) in itertools.pairwise(periods):
        assert abs(e - s) <= 1


def test_oos_split_70_30():
    ts_min, ts_max = 0, 700 * 86400
    (tr_s, tr_e), (te_s, te_e) = split_train_test(ts_min, ts_max, 0.70)
    assert tr_s == ts_min
    assert te_e == ts_max
    assert tr_e == te_s  # непрерывность
    assert te_s - tr_s == pytest.approx((ts_max - ts_min) * 0.70, abs=1)
    assert te_e - te_s == pytest.approx((ts_max - ts_min) * 0.30, abs=1)


def test_shuffle_test_detects_random():
    # реальный и shuffled Sharpe почти одинаковы -> сигнала нет.
    ratio = shuffle_ratio(5.0, 5.1)
    assert ratio == pytest.approx(5.0 / 5.1, abs=1e-9)
    assert shuffle_status(ratio) == "fail"


def test_shuffle_test_detects_signal():
    # shuffled кратно слабее реального -> сигнал есть.
    ratio = shuffle_ratio(6.6, 1.9)
    assert ratio > 2.0
    assert shuffle_status(ratio) == "ok"
    # слабый сигнал 1.2-2.0 -> WARN
    assert shuffle_status(shuffle_ratio(6.6, 4.0)) == "warn"


def test_verdict_go():
    verdict, reason = assemble_verdict(_checks())
    assert verdict == "GO"
    assert reason
    # sample 500 сделок, net Sharpe 3.2, всё ок
    assert sample_status(500) == "ok"
    assert fees_status(3.2) == "ok"
    assert oos_status(6.0, 5.5) == "ok"
    assert wf_status([2.0, 2.5, 3.0], [10.0, 12.0, 11.0]) == "ok"


def test_verdict_no_go_low_n():
    checks = _checks(sample="fail")
    verdict, reason = assemble_verdict(checks)
    assert verdict == "NO-GO"
    assert "мало сделок" in reason
    assert sample_status(50) == "fail"
    assert sample_status(150) == "warn"


def test_verdict_no_go_fees_kill():
    # gross Sharpe 6.0, net 0.5 — издержки съедают весь edge.
    checks = _checks(fees="fail")
    verdict, reason = assemble_verdict(checks)
    assert verdict == "NO-GO"
    assert "комиссии" in reason
    assert fees_status(0.5) == "fail"
    assert fees_status(2.0) == "warn"