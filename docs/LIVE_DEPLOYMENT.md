# Live Deployment

## v1.1 Adaptive Barriers (current)
- TP/SL now derived from Volume Profile levels (VAH/POC/VAL) + structure
- Dynamic R:R ratio: avg 2.22 (range 1.0-4.5)
- SL placement: nearest DOWN level (for longs) / nearest UP level (for shorts)
- TP placement: first level with R:R >= 1.0 within 2.5xATR
- Adaptive filter: rejects signals with R:R < 1.0 or TP prob < 0.25
- Expected live metrics: Sharpe 1.4-1.7, MaxDD 8-12%, WR 25-28%, ~15 trades/day

## Risk parameters
- Position size: 0.51% per trade (target MaxDD 20%, actual expected ~10%)
- All other params unchanged from v1.0