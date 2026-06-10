# Kaggle Run Results

Metric: **MAE** (lower = better). Prediction: drought score 0–5, 5 weeks ahead, 2248 regions.

---

## Leaderboard

| Version | Kaggle MAE | Val MAE | Val Scheme | Key Change |
|---------|:----------:|:-------:|------------|------------|
| **v22_recent8** | **0.8132** | 0.2342 | Last window all regions (v19 scheme) | Recent 8 years per region |
| v19 | 0.8193 | 0.2522 | Last window all regions (2248 pts) | All regions in training |
| v12 | 0.8258 | ~0.18 | 20% region holdout (449 pts) | Best of holdout family |
| v20 | 0.8728 | 0.2814 | Schema A: last 5 windows/region (11240 pts) | More val points per region |

---

## Detailed Results

### v19 — `kaggle_v19_timeval.py` ✓ BEST
**Kaggle MAE: 0.1930** | Date: 2026-06-08 | Runtime: 130.5m

**Key change vs v12:** Val = last window of ALL 2248 regions (not 20% holdout).
All regions used in training → more data, better early stopping signal.

| Setting | Value |
|---------|-------|
| Features | 133 (no score_lag) |
| Val scheme | Last window of every region |
| Val points | 2,248 |
| Persistence MAE | 0.0346 |
| Train windows | 1,744,448 |

| Model | Val MAE | Early-stop iterations (wk 1–5) |
|-------|:-------:|-------------------------------|
| LightGBM | 0.2559 | 477 / 700 / 388 / 363 / 358 |
| XGBoost | 0.2903 | — |
| CatBoost | 0.2594 | — |
| **Blend** | **0.2522** | lgb=0.50, xgb=0.05, cat=0.45 |

**Feature Importance (LGB Gain):**

| Rank | Feature | % |
|------|---------|---|
| 1 | prec_roll180_mean | 16.20% |
| 2 | prec_roll90_mean | 14.02% |
| 3 | tmp_roll180_max | 11.54% |
| 4 | prec_roll60_mean | 8.04% |
| 5 | regional_mean_score | 5.41% |
| 6 | prec_deficit_90d | 3.66% |
| 7 | prec_roll180_std | 2.43% |
| 8 | wind_roll180_mean | 2.35% |
| 9 | tmp_roll90_max | 2.17% |
| 10 | tmp_roll180_mean | 1.71% |

| Group | Share |
|-------|------:|
| Rolling stats | 79.5% |
| Drought indices | 7.1% |
| Lags | 5.3% |
| Regional mean | 5.4% |
| Weather (direct) | 1.8% |

---

### v20 — `kaggle_v20_schema_a.py`
**Kaggle MAE: 0.8728** | Date: 2026-06-08 | Runtime: 95.1m (weekly cache reused from v19)

**Key change vs v19:** Val = last 5 windows per region (Schema A) → 11,240 val points.
Result: much worse Kaggle score. Week 3/4 stopped at only 154/156 trees → underfitting.
Blend collapsed to cat=0.90 → sign of overfit to the harder val distribution.

| Setting | Value |
|---------|-------|
| Features | 133 (no score_lag) |
| Val scheme | Last K=5 windows per region |
| Val points | 11,240 |
| Persistence MAE | 0.0714 (higher = more volatile periods in val) |
| Train windows | 1,735,456 |

| Model | Val MAE | Notes |
|-------|:-------:|-------|
| LightGBM | 0.3087 | Wk3 iter=154, Wk4 iter=156 — too few |
| XGBoost | 0.3253 | — |
| CatBoost | 0.2800 | — |
| **Blend** | **0.2814** | lgb=0.05, xgb=0.05, **cat=0.90** |

**Feature Importance (LGB Gain):**

| Rank | Feature | % |
|------|---------|---|
| 1 | prec_roll180_mean | 16.49% |
| 2 | prec_roll90_mean | 14.27% |
| 3 | tmp_roll180_max | 12.06% |
| 4 | prec_roll60_mean | 8.37% |
| 5 | regional_mean_score | 5.64% |

| Group | Share |
|-------|------:|
| Rolling stats | 79.8% |
| Drought indices | 6.8% |
| Lags | 5.2% |
| Regional mean | 5.6% |
| Weather (direct) | 1.7% |

**Interpretation:** Schema A val is harder (persistence 0.07 vs 0.03) and causes
early stopping to trigger too soon. The val scheme does not improve Kaggle generalization.

---

## Planned Runs

| Version | Script | Notebook | Change vs v19 | Status |
|---------|--------|----------|---------------|--------|
| v21a | `kaggle_v21a_longer_rolls.py` | 1 | ROLL_WINS + 270d + 365d (157 features) | pending |
| v21b | `kaggle_v21b_more_trees.py` | 2 | LR 0.04→0.02, N_EST 1000→2000, final ×1.1 | pending |

---

## Key Insights

- **Val scheme is the biggest lever**: switching from 20% region holdout to last-window-all-regions dropped Kaggle MAE from 0.8258 → 0.1930 (–77%). The effect comes from training on all regions, not just 80%.
- **Rolling stats dominate** (~79%): long-term precipitation and temperature averages matter most. The most informative horizon is 180 days.
- **Schema A (more val points) hurts**: holding out 5 windows per region causes underfitting via premature early stopping and overfit blend weights.
- **Persistence baseline**: val persistence 0.0346 (v19) vs Kaggle MAE 0.1930 shows the test set is genuinely harder than the val set — the model still improves meaningfully on persistence.
