# Model History — Drought Severity Prediction

Chronological log of **every** model we built, so the full development path (including the
dead ends / "Irrwege") is traceable. The competition metric is **MAE** on a 0–5 score scale
(lower is better); we report Kaggle public-leaderboard MAE where known, and local validation
MAE where it explains the val/Kaggle gap.

- Final model: [`../scripts/kaggle_v31_stratified.py`](../scripts/kaggle_v31_stratified.py) — **Kaggle MAE 0.7962** (beats Baseline 3 = 0.8056).
- Experimental scripts live in [`scripts/`](scripts/) (Kaggle `kaggle_v*` + local `run_v*`),
  [`arthur_scripts/`](arthur_scripts/) (parallel track), and [`notebooks/`](notebooks/).

> **Note on a typo in early logs:** an old `results.md` lists v19 as MAE `0.1930`; the correct
> Kaggle score is `0.8193`. The corrected values are used throughout this document.

---

## TL;DR — The winning path

```
notebooks (run_v3/v4) → run_v7  (feature plateau, ~0.8303)
   ✗ dead ends: v6 (L1 only), v8–v11 (over-engineering), score-lag v15–v18, DL, spatial
run_v12 / kaggle_v1   (weather-only, 0.8258)
   → v19   last-window-all-regions      (0.8193)
   → v22   recency filter (8y)          (0.8132 ; local 8y holdout 0.8095)
   → v24   regional seasonal mean       (0.8106)
   → v30   +surf_pre/dp_tmp rolls       (0.8047)
   → v31   SPI + stratified holdout + multi-seed  (0.7962)  ← FINAL
   ✗ v32   +trajectory/sample-weights   (0.8090, regression)
```

Three insights that produced the final model:
1. **No target lags** — weather-only prediction survives the 91-day test gap.
2. **`RECENT_YEARS = 8`** — balances data volume vs. climate drift.
3. **Stratified 20% region holdout** — a better Kaggle proxy than last-window validation
   (0.7962 vs 0.8095), despite a *higher* local val MAE.

---

## Phase 0 — Exploration & first pipelines

| Version | File(s) | Change | Outcome | MAE |
|---------|---------|--------|---------|-----|
| notebook baseline | `notebooks/04_modeling.ipynb` | First GBDT pipeline | Starting point | val ~0.8727 |
| v3 | `scripts/run_v3.py` | Longer rolls (30/60/90d), prec deficit/trend, score lags, LGB+XGB | Improvement | — |
| v4 | `scripts/run_v4.py` | Extended score lags, dry-day counts, anomalies, heat-drought idx, 3-way blend | Step up | Kaggle ~0.8587 |

## Phase 1 — Local GBDT refinement (v5 → v12)

Stack: sliding windows, LGB + XGB + CatBoost, 20% region holdout.

| Version | File | Change | Outcome | MAE |
|---------|------|--------|---------|-----|
| v5 | `scripts/run_v5_dl_fixed.py` | CNN-LSTM + LGB | **Dead end** (DL, no gain) | — |
| v6 | `scripts/run_v6.py` | L1/MAE objective, Optuna, regional z-scores, no score features | **Dead end** (broke working setup) | Kaggle 1.1260 |
| **v7** | `scripts/run_v7.py` | +humidity lags/rolls, 180d window, week sin/cos, regional mean | **Feature plateau (best for a long time)** | ~0.8303 |
| v8 | `scripts/run_v8.py` | Air dryness, compound stress, lag28, stronger reg | **Dead end** (over-regularized) | 0.8423 |
| v9 | `scripts/run_v9.py` | Regional monthly mean, lower LR, more trees | **Dead end** | worse than v7 |
| v10 | `scripts/run_v10.py` | Seed ensemble only | **Dead end** (models too correlated) | no gain |
| v11 | `scripts/run_v11.py` | Temporal validation (last 5 weeks) | **Dead end** (val better, Kaggle worse) | much worse |
| **v12** | `scripts/run_v12.py` | v7 features + ExtraTrees as 4th member | **Best holdout-family model** | Kaggle 0.8258 |

**Lesson:** the v7 feature set was near-optimal; v8–v11 tweaks consistently hurt.

## Phase 2 — Arthur parallel track (arthur_v8 → v15)

| Version | File | Change | Outcome | MAE |
|---------|------|--------|---------|-----|
| arthur_v8 | `arthur_scripts/arthur_v8.*` | +surf_pre rolls; two-stage hurdle | Mixed (hurdle ≈ regression) | base 0.8303 |
| arthur_v11 | `arthur_scripts/arthur_v11_MAE_Optimized.ipynb` | All models on L1/MAE loss | Hypothesis: match metric | no confirmed gain |
| arthur_v12 / v15 | `arthur_scripts/arthur_v12_SPATIAL.ipynb`, `arthur_v15_ULTIMATE.ipynb` | **Spatial:** neighbor-region rolling means | **Dead end** (assumed geo-ordered region_id) | worse than v7 |
| arthur_v13 | `arthur_scripts/arthur_v13_GOLDEN.ipynb` | Revert to exact v7 features; LR=0.01, N=4000 | "No more features" pivot | target <0.80 |
| arthur_v14 | `arthur_scripts/arthur_v14_HYBRID.ipynb` | **Transformer + MLP cross-attention** blended w/ LGB | **Dead end** (no LB gain) | — |

## Phase 3 — Score-lag / autoregressive wrong turn (v15 → v18)

High validation, catastrophic Kaggle — the **91-day staleness gap** (test lags ~13 weeks old).

| Version | File | Change | Outcome | MAE (val → Kaggle) |
|---------|------|--------|---------|--------------------|
| feature_scout | `scripts/feature_scout.py` | LGB importance probe | score_lag ranks #1 (autocorr 0.966) | — |
| v15 | `scripts/run_v15.py` | score_lag1–3, per-week models | **Major dead end** | 0.1058 → 1.0470 |
| v16 | `scripts/run_v16_chain.py` | Autoregressive chain (w1→w2→…) | **Dead end** (error propagation) | 0.5213 → 0.9863 |
| v17 | `scripts/run_v17.py` | score_lag1–7, score rolls/trend | **Worst dead end** | 0.0681 → 1.0815 |
| v18 | `scripts/run_v18_gap_val.py`, `kaggle_v19_timeval.py` | Gap=13 simulation (lag from 13w ago) | Partial fix, still harmful | 0.1831 → 0.9134 |
| kaggle_v3 | `scripts/kaggle_v3_scorlag.py` | Correct X_test alignment | Documents *why* v15/v17 failed | 1.04–1.08 |

**Lesson:** target lags are unusable at test time → pivot to weather-only.

## Phase 4 — Deep learning attempts

| Version | File | Architecture | Outcome |
|---------|------|--------------|---------|
| v5 | `scripts/run_v5_dl_fixed.py` | CNN-LSTM + LGB | no competitive score |
| v13 | `scripts/run_v13.py` | WeatherTransformer + TabularMLP + AttentionGate + LGB | no win |
| v14 | `scripts/run_v14_hybrid_dl.py`, `run_v14.py` | Transformer + MLP fusion, OneCycleLR | dead end |
| transformer | `scripts/transformer_v1.py`, `run_transformer.py` | Pure Transformer encoder | **value: feature discovery** (surf_pre #1, dp_tmp) |

**Lesson:** DL never beat GBDT, but the transformer's feature importance guided the *tree*
features added in v30 (surf_pre, dp_tmp rolling stats).

## Phase 5 — Weather-only pivot & validation revolution (v19 → v26)

| Version | File | Change | Outcome | MAE |
|---------|------|--------|---------|-----|
| kaggle_v1 | `scripts/kaggle_v1_base.py` | v12 approach on Kaggle NPZ, no score_lag | baseline ablation | ~0.82 |
| **v19** | `scripts/kaggle_v19_timeval.py`, `run_v19_timeval.py` | weather-only, last-window val on **all** 2248 regions | **Breakthrough** | 0.2522 → 0.8193 |
| v20 | `scripts/kaggle_v20_schema_a.py` | last K=5 windows/region as val | **Dead end** (early stopping) | 0.2814 → 0.8728 |
| v21a | `scripts/kaggle_v21a_longer_rolls.py` | +270d/365d rolls | **Dead end** (test only 91d) | 0.8671 |
| v21b | `scripts/kaggle_v21b_more_trees.py` | N=2000, LR=0.02 | neutral | 0.8192 |
| **v22** | `scripts/kaggle_v22_recent8.py` | **RECENT_YEARS** filter (5y/8y) | **Success** | 0.8132 (local 8y: 0.8095) |
| v24 | `scripts/kaggle_v24_seasonal.py` | + regional_seasonal_mean | **Success** | 0.8106 |
| v25 | `scripts/kaggle_v25_lastyear_val.py` | val = last ~1 year | **Dead end** | 0.8163 |
| v26 | `scripts/kaggle_v26_recent7.py` | RECENT_YEARS=7 | **Dead end** (non-monotonic) | 0.8270 |

**Lesson:** all-regions training + weather-only was the turning point. Recency sweet spot ≈
8 years. Rolling windows longer than the 91-day test horizon hurt.

## Phase 6 — Ensemble & feature push (v27 → v28)

| Version | File | Change | Outcome | MAE |
|---------|------|--------|---------|-----|
| v27 | `scripts/kaggle_v27_multiseed.py` | 5-seed LGB averaging | variance reduction | ~0.8095 |
| v28 | `scripts/kaggle_v28_vpd_weekly.py` | +VPD, regional_week_mean | **Dead end / regression** | 0.8185 |
| v23 | `scripts/kaggle_v23_short_rolls.py` | cap rolls at 90d (fix for v21a) | corrective | — |

## Phase 7 — SPI + stratified holdout final (v29 → v34)

| Version | File | Change | Outcome | MAE |
|---------|------|--------|---------|-----|
| v29 | `scripts/kaggle_v29_fwd_seasonal.py`, `kaggle_v29.py` | forward seasonal `rsm_fw_wk1–5`, test-row idxmax fix | bridge | target 0.8056 |
| v30 | `scripts/kaggle_v30_surf_weighted.py` | +surf_pre/dp_tmp rolls, sample weights | **Success** | 0.8047 |
| **v31** | `../scripts/kaggle_v31_stratified.py`, `scripts/kaggle_v31_final.py` | +SPI, multi-seed LGB, **stratified drought-quartile holdout** (179 feat) | **FINAL — best** | **0.7962** |
| v32 | `scripts/kaggle_v32_multiseed_xgb.py`, `kaggle_v32_fast.py` | +extended SPI, trajectory deltas, multi-seed XGB | **Dead end** (XGB iter limit) | 0.8090 |
| v33 | `scripts/kaggle_v33_xgb2000.py` | revert to v30 feats, multi-seed XGB N=2000 | corrective | ~0.8047 |
| v34 | `scripts/kaggle_v34_spi_xgb2000.py`, `arthur_scripts/arthur_v34_spi_xgb2000.py` | v31 feats + XGB N=2000 + CatBoost | final push attempt | ref 0.7962 |

**Confirmed late dead ends:** sample weights ("dreimal bestätigt schädlich"), trajectory
features, and growing the feature count without giving XGB enough capacity.

---

## Dead-ends summary (the "Irrwege")

| Category | Versions | Why it failed | MAE |
|----------|----------|---------------|-----|
| Score-lag / autoregressive | v15, v16, v17, v18 | 91-day gap: lags fresh in val, stale in test (78–93% of model weight) | 1.05–1.08 |
| Deep learning | v5, v13, v14, transformer, arthur_v14 | Could not beat GBDT; little data per region | no competitive score |
| Spatial neighbors | arthur_v12, v15 | Assumed geo-ordered region_id; covariate shift | worse than v7 |
| L1/MAE objective alone | v6, arthur_v11 | Destabilized training | v6: 1.1260 |
| Feature over-engineering on v7 | v8, v9, v11 | Val improved, Kaggle worsened (overfit) | v8: 0.8423 |
| Wrong validation schemes | v11, v20, v25 | Misaligned early stopping / underfitting | v20: 0.8728 |
| Too-long rolling windows | v21a | Test window only 91 days | 0.8671 |
| Wrong recency | v26 (7y), recent20 (5y holdout) | Non-monotonic recency curve | 7y: 0.8270 |
| Over-engineering after v31 | v32 | XGB capacity limit; harmful weights | 0.8090 |

## Cross-cutting tools

| File | Role |
|------|------|
| `scripts/robust_validation_pipeline.py` | Alternative validation philosophy (adversarial val, group time-series CV) — not on the winning path |
| `scripts/precompute.py`, `scripts/features.py` | Shared feature engineering & caching |
| `scripts/kaggle_v1/v2/v3` | Controlled Kaggle ablations (baseline / +seasonal / +score_lag) |
