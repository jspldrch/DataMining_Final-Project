# Preprocessing v2 (Erweiterung zu Notebook 03)

> **Dokumentation Nr. 10** · [Lesereihenfolge](README.md)

**Ziel:** Kaggle-MAE Richtung **≤ 0,8** — ohne die bestehende v1-Pipeline zu überschreiben.

## Dateien (v1 vs. v2)

| | **v1 (Original)** | **v2 (Neu)** |
|---|-------------------|--------------|
| Notebook | `03_preprocessing.ipynb` | **`03b_preprocessing_v2.ipynb`** |
| Features | `scripts/features.py` | **`scripts/features_v2.py`** |
| Streaming | `scripts/preprocess_streaming.py` | **`scripts/preprocess_streaming_v2.py`** |
| Train-Parquet | `train_labeled.parquet` | **`train_labeled_v2.parquet`** |
| Test-Parquet | `test_features.parquet` | **`test_features_v2.parquet`** |
| Modeling | `04_modeling.ipynb` | **`04b_modeling_v2.ipynb`** |

v1 bleibt unverändert → direkter Vergleich in Reports (Ablation).

## Was v2 zusätzlich macht

### 1. Score-Historie (Baseline 1 im Modell)
- `score_persist7` — war in v1 schon **gespeichert**, aber **nicht** in `feature_columns()`
- `score_lag14`, `score_lag21`, `score_lag28`, `score_lag35` — Wochen-Rhythmus + 5-Wochen-Horizont

### 2. Region-Target-Encoding (nur Train-Labels)
- `region_score_mean`, `region_score_median`, `region_score_std`
- Ein Pass über `train.csv` (`scripts/region_stats.py`), kein Test-Leakage

### 3. 91-Tage-Test-Fenster (Kaggle-Input)
Pro Region, aus rohem `test.csv`:
- z. B. `test91_mean_tmp_range`, `test91_max_tmp_max`, `test91_min_surf_pre`, …
- Auf **allen** Test-Zeilen gesetzt; Modell nutzt in 04 weiter die **letzte Zeile** pro Region

### 4. Unverändert von v1
- Wetter-Lags/Rolls, Kalender sin/cos, `region_id`
- Streaming pro Region (RAM ~5k Zeilen/Region)
- Train+Test kombiniert für Lags (Test sieht Train-Ende)

## Ablauf

```text
03  → train_labeled.parquet     → 04  → submission (Baseline / v1 MAE)
03b → train_labeled_v2.parquet   → 04b → submission_v2 (Ziel ≤ 0.8 MAE)
```

## Lokal / Colab

Gleiches Setup wie 03 (`scripts/notebook_init.setup()`).  
Outputs liegen weiter unter `outputs/processed/` (lokal) bzw. Drive (Colab).

## Parallelisierung (03b / 04b)

| Stelle | Was |
|--------|-----|
| **03b** | Regionen parallel (`ProcessPoolExecutor`, Standard ≈ CPU−1) |
| **04b** | Sliding-Fenster pro Region parallel; 5 Wochen-Modelle parallel in Validierung |
| **Steuerung** | `export DM_WORKERS=4` oder `DM_WORKERS=1` (aus) |

Colab: Zelle 2 kopiert CSVs nach `/content/` (größerer Speedup als nur Pro).

## Wann 03 neu laufen?

| Situation | Aktion |
|-----------|--------|
| v1-Parquets existieren, nur v2 testen | **Nur 03b** (~20–40 Min. full) |
| v1 behalten + v2 parallel | **03b** zusätzlich |
| Feature in v1 zurückportieren | optional später — bis dahin getrennt halten |

## Report-Formulierung (Vorschlag)

> We extended preprocessing with explicit persistence lags, regional score statistics, and 91-day test-window aggregates (v2), improving public MAE from 0.91 to X.XX while keeping the v1 pipeline for ablation.
