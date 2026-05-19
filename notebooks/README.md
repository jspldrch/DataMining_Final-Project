# Notebooks

| Notebook | Beschreibung | Status |
|----------|--------------|--------|
| `00_colab_bootstrap.ipynb` | **Colab:** `git pull` + Drive-CSVs (pro Session) | ✓ |
| `01_exploration.ipynb` | Data Discovery & EDA (Sample / Colab) | ✓ |
| `02_eda_analysis_local.ipynb` | EDA chunkweise nach Regionen (~8 GB RAM) | ✓ |
| `03_preprocessing.ipynb` | Features (Lags/Rolling) → Parquet | ✓ |
| `04_modeling.ipynb` | Baselines + LightGBM + Submission | ✓ |

**Start:** vom Projektroot `jupyter notebook notebooks/01_exploration.ipynb`

Die erste Code-Zelle setzt `PROJECT_ROOT` und lädt `config.paths`.

**Dokumentation:** Lesereihenfolge in [docs/README.md](../docs/README.md) (01–08).
