# Notebooks

Gleiche Pipeline **lokal** und in **Colab** — nur Pfade unterscheiden sich (siehe `scripts/project_env.py`).

| Nr. | Notebook | Setup-Zelle | Daten |
|-----|----------|-------------|--------|
| 01 | `01_exploration.ipynb` | `config.paths.setup_environment()` | `data/train.csv` oder Sample |
| 02 | `02_eda_analysis_local.ipynb` | `config.paths` | Chunked EDA |
| 03 | `03_preprocessing.ipynb` | `setup()` | v1: `train_labeled.parquet` |
| **03b** | `03b_preprocessing_v2.ipynb` | `setup()` | v2: `train_labeled_v2.parquet` ([Diff](../docs/10_PREPROCESSING_V2.md)) |
| 04 | `04_modeling.ipynb` | `setup()` | Submission v1 |
| **04b** | `04b_modeling_v2.ipynb` | `setup()` | Submission `*_v2.csv` (nach 03b) |

## Lokal starten

```bash
cd DataMining_Final-Project
source venv/bin/activate
pip install -r requirements.txt
# data/train.csv + data/test.csv ablegen (wie Colab auf Drive)
jupyter notebook notebooks/03_preprocessing.ipynb
```

`MODE=full` wenn `data/train.csv` existiert (wie Colab). Optional `export DM_MODE=sample` erzwingt Sample.

## Colab

[02_COLAB_SETUP.md](../docs/02_COLAB_SETUP.md) — Zelle 1: `git clone`/`pull` + Drive-CSVs, danach identischer Code wie lokal.
