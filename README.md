# Data Mining Final Project – Weather Score Forecasting

Vorhersage des Wetter-Impact-Scores (`score`) aus täglichen Meteorologie-Daten über 2.248 Regionen.

## Repository-Struktur

```
├── config/              # Zentrale Pfade (lokal vs. Colab)
│   └── paths.py
├── data/                # Train/Test CSV (nicht in Git – siehe Setup)
├── docs/
│   ├── README.md            # Lesereihenfolge 01–08
│   ├── 01_DATA_SETUP.md … 08_PROGRESS_LOG.md
│   └── presentation/        # Folien (PDF)
├── notebooks/
│   └── 01_exploration.ipynb # EDA (lokal + Colab)
├── outputs/
│   └── figures/             # Generierte Plots aus Notebooks
├── scripts/
│   └── create_sample.py     # 10k-Sample für schwachen PC
├── requirements.txt
└── CONTRIBUTING.md          # Regeln für Teamarbeit
```

## Quick Start

```bash
# 1. Repo klonen & venv
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Daten ablegen (siehe docs/01_DATA_SETUP.md)
#    → data/train.csv, data/test.csv

# 3. Sample für lokales Arbeiten
python scripts/create_sample.py

# 4. Notebook starten (vom Projektroot)
jupyter notebook notebooks/01_exploration.ipynb

# Optional: voller Train chunkweise (~8 GB RAM)
jupyter notebook notebooks/02_eda_analysis_local.ipynb
```

## Lokal vs. Google Colab

| | Lokal | Colab |
|---|--------|--------|
| Code | Git / lokaler Clone | `git clone` / `git pull` → `/content/DataMining_Final-Project` |
| Train/Test CSV | `data/` im Repo-Ordner | **nur** auf Drive: `MyDrive/DataMining/data/` |
| Start | `01_exploration` | **`00_colab_bootstrap`** → `03` → `04` |

Siehe [docs/02_COLAB_SETUP.md](docs/02_COLAB_SETUP.md) — **kein** manuelles Hochladen von Code auf Drive.

## Dokumentation

**Lesereihenfolge:** [docs/README.md](docs/README.md)

| Nr. | Datei | Inhalt |
|-----|--------|--------|
| 01 | [01_DATA_SETUP.md](docs/01_DATA_SETUP.md) | Daten herunterladen & ablegen |
| 02 | [02_COLAB_SETUP.md](docs/02_COLAB_SETUP.md) | Google Colab |
| 03 | [03_PROJECT_PLAN.md](docs/03_PROJECT_PLAN.md) | Meilensteine & Aufgaben |
| 04 | [04_TRAIN_DATA_ANALYSIS.md](docs/04_TRAIN_DATA_ANALYSIS.md) | Train-Set (Referenz) |
| 05 | [05_TEST_DATA_ANALYSIS.md](docs/05_TEST_DATA_ANALYSIS.md) | Test-Set (Referenz) |
| 06 | [06_EDA_ANALYSIS.md](docs/06_EDA_ANALYSIS.md) | Gesamtfazit & Modell-Empfehlungen |
| 07 | [07_LOCAL_EDA_ANALYSIS.md](docs/07_LOCAL_EDA_ANALYSIS.md) | Lokale Chunked-EDA (Vollbild) |
| 08 | [08_PROGRESS_LOG.md](docs/08_PROGRESS_LOG.md) | Fortschritt fürs Team |

## Team

- Änderungen an `docs/08_PROGRESS_LOG.md` bei größeren Schritten
- Keine CSVs committen (`.gitignore`)
- Notebooks nummeriert: `01_`, `02_`, …
