# Data Mining Final Project – Weather Score Forecasting

Vorhersage des Wetter-Impact-Scores (`score`) aus täglichen Meteorologie-Daten über 2.248 Regionen.

## Repository-Struktur

```
├── config/              # Zentrale Pfade (lokal vs. Colab)
│   └── paths.py
├── data/                # Train/Test CSV (nicht in Git – siehe Setup)
├── docs/
│   ├── PROJECT_PLAN.md      # Meilensteine & Aufgaben
│   ├── PROGRESS_LOG.md      # Chronologisches Projekttagebuch
│   ├── EDA_ANALYSIS.md      # Auswertung der Exploration
│   ├── DATA_SETUP.md        # Daten herunterladen & ablegen
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

# 2. Daten ablegen (siehe docs/DATA_SETUP.md)
#    → data/train.csv, data/test.csv

# 3. Sample für lokales Arbeiten
python scripts/create_sample.py

# 4. Notebook starten (vom Projektroot)
jupyter notebook notebooks/01_exploration.ipynb
```

## Lokal vs. Google Colab

| | Lokal | Colab |
|---|--------|--------|
| Train | `data/train_sample.csv` (10k) | `data/train.csv` (voll) |
| Test | `data/test.csv` | `data/test.csv` |
| Pfade | `config/paths.py` | Drive-Pfad in `config/paths.py` anpassen |

Notebooks erkennen die Umgebung automatisch (`setup_environment()`).

## Dokumentation

| Datei | Inhalt |
|-------|--------|
| [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md) | Projektplan |
| [docs/PROGRESS_LOG.md](docs/PROGRESS_LOG.md) | Fortschritt fürs Team |
| [docs/EDA_ANALYSIS.md](docs/EDA_ANALYSIS.md) | EDA-Ergebnisse & Modell-Empfehlungen |
| [docs/DATA_SETUP.md](docs/DATA_SETUP.md) | Daten-Setup |

## Team

- Änderungen an `docs/PROGRESS_LOG.md` bei größeren Schritten
- Keine CSVs committen (`.gitignore`)
- Notebooks nummeriert: `01_`, `02_`, …
