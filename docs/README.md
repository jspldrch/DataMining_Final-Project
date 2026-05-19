# Dokumentation – Lesereihenfolge

Alle Analyse-Dokumente sind mit **`01_` … `10_`** nummeriert. In dieser Reihenfolge lesen (außer `08`, das ist ein laufendes Team-Tagebuch).

| Nr. | Datei | Wann lesen |
|-----|--------|------------|
| **01** | [01_DATA_SETUP.md](01_DATA_SETUP.md) | Zuerst: Daten laden & Ordnerstruktur |
| **02** | [02_COLAB_SETUP.md](02_COLAB_SETUP.md) | Colab: Git für Code, Drive nur für CSVs |
| **03** | [03_PROJECT_PLAN.md](03_PROJECT_PLAN.md) | Überblick: Meilensteine & nächste Schritte |
| **04** | [04_TRAIN_DATA_ANALYSIS.md](04_TRAIN_DATA_ANALYSIS.md) | Train-Set verstehen (Struktur, Score, Regionen) |
| **05** | [05_TEST_DATA_ANALYSIS.md](05_TEST_DATA_ANALYSIS.md) | Test-Set & Fallstricke (Datum, 91 Tage/Region) |
| **06** | [06_EDA_ANALYSIS.md](06_EDA_ANALYSIS.md) | **Gesamtfazit** EDA + Modell-Empfehlungen |
| **07** | [07_LOCAL_EDA_ANALYSIS.md](07_LOCAL_EDA_ANALYSIS.md) | Regionale Details (Chunked-Vollbild, optional vertiefend) |
| **08** | [08_PROGRESS_LOG.md](08_PROGRESS_LOG.md) | Chronik fürs Team – jederzeit nachschlagen |
| **09** | [09_WEEKLY_MODELING.md](09_WEEKLY_MODELING.md) | Warum 04 täglich→wöchentlich aggregiert (RAM + Kaggle) |
| **10** | [10_PREPROCESSING_V2.md](10_PREPROCESSING_V2.md) | Preprocessing v2 vs. v1 (03b / 04b) |

## Kurz-Pfad je Ziel

- **Neu im Projekt:** 01 → 03 → 04 → 05 → 06  
- **Nur Colab einrichten:** 01 → 02  
- **Preprocessing / Modell:** 06 (Pflicht), 04 + 05 bei Bedarf  
- **Regionale Unterschiede:** 07 nach 06  

## Notebooks (parallel zu den Docs)

| Nr. | Notebook | Passende Docs |
|-----|----------|----------------|
| 01 | `notebooks/01_exploration.ipynb` | 04, 05, 06 |
| 02 | `notebooks/02_eda_analysis_local.ipynb` | 07 |
| 03 | `03_preprocessing.ipynb` | 06 – Features v1 → Parquet |
| **03b** | `03b_preprocessing_v2.ipynb` | **10** – erweiterte Features → `*_v2.parquet` |
| 04 | `04_modeling.ipynb` | 06, 09 – Modell v1 |
| **04b** | `04b_modeling_v2.ipynb` | **10** – Modell auf v2-Parquets |

## Sonstiges

- [../CONTRIBUTING.md](../CONTRIBUTING.md) – Team-Regeln  
- [presentation/](presentation/) – Folien (PDF)
