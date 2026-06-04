# Daten-Setup

> **Dokumentation Nr. 01** · [Lesereihenfolge](README.md)

Die Rohdaten werden **nicht** im Git-Repository versioniert (zu groß).

## Benötigte Dateien

| Datei | Größe (ca.) | Pflicht |
|-------|-------------|---------|
| `data/train.csv` | ~1,1 GB | Ja |
| `data/test.csv` | ~19 MB | Ja |
| `data/train_sample.csv` | ~1 MB | Optional (via Script) |

## Schritte

### 1. Ordner `data/` anlegen

```text
DataMining_Final-Project/
└── data/
    ├── train.csv
    └── test.csv
```

### 2. Dateien vom Kurs / Wettbewerb kopieren

Download-Link oder LMS-Pfad vom Dozenten verwenden.

### 3. Sample für lokalen PC

```bash
python scripts/create_sample.py
```

Erzeugt `data/train_sample.csv` (erste 10.000 Zeilen).

## Legacy-Pfad

Falls du noch den alten Ordner hast:

```text
data-mining-2026-final-project/data/train.csv
```

Das funktioniert weiterhin (`config/paths.py` erkennt beide Orte). Empfohlen ist die Migration nach `data/`:

```bash
mkdir -p data
cp data-mining-2026-final-project/data/*.csv data/
```

## Google Colab

Details: **[02_COLAB_SETUP.md](02_COLAB_SETUP.md)**

1. **Dieselben** `train.csv` und `test.csv` auf Drive legen:  
   `MyDrive/DataMining/DataMining_Final-Project/data/`
2. Notebooks **03** / **04**: Zelle 1 `setup()` — gleiche Pipeline wie lokal (`scripts/project_env.py`)
3. EDA optional: `01_exploration.ipynb`

## Symlink (macOS/Linux, optional)

Wenn die CSVs nur im Legacy-Ordner liegen:

```bash
ln -s data-mining-2026-final-project/data data
```
