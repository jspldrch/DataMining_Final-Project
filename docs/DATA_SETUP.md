# Daten-Setup

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

1. `train.csv` und `test.csv` auf Google Drive hochladen, z. B.  
   `MyDrive/DataMining/data/`
2. In `config/paths.py` ggf. `COLAB_DATA_DIR` anpassen
3. Notebook in Colab öffnen: `notebooks/01_exploration.ipynb`

## Symlink (macOS/Linux, optional)

Wenn die CSVs nur im Legacy-Ordner liegen:

```bash
ln -s data-mining-2026-final-project/data data
```
