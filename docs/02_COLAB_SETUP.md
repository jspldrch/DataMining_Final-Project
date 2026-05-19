# Google Colab – Setup

> **Dokumentation Nr. 02** · [Lesereihenfolge](README.md)

Zwei getrennte Uploads: **Code** (klein) und **Daten** (groß, ~1,1 GB).

## Zielstruktur auf Google Drive

```
My Drive/
└── DataMining/
    ├── data/
    │   ├── train.csv      (~1,1 GB)
    │   └── test.csv       (~19 MB)
    └── DataMining_Final-Project/    ← Repo (ohne venv, ohne CSVs)
        ├── config/
        ├── notebooks/
        ├── requirements.txt
        └── ...
```

Der Pfad `MyDrive/DataMining/data` ist in `config/paths.py` voreingestellt (`COLAB_DATA_DIR`).

---

## Schritt 1: Daten auf Drive hochladen

1. Im Browser [Google Drive](https://drive.google.com) öffnen.
2. Ordner anlegen: `DataMining` → darin `data`.
3. `train.csv` und `test.csv` in `DataMining/data/` hochladen.  
   (1,1 GB dauert – ggf. über Desktop-App oder zip entpacken auf Drive.)

**Nicht nötig in Colab:** `train_sample.csv` (nur für lokalen PC).

---

## Schritt 2: Code auf Drive (oder GitHub)

### Variante A – GitHub (empfohlen fürs Team)

Repo pushen, dann in Colab:

```python
!git clone https://github.com/<DEIN-USER>/DataMining_Final-Project.git
%cd DataMining_Final-Project
```

### Variante B – Ordner manuell hochladen

Projektordner als Zip packen (**ohne** `venv/`, **ohne** `data/*.csv`), nach  
`My Drive/DataMining/DataMining_Final-Project/` entpacken.

Wichtig: Ordner `config/` muss dabei sein (für `from config.paths import …`).

---

## Schritt 3: Notebook in Colab öffnen

1. [colab.research.google.com](https://colab.research.google.com)
2. **Datei → Notebook hochladen**  
   Oder: `DataMining_Final-Project/notebooks/01_exploration.ipynb` in Drive öffnen → **Mit Google Colaboratory öffnen**
3. **Runtime → Change runtime type** → CPU reicht für EDA; **RAM**: „High-RAM“ falls verfügbar (voller `train.csv`).

---

## Schritt 4: Erste Zellen ausführen

Das Notebook macht automatisch:

1. `drive.mount("/content/drive")`
2. `setup_environment()` → nutzt `train.csv` auf Drive, `USE_CHUNKED_TRAIN = True`

**Vor dem Lauf prüfen:** Ausgabe von Zelle „Umgebung“:

```
✓ Train gefunden (1100.x MB)
✓ Test gefunden (19.x MB)
```

### Anderer Drive-Pfad?

In Colab, direkt nach dem Mount, einmalig:

```python
import config.paths as cp
cp.COLAB_DATA_DIR = "/content/drive/MyDrive/DEIN/ORDNER/data"
```

Dann `setup_environment()` erneut ausführen.

---

## Schritt 5: Requirements installieren

Einmal pro Colab-Sitzung (neue Zelle oben oder nach `%cd` ins Projekt):

```python
%cd /content/DataMining_Final-Project   # oder dein Clone-Pfad
!pip install -r requirements.txt
```

---

## Typischer Ablauf (Copy-Paste, neue Colab-Session)

```python
from google.colab import drive
drive.mount("/content/drive")

# Nur bei GitHub-Clone:
# !git clone https://github.com/<USER>/DataMining_Final-Project.git

import os
os.chdir("/content/drive/MyDrive/DataMining/DataMining_Final-Project")

!pip install -q -r requirements.txt
```

Danach alle Zellen in `01_exploration.ipynb` ausführen.

---

## Häufige Probleme

| Problem | Lösung |
|---------|--------|
| `ModuleNotFoundError: config` | `%cd` ins Projektroot, wo `config/` liegt |
| Train nicht gefunden | Pfad in Drive prüfen; `COLAB_DATA_DIR` anpassen |
| Out of Memory | Runtime mit mehr RAM; in Notebook bleibt Chunk-Loading aktiv |
| Langsames Laden | Normal (~5–10 Min für 12M Zeilen beim ersten `load_train`) |

---

## Was du nicht committen / hochladen musst

- `venv/`
- `data/train.csv` (nur auf Drive)
- `.ipynb_checkpoints/`
