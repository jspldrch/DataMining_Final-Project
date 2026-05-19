# Google Colab – Setup

> **Dokumentation Nr. 02** · [Lesereihenfolge](README.md)

## Prinzip: Git für Code, Drive nur für CSVs

| Was | Wo | Wie aktualisieren |
|-----|-----|-------------------|
| Code, Notebooks, `scripts/` | **GitHub** → `git clone` / `git pull` in Colab | `git push` vom Mac |
| `train.csv`, `test.csv` | **Google Drive** (einmal hochladen) | Manuell / Drive-App |
| Parquet, Submissions | **Drive** `outputs/` (Symlink) | Entstehen in Notebook 03/04 |

**Du musst keinen Projektordner mehr auf Drive pflegen** — nur noch den Datenordner.

---

## Einmalig: CSVs auf Drive

```
My Drive/
└── DataMining/
    ├── data/
    │   ├── train.csv   (~1,1 GB)
    │   └── test.csv    (~19 MB)
    └── outputs/        (wird von Colab angelegt)
        ├── processed/
        └── submissions/
```

Pfad in Colab: `/content/drive/MyDrive/DataMining/data/`

---

## Jede Colab-Session (3 Schritte)

### 1. Runtime

- **CPU** (keine GPU nötig)
- Optional: High-RAM

### 2. Bootstrap ausführen

**Option A – Notebook** (empfohlen):

1. GitHub → `notebooks/00_colab_bootstrap.ipynb` → **Open in Colab**
2. Einzige Code-Zelle ausführen

**Option B – Copy-Paste in eine Zelle:**

```python
from google.colab import drive
drive.mount("/content/drive")

REPO = "/content/DataMining_Final-Project"
!test -d "$REPO/.git" && (cd "$REPO" && git pull) || git clone https://github.com/jspldrch/DataMining_Final-Project.git "$REPO"

%cd $REPO
!pip install -q -r requirements.txt

import os
from pathlib import Path
DATA = Path("/content/drive/MyDrive/DataMining/data")
assert (DATA / "train.csv").exists(), f"CSVs fehlen in {DATA}"
print("OK — Code von Git, Daten auf Drive")
```

### 3. Notebooks aus dem Clone öffnen

Nach Bootstrap liegt alles unter:

```
/content/DataMining_Final-Project/
├── notebooks/03_preprocessing.ipynb
└── notebooks/04_modeling.ipynb
```

In Colab: **File → Open notebook** → Tab **Google Drive** ist nicht nötig — Pfad:

```
/content/DataMining_Final-Project/notebooks/03_preprocessing.ipynb
```

Oder im Dateibaum links unter `/content/DataMining_Final-Project/`.

---

## Workflow Mac ↔ Colab

```text
Mac:     ändern → git commit → git push
Colab:   00_colab_bootstrap → git pull → 03 → 04
Drive:   nur CSVs (unverändert)
```

**Kein** manuelles Hochladen von `scripts/`, `config/`, Notebooks nach Drive.

---

## Outputs (Parquet) auf Drive

Bootstrap verlinkt `outputs/processed` → `MyDrive/DataMining/outputs/processed`.

→ Notebook 03 kann unterbrochen werden; Parquet bleibt auf Drive für Notebook 04.

---

## Repo in Colab öffnen (Alternative)

[colab.research.google.com](https://colab.research.google.com) → **GitHub** →  
`jspldrch/DataMining_Final-Project` → Notebook wählen.

Trotzdem **zuerst** Bootstrap-Zelle oder `drive.mount` + prüfen, dass CSVs auf Drive liegen — GitHub enthält **keine** CSVs.

---

## Häufige Probleme

| Problem | Lösung |
|---------|--------|
| `ModuleNotFoundError: scripts` | Bootstrap ausführen; Arbeitsverzeichnis = `/content/DataMining_Final-Project` |
| `train.csv` nicht gefunden | CSVs nach `MyDrive/DataMining/data/` |
| Alter Code in Colab | Bootstrap erneut → `git pull` |
| RAM voll bei 03 | Streaming-Modus (`MODE=full`) — siehe `03_preprocessing` |
| Drive-Ordner `DataMining_Final-Project` | **Nicht mehr nötig** für Code (optional löschen/archivieren) |

---

## Was nicht auf Drive / Git

| Datei | Git | Drive |
|-------|-----|-------|
| `train.csv`, `test.csv` | nein | **ja** |
| `outputs/*.parquet` | nein | ja (via Symlink) |
| `venv/` | nein | nein |
| Code, Docs, Notebooks | **ja** | nein |
