# Google Colab – Setup

> **Dokumentation Nr. 02** · [Lesereihenfolge](README.md)

## Kurz

| Was | Lokal | Colab |
|-----|--------|--------|
| Code | Repo-Clone | `git pull` → `/content/DataMining_Final-Project` |
| `train.csv`, `test.csv` | `data/` im Projekt | **dieselben Dateien** auf Drive |
| Outputs | `outputs/` | `MyDrive/.../outputs/` |

Setup in **03** und **04**: eine Zelle `from scripts.notebook_init import setup` → `env = setup()` (siehe `scripts/project_env.py`).

```
MyDrive/DataMining/DataMining_Final-Project/
├── data/
│   ├── train.csv
│   └── test.csv
└── outputs/          ← entsteht durch Notebook 03/04
    ├── processed/
    └── submissions/
```

Lokal dieselbe Struktur unter dem Repo-Root.

---

## Ablauf

1. [03_preprocessing.ipynb](https://colab.research.google.com/github/jspldrch/DataMining_Final-Project/blob/main/notebooks/03_preprocessing.ipynb) → **Run all**
2. [04_modeling.ipynb](https://colab.research.google.com/github/jspldrch/DataMining_Final-Project/blob/main/notebooks/04_modeling.ipynb) → **Run all**

Zelle 1 macht automatisch: `drive.mount()` · `git clone`/`pull` · `pip install` · Pfade (Drive-Daten, Drive-Outputs).

**Kein** manuelles Hochladen von `scripts/`. Lokal: dieselbe Zelle, Daten aus `data/`.

---

## Nach `git push` am Mac

In Colab: Notebook neu öffnen oder Zelle 1 erneut ausführen → `git pull` holt aktuelle `scripts/`.

---

## Runtime

- **CPU** (keine GPU)
- Optional: High-RAM bei Problemen in 04

---

## Häufige Probleme

| Problem | Lösung |
|---------|--------|
| CSV nicht gefunden | `MyDrive/DataMining/DataMining_Final-Project/data/train.csv` prüfen |
| RAM voll in 03 | Streaming läuft automatisch bei `MODE=full` |
| RAM voll in 04/04b | Zelle 1 kopiert Parquet → `/content/`, aggregiert täglich→wöchentlich; Colab-Default `DM_WORKERS=1` |
| Drive `Errno 107` | `read_parquet_notebook()` in Zelle 1 (automatisch) |
| 04: Parquet fehlt | Zuerst 03 komplett durchlaufen |
| Alter Code | Zelle 1 nochmal → `git pull` |
