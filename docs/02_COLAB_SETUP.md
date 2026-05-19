# Google Colab – Setup

> **Dokumentation Nr. 02** · [Lesereihenfolge](README.md)

## Kurz

| Was | Wo |
|-----|-----|
| Notebooks öffnen | **GitHub → Open in Colab** (Links in 03 & 04) |
| Python-Code (`scripts/`) | `git pull` in Zelle 1 (automatisch) |
| `train.csv`, `test.csv` | **einmalig** auf Drive |

```
MyDrive/DataMining/DataMining_Final-Project/
├── data/
│   ├── train.csv
│   └── test.csv
└── outputs/          ← entsteht durch Notebook 03/04
    ├── processed/
    └── submissions/
```

---

## Ablauf

1. [03_preprocessing.ipynb](https://colab.research.google.com/github/jspldrch/DataMining_Final-Project/blob/main/notebooks/03_preprocessing.ipynb) → **Run all**
2. [04_modeling.ipynb](https://colab.research.google.com/github/jspldrch/DataMining_Final-Project/blob/main/notebooks/04_modeling.ipynb) → **Run all**

Jedes Notebook macht in **Zelle 1** automatisch:

- `drive.mount()`
- `git clone` / `git pull` → `/content/DataMining_Final-Project`
- `pip install -r requirements.txt`
- Pfade zu CSVs auf Drive

**Kein** separates Bootstrap-Notebook, **kein** manuelles Hochladen von `scripts/`.

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
| 04: Parquet fehlt | Zuerst 03 komplett durchlaufen |
| Alter Code | Zelle 1 nochmal → `git pull` |
