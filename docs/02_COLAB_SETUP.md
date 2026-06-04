# Google Colab – Setup

> **Dokumentation Nr. 02** · [Lesereihenfolge](README.md)

## Kurz

| Was | Lokal | Colab (Browser) | Colab-Erweiterung (Cursor/VS Code) |
|-----|--------|-----------------|-------------------------------------|
| Code | Repo auf dem Mac | `git pull` → `/content/DataMining_Final-Project` | **synchronisiertes Repo** (dein Ordner) |
| `train.csv`, `test.csv` | `data/` | Drive `MyDrive/.../data/` | Drive (empfohlen) oder `data/` im Repo |
| Outputs | `outputs/` | `MyDrive/.../outputs/` | Drive wenn Daten auf Drive, sonst `outputs/` im Repo |

Setup in **03** / **03b** / **04** / **04b**: Zelle 1 → `env = setup()` (siehe `scripts/project_env.py`).

---

## Colab-Erweiterung in Cursor / VS Code

Notebook **lokal** öffnen, Kernel **„Colab“** wählen (Remote-Runtime).

### Was sich ändert (ab Mai 2026 im Repo)

- **Kein** manueller `git clone` mehr in den Notebooks.
- `setup()` erkennt Colab und nutzt zuerst dein **synchronisiertes Workspace-Repo** (`scripts/features.py` unter `cwd`).
- Nur wenn kein Repo gefunden wird: Fallback wie früher (`/content/DataMining_Final-Project` + `git pull`).
- **Drive** wird weiter gemountet (`drive.mount`) — CSVs bleiben auf `MyDrive/.../data/` wie bisher.

### Ablauf

1. Colab-Erweiterung installieren, mit Google anmelden.
2. Projektordner in Cursor öffnen (Root mit `scripts/`, `notebooks/`).
3. `.ipynb` öffnen → Kernel: **Colab** → CPU (oder High-RAM für 04).
4. Zelle 1 ausführen; Ausgabe prüfen:
   - `Colab Extension / Workspace: Code aus …` → dein Sync-Ordner
   - `git pull OK → /content/…` → Browser-Fallback (oder `DM_FORCE_GIT_COLAB=1`)
5. `Train:` / `Test:` **OK** — sonst Drive-Pfad prüfen (unten).

### Wichtig

| Thema | Empfehlung |
|--------|------------|
| Code-Änderungen | Lokal speichern → Kernel nutzt **diesen** Stand (nicht GitHub, außer Fallback). |
| Große CSVs | Weiter auf **Drive** legen, nicht ins Repo committen. |
| Outputs | Bei Drive-Daten → `MyDrive/.../outputs/`; bei `data/` im Repo → `outputs/` lokal im Projekt. |
| `git push` | Nur nötig für Teammates / Browser-Colab; Extension braucht es nicht für deinen Code. |

### Optionale Umgebungsvariablen (vor `setup()`)

```python
import os
# os.environ["DM_PROJECT_ROOT"] = "/content/drive/MyDrive/.../DataMining_Final-Project"
# os.environ["DM_FORCE_GIT_COLAB"] = "1"   # erzwingt git pull nach /content/...
# os.environ["DM_MODE"] = "sample"         # oder full
```

---

## Colab im Browser (klassisch)

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
| RAM voll in 04/04b | In 04/04b: Parquet → `/content/`, wöchentlich aggregieren; optional `DM_WORKERS=1` |
| Drive `Errno 107` | `read_parquet_notebook()` in Zelle 1 (automatisch) |
| 04: Parquet fehlt | Zuerst 03 komplett durchlaufen |
| Alter Code | Zelle 1 nochmal → `git pull` |
