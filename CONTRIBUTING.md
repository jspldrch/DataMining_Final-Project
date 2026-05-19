# Zusammenarbeit im Team

## Branch-Workflow

1. `main` bleibt stabil (läuft lokal mit Sample + dokumentiert)
2. Feature-Arbeit auf Branches: `feature/02-preprocessing`, `feature/baseline-model`, …
3. Pull Request mit kurzer Beschreibung + Verweis auf `docs/08_PROGRESS_LOG.md`

## Was wohin gehört

| Änderung | Ort |
|----------|-----|
| Exploration / Modell | `notebooks/NN_name.ipynb` |
| Wiederverwendbarer Code | `src/` (wenn wir Module extrahieren) |
| Einmal-Skripte | `scripts/` |
| Pfade & Colab-Konfig | `config/paths.py` |
| Plots aus Notebooks | `outputs/figures/` |
| Planung & Erkenntnisse | `docs/` |
| Rohdaten | `data/` (lokal, **nie** committen) |

## Notebooks

- Immer vom **Projektroot** starten oder erstes Setup-Zelle ausführen (`PROJECT_ROOT` wird gesetzt)
- Pfade nur über `config.paths` oder relative Pfade ab `PROJECT_ROOT`
- Outputs nach `outputs/figures/` speichern (`FIGURES_DIR`)
- Vor Commit: Kernel → Restart & Run All (wenn möglich), oder Outputs clearen bei riesigen Diffs

## Progress dokumentieren

Nach jedem größeren Schritt Eintrag in `docs/08_PROGRESS_LOG.md`:

```markdown
## Step N: Titel (Datum)
**Goal:** …
**Action:** …
**Key Findings:** …
```

## Daten

- Siehe `docs/01_DATA_SETUP.md`
- Jeder legt `data/train.csv` lokal ab (~1,1 GB)
- `python scripts/create_sample.py` für 10k-Zeilen-Sample

## Code-Stil

- Python 3.10+
- Klare Variablennamen, wenig Magie
- Kommentare nur für nicht-offensichtliche Logik (z. B. Datums-Workaround)

## Review-Checkliste

- [ ] Keine CSVs / Secrets im Commit
- [ ] Pfade funktionieren lokal (`data/`) und sind in Colab dokumentiert
- [ ] `PROGRESS_LOG` aktualisiert bei relevanten Findings
