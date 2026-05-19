# Lokale EDA – Auswertung (`02_eda_analysis_local.ipynb`)

> **Dokumentation Nr. 07** · [Lesereihenfolge](README.md)

*Vollständiger Train per Chunking (62 Chunks × 200k, ~2 Min.) auf ~8 GB RAM*  
*Stand: Mai 2026*

**Outputs:** `outputs/regional/region_summary.csv` · Plots in `outputs/figures/local_*.png`

---

## 1. Durchführung & Validierung

| Check | Ergebnis |
|-------|----------|
| Modus | `chunked` über volles `train.csv` |
| Chunks | 62 × 200.000 Zeilen |
| Laufzeit | ~2 Min. |
| Regionen erfasst | **2.248** |
| Zeilen-Summe | **12.319.040** ✓ |
| Zeilen pro Region | **5.480** (überall identisch) |
| Labels pro Region | **782** (überall identisch) |
| Score-Rate | **14,2701 %** (überall identisch) |

→ Chunk-Strategie liefert **dieselben Kennzahlen** wie die frühere Voll-Scan-Analyse in [04_TRAIN_DATA_ANALYSIS.md](04_TRAIN_DATA_ANALYSIS.md). Lokales EDA auf 8 GB RAM ist damit **validiert**.

---

## 2. Zielvariable `score` – neu: Unterschiede zwischen Regionen

Bisher bekannt: global zero-inflated, wöchentlicher Rhythmus, 14,27 % Labels.

**Neu aus Regional-Aggregation** (Ø-Score über 782 gelabelte Tage pro Region):

| Statistik | Ø-Score pro Region |
|-----------|-------------------|
| Mittel | 0,84 |
| Median | 0,74 |
| Min | **0,08** (R1145) |
| Max | **2,26** (R1714) |
| Std | 0,48 |

### Verteilung (Regionen)

| Ø-Score-Bereich | Anzahl Regionen |
|-----------------|-----------------|
| 0,0 – 0,3 | 325 |
| 0,3 – 0,5 | 305 |
| 0,5 – 0,7 | 431 |
| 0,7 – 0,9 | 289 |
| 0,9 – 1,1 | 216 |
| 1,1 – 1,5 | 447 |
| 1,5 – 2,0 | 210 |
| 2,0 – 5,0 | 25 |

**Interpretation:** Die Label-**Dichte** ist überall gleich, aber der **durchschnittliche Impact** unterscheidet sich stark nach Region. `region_id` ist ein zentraler Prädiktor (Target Encoding / Kategorie).

### Extreme Regionen

**Höchster Ø-Score (Auszug):**

| Region | Ø score | Ø tmp | Ø tmp_max | Ø tmp_range | Ø surf_pre |
|--------|---------|-------|-----------|-------------|------------|
| R1714 | 2,26 | 12,5 | 19,3 | 13,2 | 85,8 |
| R1726 | 2,22 | 11,9 | 18,6 | 13,0 | 85,7 |
| R68 | 2,19 | 11,7 | 19,5 | 14,7 | 80,4 |
| R173 | 2,15 | 18,9 | 26,9 | 14,8 | 99,4 |

**Niedrigster Ø-Score (Auszug):**

| Region | Ø score | Ø tmp | Ø tmp_max | Ø tmp_range |
|--------|---------|-------|-----------|-------------|
| R1145 | 0,08 | 3,7 | 8,2 | 9,4 |
| R1156 | 0,09 | 4,1 | 8,5 | 9,0 |
| R2089 | 0,10 | 10,7 | 15,9 | 10,4 |

→ Hohe Scores: oft **große Temperaturamplitude** + teils niedriger Luftdruck.  
→ Niedrige Scores: oft **sehr kühle** Regionen (Ø tmp unter ~5 °C).

---

## 3. Klima über 2.248 Regionen

| Feature (Regional-Mittel) | Min | Max | Ø über Regionen |
|---------------------------|-----|-----|-----------------|
| tmp | −0,32 °C | 25,4 °C | 13,0 °C |
| tmp_max | 6,0 | 30,8 | 18,9 |
| tmp_range | 1,9 | 15,4 | 11,3 |
| prec | 0,25 | 6,0 | 2,6 |
| surf_pre | **68,8** | 101,8 | 95,9 |
| wind | 1,5 | 6,8 | 3,7 |

- **317 Regionen** mit Ø-Luftdruck **< 90** (Tiefdrucklagen im Mittel).
- Nur **1 Region** mit Ø-Temperatur **> 25 °C**.
- Keine Region mit Ø-Niederschlag > 10.

---

## 4. Zeitachse pro Region

| | Wert |
|---|------|
| `year_min` global | 3.004 |
| `year_max` global | 58.061 |
| Spanne pro Region (`year_max − year_min`) | **immer 15 Jahre** |

Jede Region hat ein **eigenes 15-Jahres-Fenster** in unterschiedlichen synthetischen Jahren, z. B.:

| Region | Jahre |
|--------|-------|
| R1 | 3004 – 3019 |
| R1001 | 23086 – 23101 |
| R1002 | 23088 – 23103 |

→ Kein gemeinsamer Kalender über Regionen; Modell braucht **region_id** und darf Jahre nicht als globalen Trend missbrauchen.

---

## 5. Korrelationen

### A) Punkt-Ebene (30k Stichprobe gelabelter Zeilen)

Aus demselben Chunk-Lauf (repräsentativ für Vollbild):

| Feature | r mit `score` |
|---------|---------------|
| **tmp_range** | **+0,170** |
| **tmp_max** | +0,160 |
| surf_tmp | +0,130 |
| tmp | +0,127 |
| **month** | +0,126 |
| tmp_min | +0,095 |
| surf_pre | −0,087 |
| year | +0,055 |
| prec | ≈ 0 |

→ **Bestätigt** die Ergebnisse aus `01_exploration` / [04_TRAIN_DATA_ANALYSIS.md](04_TRAIN_DATA_ANALYSIS.md).  
→ `month` fällt im Vollbild-Sample stärker auf als im 10k-Sample.

### B) Regionen-Ebene (Ø-Werte vs. Ø-Score)

| Feature (Regional-Ø) | r mit Ø-Score |
|----------------------|---------------|
| tmp_range_mean | **+0,66** |
| prec_mean | −0,53 |
| tmp_max_mean | +0,49 |
| surf_pre_mean | −0,42 |
| tmp_mean | +0,35 |

**Vorsicht:** Aggregations-Effekt (Ökologischer Fehlschluss). Regionen mit hoher Amplitude haben im Mittel höhere Scores – das **ersetzt nicht** Lag-Features auf Tagesebene, zeigt aber, dass Klima-Cluster und `region_id` zusammenhängen.

---

## 6. Detail-Zeitreihen (R1714 vs. R1145)

Notebook vergleicht automatisch Region mit höchstem vs. niedrigstem Ø-Score:

- **R1714** (hoch): moderate Temp., deutliche Score-Punkte
- **R1145** (niedrig): sehr kühles Klima, kaum hohe Scores

Plot: `outputs/figures/local_region_compare.png`

---

## 7. Vergleich: 10k-Sample vs. Chunked-Vollbild

| Aspekt | `01_exploration` (10k) | `02_eda_analysis_local` (chunked) |
|--------|------------------------|-------------------------------------|
| Regionen | 3 | **2.248** |
| Score-Rate | 14,27 % | 14,27 % ✓ |
| R1 Ø-Score | ~1,10 | **0,997** |
| tmp_range ↔ score | ~0,17 | **0,170** ✓ |
| Regionale Unterschiede | nicht sichtbar | **stark** (0,08 – 2,26) |

Das 10k-Sample am Dateianfang war **strukturell korrekt**, aber **geografisch nicht repräsentativ**.

---

## 8. Modellierung – aktualisierte Empfehlungen

1. **`region_id` Pflicht** – größter struktureller Faktor neben Wetter.
2. **Features:** `tmp_range`, `tmp_max`, Lags, Rolling `prec`/`wind`, `month` (zyklisch).
3. **Nicht:** globaler `year`-Trend; nicht nur Regional-Mittelwerte als einzige Features.
4. **CV:** Walk-forward pro Region; 7-Tage-Gap wegen Label-Rhythmus.
5. **Kalte Regionen** (niedriger Ø-Score) vs. **hohe Amplitude** (hoher Ø-Score) – Modell muss beide Regime lernen.
6. **317 Regionen** mit niedrigem Ø-Druck – `surf_pre` und ggf. Region-Cluster beachten.

---

## 9. Nächste Schritte

| Priorität | Aufgabe |
|-----------|---------|
| ✓ | Lokales EDA Vollbild – erledigt |
| → | `03_preprocessing` / Features inkl. `region_id` + Lags |
| → | Colab: exakte Korrelation auf allen 1,76 Mio. Labels (optional) |
| → | `region_summary.csv` für Team-Plots / Präsentation nutzen |

---

*Quellen: `outputs/regional/region_summary.csv`, Notebook-Outputs Zellen 7–13, Plots `local_regional_overview.png`, `local_score_corr_sample.png`, `local_region_compare.png`*
