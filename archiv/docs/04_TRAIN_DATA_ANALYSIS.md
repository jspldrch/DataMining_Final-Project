# Train-Set Analyse (`train.csv`)

> **Dokumentation Nr. 04** · [Lesereihenfolge](README.md)

*Referenzdokument – gesammelt aus `01_exploration.ipynb` (Sample) + chunkweiser Vollanalyse*  
*Stand: Mai 2026*

Verwandte Dokumente: [05_TEST_DATA_ANALYSIS.md](05_TEST_DATA_ANALYSIS.md) · [06_EDA_ANALYSIS.md](06_EDA_ANALYSIS.md) (Gesamtfazit & Modellierung)

---

## Kurzfassung

| | Wert |
|---|------|
| **Zeilen gesamt** | 12.319.040 |
| **Regionen** | 2.248 |
| **Zeilen pro Region** | 5.480 (~15 Jahre täglich) |
| **Mit `score` (Label)** | 1.757.936 (**14,27 %**) |
| **Ohne `score`** | 10.561.104 (85,73 %) |
| **Jahre (numerisch)** | 3.004 – 58.061 |
| **Wetter-Features fehlend** | 0 % |

**Problemtyp:** Semi-supervised Panel – tägliches Wetter für alle Zeilen, `score` nur ~jede 7. Tag und nur in 14,27 % der Zeilen.

---

## 1. Struktur & Spalten

```
region_id, date,
prec, surf_pre, humidity, tmp, dp_tmp, wb_tmp,
tmp_max, tmp_min, tmp_range, surf_tmp,
wind, wind_max, wind_min, wind_range,
score
```

| Prüfung | Ergebnis |
|---------|----------|
| Duplikate `(region_id, date)` | 0 (Vollbild erwartet) |
| Fehlende Werte Wetter | 0 |
| Fehlende Werte `score` | 85,73 % (strukturell) |
| Regionen | 2.248, je **5.480** Zeilen |
| Label pro Region | **782 / 5.480** = 14,27 % (**überall gleich**) |

---

## 2. Datumsangaben

### Synthetischer Kalender

- Jahre laufen von **3.004 bis 58.061** – kein realer Gregorianischer Kalender.
- **`pd.to_datetime()` funktioniert nicht** (Grenze ~2262) → Split: `year`, `month`, `day` als Integer.

### String-Sort-Falle

| Sortierung | Min | Max |
|------------|-----|-----|
| **`date` als String** | `10004-12-31` | `8133-12-31` |
| **Chronologisch (Ordinal)** | ca. ab Jahr 3.004 | ca. bis Jahr 58.061 |

→ Für Min/Max und Filter **nie** `df["date"].min()` allein verwenden.

### Pro Region (Beispiel R1)

| | Wert |
|---|------|
| Train-Zeitraum | 3004-12-31 – 3019-12-31 |
| Zeilen | 5.480 |
| Kleine Kalenderlücken | max. ~3 Tage (Sample-Check) |

---

## 3. Zielvariable `score`

### Verteilung (gelabelte Zeilen, Vollbild)

| Kennzahl | Wert |
|----------|------|
| Anzahl | 1.757.936 |
| Mittelwert | **0,91** |
| Median | **0** |
| Min / Max | 0 / 5 |
| Anteil `score == 0` | **58,0 %** |

**Häufigkeit (Vollbild):**

| Score | Anzahl (ca.) |
|-------|----------------|
| 0 | 331.093 |
| 1 | 93.443 |
| 2 | 63.058 |
| 3 | 43.561 |
| 4 | 27.716 |
| 5 | 12.076 |

→ **Zero-inflated:** die meisten gelabelten Tage sind 0, hohe Scores (3–5) sind selten.

### Label-Rhythmus (wöchentlich)

Aus Notebook-Sample (1.427 gelabelte Zeilen):

| Statistik | Tage zwischen Score-Meldungen |
|-----------|-------------------------------|
| Median | **7** |
| Modus | **7** (≈ 90 % der Intervalle) |
| Mittel | 7,13 |

**Bedeutung:** `score` wird etwa **einmal pro Woche** erhoben. Die übrigen Tage haben kein Label by design (nicht MCAR).

**Modellierung:** Nur Zeilen mit `score` für Training; Time-Series-CV mit Gap ≥ 7 Tagen; Walk-Forward **pro `region_id`**.

### Regionale Unterschiede

| Aspekt | Vollbild | Sample (nur R1–R3) |
|--------|----------|---------------------|
| Label-Rate | überall 14,27 % | ~14,27 % |
| Ø `score` pro Region | (in Colab nachrechnen) | R1: 1,10; R2: 0,78; R3: 0,24 |

Sample ist **nicht repräsentativ** für regionale Score-Höhen – nur für globale Struktur. Vollständige regionale Auswertung: [07_LOCAL_EDA_ANALYSIS.md](07_LOCAL_EDA_ANALYSIS.md).

---

## 4. Wettervariablen (Sample 10k – indikativ)

Lokales Sample (erste 10k Zeilen, Jahre 3004–3019, 3 Regionen):

| Feature | Min | Max | Median (ca.) |
|---------|-----|-----|--------------|
| tmp | −6,4 °C | 33,0 °C | 19,7 |
| tmp_max | −2,1 | 41,6 | 25,2 |
| tmp_range | 1,4 | 21,9 | 10,8 |
| prec | 0 | **137,6** | 0,19 |
| surf_pre | 98,2 | 103,6 | 100,8 |
| wind | 0,5 | 12,0 | 2,2 |

- **Niederschlag:** stark rechtsschief (meiste Tage trocken, seltene Extreme).
- **Luftdruck:** eng um ~101 hPa im Sample.
- **99,9 %-Quantil `prec`:** ~76 (Sample).

---

## 5. Korrelation mit `score` (Sample, nur gelabelte Zeilen)

| Feature | r mit `score` | Richtung |
|---------|---------------|----------|
| **tmp_range** | **+0,171** | Große Tagesamplitude |
| **tmp_max** | +0,149 | Hitze |
| **surf_pre** | −0,111 | Tiefdruck / Sturm |
| surf_tmp, tmp | ~+0,11 | Wärme |
| month | +0,104 | leichte Saisonalität |
| **prec** (gleicher Tag) | **+0,015** | fast keine lineare Koppelung |
| **year** | ≈ 0 | kein Trend |

---

## 6. Lags & Rolling (Sample)

### Beste Lag-Korrelationen mit `score`

| Feature | Beste Lag | r |
|---------|-----------|---|
| tmp_range | 21 Tage | **+0,240** |
| tmp_range | 3 Tage | +0,236 |
| tmp_range | 14 Tage | +0,227 |
| prec | 3 Tage | −0,103 |
| surf_pre | 7 Tage | −0,100 |

Same-day `tmp_range` (+0,17) → Lags bis **+0,24** – Lags sind Pflicht-Features.

### Rolling 7 Tage

| Feature | r mit `score` |
|---------|---------------|
| prec_roll7_mean | **−0,150** |
| prec_roll7_std | −0,145 |
| wind_roll7_mean | −0,117 |
| tmp_roll7_mean | +0,087 |

---

## 7. Extreme Wetter vs. `score` (Sample, 95 %-Schwelle)

| Ereignis | Ø Score extrem | Ø Score normal |
|----------|----------------|----------------|
| tmp_max ≥ 35,1 °C | **2,28** | 0,85 |
| tmp_range ≥ 16,4 °C | **1,67** | 0,88 |
| prec ≥ 20,3 mm | 0,94 | 0,91 |
| wind_max ≥ 5,7 | 0,86 | 0,91 |

Hitze und Temperaturspreizung trennen die Score-Verteilung am stärksten.

---

## 8. Saisonalität (Sample)

- Score-Meldungen **gleichmäßig über Monate** (~116–128 pro Monat).
- Kein relevanter **Jahrestrend** (`year` ≈ 0).

---

## 9. Sample vs. Vollbild

| | `train_sample.csv` (10k) | `train.csv` (voll) |
|---|--------------------------|---------------------|
| Zweck | Lokales Notebook, schnell | Colab, finales Modell |
| Regionen | 3 | 2.248 |
| Jahre | 3004–3019 | 3004–58061 |
| Score-Rate | 14,27 % | 14,27 % ✓ |
| Korrelationen / Plots | indikativ | in Colab validieren |

**Regel:** Strukturelle Erkenntnisse (14,27 %, 7-Tage-Rhythmus, Feature-Ranking) gelten fürs Vollbild. Regionale Score-Mittel und exakte Korrelationen auf vollem Train in Colab prüfen.

---

## 10. Wichtigste Regeln fürs Team

1. Daten nur mit `year` / `month` / `day` oder Ordinal sortieren – **nicht** `date` als String.
2. Training auf Zeilen mit `score.notna()` (~1,76 Mio.).
3. **7-Tage-Rhythmus** in CV und Feature-Design berücksichtigen.
4. Features priorisieren: `tmp_range`, `tmp_max`, Lags, `prec`/`wind` Rolling-7d, `region_id`.
5. `year` nicht als Trend-Feature; `prec` am selben Tag allein reicht nicht.

---

## Quellen

| Quelle | Inhalt |
|--------|--------|
| `01_exploration.ipynb` | EDA, Plots, Sample-Korrelationen |
| `02_eda_analysis_local.ipynb` | Chunked Regional-Aggregation lokal → `outputs/regional/region_summary.csv` |
| Chunk-Scan `train.csv` | Vollbild-Zeilen, Score-Verteilung, Regionen |
| `08_PROGRESS_LOG.md` Step 3–6 | Chronologie der Findings |

*Bei Aktualisierung nach Colab-Lauf: Abschnitt 9 und Korrelationstabellen mit Vollbild-Werten ergänzen.*
