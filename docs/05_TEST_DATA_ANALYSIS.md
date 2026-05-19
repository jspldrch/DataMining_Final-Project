# Test-Set Analyse (`test.csv`)

> **Dokumentation Nr. 05** · [Lesereihenfolge](README.md)

*Analyse vom Mai 2026 – 204.568 Zeilen, 2.248 Regionen*

---

## Kurzfassung

Das Test-Set wirkt auf den ersten Blick „komisch“, ist aber **strukturell konsistent**. Die Auffälligkeiten kommen vor allem von:

1. **Synthetischen Jahreszahlen** (3.020 – 58.063) – wie im Train, kein echter Kalender  
2. **Falschem Sortieren** von `date` als String (Min/Max irreführend)  
3. **Festem 91-Tage-Fenster pro Region** – jede Region hat ein anderes „Jahr“ und Zeitfenster  
4. **Stärkeren Extremwerten** als im lokalen Train-Sample (Kälte, Tiefdruck)

Es gibt **keine Duplikate, keine Nulls, keine logischen Widersprüche** (`tmp_max` ≥ `tmp_min` überall).

---

## 1. Struktur

| Kennzahl | Wert |
|----------|------|
| Zeilen | 204.568 |
| Regionen | 2.248 (identisch mit Train) |
| Zeilen pro Region | **exakt 91** (jede Region) |
| Spalten | 16 (wie Train **ohne** `score`) |
| Duplikate `(region_id, date)` | 0 |
| Fehlende Werte | 0 |

**Bedeutung:** Test = für jede Region genau **91 aufeinanderfolgende Tage** (mit vereinzelten 1-Tages-Lücken), für die `score` vorhergesagt werden soll.

---

## 2. „Komische“ Datumsangaben

### Synthetische Jahre

- Numerisch: Jahre **3.020 – 58.063** (Mittel ~32.311)
- **94 %** der Zeilen haben `year > 10.000`
- Train deckt **3.004 – 58.061** ab – gleiche künstliche Zeitachse

`pd.to_datetime()` ist deshalb ungeeignet (Notebook: Split in `year`, `month`, `day`).

### String-Sort-Falle (wichtig!)

| Sortierung | Min | Max |
|------------|-----|-----|
| **Als String** | `10021-08-12` | `8135-04-28` |
| **Als Ordinal** (year×372 + month×31 + day) | `3020-09-18` | `58063-10-27` |

→ `df["date"].min()` / `.max()` **nicht** für Analysen verwenden.

### Pro Region: eigenes Fenster

Beispiele:

| Region | Test-Zeitraum | Jahr (konstant pro Region) |
|--------|---------------|----------------------------|
| R1 | 3020-09-18 – 3020-12-17 | 3020 |
| R1001 | 23102-07-17 – 23102-10-15 | 23102 |
| R1591 | u. a. 32072-03-02 – 04 (Kälteextreme) | 32072 |

Jede Region hat **ein anderes synthetisches Jahr** und ein **eigenes 91-Tages-Fenster** (~3 Monate).

### Lücken im Fenster

In Stichproben (z. B. R1): **1–2 Kalendertage fehlen** im 91-Tage-Block (Ordinal-Sprung > 1). Für Lags/Rolling: kleine Lücken tolerieren oder pro Region interpolieren.

---

## 3. Train vs. Test (pro Region)

Beispiel **R1**:

| | Train | Test |
|---|-------|------|
| Zeilen | 5.480 (~15 Jahre) | 91 |
| Zeitraum | 3004-12-31 – **3019-12-31** | **3020-09-18** – 3020-12-17 |
| Chronologie | Test beginnt **nach** Train-Ende | ✓ |

→ Aufgabe: Wetter aus der **Zukunft** relativ zum Train-Ende jeder Region; kein zufälliger Split.

Alle 2.248 `region_id` aus Test kommen auch im Train vor (0 nur-in-test / nur-in-train).

---

## 4. Wetterwerte – Extrem vs. Train-Sample

| Feature | Train-Sample (10k) | Test |
|---------|-------------------|------|
| tmp | −6,4 … 33,0 °C | **−25,9 … 39,7 °C** |
| tmp_min | −10,1 … 27,4 | **−34,8 … 32,3** |
| surf_pre | 98,2 … 103,6 | **67,9 … 103,8** |
| wind_max | bis 15,7 | bis **21,9** |

### Auffällige Teilmengen

| Phänomen | Anzahl | Details |
|----------|--------|---------|
| `surf_pre < 80` | 6.722 (3,3 %) | 94 Regionen; Minimum **67,87** (z. B. R249, Jahr 10085) |
| `tmp_min < −30` | **3** | Alle Region R1591, `32072-03-02` bis `04` |
| `tmp_max < tmp_min` | 0 | – |
| `tmp` außerhalb [min, max] | 0 | – |

Niedrige `surf_pre`-Werte sind **meteorologisch möglich** (starkes Tief), wirken aber im Vergleich zum Train-Sample **selten und extrem**.

---

## 5. Sind das „Fehler“?

| Beobachtung | Einschätzung |
|-------------|--------------|
| Jahre 30xx–58xxx | **Design** des Datensatzes, kein Parse-Fehler |
| String min/max Datum | **Analyse-Artefakt** – Ordinal nutzen |
| 91 Tage/Region | **Test-Design** der Challenge |
| Verschiedene Jahre pro Region | **Normal** in diesem Datensatz |
| surf_pre / Kälte-Extreme | **Echte Ausreißer** im Test – Modell muss robust sein |
| Kleine Datums-Lücken | **Vorsicht** bei Lags (min_periods / Interpolation) |

---

## 6. Empfehlungen fürs Modell

1. **Nie** `date` als String sortieren oder als datetime parsen.  
2. Features: `year`, `month`, `day` oder Ordinal + **region_id**.  
3. Training: nur Zeilen mit `score`; Inference: 91-Tage-Blöcke pro Region aus Test.  
4. **Distribution Shift** einplanen (extremere Wetterlage im Test).  
5. Optional: `surf_pre`, `tmp_min` cappen oder RobustScaler; Extrem-Flags als Features.  
6. Submission: genau **204.568** Zeilen, eine Prediction pro `(region_id, date)` im Test.

---

*Siehe auch: [06_EDA_ANALYSIS.md](06_EDA_ANALYSIS.md), [04_TRAIN_DATA_ANALYSIS.md](04_TRAIN_DATA_ANALYSIS.md), `01_exploration.ipynb` Abschnitt 15 (Test-Vergleich).*
