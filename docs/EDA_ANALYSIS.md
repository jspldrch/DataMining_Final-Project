# EDA-Analyse – Weather Forecasting (Data Mining Final Project)

*Auswertung von `01_exploration.ipynb` (lokal, 10k-Sample) ergänzt um validierte Kennzahlen des vollen `train.csv`.*

---

## 1. Executive Summary

Der Datensatz ist ein **großskaliges Panel-Zeitreihen-Problem** mit **2.248 Regionen**, täglichen Wetterbeobachtungen und einer **sparse gelabelten Zielvariable `score` (14,27 %)**. Es handelt sich faktisch um ein **semi-supervised Forecasting-/Regression-Problem**: ~10,6 Mio. Zeilen ohne Label, ~1,76 Mio. mit Label.

Die EDA zeigt drei zentrale Erkenntnisse für die Modellierung:

1. **`score` wird wöchentlich gemeldet** (Abstand zwischen Labels ≈ 7 Tage) – kein tägliches Supervised-Learning im klassischen Sinn.
2. **Temperatur-Amplitude (`tmp_range`) und Hitze (`tmp_max`)** sind die stärksten Prädiktoren; **Niederschlag wirkt verzögert** über Lags und Rolling-Features.
3. **Train und Test unterscheiden sich** in Wertebereichen (extremere Wetterlagen im Test) – Generalisierung und robuste Features sind entscheidend.

---

## 2. Datensatz-Überblick

| | Train (voll) | Train (Sample, Notebook) | Test |
|---|-------------|--------------------------|------|
| Zeilen | 12.319.040 | 10.000 | 204.568 |
| Regionen | 2.248 | 3 (R1, R2, R3) | 2.248 |
| Zeilen/Region | 5.480 (~15 Jahre) | ungleich (4.750 / 4.750 / 500) | ~91 pro Region* |
| `score` vorhanden | 14,27 % | 14,27 % | nein |
| Wetter-Features fehlend | 0 % | 0 % | 0 % |
| Datumsbereich | 10004 – 8133 | 3004 – 3019 | 10021 – 8135 |

\*Test deckt einen anderen Zeitraum ab als der Sample-Ausschnitt; vollständige Test-Analyse basiert auf allen 204k Zeilen.

**Spalten:** `region_id`, `date`, 14 Wettervariablen (`prec`, `surf_pre`, `humidity`, `tmp`, `dp_tmp`, `wb_tmp`, `tmp_max`, `tmp_min`, `tmp_range`, `surf_tmp`, `wind`, `wind_max`, `wind_min`, `wind_range`), Ziel `score`.

**Datums-Parsing:** `pd.to_datetime()` scheitert (Jahre bis 58.063). Lösung: Split in `year`, `month`, `day` als Integer – chronologische Sortierung pro Region bleibt erhalten.

---

## 3. Zielvariable `score`

### 3.1 Verteilung

| Kennzahl | Sample (Notebook) | Vollständiger Train |
|----------|-------------------|---------------------|
| Mittelwert | 0,91 | 0,91 |
| Median | 0 | 0 |
| Anteil = 0 | 53,6 % | 58,0 % |
| Bereich | 0 – 5 | 0 – 5 |
| Häufigste Werte | 0 > 1 > 2 > 3 > 4 > 5 | gleiches Muster |

**Interpretation:** Stark **zero-inflated** verteilt. Die meisten gelabelten Tage haben keinen oder geringen „Impact“; höhere Scores (3–5) sind selten. Das spricht für:

- Metriken, die Nullen nicht dominieren lassen (z. B. MAE auf gelabelten Tagen, oder Fokus auf Score > 0), oder
- explizite Behandlung der Null-Inflation (Two-Part-Model, Zero-Inflated Regression, ggf. Klassifikation 0 vs. >0 + Regression).

### 3.2 Label-Rhythmus (kritisch!)

Im Sample: Abstand zwischen aufeinanderfolgenden Score-Meldungen:

| Statistik | Wert |
|-----------|------|
| Mittel | 7,13 Tage |
| Median | **7 Tage** |
| Modus | **7 Tage** (1.287 von 1.424 Intervallen) |

**→ `score` wird in wöchentlichem Turnus erhoben**, nicht täglich. An den ~85 % Zeilen ohne `score` fehlt das Label strukturell, nicht zufällig (MNAR durch Design).

**Konsequenz:** Beim Training nur Zeilen mit `score` nutzen – oder explizit ein 7-Tage-Forecasting-Setup bauen. Time-Series-CV muss **mindestens 7 Tage** zwischen Train- und Validierungsfenstern respektieren; Walk-Forward pro Region ist Pflicht.

### 3.3 Regionale Label-Rate

| | Sample | Vollständiger Train |
|---|--------|---------------------|
| Score-Rate pro Region | R1/R2: 14,27 %; R3: 14,20 % | **überall 782/5.480 = 14,27 %** |

Die Label-Dichte ist im Gesamtdatensatz **perfekt uniform** über Regionen. Regionale Modelle sind wegen unterschiedlicher Klimata sinnvoll, nicht wegen unterschiedlicher Label-Verfügbarkeit.

Im Sample weichen die **durchschnittlichen Scores** dennoch ab (R1: 1,10; R2: 0,78; R3: 0,24) – das Sample ist geografisch nicht repräsentativ; regionale Effekte im Vollbild neu validieren.

---

## 4. Datenqualität & Zeitstruktur

### 4.1 Fehlende Werte

- **Wetter:** 0 % missing (Sample und Vollbild).
- **Score:** ausschließlich strukturell fehlend (~85,73 %).

Kein Imputationsbedarf für Wettervariablen; ggf. Interpolation nur bei vereinzelten Kalenderlücken.

### 4.2 Kalenderlücken (Sample)

| Region | Tage | Lücken | Max. Lücke |
|--------|------|--------|------------|
| R1, R2 | 4.750 | 65 | 3 Tage |
| R3 | 500 | 7 | 3 Tage |

Die Zeitreihen sind **nahezu lückenlos** (max. 3 fehlende Tage). Für Lags/Rolling-Fenster: `min_periods` setzen oder kleine Lücken forward-fill pro Region.

### 4.3 Zeitraum

- **Voll-Train:** Jahrtausende 10004 – 8133 (synthetischer/verschobener Kalender).
- **Sample:** nur 3004 – 3019, 3 Regionen – **nicht repräsentativ** für globale Muster.
- **Test:** 10021 – 8135, alle 2.248 Regionen – deutlich **zukunftsorientierter** als der Sample-Train-Ausschnitt.

---

## 5. Wettervariablen

### 5.1 Verteilungen (Sample)

- **Temperatur:** Saisonaler Schwung (tmp ca. −6 °C bis 33 °C); `tmp_range` typisch 8–13 °C.
- **Niederschlag:** Stark rechtsschief (Median 0,19; Max 137,6) – die meisten Tage trocken, seltene Extremereignisse.
- **Wind:** Moderat (Median ~2,2); Max ~12.
- **Luftdruck `surf_pre`:** Eng um ~101 hPa (98 – 104).

Extremwerte (99,9 %-Quantil im Sample): `prec` bis 76; `tmp_max` bis 41; `wind` bis 7 – Ausreißer sind real und modellrelevant.

### 5.2 Korrelation mit `score` (gleicher Tag, Sample)

| Feature | r mit score | Richtung |
|---------|-------------|----------|
| tmp_range | **+0,171** | Größere Tagesamplitude → höherer Score |
| tmp_max | +0,149 | Hitze |
| surf_pre | −0,111 | Tiefdruck / Sturmlage |
| surf_tmp, tmp | ~+0,11 | Wärme |
| month | +0,104 | Schwache Saisonalität |
| prec | +0,015 | **Keine lineare Tageskorrelation** |
| year | −0,004 | Kein Trend |

**Kernthese:** Der Score reagiert auf **thermische Extremität und Schwankung**, nicht auf den durchschnittlichen Regen am selben Tag.

---

## 6. Verzögerte Effekte (Lag & Rolling)

### 6.1 Lag-Korrelationen (stärkste)

| Feature | Beste Lag-Korrelation | vs. Same-Day |
|---------|----------------------|--------------|
| tmp_range | **+0,240** (lag 21) | 0,171 |
| tmp_range | +0,236 (lag 3) | |
| prec | **−0,103** (lag 3) | 0,015 |
| surf_pre | −0,100 (lag 7) | −0,111 |
| wind | −0,088 (lag 3) | −0,055 |

**`tmp_range`-Lags** verbessern die lineare Vorhersage um ~40 % relativ – unbedingt als Features (1, 3, 7, 14, 21 Tage).

**Niederschlag:** Negativ verzögert (mehr Regen vor einigen Tagen → niedrigerer Score?) – Hypothese: Erholung nach Wetterereignis oder konfundiert mit Jahreszeit; im Modell testen, nicht nur linear interpretieren.

### 6.2 Rolling 7-Tage-Fenster

| Feature | r mit score |
|---------|-------------|
| prec_roll7_mean | **−0,150** |
| prec_roll7_std | −0,145 |
| wind_roll7_mean | −0,117 |
| tmp_roll7_mean | +0,087 |

Eine **regenreiche oder windige Woche** geht mit niedrigerem Score einher – stärker als der Tages-`prec`-Wert allein. Rolling-Statistiken (mean, std, max) für `prec`, `wind`, `tmp` sind Pflicht-Features.

---

## 7. Extreme Wetterereignisse vs. Score (Sample, 95 %-Schwelle)

| Ereignis | Ø Score (extrem) | Ø Score (normal) | n (extrem) |
|----------|------------------|------------------|------------|
| tmp_max ≥ 35,1 °C | **2,28** | 0,85 | 60 |
| tmp_range ≥ 16,4 °C | **1,67** | 0,88 | 60 |
| prec ≥ 20,3 mm | 0,94 | 0,91 | 69 |
| wind_max ≥ 5,7 | 0,86 | 0,91 | 83 |

**Hitzetage und große Temperaturspreizung** trennen die Score-Verteilung am deutlichsten. Regen- und Winde extreme sind im Sample schwächer – ggf. mit mehr Daten oder Schwellen > 95 % nachschärfen.

→ Nichtlineare Modelle (LightGBM/XGBoost) oder Schwellen-Features (`tmp_max > Q95`) sind sinnvoll.

---

## 8. Saisonalität

- **Score-Meldungen pro Monat:** nahezu gleichmäßig (116–128 pro Monat im Sample) – kein starkes Reporting-Bias über Monate.
- **Korrelation `month` mit score:** +0,10 – leichte Saisonalität, aber kein dominanter Faktor.
- **Jahrestrend:** praktisch null (`year` ≈ 0).

Saisonale Features (`month`, ggf. zyklische Encoding sin/cos) ja; linearer Jahrestrend nein.

---

## 9. Train vs. Test (Distribution Shift)

| Feature | Train-Sample Max | Test Max | Auffälligkeit |
|---------|------------------|----------|---------------|
| tmp | 32,97 | **39,67** | Test extremer |
| tmp_min | −10,13 | **−34,82** | Test viel kälter |
| wind_max | 15,71 | **21,90** | Test windiger |
| surf_pre (Mittel) | 100,78 | **95,94** | Test niedrigerer Druck |
| prec (Mittel) | 3,87 | **1,93** | Sample regenreicher (Selektionsbias) |

Der **10k-Sample-Ausschnitt** (erste Zeilen, wenige Regionen, Jahre 3004–3019) ist **klimatisch nicht identisch** mit dem Test-Zeitraum. Plots und Korrelationen aus dem Notebook sind für **Feature-Ideen** gültig; finale Modell- und Metrik-Entscheidungen brauchen **Colab/Volltrain**.

Test = reines Prediction-Set (kein `score`) über **alle Regionen** und einen **weiten Zukunftszeitraum**.

---

## 10. Modellierungs-Empfehlungen (aus EDA abgeleitet)

### 10.1 Problemformulierung

| Aspekt | Empfehlung |
|--------|------------|
| Task | Regression auf `score` ∈ [0, 5] (ggf. gerundet) |
| Trainingseinheit | Nur Zeilen mit `score.notna()` (~1,76 Mio.) |
| Panel-Struktur | `region_id` als Gruppe; nie zufälliger Split |
| CV | Time-Series Walk-Forward **pro Region**, Gap ≥ 7 Tage |

### 10.2 Feature Engineering (Priorität)

1. **Lags** (1, 3, 7, 14, 21): `tmp_range`, `tmp_max`, `tmp`, `prec`, `wind`, `surf_pre`
2. **Rolling 7d/14d:** mean, std, max von `prec`, `wind`, `tmp`
3. **Kalender:** `month`, `day` (zyklisch encodiert)
4. **Region:** Target Encoding oder `region_id` als Kategorik für LightGBM
5. **Extrem-Flags:** z. B. `tmp_max > Q95` pro Region oder global
6. **Nicht nutzen:** `year` als Trend; roher Tages-`prec` allein

### 10.3 Modell & Baseline

| Stufe | Modell |
|-------|--------|
| Baseline 1 | Persistence: letzter bekannter `score` (7 Tage zurück) |
| Baseline 2 | Region-Median von `score` |
| Hauptmodell | **LightGBM** (Skalierung, Sparsity, viele Kategorien) |
| Validierung | MAE/RMSE nur auf gelabelten Zeilen im CV |

### 10.4 Risiken

| Risiko | Mitigation |
|--------|------------|
| Sample ≠ Vollbild | Training in Colab auf vollem `train.csv` |
| Weekly labels | 7-Tage-Gap in CV; Features aus Tagen ohne Label nutzen |
| Zero-inflation | Gewichtung, Huber-Loss, oder Two-Stage-Modell |
| Test-Shift | Robustheit via viele Regionen, Extrem-Features, Regularisierung |

---

## 11. Fazit

Die Exploration zeigt ein **gut strukturiertes, aber semi-supervised Wetter-Panel**: tägliches Wetter für alle Regionen, **wöchentliche** Score-Labels mit klarer Null-Inflation. Der predictive Signal liegt primär in **Temperaturdynamik** (Range, Max, Lags) und **aggregiertem Regen/Wind über die Vorwoche**, nicht in punktuellem Tagesregen.

Das lokale Notebook liefert **valide strukturelle Erkenntnisse** (14,27 % Labels, 7-Tage-Rhythmus, Feature-Ranking), die mit dem Voll-Scan übereinstimmen. Für die **Präsentation und das finale Modell** sollten Korrelationen und regionale Score-Mittelwerte einmal auf dem **vollen Datensatz in Colab** repliziert werden – der Sample deckt nur 3 von 2.248 Regionen und einen kurzen Zeitraum ab.

**Nächster Schritt laut Projektplan:** Preprocessing + Feature-Pipeline (`02_features.ipynb` o. ä.) → Baseline → LightGBM mit Time-Series-CV in Colab.

---

*Erstellt: Mai 2026 | Basis: `01_exploration.ipynb` (lokal) + chunkweise Analyse `train.csv`*
