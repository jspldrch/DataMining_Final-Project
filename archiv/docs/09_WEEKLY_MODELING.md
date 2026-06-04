# Weekly Modeling (Notebook 04)

> **Dokumentation Nr. 09** · [Lesereihenfolge](README.md)

## Warum diese Schicht existiert

| Stufe | Granularität | Grund |
|-------|--------------|--------|
| **03 Preprocessing** | Tägliche Zeilen mit Features | Streaming über 12M Zeilen; Lags/Rolls brauchen volles Panel |
| **04 Modeling** | **Wochen**-Samples | Kaggle: `pred_week1`…`pred_week5`; EDA: ~7-Tage-Score-Rhythmus |

Ohne Wochen-Collapse in 04 entstehen aus ~782 gelabelten Tagen/Region ~776 Sliding-Fenster → **>1M Samples** und Colab-OOM.

## Was `daily_to_weekly` macht

- Bucket: `ordinal // 7` pro `region_id`
- Pro Bucket: **letzter gelabelter Tag** (Features + `score` von diesem Tag)
- Erwartung: ~100–120 Wochen/Region statt 782 Tage

## Training vs. Submission

- **Train:** Fenster auf Wochenreihen → X an Woche *i*, y = Scores *i+1…i+5*
- **Valid:** 20 % Regionen, letzte 5 Wochen als Holdout
- **Test:** eine Zeile/Region = **letzter Tag** der 91 Test-Tage (Features aus 03), Vorhersage der **nächsten 5 Wochen**

## 03 erneut?

**Nein**, solange `train_labeled.parquet` und `test_features.parquet` existieren. Wochenlogik lebt nur in `scripts/weekly_model.py` + Notebook 04.

## Dateien

- `scripts/weekly_model.py` — Aggregation, Samples, Submission
- `notebooks/04_modeling.ipynb` — Colab-Pipeline
