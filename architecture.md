# Architecture Overview

Visual companion to [`SYSTEMDESIGN.md`](SYSTEMDESIGN.md). Two planes meet at exactly
one artifact: `web/lib/model-params.json`.

> Educational software simulation only. All data is synthetic. Not a medical device.

---

## 1. Offline training plane

```
                     config.yaml  (seed 42, hyperparameters, paths)
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        ▼                          ▼                          ▼
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│ Tabular Generator│      │Temporal Generator│      │ Signal Generator │
│ 100,000 records  │      │ 20,000 sequences │      │ 20,000 waveforms │
│ 25,000 × 4 class │      │ Markov dynamics  │      │ P/QRS/T @ 500 Hz │
└────────┬─────────┘      └────────┬─────────┘      └────────┬─────────┘
         │                         │                         │
         ▼                         ▼                         ▼
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│   Validation     │      │  ID-based split  │      │ Stratified split │
│ schema · ranges  │      │    70/15/15      │      │    70/15/15      │
│ duplicates       │      │  (no leakage)    │      │                  │
└────────┬─────────┘      └────────┬─────────┘      └────────┬─────────┘
         │                         │                         │
         ▼                         │                         │
┌──────────────────┐               │                         │
│ Feature Eng.     │               │                         │
│ 13 vitals + 9    │               │                         │
│ derived = 22     │               │                         │
│ Impute → Scale   │               │                         │
└────────┬─────────┘               │                         │
         │                         │                         │
         ▼                         ▼                         ▼
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│ Classical ML     │      │  GRU  ·  LSTM    │      │     1D CNN       │
│ KNN LR DT RF     │      │   (PyTorch)      │      │    (PyTorch)     │
└────────┬─────────┘      └────────┬─────────┘      └────────┬─────────┘
         │                         │                         │
         └─────────────────────────┼─────────────────────────┘
                                   ▼
                        ┌──────────────────────┐
                        │   Evaluation Suite   │
                        │ accuracy · macro-F1  │
                        │ per-class · confusion│
                        └──────────┬───────────┘
                                   ▼
                  ┌────────────────────────────────────┐
                  │ Persisted artifacts                │
                  │ models/classical/*.joblib          │
                  │ models/deep/*.pt                   │
                  │ models/preprocessing_pipeline.pkl  │
                  │ experiments/results/*.json         │
                  └────────────────┬───────────────────┘
                                   │
                                   ▼
                  ┌────────────────────────────────────┐
                  │ Inference + Alert Engine (Python)  │
                  │ risk class → P0 / P1 / P2 / P3     │
                  └────────────────────────────────────┘
```

---

## 2. Model ↔ input modality map

| Modality | Input shape | Models | Test accuracy |
|---|---|---|---|
| Tabular vitals | `(n, 22)` | KNN · Logistic Regression · Decision Tree · Random Forest | 95.86 – 98.05% |
| Temporal sequence | `(n, 60, 6)` | GRU · LSTM | 93.70% · 92.40% |
| ECG-like signal | `(n, 1, 500)` | 1D CNN | 99.73% |

Only the tabular family can be driven from a 13-field web form, which is why the hosted
demo serves Logistic Regression. See `SYSTEMDESIGN.md` §9.

---

## 3. Model export boundary

```
models/preprocessing_pipeline.pkl ─┐
models/classical/logistic_regression.joblib ─┤
                                   ▼
                 scripts/export_web_model_params.py
                 (writes JSON, scores it on the held-out split,
                  exits non-zero below 0.95)
                                   ▼
                      web/lib/model-params.json   ≈ 4.4 KB
                      imputer medians · scaler mean/scale
                      4 × 22 coefficients · 4 intercepts
```

This is the **only** artifact that crosses from Python to JavaScript.

---

## 4. Online serving plane (Vercel)

```
Browser
  │  POST /api/predict  {13 vitals}
  ▼
Next.js serverless function (Node.js)
  │
  ├─ 1. normalize     fill defaults · coerce numbers · °F → °C
  ├─ 2. validate      22-range table → 400 with per-field reasons
  ├─ 3. engineer      + 9 derived features → 22 columns
  ├─ 4. impute        median substitution
  ├─ 5. standardize   (x − mean) / scale
  ├─ 6. score         4 logits = W·z + b
  ├─ 7. softmax       probabilities
  └─ 8. decide        argmax → class → priority P0–P3 → colour
  ▼
JSON response  {risk class, priority, confidence, probabilities, disclaimer}
```

Step-by-step detail with a worked example:
[`SYSTEMDESIGN.md` §8](SYSTEMDESIGN.md#8-production-inference-pipeline-input-to-output).

---

## 5. What runs where

| Component | Offline | Vercel |
|---|---|---|
| Python, NumPy, pandas, scikit-learn | ✅ | — |
| PyTorch and `.pt` checkpoints | ✅ | — |
| `joblib` model files | ✅ | — |
| `web/lib/model-params.json` | produced | ✅ |
| TypeScript inference | — | ✅ |
| Vitals validation | mirrored | ✅ |
| Alert priority mapping | ✅ | ✅ |

---

## 6. Reading guide

| Question | Document |
|---|---|
| Why does this project exist? What was achieved? | [`README.md`](README.md) |
| How is it designed, trained, served and verified? | [`SYSTEMDESIGN.md`](SYSTEMDESIGN.md) |
| What is the current state? | [`BUILD_STATUS.md`](BUILD_STATUS.md) |
| How do the planes fit together visually? | this file |
