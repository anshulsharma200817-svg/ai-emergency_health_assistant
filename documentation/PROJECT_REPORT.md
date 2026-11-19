# Project Report

## AI-Based Emergency Health Detection and Alert System

**An educational, end-to-end machine-learning pipeline for four-class emergency risk
stratification on fully synthetic physiological data.**

| | |
|---|---|
| **Project title** | AI-Based Emergency Health Detection and Alert System |
| **Submission type** | Final Minor Project |
| **Team** | Anshul Sharma (BETN1AI25061), Yuvraj Tayal (BETN1AI25101) |
| **Repository** | <https://github.com/anshulsharma200817-svg/ai-emergency_health_assistant> |
| **Live demo** | <https://ai-emergency-health-assistant-kappa.vercel.app/> |
| **Technical design** | [`../SYSTEMDESIGN.md`](../SYSTEMDESIGN.md) |

> **Disclaimer.** This is an educational software simulation built on 100% synthetic
> data. It is not a medical device and must not be used for diagnosis, triage or any
> clinical decision.

---

## 1. Abstract

Emergency triage turns a handful of vital signs into an urgency level under time
pressure. This project builds that pipeline end to end on **fully synthetic data**:
three clinical data modalities (tabular vitals, 60-step temporal trajectories, and
500-sample ECG-like waveforms), 22 leakage-free features, **seven trained models**
(four classical, three deep), a safety-conscious inference layer, and a public web
demo that runs the served model in TypeScript with no Python at runtime.

The served model is a multinomial **Logistic Regression** with **97.03%** held-out
accuracy and **0.9703** macro F1 on 15,000 unseen records. The best tabular model is a
Random Forest at **98.05%**, and the best model overall is a 1D CNN at **99.87%** on
synthetic ECG waveforms; §7 explains, with measured costs, why neither is the model we
serve. The exported parameters reproduce scikit-learn on the full test set with **0
label differences** and a maximum probability difference of **1.27 × 10⁻⁶**. All 81
automated tests pass, a 160-check HTTP edge sweep passes against the deployed API, and
every published number is reproducible from seed 42.

---

## 2. Problem statement

Four risk levels, overlapping symptoms, asymmetric error costs and no tolerance for
confident wrong answers make triage a hard classification problem. Real patient data is
private and regulated, so the honest scope for a student project is synthetic data.

The project therefore answers one engineering question:

> Can the complete lifecycle — generation, validation, feature engineering, training,
> evaluation, serving and verification — be built on synthetic data and published
> without exaggerating any part of it?

Concretely, the system must:

1. Map 13 raw vitals and symptoms to `NORMAL`, `MODERATE`, `HIGH` or `CRITICAL`.
2. Map the class to an alert priority (P0–P3) so urgent cases surface first.
3. Reject malformed or implausible input instead of defaulting to a reassuring answer.
4. Serve predictions from a public URL in milliseconds, with no Python runtime, and
   preserve training-time behaviour exactly.

---

## 3. Objectives

| # | Objective | Target | Achieved |
|---|---|---|---|
| 1 | Synthetic data at research scale | 100K / 20K / 20K | ✅ 100,000 tabular, 20,000 temporal, 20,000 signals |
| 2 | Three modalities | Tabular, sequence, signal | ✅ Three generators, three model families |
| 3 | Four classical models | KNN, LR, DT, RF | ✅ Trained on 22 features |
| 4 | Three deep models | 1D CNN, GRU, LSTM | ✅ 20 epochs each with checkpointing |
| 5 | Tabular accuracy | ≥ 95% macro F1 | ✅ 0.9805 (RF), 0.9703 (LR, served) |
| 6 | Signal accuracy | ≥ 99% | ✅ 99.87% (1D CNN) |
| 7 | Reproducibility | Deterministic reruns | ✅ Seed 42; metrics reproduced exactly |
| 8 | Automated tests | All passing | ✅ 81/81 |
| 9 | Runtime independence | No Python on the server | ✅ 4.4 KB frozen parameters + TypeScript |
| 10 | Python↔web parity | No behavioural drift | ✅ 0/15,000 label diffs; Δp ≤ 1.27 × 10⁻⁶ |
| 11 | Safety guardrails | Reject implausible input | ✅ Range, type and cross-field validation |
| 12 | Honest documentation | Decisions and limits recorded | ✅ README, SYSTEMDESIGN, BUILD_STATUS, this report |

---

## 4. Literature and design rationale

The design borrows three well-established ideas rather than inventing new modelling:

- **Stratified, leakage-free evaluation.** Scaling and imputation are fit on the
  training split only; temporal and signal splits are grouped by case so a trajectory
  cannot appear in two splits. Published metrics therefore measure generalisation, not
  memorisation.
- **Model complexity must be justified by deployment cost.** A tree ensemble that is
  1 pp more accurate but 60,000× larger is not automatically the better system. §7
  quantifies this instead of asserting it.
- **Fail-closed interfaces.** In safety-adjacent software, an unparseable value should
  produce an error, never a default. The API returns `400` for present-but-invalid
  input and only applies defaults when a field is genuinely absent.

Feature engineering follows standard triage practice: pulse pressure, mean arterial
pressure, shock index, age band, SpO₂ severity band, and binary criteria for
tachycardia, hypotension, fever and tachypnea.

---

## 5. System architecture

The system is two planes joined by one artifact.

**Training plane (Python, offline).** `generate_data.py` writes three datasets; the
validator checks schema, ranges, duplicates and class balance; the preprocessing
pipeline imputes, engineers and scales; the trainers fit four classical and three deep
models; the evaluator reports accuracy, macro F1, per-class metrics and confusion
matrices; `scripts/export_web_model_params.py` freezes the served model to JSON.

**Serving plane (TypeScript, Vercel).** A Next.js 16 App Router app renders the home
screen, the live demo and the training-results charts. `POST /api/predict` reimplements
preprocessing and the softmax decision function in `web/lib/inference.ts`;
`GET /api/metrics` serves the published results.

**The joining artifact** is `web/lib/model-params.json` — imputer medians, scaler
mean/scale, a 4×22 coefficient matrix and 4 intercepts, 4.4 KB in total. A second
generated artifact, `web/lib/training-results.json`, keeps the home-screen charts and
the metrics API in sync with the fitted models.

```
data/raw ─▶ validate ─▶ features ─▶ split ─▶ fit ─▶ evaluate ─▶ export JSON
                                                                    │
                                        web/lib/*.json ─▶ /api/predict, /api/metrics
                                                                    │
                                                             Next.js on Vercel
```

---

## 6. Data design

### 6.1 Classes and balance

| Class | Meaning | Rows (tabular) |
|---|---|---|
| `NORMAL` | Stable vitals, no red flags | 25,000 |
| `MODERATE` | Abnormal but not urgent | 25,000 |
| `HIGH` | Deteriorating, needs prompt review | 25,000 |
| `CRITICAL` | Emergency pattern | 25,000 |

### 6.2 Three modalities

| Modality | Method | Volume | Shape |
|---|---|---|---|
| Tabular | Scenario-based vitals + symptoms with bounded noise | 100,000 | 14 raw columns → 22 features |
| Temporal | Markov latent-state trajectories over 6 physiological channels | 20,000 | 60 × 6 |
| Signal | P/QRS/T waveform synthesis at 500 Hz with baseline wander and noise | 20,000 | 1 × 500 |

### 6.3 Raw and derived features

13 raw inputs: `age`, `sex`, `heart_rate`, `systolic_bp`, `diastolic_bp`, `spo2`,
`temperature`, `respiratory_rate`, `chest_pain`, `breathing_difficulty`, `dizziness`,
`unconsciousness`, `bleeding_injury` (plus `case_id`, which is never a feature).

9 derived features: `pulse_pressure`, `mean_arterial_pressure`, `shock_index`,
`age_group`, `spo2_severity`, `tachycardia_flag`, `hypotension_flag`, `fever_flag`,
`tachypnea_flag`.

### 6.4 A deliberate data artifact

15 of 100,000 rows have `diastolic_bp == systolic_bp` because one-decimal rounding
collapsed a near-equal pair. The dataset is frozen so published metrics stay
reproducible; the API accepts equality and rejects strict inversion.

---

## 7. Model selection and results

### 7.1 Classical models (tabular, 15,000-record test set)

| Model | Test accuracy | Test macro F1 | Train time |
|---|---|---|---|
| K-Nearest Neighbors (k=5) | 95.86% | 0.9586 | 0.01 s |
| **Logistic Regression** (served) | **97.03%** | **0.9703** | 1.46 s |
| Decision Tree (depth 10) | 96.61% | 0.9662 | 1.0 s |
| Random Forest (200 trees) | **98.05%** | **0.9805** | 40.7 s |

### 7.2 Deep models (PyTorch, 3,000-record test set per modality)

| Model | Input | Test accuracy | Test macro F1 | Train time |
|---|---|---|---|---|
| **1D CNN** | ECG signal (1×500) | **99.87%** | **0.9987** | 132 s |
| GRU (2-layer) | 60-step × 6 channels | 95.53% | 0.9545 | 647 s |
| LSTM (2-layer) | 60-step × 6 channels | 95.03% | 0.9496 | 481 s |

### 7.3 Served model and confusion matrix

Logistic Regression on the 15,000-record tabular test set:

| actual ↓ / predicted → | NORMAL | MODERATE | HIGH | CRITICAL |
|---|---|---|---|---|
| **NORMAL** | 3,707 | 42 | 1 | 0 |
| **MODERATE** | 45 | 3,592 | 109 | 4 |
| **HIGH** | 0 | 95 | 3,572 | 83 |
| **CRITICAL** | 0 | 0 | 67 | 3,683 |

Per-class performance:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| NORMAL | 0.9880 | 0.9885 | 0.9883 | 3,750 |
| MODERATE | 0.9633 | 0.9579 | 0.9606 | 3,750 |
| HIGH | 0.9528 | 0.9525 | 0.9527 | 3,750 |
| CRITICAL | 0.9769 | 0.9821 | 0.9795 | 3,750 |

**No CRITICAL case is predicted NORMAL or MODERATE** — the errors that do occur are
adjacent-class confusions (MODERATE↔HIGH), which is the behaviour a triage system
should have.

### 7.4 Why Logistic Regression is served

| Criterion | Logistic Regression | Random Forest | 1D CNN |
|---|---|---|---|
| Test accuracy | 97.03% | 98.05% | 99.87% |
| Artifact size | 1.2 KB joblib → 4.4 KB JSON | 76.4 MB joblib, 795,402 nodes | PyTorch checkpoint |
| Runtime | Pure arithmetic | Tree traversal (~32 MB compact export) | PyTorch runtime |
| Input available in the demo | 13 vitals + symptoms | 13 vitals + symptoms | 1×500 ECG waveform |
| Cold start / bundle impact | Negligible | Dominates the function bundle | Not deployable as-is |
| Explainability | 4×22 coefficients | 200 trees | Opaque |

The 1.02 pp gap to Random Forest is not worth a 60,000× larger artifact and a
traversal engine, and the 2.84 pp gap to the CNN is not reachable at all because the
demo does not collect ECG waveforms. Logistic Regression is the model the deployment
can actually run well, and it is the model we serve.

### 7.5 A red-flag constraint the model cannot break

In the training data, `unconsciousness = 1` appears in 1,490 records — 1,330 CRITICAL
and 160 HIGH — and never in NORMAL or MODERATE. The linear model nonetheless learned a
*positive* NORMAL coefficient for it, so "healthy vitals + unconscious" was returned as
`NORMAL` at 99.9%. The serving layer now restricts that record to HIGH/CRITICAL and
renormalises. Because the constraint agrees with the data, it changes **0 of 15,000**
test predictions and accuracy stays exactly 97.03%.

---

## 8. Implementation

### 8.1 Generation and validation

`src/data/` contains three generators (tabular, temporal, signal) driven by
`config.yaml`, plus a validator that enforces the schema, value ranges, duplicate
freedom and class balance. The pipeline aborts before training if validation fails.

### 8.2 Preprocessing

`src/preprocessing/` performs median imputation and standard scaling fitted on the
training split only, derives the 9 engineered features, and writes `.npy` splits
(70/15/15, stratified). Temporal and signal splits are grouped by `case_id`.

### 8.3 Training and evaluation

`src/training/` fits KNN, Logistic Regression, Decision Tree and Random Forest, and
`train_deep.py` runs 20-epoch PyTorch loops for the CNN, GRU and LSTM with per-epoch
train/validation tracking and best-checkpoint saving. Random seeds are fixed per
architecture so a rerun reproduces the same metrics.

### 8.4 Inference and alerting

`HealthRiskPredictor` supports single and batch inference and mirrors the web
validation rules exactly, including half-up 4-decimal rounding. `SimulatedAlertEngine`
maps classes to P0–P3 and raises on unknown or missing classes rather than defaulting
to the least-urgent tier.

### 8.5 Web deployment

The Next.js app serves a static home page and two serverless functions. Vercel builds
from the `web/` Root Directory with no environment variables and no database.

---

## 9. Testing and verification

| Check | Result |
|---|---|
| `pytest tests -q` | **81 passed** in ~7 s |
| HTTP edge sweep (`scripts/api_edge_check.py`) | **160 checks, 0 failures** |
| Python↔web parity, full test set | **0/15,000 label differences**, max Δp 1.27 × 10⁻⁶ |
| HTTP parity, 1,282 raw records | **0 label mismatches**, max Δp 1 × 10⁻⁴ (rounding ties) |
| Determinism | 5 sequential and 48 concurrent requests identical |
| Mobile layout | 390 px viewport, no horizontal overflow |

Automated coverage includes generators, the preprocessing pipeline (including a
leakage guard), splitters, classical and deep training, inference (constraints, batch
mode, string coercion, malformed/out-of-range/inverted-BP rejection), the alert engine,
the data validator and annotation resolution.

Defects found and fixed during verification are listed in
[`../BUILD_STATUS.md`](../BUILD_STATUS.md) §3, including a test run that had
overwritten production models with 200-sample artifacts, malformed input that produced
a confident `NORMAL`, and a stale result panel that could display a verdict beside
inputs it never scored.

---

## 10. Results summary

- **97.03%** held-out accuracy and **0.9703** macro F1 for the served tabular model.
- **98.05%** for the best tabular model (Random Forest, not served) and **99.87%** for
  the best model overall (1D CNN, needs ECG input and a PyTorch runtime).
- **Zero** CRITICAL→NORMAL/MODERATE errors on 15,000 held-out records.
- **4.4 KB** deployed model, no Python at runtime, sub-100 ms inference.
- **81/81** tests and **160/160** API checks passing.
- Every number reproducible from seed 42.

---

## 11. Limitations

- Synthetic data reproduces realistic *shapes*, not real physiology; nothing here
  transfers to clinical use.
- No authentication or rate limiting on the public demo.
- Deep models are trained and measured but not served, because the demo collects
  tabular vitals rather than ECG waveforms.
- No persistence: predictions are not logged, since storing health-like data would
  require a privacy design the project deliberately avoids.

## 12. Future work

1. Serve the 1D CNN behind a separate ECG-input demo with a small model runtime.
2. Add authentication, rate limiting and audit logging before any public promotion.
3. Surface per-prediction SHAP attributions in the UI instead of offline-only analysis.
4. Re-evaluate the same pipeline on consented, de-identified data under IRB approval —
   and still not as a medical device.

---

## 13. Conclusion

The project meets its objective: a complete, reproducible, honestly documented ML
lifecycle on synthetic data, with a deployed demo whose behaviour matches the training
pipeline to within floating-point rounding. The most valuable outcome is not the
accuracy number but the engineering discipline around it — leakage-free evaluation,
measured deployment trade-offs, fail-closed input handling, and verification that the
model served in production is exactly the model that was evaluated.

---

## References

1. scikit-learn developers. *scikit-learn: Machine Learning in Python.*
   <https://scikit-learn.org>
2. Paszke, A. et al. *PyTorch: An Imperative Style, High-Performance Deep Learning
   Library.* NeurIPS 2019.
3. Pedregosa, F. et al. *Scikit-learn: Machine Learning in Python.* JMLR 12, 2011.
4. Next.js documentation. <https://nextjs.org/docs>
5. Vercel documentation — Monorepos and Root Directory.
   <https://vercel.com/docs/monorepos>
6. Lundberg, S. & Lee, S.-I. *A Unified Approach to Interpreting Model Predictions.*
   NeurIPS 2017 (SHAP).
