# System Design — AI-Based Emergency Health Detection and Alert System

**Document type:** Technical system design and implementation record
**Project:** AI-Based Emergency Health Detection and Alert System
**Stack:** Python 3.10+ (NumPy, pandas, scikit-learn, PyTorch), Next.js 16 (TypeScript), Vercel
**Data:** 100% synthetic — 100,000 tabular records, 20,000 temporal sequences, 20,000 ECG-like signals
**Status:** Feature complete, trained, tested, and deployed
**Companion document:** [`README.md`](README.md) — goals, objectives, and achievements

> [!IMPORTANT]
> **EDUCATIONAL SOFTWARE SIMULATION ONLY.**
> This system is an educational simulation built entirely on procedurally generated
> synthetic data. It is **not** a medical device. It does **not** diagnose, recommend
> treatment, dispatch ambulances, ingest real sensor data, or integrate with any
> hospital system. No real patient data is used anywhere in this repository.

---

## 0. Purpose and governing constraints

The project demonstrates a complete, reproducible machine-learning lifecycle:
**synthetic data generation → validation → preprocessing → training → evaluation →
inference → alert routing → web deployment.** Every decision below follows from four
constraints:

| Constraint | Consequence for the design |
|---|---|
| Educational/research simulation only | Synthetic generators, explicit disclaimers in every prediction payload, no clinical claims |
| Must be reproducible on a laptop | Single seed (42), deterministic splits, CPU-only training, no external services |
| Must cover multiple clinical data modalities | Three generators and three model families rather than one narrow demo |
| Must be publicly demonstrable | A dependency-free Next.js deployment that does not ship Python, pickles, or PyTorch weights |

The central engineering judgment is **where computation happens**: training is offline
and heavy (scikit-learn/PyTorch, 100K records), while serving is online and must be
trivial (a 4 KB parameter file and ~40 lines of TypeScript arithmetic). Section 8
explains that boundary in detail.

---

## 1. System architecture

```
┌─────────────────────────────── config.yaml (seed: 42) ──────────────────────────────┐
│                                                                                       │
│      ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐            │
│      │  Tabular         │    │  Temporal         │    │  Signal           │            │
│      │  Generator       │    │  Generator        │    │  Generator        │            │
│      │  100K records    │    │  20K sequences    │    │  20K waveforms    │            │
│      │  25K × 4 classes │    │  Markov dynamics  │    │  P/QRS/T waves    │            │
│      └────────┬─────────┘    └────────┬──────────┘    └────────┬─────────┘            │
│               │                       │                        │                       │
│               ▼                       ▼                        ▼                       │
│      ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐            │
│      │  Feature Eng.    │    │  ID-based Split  │    │  Stratified       │            │
│      │  22 total feats  │    │  70/15/15        │    │  Split 70/15/15   │            │
│      │  Standard Scale  │    │  (no leakage)    │    │                   │            │
│      └────────┬─────────┘    └────────┬──────────┘    └────────┬─────────┘            │
│               │                       │                        │                       │
│               ▼                       ▼                        ▼                       │
│      ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐            │
│      │  Classical ML    │    │  RNN / LSTM      │    │  1D CNN           │            │
│      │  KNN  LR DT RF   │    │  (PyTorch)       │    │  (PyTorch)        │            │
│      └────────┬─────────┘    └────────┬──────────┘    └────────┬─────────┘            │
│               └───────────────────────┼────────────────────────┘                       │
│                                       ▼                                               │
│                              ┌──────────────────┐                                     │
│                              │  Evaluation Suite │                                     │
│                              │  Accuracy, Macro  │                                     │
│                              │  F1, Precision,   │                                     │
│                              │  Recall, per-class│                                     │
│                              └────────┬──────────┘                                     │
│                                       ▼                                               │
│                              ┌──────────────────┐                                     │
│                              │  Inference        │                                     │
│                              │  Predictor        │                                     │
│                              │  + Alert Engine   │                                     │
│                              │  P0 P1 P2 P3      │                                     │
│                              └──────────────────┘                                     │
└───────────────────────────────────────────────────────────────────────────────────────┘
```

Two runtime planes:

| Plane | Runs where | What it does |
|---|---|---|
| **Offline training plane** | Local machine / Docker | Generate data, engineer features, fit 7 models, evaluate, write artifacts |
| **Online serving plane** | Vercel serverless | Accept vitals JSON, run frozen model parameters in TypeScript, return risk + priority |

The two planes meet at exactly one artifact: `web/lib/model-params.json`.

---

## 2. Data design

### 2.1 Target classes

| Class | Label | Records (tabular) |
|---|---|---|
| NORMAL | 0 | 25,000 |
| MODERATE | 1 | 25,000 |
| HIGH | 2 | 25,000 |
| CRITICAL | 3 | 25,000 |

### 2.2 Three synthetic modalities

| Modality | Generator | Volume | Shape | Consumed by |
|---|---|---|---|---|
| **Tabular** | `src/data/tabular_generator.py` | 100,000 records | `(100000, 22)` after engineering | KNN, LR, DT, RF |
| **Temporal** | `src/data/temporal_generator.py` | 20,000 sequences | `(20000, 60, 6)` | GRU, LSTM |
| **Signal** | `src/data/signal_generator.py` | 20,000 waveforms | `(20000, 1, 500)` | 1D CNN |

Generation strategies:

- **Tabular** — scenario-based sampling. Each class is produced from a different
  physiological scenario (healthy baseline, compensated stress, decompensating, and
  peri-arrest), with bounded noise so distributions overlap realistically rather than
  being trivially separable.
- **Temporal** — Markov latent-state trajectories over six vital channels for 60
  timesteps, so deterioration is a *process* rather than a single point. Risk-class
  labels are sampled uniformly (5,000 per class); within MODERATE, the generator
  rotates through its three trajectory types (gradual deterioration, recovery, noisy).
- **Signal** — ECG-like P/QRS/T waveform synthesis at 500 Hz for one second
  (500 samples), with four pattern families matching the four risk classes.

### 2.3 Raw features (14 columns)

`case_id`, `age`, `sex`, `heart_rate`, `systolic_bp`, `diastolic_bp`, `spo2`,
`temperature`, `respiratory_rate`, `chest_pain`, `breathing_difficulty`, `dizziness`,
`unconsciousness`, `bleeding_injury`

`case_id` is an identifier, never a model input.

### 2.4 Derived features (9)

| Feature | Formula | Rationale |
|---|---|---|
| `pulse_pressure` | SBP − DBP | Perfusion proxy |
| `mean_arterial_pressure` | DBP + PP/3 | Standard MAP estimate |
| `shock_index` | HR / SBP | Early decompensation marker |
| `age_group` | ≤18 → 0, ≤65 → 1, else 2 | Ordinal risk cohort |
| `spo2_severity` | ≤88 → 0, ≤94 → 1, else 2 | Hypoxia banding |
| `tachycardia_flag` | HR > 100 | Binary clinical criterion |
| `hypotension_flag` | SBP < 90 | Binary clinical criterion |
| `fever_flag` | Temp > 38.0 °C | Binary clinical criterion |
| `tachypnea_flag` | RR > 20 | Binary clinical criterion |

**Total feature dimension: 22** (13 numeric vitals + 9 derived).

The bin edges are right-closed, matching `pandas.cut`, and are reimplemented with
identical semantics in TypeScript (Section 9).

---

## 3. Preprocessing and splitting

| Stage | Implementation | Notes |
|---|---|---|
| Splitting | `src/preprocessing/splitters.py` | 70/15/15, stratified; temporal splits are **ID-based** so one trajectory cannot appear in two splits |
| Imputation | `SimpleImputer(strategy="median")` | Fitted on train only |
| Scaling | `StandardScaler` | Fitted on train only; `(x − mean) / scale` |
| Label encoding | Fixed mapping | NORMAL=0, MODERATE=1, HIGH=2, CRITICAL=3 |
| Leakage guard | Transform-only on val/test | Enforced by the pipeline API and covered by a test |

Split sizes:

| Modality | Train | Val | Test |
|---|---|---|---|
| Tabular | 70,000 | 15,000 | 15,000 |
| Temporal | 14,000 | 3,000 | 3,000 |
| Signal | 14,000 | 3,000 | 3,000 |

The fitted pipeline (imputer + scaler) is serialized to
`models/preprocessing_pipeline.pkl` and is what the export script reads.

### 3.1 Data-quality gate

`run_pipeline.py` treats the validator's report as a **gate, not a log line**.
Training aborts with exit code 1 when the tabular file has an invalid schema,
duplicate rows, any missing value, or any value outside the configured physiological
ranges. Before this gate existed, a corrupt CSV still produced a model and a full set
of metrics. Duplicates are fatal because they can straddle the train/test split and
inflate held-out accuracy. `--allow-data-issues` downgrades the gate to a warning for
deliberate experiments. On the committed 100,000-row dataset it passes with
0 duplicates, 0 nulls and 0 out-of-range values.

The generator itself contains one known rounding artifact: it enforces
`diastolic_bp < systolic_bp` before rounding, but one-decimal rounding collapses
15 near-equal pairs to equality, so 15 of 100,000 rows have a pulse pressure of 0.
The dataset was frozen rather than regenerated because every published metric and the
exported parameters are derived from it; the API accordingly rejects only a
**strictly inverted** pair (Section 10.1).

## 4. Models

### 4.1 Classical ML (tabular)

| Model | Hyperparameters | Val Acc | Test Acc | Test Macro F1 |
|---|---|---|---|---|
| KNN | `n_neighbors=5` | 95.86% | 95.86% | 0.9586 |
| Logistic Regression | `max_iter=1000`, `C=1.0`, multinomial, lbfgs | 97.06% | 97.03% | 0.9703 |
| Decision Tree | `max_depth=10` | 96.41% | 96.61% | 0.9662 |
| **Random Forest** ⭐ | `n_estimators=200`, `max_depth=20` | **98.09%** | **98.05%** | **0.9805** |

Random Forest is the most accurate tabular model; Logistic Regression is the one that
is **served** (Section 10 explains why that is not a contradiction).

### 4.2 Deep learning (PyTorch)

| Model | Input | Architecture | Val Acc | Test Acc | Test Macro F1 | Train time |
|---|---|---|---|---|---|---|
| **Physio1DCNN** ⭐ | `(batch, 1, 500)` | Conv1D 1→16 (k=7) → BN → ReLU → Pool → Conv1D 16→32 → BN → ReLU → Pool → AdaptiveAvgPool(16) → Dropout(0.3) → FC | **99.80%** | **99.87%** | **0.9987** | 132 s |
| **PhysioRNN** (2-layer GRU) | `(batch, 60, 6)` | GRU (hidden 128, 2 layers) → last timestep → Dropout(0.2) → FC | 95.53% | 95.53% | 0.9545 | 647 s |
| **PhysioLSTM** | `(batch, 60, 6)` | LSTM (hidden 128, 2 layers) → last timestep → Dropout(0.2) → FC | 94.97% | 95.03% | 0.9496 | 481 s |

Training protocol: 20 epochs, Adam (lr = 1e-3), CrossEntropyLoss, batch size 128, best
checkpoint selected by validation accuracy, CPU-only PyTorch 2.15. Kernel size for the
CNN is 7; hidden size for both recurrent models is 128. Full run ≈ 18 minutes.

Artifacts:

```
models/deep/cnn_1d_best.pt    30 KB
models/deep/rnn_best.pt      598 KB
models/deep/lstm_best.pt     794 KB
experiments/results/deep_results.json   # per-epoch history + final metrics
```

All three checkpoints were verified to reload and emit `[batch, 4]` logits. Weights are
git-ignored, so a clone must retrain:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
python3 run_pipeline.py --deep-only --skip-generate
```

---

## 5. Evaluation methodology

Metrics are computed by `src/evaluation/metrics.py` with scikit-learn:
accuracy, macro-F1, precision, recall, per-class support, and a confusion matrix.

| Stage | Data | Reported |
|---|---|---|
| Fit | train split | training accuracy / macro-F1 |
| Model selection | validation split | validation accuracy (best checkpoint) |
| Final claim | test split, never seen during fit or selection | accuracy, macro-F1, per-class report |

Macro-F1 is the headline metric because the four classes are equally important and a
model that ignores CRITICAL to optimize accuracy would be unacceptable. Results are
persisted to `experiments/results/{classical,deep}_results.json` and served by
`/api/metrics`.

---

## 6. Alert priority mapping

| Predicted risk | Priority | Simulated alert level |
|---|---|---|
| CRITICAL | **P0** | `EMERGENCY_SIMULATION_CRITICAL` |
| HIGH | **P1** | `EMERGENCY_SIMULATION_HIGH` |
| MODERATE | **P2** | `SIMULATION_ADVISORY_MODERATE` |
| NORMAL | **P3** | `SIMULATION_STATUS_NORMAL` |

Implemented in `src/inference/alert_engine.py` (Python) and mirrored in
`web/lib/inference.ts` (`PRIORITY_MAP`, `ALERT_COLOR`) so offline and hosted behaviour
agree. Every payload carries
`"disclaimer": "EDUCATIONAL SIMULATION ONLY — NOT FOR CLINICAL USE"`.

The Python engine **fails closed**: `generate_alert` raises `ValueError` for any class
it does not recognise — including a missing `predicted_risk_class` — instead of
defaulting to P3 "normal". A misspelled or absent label is an upstream bug, and
reporting the least-urgent tier would be the most dangerous possible default.

---

## 7. Serving architecture (Vercel)

### 7.1 Topology

```
Browser ──POST /api/predict──► Next.js serverless function (Node.js)
                                     │
                                     ▼
                            web/lib/inference.ts      ← pure TypeScript
                                     │
                                     ▼
                            web/lib/model-params.json ← frozen model (4.4 KB)
```

Only `web/` is deployed. `data/`, `models/`, `experiments/`, `.venv`, and the training
source are git-ignored and never uploaded. **No Python, scikit-learn, PyTorch, joblib,
or pickle executes on Vercel.**

| Route | Type | Purpose |
|---|---|---|
| `/` | Static (prerendered) | Interactive vitals form and results dashboard |
| `/api/predict` | Dynamic function | POST inference; GET health check + accepted ranges |
| `/api/metrics` | Dynamic function | Classical + deep metrics and dataset stats |

### 7.2 The frozen model artifact

`web/lib/model-params.json` is generated by
`scripts/export_web_model_params.py` from the fitted sklearn objects:

| Source (sklearn) | Exported key | Size |
|---|---|---|
| `SimpleImputer.statistics_` | `imputer_statistics` | 22 numbers |
| `StandardScaler.mean_` | `scaler_mean` | 22 numbers |
| `StandardScaler.scale_` | `scaler_scale` | 22 numbers |
| `LogisticRegression.coef_` | `coef` | 4 × 22 numbers |
| `LogisticRegression.intercept_` | `intercept` | 4 numbers |
| `LogisticRegression.classes_` | `classes` | 4 integers |

Total ≈ 4.4 KB, committed to git, versioned with the code. The export script refuses to
write a model that scores below 0.965 on a held-out split of at least 8,000 rows
(see Section 12).

---

## 8. Production inference pipeline: input to output

This is the exact path a request takes on the deployed system. Steps 1–9 are in
`web/lib/inference.ts`; the HTTP wrapper is `web/app/api/predict/route.ts`.

### Step 0 — Browser

`web/app/page.tsx` holds 13 vitals in React state (number inputs for seven measured
values, a sex selector, five symptom toggles). Each input is validated on change against
the same `VITALS_RANGES` table the server uses, so an impossible value is flagged
immediately and blocks submission. Editing any vitals **discards the previous result**
and shows "Vitals changed — run the prediction again", and a rendered result is stamped
with the exact values it was scored for. Neither behaviour is cosmetic: without them the
panel could display a confident verdict beside inputs it never saw. On submit the page
issues:

```http
POST /api/predict
Content-Type: application/json

{"age":67,"sex":1,"heart_rate":142,"systolic_bp":72,"diastolic_bp":44,
 "spo2":84,"temperature":39.4,"respiratory_rate":34,
 "chest_pain":1,"breathing_difficulty":1,"dizziness":1,
 "unconsciousness":0,"bleeding_injury":0}
```

The route validates that the body is a JSON object, then calls `predictRisk`.

### Step 1 — Normalize (`normalizeVitals`)

- Missing fields fall back to the healthy-adult baseline (`DEFAULT_VITALS`).
- Numeric strings are coerced (`"72"` → `72`).
- `temperature_unit: "F"` (or `"fahrenheit"`, `"°F"`) converts
  `(F − 32) × 5/9` before anything else. The default is Celsius.

### Step 2 — Validate (`validateVitals`)

Every field is checked against `VITALS_RANGES`; the symptom flags must be exactly 0 or
1. Any violation throws `VitalsValidationError`, which the route converts to
**HTTP 400** with a per-field explanation. The model never sees an out-of-range record.
(Section 11 lists the ranges and the rationale for each bound.)

### Step 3 — Engineer features (`engineerFeatures`)

13 normalized vitals + 9 derived features = the same 22 columns used in training:

```
pulse_pressure         = systolic_bp − diastolic_bp
mean_arterial_pressure = diastolic_bp + pulse_pressure / 3
shock_index            = heart_rate / systolic_bp
age_group              = age ≤ 18 → 0, ≤ 65 → 1, else 2
spo2_severity          = spo2 ≤ 88 → 0, ≤ 94 → 1, else 2
tachycardia_flag       = heart_rate > 100
hypotension_flag       = systolic_bp < 90
fever_flag             = temperature > 38.0
tachypnea_flag         = respiratory_rate > 20
```

The ordering is taken from `params.features`, so the model parameters and the feature
construction can never drift apart.

### Step 4 — Impute

`value = Number.isFinite(raw) ? raw : imputer_statistics[i]`

This mirrors `SimpleImputer(strategy="median")`. With validation in place it is
effectively unreachable for HTTP traffic, but it is retained for exact parity with the
training pipeline (e.g. a zero systolic BP would produce a non-finite shock index).

### Step 5 — Standardize

`z_i = (x_i − scaler_mean_i) / scaler_scale_i`

Exactly `StandardScaler.transform`, including the guard for a zero scale factor.

### Step 6 — Score four logits

`logit_c = Σ_i coef[c][i] × z_i + intercept[c]`

Four classes × 22 weights = 88 multiplications plus 4 intercepts. This is the entire
"model" at runtime.

### Step 7 — Softmax

`P(c) = exp(logit_c − max) / Σ exp(logit_k − max)`

The max-subtraction is a numerical-stability measure and does not change the result.

### Step 7b — Apply the red-flag constraint

One feature encodes a symptom that the linear model cannot separate safely:
`unconsciousness`. In the 100,000-record training set it appears in **35.1 % of
CRITICAL** and **4.9 % of HIGH** records — and in **zero** NORMAL or MODERATE
records. Maximum-likelihood logistic regression therefore has no finite optimum for
that coefficient (quasi-complete separation); the fitted weight lands at
**+0.557 on NORMAL**, so *healthy vitals + unconscious* produced `NORMAL 100 %`, and
ticking all five symptoms *raised* NORMAL confidence to 99.9 %.

The serving layer restores the constraint the data actually encodes: when
`unconsciousness = 1`, the probabilities of NORMAL and MODERATE are set to zero and
the remaining mass is renormalized over HIGH and CRITICAL. When the model put less
than 0.1 % mass on both allowed classes it is extrapolating, and the empirical
`P(class | unconsciousness = 1)` split is used instead (**HIGH 12.3 % / CRITICAL
87.7 %**, from the held-out counts 160 / 1,330). The response reports
`applied_constraint: "unconsciousness"` so the rule is visible rather than silent,
and the UI prints a note under "Scored for" that also discloses when the empirical
split is being shown.

This is a data-derived invariant, not a tuned threshold: on the full 15,000-row test
set 1,490 records have `unconsciousness = 1`, and the unconstrained model already
predicted HIGH or CRITICAL for **all** of them — the mask changes **0 predictions**
and leaves accuracy at exactly **0.9703**. The identical rule is implemented in
`applyDataConstraints` (`web/lib/inference.ts`) and `apply_data_constraints`
(`src/inference/predictor.py`), and both paths are covered by the API sweep and the
Python suite.

### Step 8 — Decide and map to an alert

`predicted = argmax(P)` → class label → priority `P0–P3`, alert colour, and the
educational disclaimer.

### Step 9 — Flag extrapolation

Independently of the prediction, `outsideTrainingSupport` compares the record with
`TRAINING_SUPPORT` (the observed min/max of each vital in the 100,000-record training
set). Values inside the accepted range but outside that support still produce a score;
the response then carries `distribution_warning` and `outside_training_range`, and the UI
prints an amber banner above the probability bars. Step 9 never changes the class — it
only stops an extrapolation from being presented as an ordinary answer.

### Worked example — critical preset

Input: 67 y, male, HR 142, BP 72/44, SpO₂ 84, 39.4 °C, RR 34, chest pain +
breathing difficulty + dizziness.

| Stage | Values |
|---|---|
| Raw (first 8) | `67, 1, 142, 72, 44, 84, 39.4, 34` |
| Standardized z (first 8) | `0.65, 1.00, 1.08, −1.08, −1.11, −0.89, 0.95, 1.29` |
| Logits | NORMAL −38.02 · MODERATE −2.66 · HIGH 17.97 · CRITICAL 22.70 |
| Softmax | 0.0000 · 0.0000 · 0.0088 · **0.9912** |
| Output | `CRITICAL`, `P0`, confidence 0.9912 |

The deployed API returns exactly these numbers, which is what the parity test in
Section 12 asserts.

### Why this design

The model is **not** "called" from a Python process at runtime; it is re-expressed as
arithmetic. Consequences:

- cold start is a JSON import, not a model load;
- the function has no native dependencies and cannot break on a wheel/Python upgrade;
- the served decision function is auditable line by line;
- the only thing that can go stale is the JSON, which the export guard now checks.

---

## 9. Model selection for serving

| Model | Test Acc | Served? | Reason |
|---|---|---|---|
| **Logistic Regression** | 97.03% | ✅ | 1.2 KB joblib → 4.4 KB JSON, ~40 lines of arithmetic, 0 label diffs / 1.3 × 10⁻⁶ probability parity |
| Random Forest | 98.05% | ❌ | 200 trees, **795,402 nodes**, 76 MB joblib → ≈32 MB compact JSON plus a traversal evaluator |
| KNN | 95.86% | ❌ | 6.5 MB joblib (70,000 × 22 training vectors) and per-request distance computation |
| Decision Tree | 96.61% | ❌ | 80 KB and portable, but less accurate than LR |
| 1D CNN | 99.87% | ❌ | needs a PyTorch runtime and a 500-sample ECG, which the vitals form cannot produce |
| GRU | 95.53% | ❌ | needs a 60-step × 6-channel trajectory, not available from the form |
| LSTM | 95.03% | ❌ | same as GRU |

Serving cost is measured, not estimated: Random Forest hits its `max_depth=20` cap in
every tree, and a compact export (feature/threshold/left/right arrays plus leaf class
distributions) is still tens of megabytes — that would dominate both the function bundle
and cold-start time.

**The accuracy gap is also smaller where it matters.** On the 15,000-record test set the
two models are effectively tied on CRITICAL recall, and Random Forest's +1.02 pp comes
almost entirely from the easier NORMAL and MODERATE classes:

| Class | LR recall | RF recall |
|---|---|---|
| NORMAL | 0.9885 (3,707/3,750) | 0.9973 (3,740/3,750) |
| MODERATE | 0.9579 (3,592/3,750) | 0.9813 (3,680/3,750) |
| HIGH | 0.9525 (3,572/3,750) | 0.9616 (3,606/3,750) |
| **CRITICAL** | **0.9821 (3,683/3,750)** | 0.9819 (3,682/3,750) |
| Total errors | 446 | 292 |

The decision is **modality- and deployability-driven**, not accuracy-driven:

1. The form produces tabular vitals, so only the four classical models are eligible.
2. Among those, LR is the only one whose complete decision function fits in a small
   JSON file and runs as plain arithmetic.
3. Softmax probabilities are needed for confidence and alert thresholds; LR provides
   them directly.
4. The 1.02 pp gap to Random Forest is the price of a zero-dependency deployment, and
   it is stated openly rather than hidden.

The deep models are **real and trained** — they are simply not the right artifact for a
13-field web form. Their metrics are exposed through `/api/metrics` with the input
modality spelled out.

---

## 10. Input validation contract

### 10.1 Accepted ranges

| Field | Min | Max | Unit / notes |
|---|---|---|---|
| `age` | 0 | 120 | years |
| `sex` | 0 | 1 | 0 = female, 1 = male |
| `heart_rate` | 20 | 300 | bpm |
| `systolic_bp` | 40 | 300 | mmHg; 40 because real critical records reach 45 |
| `diastolic_bp` | 20 | 200 | mmHg |
| `spo2` | 50 | 100 | percent |
| `temperature` | 25 | 45 | **Celsius** |
| `respiratory_rate` | 4 | 80 | breaths/min |
| `chest_pain` … `bleeding_injury` | 0 | 1 | flags only |

Design notes:

- Bounds are derived from the generated dataset wherever possible. An earlier pass used
  `systolic_bp ≥ 50` and rejected 1,991 genuine critical records (SBP 45–49), which is
  how the lower bound was corrected.
- A cross-field rule rejects only a **strictly inverted** blood pressure
  (`diastolic_bp > systolic_bp`), which never occurs in the training set. Equality is
  accepted because one-decimal rounding collapsed 15 near-equal training pairs to
  `diastolic == systolic`; those rows are inside the model's experience, even though a
  pulse pressure of zero is not clinically meaningful. The check is shared by the form
  and the API (`crossFieldError` in `web/lib/ranges.ts`).
- The **form uses the same table** (`web/lib/ranges.ts`) as the server, so the browser
  cannot silently reject a value the API would accept (an earlier version limited age to
  1–110 and temperature to 30–45 while the API allowed 0–120 and 25–45).
- Client-side validation is advisory; the server check is the authority because the API
  is public. The form is `noValidate` so its own messages — not a transient browser
  bubble — explain what is wrong, and every rejected field is marked
  `aria-invalid` with a `role="alert"` message.

### 10.2 Training support and extrapolation

Validation bounds are chosen for clinical plausibility, so they are **wider than the
data the model ever saw**. The 100,000-record synthetic training set covers:

| Field | Trained on | Accepted by API |
|---|---|---|
| `age` | 18 – 89 | 0 – 120 |
| `heart_rate` | 25 – 240 | 20 – 300 |
| `systolic_bp` | 45 – 254.7 | 40 – 300 |
| `diastolic_bp` | 25 – 155 | 20 – 200 |
| `spo2` | 55 – 100 | 50 – 100 |
| `temperature` | 30.5 – 44.49 | 25 – 45 |
| `respiratory_rate` | 5 – 58 | 4 – 80 |

A record inside the accepted range but outside this support is an **extrapolation**: the
model still answers, often with near-100% confidence, but that confidence is not
meaningful. Such responses are not rejected — they are labelled. `outside_training_range`
lists each violating field with its observed bounds, and `distribution_warning` carries a
human-readable sentence that the UI renders as an amber banner.

This was added after a live report in which `temperature 45 °C` on a 100-year-old
returned `MODERATE` at 99.2% confidence with `CRITICAL` at 0.0%. Both values are outside
the training support, so the confident answer was an artefact of extrapolation, not
evidence.

### 10.3 Red-flag invariant

The constraint described in Step 7b is a serving rule, not a validation rule, so it
never rejects a request. A record with `unconsciousness = 1` is always scored, but
its result is confined to HIGH or CRITICAL and the response sets
`applied_constraint: "unconsciousness"`. `unconsciousness = 0` (or an omitted flag)
returns `applied_constraint: null`. The rule is exercised at healthy, critical,
minimum, all-symptom and coerced (`"1"`, `true`) inputs by
`scripts/api_edge_check.py`, and in both single-record and batch paths by
`tests/test_pipeline.py`.

### 10.4 Error shape

```json
{
  "error": "Vitals out of range",
  "issues": [
    {
      "field": "temperature",
      "value": 98.6,
      "min": 25,
      "max": 45,
      "message": "temperature looks like Fahrenheit; send Celsius or add \"temperature_unit\": \"F\""
    }
  ]
}
```

Malformed JSON returns `400 {"error": "Invalid JSON body"}`; a non-object body returns
`400 {"error": "Request body must be a JSON object of vitals"}`. Unexpected failures
return `500` with a message, never a fabricated prediction.

### 10.4 Fahrenheit handling

```bash
curl -X POST .../api/predict -H 'Content-Type: application/json' \
  -d '{"temperature": 98.6, "temperature_unit": "F", ...}'
```

Without the unit, 98.6 is rejected with the hint above. This matters because 98.6 °F is
a perfectly normal body temperature and, unscaled, it reads as a lethal fever — before
validation existed it produced `CRITICAL` at 100% confidence.

---

## 11. API reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Interactive demo UI |
| `/api/predict` | POST | Vitals JSON → risk class, priority, probabilities |
| `/api/predict` | GET | Health check, `accepted_ranges`, `temperature_unit`, sample prediction |
| `/api/metrics` | GET | Classical metrics, deep metrics, dataset and pipeline stats |

All responses include permissive CORS headers (`*`) and `OPTIONS` returns `204`.

### Example request

```bash
curl -X POST https://your-app.vercel.app/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"age":67,"heart_rate":142,"systolic_bp":72,"diastolic_bp":44,
       "spo2":84,"temperature":39.4,"respiratory_rate":34,
       "chest_pain":1,"breathing_difficulty":1,"dizziness":1}'
```

### Example response

```json
{
  "predicted_risk_class": "CRITICAL",
  "priority": "P0",
  "confidence": 0.9912,
  "probabilities": {"NORMAL": 0, "MODERATE": 0, "HIGH": 0.0088, "CRITICAL": 0.9912},
  "alert_color": "#ef4444",
  "disclaimer": "EDUCATIONAL SIMULATION ONLY — NOT FOR CLINICAL USE",
  "distribution_warning": null,
  "outside_training_range": [],
  "applied_constraint": null
}
```

`applied_constraint` names a data-derived rule that reshaped the probabilities
(currently only `"unconsciousness"`, see §7 Step 7b), or `null` when no rule applied.

`distribution_warning` is a sentence and `outside_training_range` is a list of
`{field, value, observed_min, observed_max}` when the input leaves the training support
(see §10.2). Example for a 100-year-old at 45 °C:

```json
{
  "predicted_risk_class": "MODERATE",
  "confidence": 0.9919,
  "distribution_warning": "Outside training distribution: age=100 (trained on 18–89), temperature=45 (trained on 30.5–44.49). The score is an extrapolation, not a validated result.",
  "outside_training_range": [
    {"field": "age", "value": 100, "observed_min": 18, "observed_max": 89},
    {"field": "temperature", "value": 45, "observed_min": 30.5, "observed_max": 44.49}
  ]
}
```

### Offline inference (Python)

```python
import yaml
from src.inference.predictor import HealthRiskPredictor
from src.inference.alert_engine import SimulatedAlertEngine

config = yaml.safe_load(open("config.yaml"))

predictor = HealthRiskPredictor.from_saved_model(
    model_path    = "models/classical/random_forest.joblib",
    pipeline_path = "models/preprocessing_pipeline.pkl",
    config        = config,
)

record = {
    "case_id": "DEMO_001", "age": 67, "sex": 1,
    "heart_rate": 142, "systolic_bp": 72, "diastolic_bp": 44,
    "spo2": 84, "temperature": 39.4, "respiratory_rate": 34,
    "chest_pain": 1, "breathing_difficulty": 1,
    "dizziness": 1, "unconsciousness": 0, "bleeding_injury": 0,
}

result = predictor.predict_record(record)
# → {'predicted_risk_class': 'CRITICAL', 'confidence': 0.98, 'probabilities': {...}}

alert = SimulatedAlertEngine(config).generate_alert(result)
# → {'priority': 'P0', 'alert_level': 'EMERGENCY_SIMULATION_CRITICAL', ...}
```

---

## 12. Verification methodology

### 12.1 TypeScript ↔ Python parity

Parity is checked at two levels.

**A. Exported parameters vs scikit-learn, on every held-out row.** The JSON decision
function is applied to all **15,000** rows of `X_tab_test.npy` (already standardized)
and compared with `LogisticRegression.predict` / `predict_proba`:

| Check | Result |
|---|---|
| Records compared | 15,000 |
| Label differences | **0** |
| Maximum probability difference | **1.27 × 10⁻⁶** |

**B. Full HTTP chain.** 600 raw held-out records from `data/raw/tabular_raw.csv`
(120 of them with `unconsciousness = 1`, so the constraint path is included) were
posted to `/api/predict` and compared with the Python `PreprocessingPipeline` +
`LogisticRegression` + `apply_data_constraints`:

| Check | Result |
|---|---|
| Records compared | 600 (0 rejected) |
| Label mismatches | **0** |
| Maximum probability difference | **0** (both sides round to 4 dp) |
| `applied_constraint` mismatches | **0** |

Raw decision-function outputs agree to **1.27 × 10⁻⁶**. When the 4-decimal API
response is compared cell-by-cell with scikit-learn's float32 probabilities over all
15,000 rows (60,000 cells), **5 cells differ in the last digit**: each sits exactly on
an `x.xxxx5` rounding tie, where float32 and float64 land on opposite sides. No label
changes, and confidence differs in 4 rows. Rounded probabilities of 163 rows sum to
1 ± 1 × 10⁻⁴ rather than exactly 1, because four independent 4-dp roundings cannot
always close to 1. This is display rounding, not model drift.

Together these cover normalization, the strict coercion layer, feature engineering,
imputation, scaling, coefficients, softmax and the red-flag constraint — not just the
coefficients.

### 12.2 Deployed-parameter integrity

`web/lib/model-params.json` is scored against `data/processed/X_tab_test.npy` on every
export:

```
Held-out accuracy from exported params: 0.9703
```

The script exits non-zero below **0.965** accuracy or when the held-out split has
fewer than **8,000** rows (a smoke-test or partial dataset), and prints
retraining instructions. `--allow-small` overrides the row-count check. It also
generates `web/lib/training-support.json` from the raw training CSV, so the
extrapolation bounds are measured rather than hand-copied — that is how the true
temperature maximum was found to be 44.49 °C, not the 44.5 °C previously written by
hand. Both files are written atomically and only after the accuracy check passes.

**Why the guard exists.** `pytest` previously constructed `ClassicalTrainer` with the
real `config.yaml`, so each test run overwrote `models/classical/*.joblib` and
`experiments/results/classical_results.json` with 200-sample smoke models. An export
taken after such a run shipped a model trained on ~140 records: **92.05%** test
accuracy behind a page claiming 97.03%. Fixes:

1. the test's trainer now writes to `tmp_path`, verified by comparing artifact
   checksums before and after `pytest`;
2. classical models were retrained on the full dataset, reproducing
   95.86 / 97.03 / 96.61 / 98.05 exactly;
3. the JSON was re-exported (0.9703 verified);
4. the export guard was added so the failure cannot recur silently.

### 12.3 Live deployment checks

Performed with HTTP probes and headless Chrome against the Vercel URL:

| Check | Result |
|---|---|
| `/` and all 8 `_next` assets | `200` |
| Hydration / console errors | 0 errors, 0 failed requests |
| Preset buttons + Predict | functional |
| CORS preflight | `204` with expected headers |
| Invalid JSON / array body | `400` |
| Mobile layout at 390 px | no horizontal overflow |

> Operational note: the first live check was run against the stale build described in
> 12.2, so the numbers above reflect the corrected local build plus the original
> deployment's routing/asset health. Re-run the probes after the next deploy.

---

## 13. Reproducibility

| Setting | Value |
|---|---|
| Random seed | 42 (generators, NumPy, scikit-learn, PyTorch) |
| Split | 70% / 15% / 15%, stratified |
| Tabular generator | 25,000 samples per class |
| Scaling | `StandardScaler`, fit on train only |
| Label encoding | NORMAL=0, MODERATE=1, HIGH=2, CRITICAL=3 |
| Configuration | `config.yaml` (single source of truth for paths and hyperparameters) |

Retraining reproduces the published metrics exactly, which is how the 95.86 / 97.03 /
96.61 / 98.05 figures were re-confirmed after the artifact incident.

---

## 14. Test strategy — 81/81 passing

```bash
pytest tests -q      # 81 passed in ~7 s
```

| Suite | Tests | Coverage |
|---|---|---|
| `TestTabularGenerator` | 7 | shape, class balance, ranges, physiological validity |
| `TestTemporalGenerator` | 6 | shapes, class balance, label range, physiological values, dtypes |
| `TestSignalGenerator` | 5 | shape, labels, class balance, single-signal invariants |
| `TestFeatureEngineering` | 3 | derived columns, pulse-pressure formula, binary flags |
| `TestPreprocessingPipeline` | 6 | fit/transform, NaN-free output, leakage guard, save/load |
| `TestDataSplitters` | 4 | sizes, no `case_id` overlap, no temporal leakage, missing target raises |
| `TestClassicalTraining` | 3 | all four models present, metrics in [0,1], better than random |
| `TestInferencePredictor` | 13 | result keys, valid class, probabilities sum to 1, batch mode, unconsciousness constraint (single + batch), no constraint without the flag, missing-field default, malformed/out-of-range/inverted-BP rejection, equality BP, string coercion |
| `TestAlertEngine` | 12 | P0–P3 mapping, disclaimer, unknown/missing class raises, unique event ids |
| `TestDataValidator` | 5 | schema, report structure, duplicates, range violations, missing target is invalid |
| `TestDataQualityGate` | 6 | clean data passes; schema/duplicates/nulls/ranges abort; override |
| `TestValidationParity` | 1 | Python accepted ranges match `web/lib/ranges.ts` |
| Schema/config/import units | 10 | schema definitions, config parsing, module imports, annotation resolution |

Tests are **hermetic with respect to artifacts**: model training inside tests writes to
a temporary directory, never to `models/` or `experiments/`.

`scripts/api_edge_check.py` complements the unit suite with an end-to-end sweep of a
running server (160 checks): every numeric field at min−1 / min / max / max+1 and far
outside, type confusion, Fahrenheit variants, malformed bodies, CORS, cross-field blood
pressure, the extrapolation-warning boundaries, the unconsciousness red-flag constraint,
determinism under 48-way concurrency, and the metrics endpoint. It reads the generated support table, so it cannot drift from
the app:

```bash
python3 scripts/api_edge_check.py --base http://localhost:3000
```

---

## 15. Project structure

```
ai-emergency_health_assistant/
├── Dockerfile                     # Single-command build & run
├── docker-compose.yml             # Volume-mounted compose setup
├── generate_data.py               # Standalone data generation script
├── run_pipeline.py                # End-to-end pipeline orchestrator
├── config.yaml                    # All hyperparameters & paths (seed: 42)
├── requirements.txt               # Python dependencies
├── SYSTEMDESIGN.md                # This document
├── README.md                      # Goals, objectives, achievements
├── architecture.md                # Visual architecture overview
├── BUILD_STATUS.md                # Completion status + model results
├── train_classical.py             # Standalone entry points (per model family)
├── train_cnn.py
├── train_rnn.py
├── train_lstm.py
│
├── data/
│   ├── raw/                       # tabular_raw.csv, temporal_*.npy, signal_*.npy
│   ├── interim/                   # intermediate processed files
│   └── processed/                 # X_tab_*.npy, y_tab_*.npy arrays
│
├── models/
│   ├── preprocessing_pipeline.pkl # Fitted scaler + imputer + label encoder
│   ├── classical/                 # knn.joblib, logistic_regression.joblib, ...
│   └── deep/                      # cnn_1d_best.pt, rnn_best.pt, lstm_best.pt
│
├── experiments/results/
│   ├── classical_results.json
│   ├── deep_results.json
│   └── pipeline_summary.json
│
├── src/
│   ├── data/
│   │   ├── tabular_generator.py   # Scenario-based tabular synthesis
│   │   ├── temporal_generator.py  # Markov latent-state trajectory synthesis
│   │   ├── signal_generator.py    # P/QRS/T ECG waveform synthesis
│   │   ├── validator.py           # Schema + range + duplicate validation
│   │   └── schema.py              # Feature definitions & RiskClass enum
│   ├── preprocessing/
│   │   ├── feature_engineering.py # 9 derived clinical features
│   │   ├── pipeline.py            # Impute → scale → label encode
│   │   └── splitters.py           # Stratified / ID-based splits
│   ├── models/
│   │   ├── classical/             # KNN, LR, DT, RF sklearn wrappers
│   │   └── deep/                  # Physio1DCNN, PhysioRNN, PhysioLSTM
│   ├── training/
│   │   ├── train_classical.py     # Classical ML training orchestrator
│   │   └── train_deep.py          # PyTorch deep learning trainer
│   ├── evaluation/
│   │   └── metrics.py             # Accuracy, F1, confusion matrix, reports
│   ├── explainability/
│   │   ├── feature_importance.py  # Tree importances, LR coefficients
│   │   └── shap_explainer.py      # SHAP TreeExplainer integration
│   ├── inference/
│   │   ├── predictor.py           # Single + batch risk inference
│   │   └── alert_engine.py        # P0–P3 simulated alert routing
│   └── utils/
│       ├── logger.py              # Structured logging
│       ├── config_loader.py       # YAML config parsing
│       └── seed.py                # Global seed setter
│
├── web/                           # Next.js 16 demo (deployed on Vercel)
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx               # Interactive vitals form + results
│   │   └── api/
│   │       ├── predict/route.ts   # POST /api/predict — live inference
│   │       └── metrics/route.ts   # GET  /api/metrics — training results
│   ├── components/
│   │   └── TrainingResults.tsx    # Accuracy bars, confusion matrix, curves, rationale
│   ├── lib/
│   │   ├── inference.ts           # Feature engineering + LR in TypeScript
│   │   ├── model-params.json      # Exported scaler/imputer/coef parameters
│   │   └── training-results.json  # Metrics + curves exported for the charts
│   ├── public/                    # Static assets (currently empty)
│   ├── package.json
│   └── next.config.mjs
│
├── notebooks/                     # Exploration space (currently empty)
├── reports/                       # Report output directory (currently empty)
│
├── scripts/
│   ├── export_web_model_params.py # Re-export model JSON + accuracy guard
│   └── export_web_training_results.py # Refresh home-screen charts from artifacts
│
└── tests/
    ├── test_pipeline.py           # 71 integration tests
    ├── test_schema.py             # Schema & feature definition tests
    ├── test_config.py             # Config parsing tests
    └── test_imports.py            # Module import sanity tests
```

---

## 16. Deployment procedure

### Option A — Vercel dashboard (recommended)

1. Push the repository to GitHub.
2. Import it at [vercel.com/new](https://vercel.com/new).
3. **Set Root Directory to `web`** — the only change required.
4. Framework preset auto-detects Next.js; keep default build settings.
5. Deploy. No environment variables are needed.

### Option B — Vercel CLI

```bash
cd web
npm install
npx vercel --prod
```

### Requirements and verification

- Node.js ≥ 20.9 (Vercel's default Node 22 is fine); the app currently pins Next.js
  16.4.6.
- After deploying, confirm the new contract:

```bash
curl -s https://your-app.vercel.app/api/predict | jq '.status, .accepted_ranges'
```

- Do **not** deploy from the repository root: the root also contains the training
  pipeline and unrelated `backend/`/`frontend/` folders that would confuse framework
  detection.

### Re-exporting the served model after retraining

```bash
python3 run_pipeline.py --classical-only --skip-generate
python3 scripts/export_web_model_params.py      # fails if accuracy < 0.965
```

### Smoke-testing without touching production artifacts

`--quick` writes every output under a `quick/` subdirectory
(`data/raw/quick`, `data/processed/quick`, `models/quick`, `experiments/quick`).
A two-epoch run can therefore never overwrite the published datasets, checkpoints
or result JSON with a 1,000-row toy model. The export script keeps reading only
the canonical `models/` and `data/processed/` paths.

```bash
python3 run_pipeline.py --quick
```

---

## 17. Operational notes and known limitations

| Item | Status / action |
|---|---|
| Deep model weights | Git-ignored by design (≈1.4 MB); retrain locally. Not served on Vercel. |
| Served model | Logistic Regression only; other models' metrics are reported, not served. |
| `backend/.env` | Untracked but present in git history — rotate `SECRET_KEY` and the DB password. |
| Unrelated folders | `backend/`, `frontend/` are a separate auth demo; excluded from deployment by Root Directory `web`. |
| No auth on `/api/*` | Intentional for a public demo; add rate limiting before advertising the API. |
| Data provenance | Fully synthetic; no real patient data is ever used or required. |
| Zero pulse pressure | 15 / 100,000 training rows have `diastolic == systolic` from one-decimal rounding (Section 3.1); the model saw them, the API accepts equality and rejects inversion. |
| Clinical use | Not permitted. The disclaimer is present in code, UI, and every API response. |

## 18. Design decisions worth defending

| Decision | Alternative rejected | Why |
|---|---|---|
| Freeze the model into JSON parameters | Ship a pickle/ONNX/PyTorch bundle | Vercel functions stay dependency-free, cold start is negligible, and the decision function is auditable |
| Serve LR, report RF | Serve the most accurate tabular model | RF would need a tree evaluator or a multi-MB runtime for +1.02 pp |
| Validate ranges server-side | Trust the HTML form | The API is public; native constraints are bypassable |
| Celsius default + explicit unit | Accept both silently | Silent unit confusion produced a 100%-confidence CRITICAL reading for a healthy adult |
| Test artifacts in `tmp_path` | Let tests reuse `models/` | A test run previously overwrote production artifacts (Section 12.2) |
| Keep the 22-feature pipeline in TS | Reimplement engineered features differently | Parity must be exact, not approximate |
| CPU-only training | Require a GPU | Reproducibility on any laptop matters more than an 18-minute run being 5 minutes |

---

*End of technical design record. For goals, objectives, achievements, and a narrative
summary of the work, see [`README.md`](README.md).*
