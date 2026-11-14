# AI-Based Emergency Health Detection and Alert System
## Research & Educational Software Simulation Pipeline

> [!IMPORTANT]
> **EDUCATIONAL SOFTWARE SIMULATION DISCLAIMER**
> This project is strictly an **educational software simulation** built on **fully synthetic data**. It is NOT a medical device. It does NOT provide diagnosis, treatment recommendations, clinical decisions, real sensor data acquisition, ambulance dispatch, or hospital integration. All physiological records are procedurally generated and statistically calibrated — no real patient data is used anywhere.

---

## ⚡ Run Everything with One Command

```bash
docker compose up --build
```

That single command will:
1. Build the Python environment
2. Generate **100,000 tabular** + **20,000 temporal** + **20,000 signal** synthetic records
3. Train all 4 classical ML models (KNN, Logistic Regression, Decision Tree, Random Forest)
4. Run the full **45-test suite** and report results

All trained models and data files are saved to your local `data/`, `models/`, and `experiments/` directories via Docker volume mounts.

### Other Useful Commands

```bash
# Run only data generation (no training)
docker compose run pipeline python3 generate_data.py

# Run only training (uses pre-generated data)
docker compose run pipeline python3 run_pipeline.py --classical-only --skip-generate

# Quick smoke test (small dataset, fast)
docker compose run pipeline python3 run_pipeline.py --quick --classical-only

# Just run the test suite
docker compose run pipeline python3 -m pytest tests/test_pipeline.py -v

# Without Docker (local Python 3.9+)
pip install -r requirements.txt
python3 generate_data.py
python3 run_pipeline.py --classical-only --skip-generate
```

---

## 🚀 Live Web Demo

The interactive demo is a **Next.js 16** app in [`web/`](web). The trained
Logistic Regression model is exported as plain JSON parameters and served by
native Next.js API routes — no Python runtime, no pickles, no cold-start model
download.

```bash
cd web
npm install
npm run dev        # http://localhost:3000
```

Requires **Node.js 20.9+** (Vercel's default Node 22 works).

**Deploying to Vercel?** See [Section 11](#11-deploy-to-vercel) — it takes about
two minutes.

---

## 1. Project Objective

This project constructs a modular, production-quality, reproducible ML pipeline that simulates physiological anomaly detection and emergency alert priority generation across **three complementary synthetic data modalities**:

| Modality | Description | Volume | Model Type |
|---|---|---|---|
| **Tabular** | Static physiological vitals + symptoms | 100,000 records | KNN, LR, DT, RF |
| **Temporal** | 60-step vital-sign trajectories | 20,000 sequences | RNN, LSTM (PyTorch) |
| **Signals** | ECG-like waveforms at 500 Hz | 20,000 signals | 1D CNN (PyTorch) |

---

## 2. System Architecture

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

---

## 3. Synthetic Data Design

### Risk Classes
| Class | Label | Records |
|---|---|---|
| Normal | 0 | 25,000 |
| Moderate | 1 | 25,000 |
| High | 2 | 25,000 |
| Critical | 3 | 25,000 |

### Raw Features (14 columns)
`case_id`, `age`, `sex`, `heart_rate`, `systolic_bp`, `diastolic_bp`, `spo2`, `temperature`, `respiratory_rate`, `chest_pain`, `breathing_difficulty`, `dizziness`, `unconsciousness`, `bleeding_injury`

### Derived Features (9 — added during preprocessing)
| Feature | Formula |
|---|---|
| `pulse_pressure` | SBP − DBP |
| `mean_arterial_pressure` | DBP + PP/3 |
| `shock_index` | HR / SBP |
| `age_group` | Ordinal bin: 0=child, 1=adult, 2=senior |
| `spo2_severity` | Ordinal bin: 0=severe, 1=mild, 2=normal |
| `tachycardia_flag` | 1 if HR > 100 |
| `hypotension_flag` | 1 if SBP < 90 |
| `fever_flag` | 1 if Temp > 38.0 °C |
| `tachypnea_flag` | 1 if RR > 20 |

**Total feature dimension: 22**

---

## 4. Training Results (100K production dataset)

### Classical ML — Test Set Performance

| Model | Val Accuracy | Val Macro F1 | Test Accuracy | Test Macro F1 |
|---|---|---|---|---|
| KNN (k=5) | 95.86% | 0.9586 | 95.86% | 0.9586 |
| Logistic Regression | 97.06% | 0.9706 | 97.03% | 0.9703 |
| Decision Tree (depth 10) | 96.41% | 0.9641 | 96.61% | 0.9662 |
| **Random Forest** ⭐ | **98.09%** | **0.9809** | **98.05%** | **0.9805** |

> Train / Val / Test split: 70,000 / 15,000 / 15,000 (stratified by class)

---

## 5. Test Coverage — 54/54 Passing ✅

45 integration tests plus 9 schema/config/import unit tests (`pytest tests`).

```
TestTabularGenerator    7 tests  — shape, class balance, ranges, physio validity
TestTemporalGenerator   5 tests  — shapes, label range, physio values, dtypes
TestSignalGenerator     5 tests  — shape, labels, balanced classes, single signal
TestFeatureEngineering  3 tests  — derived columns, pulse pressure formula, binary flags
TestPreprocessingPipeline 6 tests — fit/transform, NaN-free, leakage guard, save/load
TestDataSplitters       3 tests  — sizes, no case_id overlap, no temporal leakage
TestClassicalTraining   3 tests  — all models present, metrics in [0,1], beat random
TestInferencePredictor  4 tests  — prediction keys, valid class, proba sum=1, batch
TestAlertEngine         5 tests  — P0/P1/P2/P3 mapping, disclaimer present
TestDataValidator       4 tests  — schema, report structure, no dups, range violations

Total: 54 passed (45 integration + 9 unit) in ~4s
```

---

## 6. Project Structure

```
ai-emergency_health_assistant/
├── Dockerfile                     # Single-command build & run
├── docker-compose.yml             # Volume-mounted compose setup
├── generate_data.py               # Standalone data generation script
├── run_pipeline.py                # End-to-end pipeline orchestrator
├── config.yaml                    # All hyperparameters & paths (seed: 42)
├── requirements.txt               # Python dependencies
│
├── data/
│   ├── raw/                       # tabular_raw.csv, temporal_*.npy, signal_*.npy
│   ├── interim/                   # intermediate processed files
│   └── processed/                 # X_tab_*.npy, y_tab_*.npy arrays
│
├── models/
│   ├── preprocessing_pipeline.pkl # Fitted scaler + imputer + label encoder
│   ├── classical/                 # knn.joblib, logistic_regression.joblib, ...
│   └── deep/                      # best_cnn.pt, best_rnn.pt, best_lstm.pt
│
├── experiments/results/
│   ├── classical_results.json
│   └── pipeline_summary.json
│
├── src/
│   ├── data/
│   │   ├── tabular_generator.py   # Scenario-based tabular synthesis
│   │   ├── temporal_generator.py  # Markov latent state trajectory synthesis
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
│   │   ├── page.tsx              # Interactive vitals form + results
│   │   └── api/
│   │       ├── predict/route.ts  # POST /api/predict — live inference
│   │       └── metrics/route.ts  # GET  /api/metrics — training results
│   ├── lib/
│   │   ├── inference.ts          # Feature engineering + LR in TypeScript
│   │   └── model-params.json     # Exported scaler/imputer/coef parameters
│   ├── package.json
│   └── next.config.mjs
│
├── scripts/
│   └── export_web_model_params.py # Re-export model JSON after re-training
│
└── tests/
    ├── test_pipeline.py           # 45 comprehensive integration tests
    ├── test_schema.py             # Schema & feature definition tests
    ├── test_config.py             # Config parsing tests
    └── test_imports.py            # Module import sanity tests
```

---

## 7. Inference Example

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

# A simulated critical patient case
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

## 8. Alert Priority Mapping

| Predicted Risk | Priority | Simulation Alert Level |
|---|---|---|
| CRITICAL | **P0** | EMERGENCY_SIMULATION_CRITICAL |
| HIGH | **P1** | EMERGENCY_SIMULATION_HIGH |
| MODERATE | **P2** | SIMULATION_ADVISORY_MODERATE |
| NORMAL | **P3** | SIMULATION_STATUS_NORMAL |

---

## 9. Deep Learning Models (PyTorch)

All three deep models are **implemented and trained** on the full synthetic
datasets (14,000 train / 3,000 val / 3,000 test):

| Model | Input | Architecture | Val Accuracy | Test Accuracy | Test Macro F1 | Train Time |
|---|---|---|---|---|---|---|
| **Physio1DCNN** ⭐ | (batch, 1, 500) signal | Conv1D → BN → Pool × 2 → AdaptivePool → FC | **99.73%** | **99.73%** | **0.9973** | 117 s |
| **PhysioRNN (GRU)** | (batch, 60, 6) sequence | 2-layer GRU → last hidden → FC | 94.03% | 93.70% | 0.9362 | 530 s |
| **PhysioLSTM** | (batch, 60, 6) sequence | 2-layer LSTM → last hidden → FC | 92.93% | 92.40% | 0.9226 | 436 s |

> 20 epochs each, Adam (lr = 0.001), CrossEntropyLoss, batch size 128, best
> checkpoint selected by validation accuracy. Training was run on CPU (PyTorch
> 2.15 CPU build); full run ≈ 18 minutes.

Best checkpoints are written to `models/deep/` (`cnn_1d_best.pt`,
`rnn_best.pt`, `lstm_best.pt`) and per-epoch history + metrics to
`experiments/results/deep_results.json`. Model weights are git-ignored, so
retrain after cloning:

```bash
pip install torch  # CPU-only: pip install torch --index-url https://download.pytorch.org/whl/cpu
python3 run_pipeline.py --deep-only --skip-generate
```

---

## 10. Reproducibility

| Setting | Value |
|---|---|
| Random seed | 42 (all generators, sklearn, numpy) |
| Train/Val/Test split | 70% / 15% / 15% |
| Tabular generator | 25,000 samples per class |
| Feature scaling | StandardScaler (fit on train only) |
| Label encoding | NORMAL=0, MODERATE=1, HIGH=2, CRITICAL=3 |

---

## 11. Deploy to Vercel

The deployment is a single Next.js 16 app. It needs **no environment variables,
no database, and no Python runtime**. The model runs as TypeScript inside the
`/api/predict` route, using parameters exported from the trained scikit-learn
pipeline.

### Option A — Vercel Dashboard (recommended)

1. Push this repository to GitHub.
2. Go to [vercel.com/new](https://vercel.com/new) and import the repository.
3. **Set Root Directory to `web`** (Edit → Root Directory).
   This is the only setting you must change.
4. Framework preset auto-detects **Next.js**; leave build settings at their
   defaults (`npm run build`, output `.next`).
5. Click **Deploy**. No environment variables are required.
6. Open the deployment URL — the live demo and `/api/predict` are ready.

### Option B — Vercel CLI

```bash
cd web
npm install
npx vercel --prod
```

Accept the defaults when prompted (link/create project, framework: Next.js).
Subsequent deploys only need `npx vercel --prod` or a `git push`.

### API endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Interactive demo UI |
| `/api/predict` | POST | Body = vitals JSON; returns risk class, priority & probabilities |
| `/api/predict` | GET | Health check + sample prediction |
| `/api/metrics` | GET | Pre-computed model metrics and dataset stats |

```bash
curl -X POST https://your-app.vercel.app/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"age":67,"heart_rate":142,"systolic_bp":72,"diastolic_bp":44,
       "spo2":84,"temperature":39.4,"respiratory_rate":34,
       "chest_pain":1,"breathing_difficulty":1}'
```

### Re-exporting the model after re-training

`web/lib/model-params.json` contains only the fitted scaler/imputer statistics
and Logistic Regression coefficients (~8 KB). If you retrain the models, refresh
it with:

```bash
python3 scripts/export_web_model_params.py
```

> The TypeScript inference was verified against the Python pipeline on 600+
> randomized records and bin-boundary edge cases: **0 label mismatches**, maximum
> probability difference < 0.0001 (JSON rounding).

### Notes

- The `web/` directory is self-contained: deploy it as the Vercel Root Directory.
- Do **not** deploy from the repository root — the root also contains the
  offline training pipeline, `backend/`, and `frontend/`, which are not part of
  the hosted demo and would confuse Vercel's framework detection.
- `data/`, `models/`, and `experiments/` are git-ignored and never shipped to
  Vercel.

