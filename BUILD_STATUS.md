# BUILD STATUS REPORT
## AI-Based Emergency Health Detection and Alert System

> [!IMPORTANT]
> **EDUCATIONAL SOFTWARE SIMULATION ONLY**:
> This project is strictly an educational software simulation for research and
> academic demonstration. It is NOT a medical device and does NOT provide
> diagnosis, treatment recommendations, ambulance dispatch, hospital
> integration, real sensor acquisition, GPS, or clinical decision-making.

- **Stage**: Feature complete · models trained · 69/69 tests passing · deployed on Vercel
- **Pipeline**: Data generation → validation → preprocessing → training →
  evaluation → inference → alert routing
- **Dataset**: 100,000 tabular + 20,000 temporal + 20,000 signal synthetic records
- **Classical models**: KNN, Logistic Regression, Decision Tree, Random Forest
  trained; Random Forest best at 98.05% test accuracy
- **Deep models**: Physio1DCNN, PhysioRNN (GRU), PhysioLSTM **trained** on the
  full 20K datasets; best is 1D CNN at 99.73% test accuracy
- **Tests**: 69/69 passing (`pytest tests`)
- **Web demo**: Next.js 16 in `web/`, model inference in TypeScript
- **Deployment**: Vercel (Root Directory = `web`, no env vars required)

---

## 1. What is completed

| Area | Status | Details |
|---|---|---|
| Synthetic data generators | ✅ | Scenario-based tabular, Markov temporal, P/QRS/T ECG signals |
| Validation & quality report | ✅ | Schema, ranges, duplicates, class balance |
| Feature engineering | ✅ | 14 raw → 22 features, leak-free |
| Preprocessing pipeline | ✅ | Median impute → StandardScaler → label encode, persisted |
| Splits | ✅ | 70/15/15 stratified, ID-based temporal split |
| Classical training | ✅ | KNN, LR, DT, RF with metrics + joblib persistence |
| Deep learning | ✅ | CNN/GRU/LSTM architectures trained for 20 epochs; checkpoints + history saved |
| Evaluation | ✅ | Accuracy, macro F1, per-class report, confusion matrix |
| Explainability | ✅ | Feature importances, LR coefficients, optional SHAP |
| Inference + alert engine | ✅ | Risk prediction + P0–P3 simulated priority routing |
| Docker pipeline | ✅ | `docker compose up --build` generates, trains, tests |
| Automated tests | ✅ | 69 passed (60 integration + 9 unit) |
| Web demo | ✅ | `web/` Next.js app with live prediction form |
| Serverless inference | ✅ | `web/app/api/predict` — dependency-free TypeScript model |
| Metrics API | ✅ | `web/app/api/metrics` |
| Vercel configuration | ✅ | Root Directory `web`, zero env vars, verified production build |
| Model export script | ✅ | `scripts/export_web_model_params.py` |

---

## 2. Model results

### Classical ML (tabular, 15,000-record test set)

| Model | Val Acc | Test Acc | Test Macro F1 |
|---|---|---|---|
| KNN | 95.86% | 95.86% | 0.9586 |
| Logistic Regression | 97.06% | 97.03% | 0.9703 |
| Decision Tree | 96.41% | 96.61% | 0.9662 |
| **Random Forest** ⭐ | **98.09%** | **98.05%** | **0.9805** |

### Deep learning (PyTorch, 3,000-record test set per modality)

| Model | Input | Val Acc | Test Acc | Test Macro F1 | Train Time |
|---|---|---|---|---|---|
| **Physio1DCNN** ⭐ | ECG signal (1×500) | **99.73%** | **99.73%** | **0.9973** | 117 s |
| PhysioRNN (2-layer GRU) | 60-step sequence (6 ch) | 94.03% | 93.70% | 0.9362 | 530 s |
| PhysioLSTM | 60-step sequence (6 ch) | 92.93% | 92.40% | 0.9226 | 436 s |

Artifacts: `models/deep/{cnn_1d,rnn,lstm}_best.pt`,
`experiments/results/deep_results.json`. Weights are git-ignored; retrain with
`python3 run_pipeline.py --deep-only --skip-generate`.

---

## 3. Deployment verification performed

1. `cd web && npm run build` — production build succeeds; `/` static,
   `/api/predict` and `/api/metrics` dynamic functions.
2. Started the production server and called both API routes — valid JSON.
3. **Parity check, two layers**: the exported JSON reproduced scikit-learn on all
   15,000 held-out rows (0 label differences, max probability Δ 1.27e-6), and 600 raw
   records (150 per class) sent to `/api/predict` matched the Python pipeline with
   **0 label mismatches**, 0 rejections and max probability difference **5e-5**.
4. `pytest tests -q` — 69 passed.
5. Live Vercel deployment checked with headless Chrome: page and all `_next`
   assets 200, hydration clean (0 console errors, 0 failed requests), presets and
   prediction button work, no horizontal overflow at 390 px.

### Fixed during this verification

- **Stale deployed model.** `pytest` instantiated `ClassicalTrainer` with the real
  `config.yaml`, so every test run overwrote `models/classical/*.joblib` and
  `experiments/results/classical_results.json` with 200-sample smoke models. The
  last export of `web/lib/model-params.json` happened after such a run, so the
  live site was serving a model trained on ~140 records (92.05% test accuracy)
  while displaying 97.03%. Fixed by:
  - pointing the test's `ClassicalTrainer` at `tmp_path` so it can never touch
    real artifacts;
  - retraining classical models on the full dataset (reproduces 95.86 / 97.03 /
    96.61 / 98.05 exactly) and re-exporting the JSON (verified 0.9703);
  - adding an accuracy guard to `scripts/export_web_model_params.py` that fails
    if the exported params score below 0.965 on a held-out split of at least 8,000 rows.
- **No input validation.** Out-of-range vitals (e.g. `temperature: 98.6` in °F,
  `spo2: 150`) were silently scored and returned `CRITICAL` at 100% confidence.
  `web/lib/inference.ts` now validates every field against physiological ranges
  and `/api/predict` returns `400` with per-field messages; Fahrenheit is
  accepted via `temperature_unit: "F"`. Ranges align with the real dataset
  (systolic BP down to 40 mmHg is valid, so the lower bound is 40).
- **Stale verdict beside edited inputs.** The result panel was only cleared when
  a preset was loaded or a new request succeeded. Typing an impossible value
  (e.g. `temperature: 0645`) left the previous prediction — a real
  `MODERATE · 99.2%` — on screen, so the page looked as if 645 °C had been
  scored. Fixed by:
  - validating each input as it is typed, using the **same** range table as the
    server (`web/lib/ranges.ts`), with inline `role="alert"` messages;
  - discarding the previous result on any change ("Vitals changed — run the
    prediction again");
  - stamping each result with the exact values it was scored for
    (`Scored for: age 100 · HR 75 · … · temp 45 °C`);
  - aligning the form's `min`/`max` with the API (it previously rejected valid
    values such as age 0 or 115 and temperature 27 °C).
- **Silent extrapolation.** Accepted ranges are wider than the training data
  (trained on ages 18–89, temperatures 30.5–44.49 °C). A 100-year-old at 45 °C
  therefore returned `MODERATE` at 99.2% with `CRITICAL` at 0.0% — a confident
  answer built on extrapolation. Responses now include
  `distribution_warning` and `outside_training_range`, surfaced as an amber
  banner above the probability bars.
- **Malformed input silently became a healthy default.** `{"heart_rate": "abc"}`,
  `"80abc"`, `"0x50"` (parsed as hex 80), arrays, objects, `"NaN"`, `"Infinity"`,
  `1e309`, `sex: 0.5`, `sex: true` and dropped symptom booleans were all replaced by
  healthy-adult defaults, so a typo produced `NORMAL` at 98.8%. Rewritten as a strict
  coercion layer: absent values get a default, present-but-unparseable values get
  `400`. Booleans are accepted for symptom flags only, and `sex` is strictly `0`/`1`.
- **Symptom toggles were mouse-only and double-fired.** `display: none` removed the
  checkboxes from the tab order, and a label `onClick` plus the forwarded input click
  cancelled each other. They are now real focusable checkboxes with `onChange` and a
  `:focus-within` outline: Space toggles, a label click toggles exactly once.
- **Double submit and late responses.** Three rapid Enters could fire two requests, and
  editing during a request let the late response repopulate the panel. A synchronous
  in-flight ref limits it to one request, and a `latestVitals` ref discards a response
  if the form changed mid-flight.
- **Inverted blood pressure accepted.** `diastolic > systolic` is now rejected (400)
  by the API and inline in the form; equality is accepted because 15 training rows
  round to it (System design §3.1).
- **The export guard wrote before verifying.** A 400-sample smoke model scoring 0.9549
  passed the old 0.95 threshold and overwrote the committed parameters. The check now
  runs **before** any write, requires 0.965 accuracy and ≥8,000 held-out rows
  (`--allow-small` overrides), and writes atomically. Both refusal paths were tested
  and left `model-params.json` byte-identical.
- **The extrapolation table was hand-copied and wrong.** `training-support.json` is now
  generated from the raw training CSV by the export script; the real maximum
  temperature is 44.49 °C, not 44.5 °C.
- **Pipeline CLI failed silently.** `--skip-generate` generated 83 MB when data was
  missing, `--classical-only --deep-only` were not mutually exclusive, and a missing
  config crashed. These now exit 1/2 with an actionable message. A data-quality gate
  aborts training on an invalid schema, duplicates, nulls or out-of-range values
  (`--allow-data-issues` overrides).
- **Symptoms could make a patient look healthier.** The displayed model gave
  `unconsciousness` a positive NORMAL coefficient (a degenerate optimum: the flag
  occurs in 0 of 50,000 low-risk training rows), so healthy vitals + unconscious
  returned `NORMAL 100%` and all five symptoms returned `NORMAL 99.9%`. The serving
  layer now masks NORMAL/MODERATE whenever that flag is set, renormalizes over
  HIGH/CRITICAL, and reports `applied_constraint`. The mask changes 0 of 15,000 test
  predictions, so 97.03% is untouched.
- **Alerting failed open.** `generate_alert` mapped an unknown or missing risk class to
  P3 `NORMAL`; it now raises `ValueError`, so a label bug cannot report the
  least-urgent tier.
- **Verification after the fixes.** 160-check API edge sweep passes (every field at
  min−1/min/max/max+1, type confusion, Fahrenheit, malformed bodies, CORS, 48-way
  concurrency, warning boundaries); exported parameters match scikit-learn on all
  15,000 held-out rows; 600 live HTTP records match with 0 label differences; a full
  classical retrain reproduces every metric and every joblib checksum exactly in ~49 s;
  and all three deep checkpoints reload to their published accuracies.

---

## 4. Deploy in two minutes

1. Push to GitHub.
2. Import the repo at [vercel.com/new](https://vercel.com/new).
3. Set **Root Directory** to `web`.
4. Deploy — no environment variables, database, or Python runtime needed.

Full instructions, API examples, and re-training notes are in
[`SYSTEMDESIGN.md` §16](SYSTEMDESIGN.md#16-deployment-procedure). For goals and
achievements, see [`README.md`](README.md).

---

## 5. Optional / future work

1. Add CNN/GRU/LSTM inference to the web API (the trained checkpoints are
   PyTorch; the hosted demo intentionally ships the tiny Logistic Regression
   model for a dependency-free deployment).
2. Add authentication or rate limiting if the demo is made public at scale.
3. Replace the synthetic data generator with consented, de-identified data in a
   research setting (requires IRB/ethics approval) — the system is not a medical
   device as shipped.
