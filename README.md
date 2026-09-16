# AI-Based Emergency Health Detection and Alert System
## Research & Educational Software Simulation Pipeline

> [!IMPORTANT]
> **EDUCATIONAL SOFTWARE SIMULATION DISCLAIMER**:
> This project is strictly an **EDUCATIONAL SOFTWARE SIMULATION**, NOT a medical device. It does **NOT** provide medical diagnosis, treatment recommendations, ambulance dispatch, hospital system integration, real physiological sensor hardware acquisition, GPS location tracking, clinical decision-making, or clinical claims.

---

## 1. Project Objective

The primary goal of this research project is to construct a modular, production-quality, reproducible Machine Learning and Deep Learning pipeline to simulate physiological anomaly detection and alert priority generation from synthetic health data. 

The pipeline explores algorithm performance across three complementary data modalities:
1. **Classical Tabular Features**: Static patient physiological vitals and symptoms.
2. **Temporal Sequences**: 60-timestep patient vitals trajectories simulating physiological deterioration/recovery dynamics.
3. **Physiological Signals**: Synthetic ECG-like wave morphology for 1D CNN pattern classification.

---

## 2. Scope & Software-Only Disclaimer

- **Strictly Fictional Synthetic Data**: All datasets are synthetically generated using scenario-based statistical models.
- **No Real Sensor Hardware**: Vitals are generated programmatically using physiological boundary distributions.
- **Simulated Priority Routing**: Alert outputs (P0, P1, P2, P3) represent software simulation flags for algorithmic experimentation only.

---

## 3. System Architecture

```
                                  +---------------------------------------+
                                  |         config.yaml (Seed: 42)        |
                                  +-------------------+-------------------+
                                                      |
            +-----------------------------------------+-----------------------------------------+
            |                                         |                                         |
            v                                         v                                         v
+-----------------------+                 +-----------------------+                 +-----------------------+
|  Tabular Generator    |                 |   Temporal Generator  |                 |    Signal Generator   |
| (100k planned cases)  |                 | (20k planned seqs)    |                 | (20k planned signals) |
+-----------+-----------+                 +-----------+-----------+                 +-----------+-----------+
            |                                         |                                         |
            v                                         v                                         v
+-----------------------+                 +-----------------------+                 +-----------------------+
| Feature Engineering   |                 | Patient-Level Split   |                 | 1D Signal Processing  |
| & Preprocessing       |                 | (70 / 15 / 15)        |                 | & Preprocessing       |
+-----------+-----------+                 +-----------+-----------+                 +-----------+-----------+
            |                                         |                                         |
            v                                         v                                         v
+-----------------------+                 +-----------------------+                 +-----------------------+
|  Classical Models     |                 |  RNN & LSTM Models    |                 |     1D CNN Model      |
|  KNN / Logistic Reg / |                 | (Temporal Trajectory) |                 | (ECG Wave Pattern)    |
|  Tree / Random Forest |                 |                       |                 |                       |
+-----------+-----------+                 +-----------+-----------+                 +-----------+-----------+
            |                                         |                                         |
            +-----------------------------------------+-----------------------------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |         Model Evaluation Suite        |
                                  | Macro/Weighted F1, Recall, Precision  |
                                  +-------------------+-------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |       Explainability & Simulation     |
                                  | Importances, Coefs, SHAP, Alerts(P0-3)|
                                  +---------------------------------------+
```

---

## 4. Data Pipeline & Schema

### Target Risk Classes
- `NORMAL` (25,000 planned cases)
- `MODERATE` (25,000 planned cases)
- `HIGH` (25,000 planned cases)
- `CRITICAL` (25,000 planned cases)

### Raw Features (14)
`case_id`, `age`, `sex`, `heart_rate`, `systolic_bp`, `diastolic_bp`, `spo2`, `temperature`, `respiratory_rate`, `chest_pain`, `breathing_difficulty`, `dizziness`, `unconsciousness`, `bleeding_injury`

### Derived Features (9)
- `pulse_pressure` ($\text{SBP} - \text{DBP}$)
- `mean_arterial_pressure` ($\text{DBP} + \frac{\text{Pulse Pressure}}{3}$)
- `shock_index` ($\frac{\text{Heart Rate}}{\text{SBP}}$)
- `age_group` (`child`, `adult`, `senior`)
- `spo2_severity` (`normal`, `mild_hypoxia`, `severe_hypoxia`)
- `tachycardia_flag` ($\text{HR} > 100$)
- `hypotension_flag` ($\text{SBP} < 90$)
- `fever_flag` ($\text{Temp} > 38.0^\circ\text{C}$)
- `tachypnea_flag` ($\text{RR} > 20$)

---

## 5. Planned Datasets & Models

### Datasets
1. **Tabular Dataset**: 100,000 cases (balanced across 4 risk classes).
2. **Temporal Sequences**: 20,000 patient trajectories (60 time steps $\times$ 6 features).
3. **Physiological Signals**: 20,000 ECG-like waveforms with P wave, QRS complex, T wave, baseline wander, and noise.

### Machine Learning Models
- **Classical ML**:
  - K-Nearest Neighbors (KNN)
  - Logistic Regression
  - Decision Tree Classifier
  - Random Forest Classifier
- **Deep Learning (PyTorch)**:
  - 1D Convolutional Neural Network (1D CNN) for signal patterns
  - Recurrent Neural Network (RNN) for temporal sequences
  - Long Short-Term Memory Network (LSTM) for temporal sequences

---

## 6. Evaluation Methodology & Explainability

- **Metrics**: Accuracy, Precision (Macro/Weighted), Recall (Macro/Weighted), Macro F1, Weighted F1, Confusion Matrix, Per-class breakdown, and Error Analysis.
- **Explainability**: Tree Feature Importances, Logistic Regression Coefficients, and optional SHAP analysis behind a feature flag (`explainability.shap_enabled`).
- **Simulated Alert Priority**:
  - `CRITICAL` $\rightarrow$ Priority **P0**
  - `HIGH` $\rightarrow$ Priority **P1**
  - `MODERATE` $\rightarrow$ Priority **P2**
  - `NORMAL` $\rightarrow$ Priority **P3**

---

## 7. Current Development Status & Future Stages

- **Current Status**: **INITIAL DEVELOPMENT STAGE**
  - Repository architecture created.
  - Complete data schema defined and implemented.
  - Preprocessing, generators, models, and evaluation skeletons established.
  - Configuration (`config.yaml`) and seed reproducibility utilities implemented.
  - Schema, config, and import unit tests verified.
- **Next Future Stages**:
  - Stage 2: Full scenario-based synthetic dataset generation (100k tabular, 20k temporal, 20k signals).
  - Stage 3: Data validation report generation & baseline classical ML training.
  - Stage 4: PyTorch deep learning model training (1D CNN, RNN, LSTM).
  - Stage 5: Explainability, SHAP analysis, and final alert simulation execution.

---

## 8. Reproducibility & Commands

### Setup Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run Unit Tests
```bash
pytest -v tests/
```
