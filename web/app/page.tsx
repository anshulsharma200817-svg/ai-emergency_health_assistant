'use client'

import { useState } from 'react'

// ─── Types ───────────────────────────────────────────────────────────────────

interface Vitals {
  age: number
  sex: number
  heart_rate: number
  systolic_bp: number
  diastolic_bp: number
  spo2: number
  temperature: number
  respiratory_rate: number
  chest_pain: number
  breathing_difficulty: number
  dizziness: number
  unconsciousness: number
  bleeding_injury: number
}

interface PredictionResult {
  predicted_risk_class: 'NORMAL' | 'MODERATE' | 'HIGH' | 'CRITICAL'
  priority: 'P0' | 'P1' | 'P2' | 'P3'
  confidence: number
  probabilities: Record<string, number>
  alert_color: string
  disclaimer: string
}

// ─── Constants ────────────────────────────────────────────────────────────────

const RISK_CONFIG = {
  NORMAL:   { color: '#22c55e', bg: 'rgba(34,197,94,0.12)',   border: 'rgba(34,197,94,0.3)',   emoji: '✅' },
  MODERATE: { color: '#f59e0b', bg: 'rgba(245,158,11,0.12)',  border: 'rgba(245,158,11,0.3)',  emoji: '⚠️' },
  HIGH:     { color: '#f97316', bg: 'rgba(249,115,22,0.12)',  border: 'rgba(249,115,22,0.3)',  emoji: '🔶' },
  CRITICAL: { color: '#ef4444', bg: 'rgba(239,68,68,0.12)',   border: 'rgba(239,68,68,0.3)',   emoji: '🚨' },
}

const MODEL_RESULTS = [
  { key: 'knn',                name: 'K-Nearest Neighbors', short: 'KNN', testAcc: 0.9586, testF1: 0.9586, color: '#6366f1', meta: 'Classical ML · 4-class · 22 features' },
  { key: 'logistic_regression',name: 'Logistic Regression',  short: 'LR',  testAcc: 0.9703, testF1: 0.9703, color: '#8b5cf6', meta: 'Classical ML · 4-class · 22 features' },
  { key: 'decision_tree',      name: 'Decision Tree',         short: 'DT',  testAcc: 0.9661, testF1: 0.9662, color: '#a78bfa', meta: 'Classical ML · 4-class · 22 features' },
  { key: 'random_forest',      name: 'Random Forest',         short: 'RF',  testAcc: 0.9805, testF1: 0.9805, color: '#7c3aed', best: true, meta: 'Classical ML · 4-class · 22 features' },
]

const DEEP_MODEL_RESULTS = [
  { key: 'cnn_1d', name: '1D CNN',  short: 'CNN',  testAcc: 0.9973, testF1: 0.9973, color: '#06b6d4', best: true, meta: 'PyTorch · ECG signal (1×500) · 4-class' },
  { key: 'rnn',    name: 'GRU-RNN', short: 'GRU',  testAcc: 0.9370, testF1: 0.9362, color: '#0ea5e9', meta: 'PyTorch · 60-step sequence · 4-class' },
  { key: 'lstm',   name: 'LSTM',    short: 'LSTM', testAcc: 0.9240, testF1: 0.9226, color: '#3b82f6', meta: 'PyTorch · 60-step sequence · 4-class' },
]

const DEFAULT_VITALS: Vitals = {
  age: 35, sex: 1, heart_rate: 75, systolic_bp: 120, diastolic_bp: 80,
  spo2: 98, temperature: 37.0, respiratory_rate: 16,
  chest_pain: 0, breathing_difficulty: 0, dizziness: 0, unconsciousness: 0, bleeding_injury: 0,
}

const CRITICAL_PRESET: Vitals = {
  age: 67, sex: 1, heart_rate: 142, systolic_bp: 72, diastolic_bp: 44,
  spo2: 84, temperature: 39.4, respiratory_rate: 34,
  chest_pain: 1, breathing_difficulty: 1, dizziness: 1, unconsciousness: 0, bleeding_injury: 0,
}

// ─── Utility ──────────────────────────────────────────────────────────────────

function pct(v: number) { return `${(v * 100).toFixed(1)}%` }

// ─── Main Page ────────────────────────────────────────────────────────────────

export default function Home() {
  const [vitals, setVitals] = useState<Vitals>(DEFAULT_VITALS)
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]   = useState<string | null>(null)

  const handleChange = (field: keyof Vitals, value: string) => {
    setVitals(prev => ({ ...prev, [field]: Number(value) }))
  }

  const toggleSymptom = (field: keyof Vitals) => {
    setVitals(prev => ({ ...prev, [field]: prev[field] ? 0 : 1 }))
  }

  const loadPreset = (preset: Vitals) => {
    setVitals(preset)
    setResult(null)
    setError(null)
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(vitals),
      })
      const data = await res.json().catch(() => null)
      if (!res.ok) {
        const issues = Array.isArray(data?.issues)
          ? data.issues.map((i: { message: string }) => i.message).join(' · ')
          : null
        setError(issues || data?.error || `API error: ${res.status}`)
        return
      }
      setResult(data)
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Unknown error'
      setError(`Prediction failed: ${msg}. Please try again.`)
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      {/* ── Navigation ── */}
      <nav>
        <div className="nav-inner">
          <a href="#" className="nav-logo">
            <div className="nav-logo-dot" />
            AI Health Detection
          </a>
          <span className="nav-badge">⚠ SIMULATION ONLY</span>
        </div>
      </nav>

      {/* ── Hero ── */}
      <section className="hero fade-in">
        <div className="hero-tag">
          <span>🔬</span> Educational ML Pipeline
        </div>
        <h1>Emergency Health<br />Risk Classification</h1>
        <p className="hero-sub">
          A production-quality synthetic data ML pipeline — 100,000 records,
          4-class risk stratification, 98%+ accuracy. Live inference demo below.
        </p>
        <div className="hero-buttons">
          <a href="#demo" className="btn btn-primary">🩺 Try Live Demo</a>
          <a href="#models" className="btn btn-outline">📊 View Results</a>
          <a
            href="https://github.com/anshulsharma200817/ai-emergency_health_assistant"
            target="_blank"
            rel="noopener noreferrer"
            className="btn btn-outline"
          >
            ⭐ GitHub
          </a>
        </div>
      </section>

      {/* ── Stats Bar ── */}
      <div className="container">
        <div className="stats-bar">
          {[
            { num: '100K', label: 'Synthetic Records' },
            { num: '98.1%', label: 'Best Tabular Accuracy' },
            { num: '54/54', label: 'Tests Passing' },
            { num: '4', label: 'Risk Classes' },
          ].map(s => (
            <div className="stat-cell" key={s.label}>
              <div className="stat-num">{s.num}</div>
              <div className="stat-label">{s.label}</div>
            </div>
          ))}
        </div>
      </div>

      {/* ── Pipeline Section ── */}
      <section id="pipeline">
        <div className="container">
          <div className="section-label">Pipeline</div>
          <h2 className="section-title">End-to-End ML Pipeline</h2>
          <p className="section-desc" style={{ marginBottom: 48 }}>
            From synthetic data generation to trained models in a single reproducible command.
          </p>
          <div className="pipeline-steps">
            {[
              { n: '01', t: 'Generate', d: '100K tabular + 20K temporal + 20K signal records with Markov dynamics' },
              { n: '02', t: 'Validate', d: '0 duplicates, 0 missing values, schema checks, physiological range checks' },
              { n: '03', t: 'Preprocess', d: 'Feature engineering: 14 → 22 features, StandardScaler, LabelEncoder' },
              { n: '04', t: 'Split', d: 'Stratified 70/15/15 split with no data leakage across trajectories' },
              { n: '05', t: 'Train', d: 'KNN, Logistic Regression, Decision Tree, Random Forest with joblib persistence' },
              { n: '06', t: 'Evaluate', d: 'Accuracy, Macro F1, per-class breakdown, alert priority routing P0–P3' },
            ].map(s => (
              <div className="pipeline-step" key={s.n}>
                <div className="step-num">{s.n}</div>
                <div className="step-title">{s.t}</div>
                <div className="step-desc">{s.d}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Model Results ── */}
      <section id="models" style={{ background: 'rgba(255,255,255,0.01)', borderTop: '1px solid rgba(255,255,255,0.05)', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
        <div className="container">
          <div className="section-label">Results</div>
          <h2 className="section-title">Model Performance</h2>
          <p className="section-desc" style={{ marginBottom: 48 }}>
            All 4 classical ML models trained on 70,000 samples, evaluated on a held-out 15,000 test set.
          </p>
          <div className="models-grid">
            {MODEL_RESULTS.map(m => (
              <div
                className="card model-card"
                key={m.key}
                style={{ '--card-color': m.color } as React.CSSProperties}
              >
                {m.best && <div className="best-badge">⭐ Best Model</div>}
                <div className="model-name">{m.name}</div>
                <div className="model-desc">{m.meta}</div>
                <div className="metric-row">
                  <span className="metric-label">Test Accuracy</span>
                  <span className="metric-value">{pct(m.testAcc)}</span>
                </div>
                <div className="metric-row">
                  <span className="metric-label">Test Macro F1</span>
                  <span className="metric-value">{m.testF1.toFixed(4)}</span>
                </div>
                <div className="accuracy-bar">
                  <div className="accuracy-fill" style={{ width: pct(m.testAcc) }} />
                </div>
              </div>
            ))}
          </div>

          <div className="section-label" style={{ marginTop: 56 }}>Deep Learning</div>
          <p className="section-desc" style={{ margin: '8px 0 32px' }}>
            PyTorch models trained for 20 epochs on 20,000 ECG signals and 20,000
            vital-sign trajectories, evaluated on a held-out 3,000-record test set each.
          </p>
          <div className="models-grid">
            {DEEP_MODEL_RESULTS.map(m => (
              <div
                className="card model-card"
                key={m.key}
                style={{ '--card-color': m.color } as React.CSSProperties}
              >
                {m.best && <div className="best-badge">⭐ Best Deep Model</div>}
                <div className="model-name">{m.name}</div>
                <div className="model-desc">{m.meta}</div>
                <div className="metric-row">
                  <span className="metric-label">Test Accuracy</span>
                  <span className="metric-value">{pct(m.testAcc)}</span>
                </div>
                <div className="metric-row">
                  <span className="metric-label">Test Macro F1</span>
                  <span className="metric-value">{m.testF1.toFixed(4)}</span>
                </div>
                <div className="accuracy-bar">
                  <div className="accuracy-fill" style={{ width: pct(m.testAcc) }} />
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Live Demo ── */}
      <section id="demo">
        <div className="container">
          <div className="section-label">Interactive Demo</div>
          <h2 className="section-title">Live Risk Prediction</h2>
          <p className="section-desc" style={{ marginBottom: 16 }}>
            Enter synthetic patient vitals and get instant risk classification from the trained Logistic Regression model.
          </p>
          <div
            style={{
              display: 'inline-flex', gap: 10, marginBottom: 32, flexWrap: 'wrap',
              padding: '10px 16px', background: 'rgba(239,68,68,0.06)',
              border: '1px solid rgba(239,68,68,0.2)', borderRadius: 8, fontSize: 12, color: '#f87171',
            }}
          >
            ⚠️ EDUCATIONAL SIMULATION ONLY — Not for clinical or medical use
          </div>
          <div style={{ display: 'flex', gap: 12, marginBottom: 24, flexWrap: 'wrap' }}>
            <button className="btn btn-outline" style={{ padding: '8px 18px', fontSize: 13 }} onClick={() => loadPreset(DEFAULT_VITALS)}>
              💚 Load Normal Preset
            </button>
            <button className="btn btn-outline" style={{ padding: '8px 18px', fontSize: 13 }} onClick={() => loadPreset(CRITICAL_PRESET)}>
              🔴 Load Critical Preset
            </button>
          </div>

          <div className="demo-grid">
            {/* ─ Form ─ */}
            <div className="card demo-form">
              <div className="form-title">Patient Vitals</div>
              <form onSubmit={handleSubmit}>
                <div className="form-grid">
                  {[
                    { field: 'age',              label: 'Age (years)',   min: 1,  max: 110, step: 1 },
                    { field: 'heart_rate',       label: 'Heart Rate (bpm)', min: 25, max: 240, step: 1 },
                    { field: 'systolic_bp',      label: 'Systolic BP (mmHg)', min: 40, max: 260, step: 1 },
                    { field: 'diastolic_bp',     label: 'Diastolic BP (mmHg)', min: 30, max: 160, step: 1 },
                    { field: 'spo2',             label: 'SpO₂ (%)', min: 50, max: 100, step: 0.1 },
                    { field: 'temperature',      label: 'Temperature (°C)', min: 30, max: 45, step: 0.1 },
                    { field: 'respiratory_rate', label: 'Resp. Rate (breaths/min)', min: 5, max: 60, step: 1 },
                  ].map(({ field, label, min, max, step }) => (
                    <div className="form-group" key={field}>
                      <label>{label}</label>
                      <input
                        type="number"
                        name={field}
                        id={`vitals-${field}`}
                        aria-label={label}
                        min={min} max={max} step={step}
                        value={vitals[field as keyof Vitals]}
                        onChange={e => handleChange(field as keyof Vitals, e.target.value)}
                      />
                    </div>
                  ))}
                  <div className="form-group">
                    <label htmlFor="vitals-sex">Sex</label>
                    <select id="vitals-sex" name="sex" value={vitals.sex} onChange={e => handleChange('sex', e.target.value)}>
                      <option value={0}>Female (0)</option>
                      <option value={1}>Male (1)</option>
                    </select>
                  </div>
                </div>

                <div style={{ marginTop: 20 }}>
                  <div style={{ fontSize: 12, color: 'var(--muted2)', fontWeight: 500, marginBottom: 10, letterSpacing: '0.5px' }}>
                    SYMPTOMS
                  </div>
                  <div className="symptom-row">
                    {[
                      { field: 'chest_pain',          label: '💔 Chest Pain' },
                      { field: 'breathing_difficulty', label: '😮‍💨 Breathing Diff.' },
                      { field: 'dizziness',            label: '😵 Dizziness' },
                      { field: 'unconsciousness',      label: '😶 Unconscious' },
                      { field: 'bleeding_injury',      label: '🩸 Bleeding/Injury' },
                    ].map(({ field, label }) => (
                      <label
                        key={field}
                        className={`symptom-toggle ${vitals[field as keyof Vitals] ? 'active' : ''}`}
                        onClick={() => toggleSymptom(field as keyof Vitals)}
                      >
                        <input type="checkbox" checked={!!vitals[field as keyof Vitals]} readOnly />
                        <div className="symptom-dot" />
                        {label}
                      </label>
                    ))}
                  </div>
                </div>

                <button type="submit" className="submit-btn" disabled={loading}>
                  {loading ? <><span className="spinner" /> Analyzing...</> : '🔍 Predict Risk Class'}
                </button>
              </form>
            </div>

            {/* ─ Result ─ */}
            <div className="result-panel">
              {!result && !error && (
                <div className="result-idle">
                  <div className="result-idle-icon">🏥</div>
                  <div style={{ fontSize: 18, fontWeight: 700, marginBottom: 8 }}>Awaiting Input</div>
                  <div className="result-idle-text">
                    Fill in patient vitals and click<br />&quot;Predict Risk Class&quot; to see results
                  </div>
                </div>
              )}

              {error && (
                <div className="card" style={{ borderColor: 'rgba(239,68,68,0.4)', background: 'rgba(239,68,68,0.06)' }}>
                  <div style={{ color: '#f87171', fontWeight: 700, marginBottom: 8 }}>⚠ Error</div>
                  <div style={{ fontSize: 14, color: 'var(--muted2)' }}>{error}</div>
                </div>
              )}

              {result && (() => {
                const rk = result.predicted_risk_class
                const cfg = RISK_CONFIG[rk]
                return (
                  <div className="result-card" style={{ borderColor: cfg.border, background: cfg.bg }}>
                    <div className="result-header">
                      <div>
                        <div className="risk-badge" style={{ background: cfg.bg, color: cfg.color, border: `2px solid ${cfg.border}` }}>
                          {cfg.emoji} {rk}
                        </div>
                        <div style={{ marginTop: 8, display: 'flex', gap: 10, alignItems: 'center' }}>
                          <span className="priority-pill" style={{ background: cfg.bg, color: cfg.color, border: `1px solid ${cfg.border}` }}>
                            {result.priority}
                          </span>
                          <span className="result-confidence">
                            Confidence: {pct(result.confidence)}
                          </span>
                        </div>
                      </div>
                    </div>

                    <div className="prob-bars">
                      <div style={{ fontSize: 12, fontWeight: 700, letterSpacing: 1, color: 'var(--muted)', marginBottom: 14 }}>
                        PROBABILITY DISTRIBUTION
                      </div>
                      {(['NORMAL', 'MODERATE', 'HIGH', 'CRITICAL'] as const).map(cls => {
                        const prob = result.probabilities[cls] ?? 0
                        const clsCfg = RISK_CONFIG[cls]
                        return (
                          <div className="prob-row" key={cls}>
                            <div className="prob-label-row">
                              <span className="prob-name" style={{ color: clsCfg.color }}>{cls}</span>
                              <span className="prob-pct">{pct(prob)}</span>
                            </div>
                            <div className="prob-bar">
                              <div className="prob-fill" style={{ width: pct(prob), background: clsCfg.color }} />
                            </div>
                          </div>
                        )
                      })}
                    </div>

                    <div className="disclaimer-box">{result.disclaimer}</div>
                  </div>
                )
              })()}

              {/* Alert Mapping Card */}
              <div className="card">
                <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 14, color: 'var(--muted2)', letterSpacing: 1 }}>
                  ALERT PRIORITY MAPPING
                </div>
                {(['CRITICAL', 'HIGH', 'MODERATE', 'NORMAL'] as const).map((cls, i) => {
                  const cfg = RISK_CONFIG[cls]
                  const priorities = ['P0', 'P1', 'P2', 'P3']
                  return (
                    <div
                      key={cls}
                      style={{
                        display: 'flex', alignItems: 'center', gap: 12,
                        borderBottom: i < 3 ? '1px solid var(--border)' : 'none',
                        background: result?.predicted_risk_class === cls ? cfg.bg : 'transparent',
                        margin: '0 -4px', padding: '10px 4px',
                        borderRadius: 6,
                      }}
                    >
                      <span style={{ fontSize: 20 }}>{cfg.emoji}</span>
                      <span style={{ flex: 1, fontSize: 14, fontWeight: 600 }}>{cls}</span>
                      <span style={{
                        padding: '3px 12px', borderRadius: 20, fontSize: 12, fontWeight: 700,
                        background: cfg.bg, color: cfg.color, border: `1px solid ${cfg.border}`,
                        fontFamily: 'JetBrains Mono, monospace',
                      }}>
                        {priorities[i]}
                      </span>
                    </div>
                  )
                })}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── Features ── */}
      <section style={{ background: 'rgba(255,255,255,0.01)', borderTop: '1px solid rgba(255,255,255,0.05)' }}>
        <div className="container">
          <div className="section-label">Architecture</div>
          <h2 className="section-title">What&apos;s Inside</h2>
          <p className="section-desc" style={{ marginBottom: 48 }}>
            A complete, modular ML system built with best practices for reproducibility and educational clarity.
          </p>
          <div className="features-grid">
            {[
              { icon: '🧬', title: 'Synthetic Data Generation', desc: 'Scenario-based Gaussian class distributions, Markov latent state trajectory dynamics, P/QRS/T ECG waveform synthesis with noise.' },
              { icon: '🔬', title: 'Feature Engineering', desc: '14 raw vitals → 22 features. Derived: pulse pressure, MAP, shock index, clinical severity flags, ordinal encodings.' },
              { icon: '🛡️', title: 'No Data Leakage', desc: 'Stratified 70/15/15 splits, pipeline fitted on train only, ID-based temporal splits prevent sequence leakage.' },
              { icon: '🤖', title: '7 ML/DL Models', desc: 'KNN, Logistic Regression, Decision Tree, Random Forest (classical), plus trained 1D CNN, GRU, and LSTM (PyTorch).' },
              { icon: '📊', title: '54/54 Tests Passing', desc: 'Full test coverage: generators, preprocessing, training, inference, alert engine, validator — all verified.' },
              { icon: '🐳', title: 'Docker Ready', desc: 'Single docker compose up command generates data, trains models, and runs the full test suite.' },
            ].map(f => (
              <div className="card" key={f.title}>
                <div className="feature-icon">{f.icon}</div>
                <div className="feature-title">{f.title}</div>
                <div className="feature-desc">{f.desc}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Quick Start ── */}
      <section id="quickstart">
        <div className="container">
          <div className="section-label">Quick Start</div>
          <h2 className="section-title">Run in 60 Seconds</h2>
          <div className="code-block" style={{ maxWidth: 700, marginTop: 32 }}>
            <div className="code-header">
              <div className="code-dot" style={{ background: '#ef4444' }} />
              <div className="code-dot" style={{ background: '#f59e0b' }} />
              <div className="code-dot" style={{ background: '#22c55e' }} />
              <span style={{ marginLeft: 8 }}>bash</span>
            </div>
            <pre>{`# Clone & run everything with one command
git clone https://github.com/anshulsharma200817/ai-emergency_health_assistant
cd ai-emergency_health_assistant
docker compose up --build

# That single command:
# 1. Generates 100K + 20K + 20K synthetic records
# 2. Trains KNN, LR, Decision Tree, Random Forest
# 3. Runs the full test suite
# 4. Saves models to ./models/`}</pre>
          </div>
          <div className="code-block" style={{ maxWidth: 700, marginTop: 20 }}>
            <div className="code-header">
              <div className="code-dot" style={{ background: '#ef4444' }} />
              <div className="code-dot" style={{ background: '#f59e0b' }} />
              <div className="code-dot" style={{ background: '#22c55e' }} />
              <span style={{ marginLeft: 8 }}>python</span>
            </div>
            <pre>{`from src.inference.predictor import HealthRiskPredictor
import yaml

config = yaml.safe_load(open("config.yaml"))
predictor = HealthRiskPredictor.from_saved_model(
    "models/classical/random_forest.joblib",
    "models/preprocessing_pipeline.pkl",
    config,
)
result = predictor.predict_record({
    "age": 67, "heart_rate": 142, "systolic_bp": 72,
    "spo2": 84, "chest_pain": 1, ...
})
# → {"predicted_risk_class": "CRITICAL", "priority": "P0"}`}</pre>
          </div>
        </div>
      </section>

      {/* ── Footer ── */}
      <footer>
        <div className="container">
          <div className="footer-logo">🏥 AI Health Detection</div>
          <div className="footer-desc">
            Educational software simulation pipeline for emergency health risk classification research.
            Built entirely on synthetic data. Not a medical device.
          </div>
          <div className="footer-links">
            <a href="#demo">Live Demo</a>
            <a href="#models">Model Results</a>
            <a href="#pipeline">Pipeline</a>
            <a href="#quickstart">Quick Start</a>
            <a href="https://github.com/anshulsharma200817/ai-emergency_health_assistant" target="_blank" rel="noopener noreferrer">GitHub</a>
          </div>
          <div className="footer-copy">
            ⚠ EDUCATIONAL SOFTWARE SIMULATION ONLY — NOT FOR CLINICAL USE &nbsp;|&nbsp;
            All data is 100% synthetic &nbsp;|&nbsp;
            Random seed: 42
          </div>
        </div>
      </footer>
    </>
  )
}
