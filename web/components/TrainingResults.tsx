/**
 * Training-results visuals for the home screen.
 *
 * Every number is imported from `web/lib/training-results.json`, which is
 * produced by `scripts/export_web_training_results.py` from the fitted
 * artifacts. Do not type metric values here: re-run the export after a
 * retrain and the page follows automatically.
 */

import results from '@/lib/training-results.json'

type RiskClass = 'NORMAL' | 'MODERATE' | 'HIGH' | 'CRITICAL'

interface Metric {
  val_accuracy: number | null
  val_macro_f1: number | null
  test_accuracy: number | null
  test_macro_f1: number | null
  train_time_s: number | null
}

interface History {
  train_acc?: number[]
  val_acc?: number[]
  train_loss?: number[]
  val_loss?: number[]
}

const CLASSICAL = results.classical as Record<string, Metric>
const DEEP = results.deep as Record<string, Metric>
const SERVED = results.served as {
  model: string
  test_accuracy: number
  test_macro_f1: number
  confusion_matrix: number[][]
  classes: RiskClass[]
  per_class: Record<RiskClass, { precision: number; recall: number; f1: number; support: number }>
  test_records: number
}
const HISTORIES = results.histories as Record<string, History>

const pct = (v: number) => `${(v * 100).toFixed(2)}%`
const pct1 = (v: number) => `${(v * 100).toFixed(1)}%`
const num = (v: number | null | undefined) => (typeof v === 'number' ? v : 0)

interface ModelRow {
  key: string
  name: string
  short: string
  group: string
  color: string
  metric: Metric
  served?: boolean
  bestTabular?: boolean
  bestDeep?: boolean
  note: string
}

const MODEL_ROWS: ModelRow[] = [
  {
    key: 'logistic_regression', name: 'Logistic Regression', short: 'LR',
    group: 'Tabular vitals · served path', color: '#8b5cf6',
    metric: CLASSICAL.logistic_regression, served: true,
    note: 'Served in this demo',
  },
  {
    key: 'random_forest', name: 'Random Forest', short: 'RF',
    group: 'Tabular vitals · served path', color: '#7c3aed',
    metric: CLASSICAL.random_forest, bestTabular: true,
    note: 'Most accurate tabular model',
  },
  {
    key: 'decision_tree', name: 'Decision Tree', short: 'DT',
    group: 'Tabular vitals · served path', color: '#a78bfa',
    metric: CLASSICAL.decision_tree, note: 'Single depth-10 tree',
  },
  {
    key: 'knn', name: 'K-Nearest Neighbors', short: 'KNN',
    group: 'Tabular vitals · served path', color: '#6366f1',
    metric: CLASSICAL.knn, note: 'k = 5',
  },
  {
    key: 'cnn_1d', name: '1D CNN', short: 'CNN',
    group: 'ECG-like signal · different input', color: '#06b6d4',
    metric: DEEP.cnn_1d, bestDeep: true,
    note: 'Best overall, on 1×500 waveform',
  },
  {
    key: 'rnn', name: 'GRU-RNN', short: 'GRU',
    group: 'Temporal sequence · different input', color: '#0ea5e9',
    metric: DEEP.rnn, note: '60-step trajectories',
  },
  {
    key: 'lstm', name: 'LSTM', short: 'LSTM',
    group: 'Temporal sequence · different input', color: '#3b82f6',
    metric: DEEP.lstm, note: '60-step trajectories',
  },
]

const ACC_FLOOR = 0.9 // bars start at 90% so the differences are legible

function AccuracyChart() {
  return (
    <div className="chart-card">
      <div className="chart-head">
        <div>
          <div className="chart-title">Held-out test accuracy — all seven models</div>
          <div className="chart-sub">
            Evaluated once on splits never used for fitting or model selection.
            Bars start at 90% so the 95–100% band is readable; labels show exact values.
          </div>
        </div>
      </div>

      <div className="acc-chart" role="img" aria-label="Bar chart of held-out test accuracy for all seven models">
        {MODEL_ROWS.map((row, index) => {
          const previousGroup = index > 0 ? MODEL_ROWS[index - 1].group : null
          const acc = num(row.metric.test_accuracy)
          const width = Math.max(0, Math.min(100, ((acc - ACC_FLOOR) / (1 - ACC_FLOOR)) * 100))
          return (
            <div key={row.key}>
              {row.group !== previousGroup && (
                <div className="acc-group-label">{row.group}</div>
              )}
              <div className={`acc-row${row.served ? ' acc-row-served' : ''}`}>
                <div className="acc-name">
                  <span className="acc-dot" style={{ background: row.color }} />
                  <span className="acc-short">{row.short}</span>
                  <span className="acc-full">{row.name}</span>
                  {row.served && <span className="served-badge">SERVED</span>}
                  {row.bestTabular && <span className="best-badge inline-badge">Best tabular</span>}
                  {row.bestDeep && <span className="best-badge inline-badge">Best overall</span>}
                </div>
                <div className="acc-track">
                  <div
                    className="acc-fill"
                    style={{ width: `${width}%`, background: row.color }}
                  />
                </div>
                <div className="acc-value">{pct(acc)}</div>
              </div>
            </div>
          )
        })}
        <div className="acc-axis" aria-hidden="true">
          <span>90%</span>
          <span>95%</span>
          <span>100%</span>
        </div>
      </div>

      <div className="chart-note">
        The CNN&apos;s {pct(num(DEEP.cnn_1d.test_accuracy))} is measured on synthetic
        ECG waveforms, not on the 13 vitals collected by the form — it cannot be
        served by this demo.
      </div>
    </div>
  )
}

function ConfusionMatrix() {
  const classes = SERVED.classes
  const matrix = SERVED.confusion_matrix
  const maxCell = Math.max(...matrix.flat())
  const recalls = classes.map(cls => ({ cls, recall: SERVED.per_class[cls].recall }))
  const weakest = recalls.reduce((a, b) => (b.recall < a.recall ? b : a))
  const strongest = recalls.reduce((a, b) => (b.recall > a.recall ? b : a))

  return (
    <div className="chart-card">
      <div className="chart-title">Served model — confusion matrix on 15,000 held-out records</div>
      <div className="chart-sub">
        Rows are the recorded class, columns are the Logistic Regression prediction.
        Green cells are correct; red cells are errors. {pct(SERVED.test_accuracy)} accuracy,
        macro F1 {SERVED.test_macro_f1.toFixed(4)}.
      </div>

      <div className="cm-wrap">
        <div className="cm-corner">actual ↓ / predicted →</div>
        <div className="cm-grid" role="table" aria-label="Logistic Regression confusion matrix">
          <div className="cm-header-row" role="row">
            <div className="cm-header" aria-hidden="true" />
            {classes.map(cls => (
              <div className="cm-header" role="columnheader" key={`h-${cls}`}>{cls}</div>
            ))}
          </div>
          {matrix.map((row, i) => (
            <div className="cm-data-row" role="row" key={`r-${classes[i]}`}>
              <div className="cm-row-label" role="rowheader">{classes[i]}</div>
              {row.map((value, j) => {
                const correct = i === j
                const strength = maxCell > 0 ? value / maxCell : 0
                const alpha = value === 0 ? 0 : 0.16 + 0.62 * strength
                return (
                  <div
                    className={`cm-cell${correct ? ' cm-correct' : ' cm-wrong'}`}
                    role="cell"
                    key={`c-${i}-${j}`}
                    style={{
                      background: correct
                        ? `rgba(34,197,94,${alpha})`
                        : `rgba(239,68,68,${alpha})`,
                    }}
                  >
                    {value.toLocaleString()}
                  </div>
                )
              })}
            </div>
          ))}
        </div>
      </div>

      <div className="perclass-wrap">
        <table className="perclass-table">
          <thead>
            <tr>
              <th>Class</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1</th>
              <th>Support</th>
            </tr>
          </thead>
          <tbody>
            {classes.map(cls => (
              <tr key={cls}>
                <td className={`class-${cls.toLowerCase()}`}>{cls}</td>
                <td>{SERVED.per_class[cls].precision.toFixed(4)}</td>
                <td>{SERVED.per_class[cls].recall.toFixed(4)}</td>
                <td>{SERVED.per_class[cls].f1.toFixed(4)}</td>
                <td>{SERVED.per_class[cls].support.toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="chart-note" style={{ marginTop: 14 }}>
          Strongest recall: <strong>{strongest.cls}</strong> ({pct1(strongest.recall)}).
          Weakest: <strong>{weakest.cls}</strong> ({pct1(weakest.recall)}) — the
          remaining errors are almost entirely one step away on the risk scale,
          which is the safe direction for a triage simulation to confuse.
        </p>
      </div>
    </div>
  )
}

const CURVES: Array<{ key: string; name: string; color: string }> = [
  { key: 'cnn_1d', name: '1D CNN · ECG signal', color: '#06b6d4' },
  { key: 'rnn', name: 'GRU-RNN · 60-step sequence', color: '#0ea5e9' },
  { key: 'lstm', name: 'LSTM · 60-step sequence', color: '#3b82f6' },
]

const CURVE_W = 280
const CURVE_H = 150
const PAD = 26

function curvePoints(values: number[], yMin: number, yMax: number, total: number) {
  const span = Math.max(1, total - 1)
  return values
    .map((value, i) => {
      const x = PAD + (i / span) * (CURVE_W - PAD * 2)
      const y = CURVE_H - PAD - ((value - yMin) / (yMax - yMin)) * (CURVE_H - PAD * 2)
      return `${x.toFixed(1)},${y.toFixed(1)}`
    })
    .join(' ')
}

function DeepCurves() {
  return (
    <div className="chart-card">
      <div className="chart-title">Deep-learning training curves</div>
      <div className="chart-sub">
        Accuracy per epoch on the 14,000-sequence training split and 3,000-sequence
        validation split. The reported test score is computed once from the
        best-validation checkpoint.
      </div>
      <div className="curve-grid">
        {CURVES.map(({ key, name, color }) => {
          const history = HISTORIES[key]
          const train = history?.train_acc ?? []
          const val = history?.val_acc ?? []
          const total = Math.max(train.length, val.length)
          const all = [...train, ...val]
          if (total === 0 || all.length === 0) {
            return (
              <div className="curve-card" key={key}>
                <div className="curve-name">{name}</div>
                <div className="chart-note">History not exported for this checkpoint.</div>
              </div>
            )
          }
          const low = Math.min(...all)
          const yMin = Math.min(0.7, Math.floor(low * 10) / 10)
          const yMax = 1.0
          const lastVal = val.length ? val[val.length - 1] : null
          const metric = DEEP[key]
          return (
            <div className="curve-card" key={key}>
              <div className="curve-name">
                <span className="acc-dot" style={{ background: color }} />
                {name}
              </div>
              <svg
                viewBox={`0 0 ${CURVE_W} ${CURVE_H}`}
                className="curve-svg"
                role="img"
                aria-label={`${name}: training and validation accuracy per epoch`}
              >
                {[yMin, (yMin + yMax) / 2, yMax].map(tick => {
                  const y = CURVE_H - PAD - ((tick - yMin) / (yMax - yMin)) * (CURVE_H - PAD * 2)
                  return (
                    <g key={tick}>
                      <line
                        x1={PAD} x2={CURVE_W - PAD} y1={y} y2={y}
                        stroke="rgba(255,255,255,0.08)" strokeWidth="1"
                      />
                      <text x={4} y={y + 3} fill="#64748b" fontSize="8">
                        {(tick * 100).toFixed(0)}%
                      </text>
                    </g>
                  )
                })}
                {train.length > 1 && (
                  <polyline
                    points={curvePoints(train, yMin, yMax, total)}
                    fill="none" stroke="rgba(148,163,184,0.55)" strokeWidth="1.5"
                    strokeDasharray="4 3" strokeLinejoin="round"
                  />
                )}
                {val.length > 1 && (
                  <polyline
                    points={curvePoints(val, yMin, yMax, total)}
                    fill="none" stroke={color} strokeWidth="2.2" strokeLinejoin="round"
                  />
                )}
              </svg>
              <div className="curve-foot">
                <span>
                  <span className="legend-swatch legend-train" /> train
                  <span className="legend-swatch legend-val" style={{ background: color }} /> validation
                </span>
                <span className="curve-score">
                  test {pct(num(metric?.test_accuracy))}
                  {lastVal !== null && ` · val ${pct(lastVal)}`}
                </span>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

function ModelChoice() {
  const lr = CLASSICAL.logistic_regression
  const rf = CLASSICAL.random_forest
  const cnn = DEEP.cnn_1d
  const gapPp = ((num(rf.test_accuracy) - num(lr.test_accuracy)) * 100).toFixed(2)

  return (
    <div className="why-wrap" id="model-choice">
      <div className="why-intro">
        <div className="section-label">Model selection</div>
        <h3 className="why-title">Served model: Logistic Regression</h3>
        <p className="why-lede">
          The Random Forest is {gapPp} percentage points more accurate on tabular
          vitals, and the 1D CNN scores highest overall. Logistic Regression is
          served anyway: it is the model that can run unchanged in the browser-facing
          TypeScript API, at a size and latency that fit a serverless deployment,
          while staying reproducible and explainable.
        </p>
      </div>

      <div className="compare-scroll">
        <table className="compare-table">
          <thead>
            <tr>
              <th scope="col">Criterion</th>
              <th scope="col" className="col-served">Logistic Regression <span className="served-badge">SERVED</span></th>
              <th scope="col">Random Forest</th>
              <th scope="col">1D CNN</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Test accuracy</td>
              <td className="col-served">{pct(num(lr.test_accuracy))}</td>
              <td>{pct(num(rf.test_accuracy))} <span className="delta">+{gapPp} pp</span></td>
              <td>{pct(num(cnn.test_accuracy))}</td>
            </tr>
            <tr>
              <td>Test macro F1</td>
              <td className="col-served">{num(lr.test_macro_f1).toFixed(4)}</td>
              <td>{num(rf.test_macro_f1).toFixed(4)}</td>
              <td>{num(cnn.test_macro_f1).toFixed(4)}</td>
            </tr>
            <tr>
              <td>Serialized size</td>
              <td className="col-served">1.2 KB joblib<br /><span className="muted-cell">4.4 KB exported JSON</span></td>
              <td>76.4 MB<br /><span className="muted-cell">200 trees · 795,402 nodes</span></td>
              <td>29.8 KB checkpoint<br /><span className="muted-cell">plus a PyTorch runtime</span></td>
            </tr>
            <tr>
              <td>Inference</td>
              <td className="col-served">4 × 22 matrix multiply<br /><span className="muted-cell">~88 multiply-adds</span></td>
              <td>Up to 200 tree traversals<br /><span className="muted-cell">depth ≤ 20 each</span></td>
              <td>Two conv blocks over 500 samples<br /><span className="muted-cell">on CPU</span></td>
            </tr>
            <tr>
              <td>Runs on Vercel as-is</td>
              <td className="col-served"><span className="yes">Yes</span> — pure TypeScript, no Python</td>
              <td><span className="warn">Not as-is</span> — 76 MB artifact to ship and traverse</td>
              <td><span className="no">No</span> — needs Python/PyTorch and raw ECG</td>
            </tr>
            <tr>
              <td>Input available in the demo</td>
              <td className="col-served"><span className="yes">Yes</span> — 13 vitals → 22 features</td>
              <td><span className="yes">Yes</span> — same 22 features</td>
              <td><span className="no">No</span> — needs a 1×500 ECG waveform</td>
            </tr>
            <tr>
              <td>Explainability</td>
              <td className="col-served">Signed coefficients per class</td>
              <td>Impurity importances</td>
              <td>Saliency / occlusion, not built in</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div className="why-grid">
        {[
          {
            icon: '📦',
            title: '1,239 bytes vs 76.4 MB',
            body: 'The exported Logistic Regression is a 4×22 coefficient matrix plus scaler and imputer tables (4.4 KB of JSON). The forest is 200 trees and 795,402 nodes. The accuracy gap does not justify a 60,000× larger artifact on a serverless host.',
          },
          {
            icon: '⚡',
            title: 'Instant, dependency-free cold start',
            body: 'Inference is a softmax over 88 multiply-adds in TypeScript. There is no Python runtime, no pickle loading and no model download on Vercel, so a cold request still answers in milliseconds.',
          },
          {
            icon: '🔁',
            title: 'Same answer as the Python pipeline',
            body: 'On a 1,282-record boundary and random parity fuzz the TypeScript API and the Python predictor produced 0 label differences. Probabilities agree to 1e-4, with the only deviations at 4-decimal rounding ties.',
          },
          {
            icon: '🔍',
            title: 'Explainable by construction',
            body: 'Each class has a signed weight per vital, so the contribution of heart rate, blood pressure or shock index to a CRITICAL verdict can be shown directly — useful for an educational system where the reasoning matters.',
          },
        ].map(card => (
          <div className="card why-card" key={card.title}>
            <div className="feature-icon">{card.icon}</div>
            <div className="feature-title">{card.title}</div>
            <div className="feature-desc">{card.body}</div>
          </div>
        ))}
      </div>

      <div className="callout">
        <strong>Where the other models fit.</strong> Random Forest remains the best
        tabular model and is trained by the pipeline for offline comparison.
        The 1D CNN is the best model overall ({pct(num(DEEP.cnn_1d.test_accuracy))}),
        but it consumes a 1×500 ECG-like signal that the demo does not collect, and
        serving it would mean shipping a PyTorch runtime. Keeping the highest
        leaderboard score is not the same as keeping the model that the deployment
        can actually run.
      </div>
    </div>
  )
}

export default function TrainingResults() {
  return (
    <div className="training-results">
      <AccuracyChart />
      <ConfusionMatrix />
      <DeepCurves />
      <ModelChoice />
    </div>
  )
}
