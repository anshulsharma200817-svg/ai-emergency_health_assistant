/**
 * Browser/Node-safe inference for the trained Logistic Regression model.
 *
 * The exported parameters in `model-params.json` are extracted verbatim from
 * the fitted scikit-learn artifacts (StandardScaler, SimpleImputer and
 * LogisticRegression) produced by the Python training pipeline. Reproducing
 * the preprocessing + decision function here keeps the Vercel deployment
 * dependency-free: no Python runtime, no pickled objects, instant cold starts.
 *
 * Educational software simulation only — NOT a medical device.
 */

import params from './model-params.json'

export type RiskClass = 'NORMAL' | 'MODERATE' | 'HIGH' | 'CRITICAL'
export type AlertPriority = 'P0' | 'P1' | 'P2' | 'P3'

export interface Vitals {
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

export interface PredictionResult {
  predicted_risk_class: RiskClass
  priority: AlertPriority
  confidence: number
  probabilities: Record<RiskClass, number>
  alert_color: string
  disclaimer: string
}

export const DISCLAIMER =
  'EDUCATIONAL SIMULATION ONLY — NOT FOR CLINICAL USE'

export const RISK_LABELS: RiskClass[] = ['NORMAL', 'MODERATE', 'HIGH', 'CRITICAL']

export const PRIORITY_MAP: Record<RiskClass, AlertPriority> = {
  NORMAL: 'P3',
  MODERATE: 'P2',
  HIGH: 'P1',
  CRITICAL: 'P0',
}

export const ALERT_COLOR: Record<RiskClass, string> = {
  NORMAL: '#22c55e',
  MODERATE: '#f59e0b',
  HIGH: '#f97316',
  CRITICAL: '#ef4444',
}

/** Defaults match the biomedical "healthy adult" baseline used by the API. */
export const DEFAULT_VITALS: Vitals = {
  age: 35,
  sex: 1,
  heart_rate: 75,
  systolic_bp: 120,
  diastolic_bp: 80,
  spo2: 98,
  temperature: 37.0,
  respiratory_rate: 16,
  chest_pain: 0,
  breathing_difficulty: 0,
  dizziness: 0,
  unconsciousness: 0,
  bleeding_injury: 0,
}

const NUMERIC_FIELDS = Object.keys(DEFAULT_VITALS) as (keyof Vitals)[]

/**
 * Coerce the incoming payload into a complete, finite vitals record.
 * Missing/invalid fields fall back to {@link DEFAULT_VITALS}.
 */
export function normalizeVitals(
  input: Partial<Vitals> | Record<string, unknown> = {},
): Vitals {
  const out = { ...DEFAULT_VITALS }
  const source = input as Record<string, unknown>

  for (const field of NUMERIC_FIELDS) {
    const raw = source[field]
    if (raw === undefined || raw === null || raw === '') continue
    const value = typeof raw === 'string' ? Number(raw) : raw
    if (typeof value === 'number' && Number.isFinite(value)) {
      out[field] = value
    }
  }

  return out
}

/**
 * Replicates `src/preprocessing/feature_engineering.py` exactly, including the
 * right-closed pandas.cut semantics:
 *   age  <= 18 -> 0, <= 65 -> 1, else 2
 *   spo2 <= 88 -> 0, <= 94 -> 1, else 2
 */
function engineerFeatures(v: Vitals): Record<string, number> {
  const pulsePressure = v.systolic_bp - v.diastolic_bp
  const meanArterialPressure = v.diastolic_bp + pulsePressure / 3.0
  const shockIndex = v.systolic_bp === 0 ? NaN : v.heart_rate / v.systolic_bp

  return {
    age: v.age,
    sex: v.sex,
    heart_rate: v.heart_rate,
    systolic_bp: v.systolic_bp,
    diastolic_bp: v.diastolic_bp,
    spo2: v.spo2,
    temperature: v.temperature,
    respiratory_rate: v.respiratory_rate,
    chest_pain: v.chest_pain,
    breathing_difficulty: v.breathing_difficulty,
    dizziness: v.dizziness,
    unconsciousness: v.unconsciousness,
    bleeding_injury: v.bleeding_injury,
    pulse_pressure: pulsePressure,
    mean_arterial_pressure: meanArterialPressure,
    shock_index: shockIndex,
    age_group: v.age <= 18 ? 0 : v.age <= 65 ? 1 : 2,
    spo2_severity: v.spo2 <= 88 ? 0 : v.spo2 <= 94 ? 1 : 2,
    tachycardia_flag: v.heart_rate > 100 ? 1 : 0,
    hypotension_flag: v.systolic_bp < 90 ? 1 : 0,
    fever_flag: v.temperature > 38.0 ? 1 : 0,
    tachypnea_flag: v.respiratory_rate > 20 ? 1 : 0,
  }
}

function softmax(logits: number[]): number[] {
  const max = Math.max(...logits)
  const exps = logits.map(z => Math.exp(z - max))
  const sum = exps.reduce((a, b) => a + b, 0)
  return exps.map(e => e / sum)
}

function round(value: number, digits: number): number {
  const factor = 10 ** digits
  return Math.round(value * factor) / factor
}

/**
 * Run the trained model on a single (synthetic) patient record.
 */
export function predictRisk(
  input: Partial<Vitals> | Record<string, unknown> = {},
): PredictionResult {
  const vitals = normalizeVitals(input)
  const engineered = engineerFeatures(vitals)

  // 1. Median imputation (pipeline.imputer.statistics_)
  const imputed = params.features.map((feature, i) => {
    const value = engineered[feature]
    return Number.isFinite(value) ? value : params.imputer_statistics[i]
  })

  // 2. Standard scaling (pipeline.scaler.mean_ / scale_)
  const scaled = imputed.map((value, i) => {
    const scale = params.scaler_scale[i]
    return scale === 0 ? 0 : (value - params.scaler_mean[i]) / scale
  })

  // 3. Multinomial logistic regression decision function
  const logits = params.coef.map((row, classIndex) =>
    row.reduce((sum, weight, i) => sum + weight * scaled[i], params.intercept[classIndex]),
  )

  const probabilities = softmax(logits)
  const bestIndex = probabilities.indexOf(Math.max(...probabilities))
  const predicted = RISK_LABELS[bestIndex]

  const probMap = RISK_LABELS.reduce(
    (acc, label, i) => {
      acc[label] = round(probabilities[i], 4)
      return acc
    },
    {} as Record<RiskClass, number>,
  )

  return {
    predicted_risk_class: predicted,
    priority: PRIORITY_MAP[predicted],
    confidence: round(probabilities[bestIndex], 4),
    probabilities: probMap,
    alert_color: ALERT_COLOR[predicted],
    disclaimer: DISCLAIMER,
  }
}
