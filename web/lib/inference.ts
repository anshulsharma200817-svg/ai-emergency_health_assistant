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
import { VITALS_RANGES, outsideTrainingSupport, type SupportViolation } from './ranges'

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
  /**
   * Set when any vital lies outside the range seen in training. The prediction
   * is then an extrapolation and must not be read as a normal result.
   */
  distribution_warning: string | null
  outside_training_range: SupportViolation[]
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

const FLAG_FIELDS = [
  'chest_pain',
  'breathing_difficulty',
  'dizziness',
  'unconsciousness',
  'bleeding_injury',
] as const

/**
 * Accepted physiological range per field, re-exported from `./ranges` so the
 * server route and the demo form share one definition.
 *
 * `temperature` is **Celsius**; pass `temperature_unit: "F"` to send Fahrenheit.
 */
export { VITALS_RANGES } from './ranges'
export type { VitalsField, VitalRange } from './ranges'

export interface VitalsIssue {
  field: string
  value: number
  min: number
  max: number
  message: string
}

/** Thrown by {@link predictRisk} when a vitals record is out of range. */
export class VitalsValidationError extends Error {
  readonly issues: VitalsIssue[]

  constructor(issues: VitalsIssue[]) {
    super(issues.map(issue => issue.message).join('; '))
    this.name = 'VitalsValidationError'
    this.issues = issues
  }
}

/**
 * Coerce the incoming payload into a complete, finite vitals record.
 * Missing/invalid fields fall back to {@link DEFAULT_VITALS}.
 *
 * `temperature_unit: "F"` (or `"fahrenheit"`) converts the supplied
 * temperature to Celsius before inference.
 */
export function normalizeVitals(
  input: Partial<Vitals> | Record<string, unknown> = {},
): Vitals {
  const out = { ...DEFAULT_VITALS }
  const source = input as Record<string, unknown>
  let temperatureSupplied = false

  for (const field of NUMERIC_FIELDS) {
    const raw = source[field]
    if (raw === undefined || raw === null || raw === '') continue
    const value = typeof raw === 'string' ? Number(raw) : raw
    if (typeof value === 'number' && Number.isFinite(value)) {
      out[field] = value
      if (field === 'temperature') temperatureSupplied = true
    }
  }

  const unit = String(
    source.temperature_unit ?? source.temperatureUnit ?? '',
  ).trim().toLowerCase()

  if (temperatureSupplied && (unit === 'f' || unit.startsWith('fahrenheit') || unit === '°f')) {
    out.temperature = ((out.temperature - 32) * 5) / 9
  }

  return out
}

/**
 * Check a normalized vitals record against {@link VITALS_RANGES}.
 * Returns an empty array when the record is acceptable.
 */
export function validateVitals(vitals: Vitals): VitalsIssue[] {
  const issues: VitalsIssue[] = []

  for (const field of NUMERIC_FIELDS) {
    if ((FLAG_FIELDS as readonly string[]).includes(field)) continue
    const { min, max } = VITALS_RANGES[field]
    const value = vitals[field]
    if (value >= min && value <= max) continue

    let message = `${field} must be between ${min} and ${max} (received ${value})`
    if (field === 'temperature' && value >= 90 && value <= 115) {
      message =
        'temperature looks like Fahrenheit; send Celsius or add "temperature_unit": "F"'
    }
    issues.push({ field, value, min, max, message })
  }

  for (const flag of FLAG_FIELDS) {
    const value = vitals[flag]
    if (value === 0 || value === 1) continue
    issues.push({
      field: flag,
      value,
      min: 0,
      max: 1,
      message: `${flag} must be 0 or 1 (received ${value})`,
    })
  }

  return issues
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
 *
 * @throws {VitalsValidationError} when any vital is outside {@link VITALS_RANGES}.
 */
export function predictRisk(
  input: Partial<Vitals> | Record<string, unknown> = {},
): PredictionResult {
  const vitals = normalizeVitals(input)

  const issues = validateVitals(vitals)
  if (issues.length > 0) {
    throw new VitalsValidationError(issues)
  }

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

  const outsideRange = outsideTrainingSupport(vitals as unknown as Record<string, number>)
  const distributionWarning =
    outsideRange.length > 0
      ? `Outside training distribution: ${outsideRange
          .map(v => `${v.field}=${v.value} (trained on ${v.observed_min}–${v.observed_max})`)
          .join(', ')}. The score is an extrapolation, not a validated result.`
      : null

  return {
    predicted_risk_class: predicted,
    priority: PRIORITY_MAP[predicted],
    confidence: round(probabilities[bestIndex], 4),
    probabilities: probMap,
    alert_color: ALERT_COLOR[predicted],
    disclaimer: DISCLAIMER,
    distribution_warning: distributionWarning,
    outside_training_range: outsideRange,
  }
}
