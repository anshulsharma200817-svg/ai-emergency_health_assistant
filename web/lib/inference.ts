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
import {
  VITALS_RANGES,
  crossFieldError,
  outsideTrainingSupport,
  type SupportViolation,
} from './ranges'

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
  /**
   * Name of a data-derived constraint applied to the class probabilities, or
   * `null`. Currently only `"unconsciousness"`, which restricts the result to
   * HIGH/CRITICAL because that symptom never occurs in NORMAL or MODERATE in
   * the training data.
   */
  applied_constraint: string | null
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
 * Strict decimal parser. Accepts `72`, `"72"`, `" 72 "`, `"37.5"`, `"1e2"`.
 * Rejects everything else — including `"80abc"`, `"0x50"`, `"Infinity"`,
 * `"NaN"`, arrays, objects and booleans. `Number()` alone is too permissive:
 * `Number("0x50")` is 80 and `Number("80abc")` is NaN, both of which used to
 * fall through to a healthy default.
 */
const DECIMAL_RE = /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/

function toFiniteNumber(raw: unknown): number | null {
  if (typeof raw === 'number') return Number.isFinite(raw) ? raw : null
  if (typeof raw !== 'string') return null
  const text = raw.trim()
  if (!DECIMAL_RE.test(text)) return null
  const value = Number(text)
  return Number.isFinite(value) ? value : null
}

/** Symptom flags accept 0/1 and booleans; JSON clients commonly send `true`. */
function toFlag(raw: unknown): number | null {
  if (raw === true) return 1
  if (raw === false) return 0
  if (raw === 0 || raw === 1) return raw
  if (raw === '0' || raw === '1') return Number(raw)
  return null
}

/** Sex is strictly binary: 0 or 1 (numeric or string, no booleans, no 0.5). */
function toSex(raw: unknown): number | null {
  if (raw === 0 || raw === 1) return raw
  if (raw === '0' || raw === '1') return Number(raw)
  return null
}

function describe(raw: unknown): string {
  if (typeof raw === 'string') return JSON.stringify(raw)
  if (raw === undefined) return 'undefined'
  try {
    return JSON.stringify(raw) ?? String(raw)
  } catch {
    return String(raw)
  }
}

/**
 * Coerce the incoming payload into a complete vitals record.
 *
 * Three outcomes per field:
 *  - **absent** (`undefined`, `null`, `""`) → healthy-adult default;
 *  - **present and coercible** → used as given, then range-checked;
 *  - **present but malformed** → a `VitalsIssue`, which the route turns into
 *    HTTP 400. It is never silently replaced by a default, because that would
 *    turn `{"heart_rate": "abc"}` into a confident `NORMAL`.
 *
 * `temperature_unit: "F"` (or `"fahrenheit"`, `"°f"`) converts the supplied
 * temperature to Celsius before inference.
 */
export function coerceVitals(
  input: Partial<Vitals> | Record<string, unknown> = {},
): { vitals: Vitals; issues: VitalsIssue[] } {
  const out: Vitals = { ...DEFAULT_VITALS }
  const issues: VitalsIssue[] = []
  const source = input as Record<string, unknown>
  let temperatureSupplied = false

  for (const field of NUMERIC_FIELDS) {
    const raw = source[field]
    if (raw === undefined || raw === null || raw === '') continue

    if ((FLAG_FIELDS as readonly string[]).includes(field)) {
      const flag = toFlag(raw)
      if (flag === null) {
        issues.push({
          field, value: Number.NaN, min: 0, max: 1,
          message: `${field} must be 0, 1, true or false (received ${describe(raw)})`,
        })
      } else {
        out[field] = flag
      }
      continue
    }

    if (field === 'sex') {
      const sex = toSex(raw)
      if (sex === null) {
        issues.push({
          field, value: Number.NaN, min: 0, max: 1,
          message: `sex must be 0 (female) or 1 (male) (received ${describe(raw)})`,
        })
      } else {
        out[field] = sex
      }
      continue
    }

    const value = toFiniteNumber(raw)
    if (value === null) {
      const { min, max } = VITALS_RANGES[field]
      issues.push({
        field, value: Number.NaN, min, max,
        message: `${field} must be a finite number (received ${describe(raw)})`,
      })
      continue
    }

    out[field] = value
    if (field === 'temperature') temperatureSupplied = true
  }

  const unit = String(
    source.temperature_unit ?? source.temperatureUnit ?? '',
  ).trim().toLowerCase()

  if (temperatureSupplied && (unit === 'f' || unit.startsWith('fahrenheit') || unit === '°f')) {
    out.temperature = ((out.temperature - 32) * 5) / 9
  }

  return { vitals: out, issues }
}

/**
 * Coerce and validate in one step. Throws {@link VitalsValidationError} when
 * the payload contains malformed or out-of-range values.
 */
export function normalizeVitals(
  input: Partial<Vitals> | Record<string, unknown> = {},
): Vitals {
  const { vitals, issues } = coerceVitals(input)
  if (issues.length > 0) throw new VitalsValidationError(issues)
  return vitals
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

  const crossField = crossFieldError(vitals)
  if (crossField) {
    issues.push({
      field: crossField.field,
      value: vitals[crossField.field],
      min: VITALS_RANGES[crossField.field].min,
      max: vitals.systolic_bp,
      message: crossField.message,
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

/**
 * Hard constraint learned from the training data.
 *
 * `unconsciousness = 1` occurs in 1,490 held-out records: 1,330 CRITICAL and
 * 160 HIGH — never in NORMAL or MODERATE (0 of 50,000 low-risk rows). The
 * linear model still gives that flag a **positive** NORMAL coefficient, so
 * "healthy vitals + unconscious" came back as `NORMAL 99.9%`, and adding all
 * five symptoms *raised* the NORMAL confidence. Restricting the softmax mass to
 * the classes the data supports fixes that corner without touching any
 * in-distribution prediction: on the full 15,000-row test set the mask changes
 * **0** predictions and accuracy stays 0.9703.
 *
 * When the model puts less than 0.1% mass on both allowed classes it is
 * extrapolating, so the empirical `P(class | unconsciousness = 1)` split is
 * used instead. `src/inference/predictor.py` applies the identical rule.
 */
const UNCONSCIOUSNESS_PRIOR: Record<'HIGH' | 'CRITICAL', number> = {
  HIGH: 0.123,
  CRITICAL: 0.877,
}

function applyDataConstraints(
  probabilities: number[],
  vitals: Vitals,
): { probabilities: number[]; constraint: string | null } {
  if (vitals.unconsciousness !== 1) {
    return { probabilities, constraint: null }
  }

  const normal = RISK_LABELS.indexOf('NORMAL')
  const moderate = RISK_LABELS.indexOf('MODERATE')
  const high = RISK_LABELS.indexOf('HIGH')
  const critical = RISK_LABELS.indexOf('CRITICAL')
  const out = [...probabilities]
  out[normal] = 0
  out[moderate] = 0

  const allowed = probabilities[high] + probabilities[critical]
  if (allowed > 1e-3) {
    out[high] = probabilities[high] / allowed
    out[critical] = probabilities[critical] / allowed
  } else {
    out[high] = UNCONSCIOUSNESS_PRIOR.HIGH
    out[critical] = UNCONSCIOUSNESS_PRIOR.CRITICAL
  }

  return { probabilities: out, constraint: 'unconsciousness' }
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
  const { vitals, issues: coercionIssues } = coerceVitals(input)

  const issues = [...coercionIssues, ...validateVitals(vitals)]
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

  const rawProbabilities = softmax(logits)
  const { probabilities, constraint } = applyDataConstraints(rawProbabilities, vitals)
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
    applied_constraint: constraint,
  }
}
