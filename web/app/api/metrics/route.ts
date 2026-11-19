import { NextResponse } from 'next/server'
import trainingResults from '@/lib/training-results.json'

export const dynamic = 'force-dynamic'

/**
 * GET /api/metrics — pre-computed training results and dataset statistics.
 *
 * Model scores are imported from `web/lib/training-results.json`, which
 * `scripts/export_web_training_results.py` writes from the fitted artifacts.
 * The API and the home screen therefore read the same numbers, so a retrain
 * cannot leave one showing stale metrics.
 */

interface ExportedMetric {
  val_accuracy: number | null
  val_macro_f1: number | null
  test_accuracy: number | null
  test_macro_f1: number | null
  train_time_s: number | null
}

const CLASSICAL = trainingResults.classical as Record<string, ExportedMetric>
const DEEP = trainingResults.deep as Record<string, ExportedMetric>

const CLASSICAL_META = {
  knn: { name: 'K-Nearest Neighbors', short: 'KNN', color: '#6366f1' },
  logistic_regression: { name: 'Logistic Regression', short: 'LR', color: '#8b5cf6' },
  decision_tree: { name: 'Decision Tree', short: 'DT', color: '#a78bfa' },
  random_forest: { name: 'Random Forest', short: 'RF', color: '#7c3aed' },
} as const

const DEEP_META = {
  cnn_1d: { name: '1D CNN', short: 'CNN', input: 'ECG signal (1×500)', color: '#06b6d4' },
  rnn: { name: 'GRU-RNN', short: 'GRU', input: '60-step sequence (6 channels)', color: '#0ea5e9' },
  lstm: { name: 'LSTM', short: 'LSTM', input: '60-step sequence (6 channels)', color: '#3b82f6' },
} as const

const METRICS = {
  models: Object.fromEntries(
    Object.entries(CLASSICAL_META).map(([key, meta]) => {
      const metrics = CLASSICAL[key]
      return [
        key,
        {
          ...meta,
          val_accuracy: metrics.val_accuracy,
          val_macro_f1: metrics.val_macro_f1,
          test_accuracy: metrics.test_accuracy,
          test_macro_f1: metrics.test_macro_f1,
          train_time_s: metrics.train_time_s,
        },
      ]
    }),
  ),
  deep_models: Object.fromEntries(
    Object.entries(DEEP_META).map(([key, meta]) => {
      const metrics = DEEP[key]
      return [
        key,
        {
          ...meta,
          val_accuracy: metrics.val_accuracy,
          test_accuracy: metrics.test_accuracy,
          test_macro_f1: metrics.test_macro_f1,
          train_time_s: metrics.train_time_s,
        },
      ]
    }),
  ),
  served_model: trainingResults.served.model,
  dataset: {
    tabular_total: 100000,
    temporal_total: 20000,
    signal_total: 20000,
    feature_dim: 22,
    num_classes: 4,
    train_size: 70000,
    val_size: 15000,
    test_size: 15000,
    classes: ['NORMAL', 'MODERATE', 'HIGH', 'CRITICAL'],
    records_per_class: 25000,
  },
  pipeline: {
    split: '70/15/15 stratified',
    scaling: 'StandardScaler (train only)',
    imputation: 'Median',
    seed: 42,
    total_time_s: 99.5,
    tests_passed: 81,
    tests_total: 81,
  },
}

export async function GET() {
  return NextResponse.json(METRICS, {
    headers: { 'Access-Control-Allow-Origin': '*' },
  })
}

export async function OPTIONS() {
  return new NextResponse(null, {
    status: 204,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    },
  })
}
