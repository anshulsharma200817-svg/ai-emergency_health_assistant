import { NextResponse } from 'next/server'

export const dynamic = 'force-dynamic'

/**
 * GET /api/metrics — pre-computed training results and dataset statistics.
 * No model loading or computation happens here.
 */
const METRICS = {
  models: {
    knn: {
      name: 'K-Nearest Neighbors',
      short: 'KNN',
      val_accuracy: 0.9586,
      val_macro_f1: 0.9586,
      test_accuracy: 0.9586,
      test_macro_f1: 0.9586,
      train_time_s: 0.01,
      color: '#6366f1',
    },
    logistic_regression: {
      name: 'Logistic Regression',
      short: 'LR',
      val_accuracy: 0.9706,
      val_macro_f1: 0.9706,
      test_accuracy: 0.9703,
      test_macro_f1: 0.9703,
      train_time_s: 0.82,
      color: '#8b5cf6',
    },
    decision_tree: {
      name: 'Decision Tree',
      short: 'DT',
      val_accuracy: 0.9641,
      val_macro_f1: 0.9641,
      test_accuracy: 0.9661,
      test_macro_f1: 0.9662,
      train_time_s: 0.52,
      color: '#a78bfa',
    },
    random_forest: {
      name: 'Random Forest',
      short: 'RF',
      val_accuracy: 0.9809,
      val_macro_f1: 0.9809,
      test_accuracy: 0.9805,
      test_macro_f1: 0.9805,
      train_time_s: 20.47,
      color: '#7c3aed',
    },
  },
  deep_models: {
    cnn_1d: {
      name: '1D CNN',
      short: 'CNN',
      input: 'ECG signal (1×500)',
      val_accuracy: 0.9973,
      test_accuracy: 0.9973,
      test_macro_f1: 0.9973,
      train_time_s: 117.1,
      color: '#06b6d4',
    },
    rnn: {
      name: 'GRU-RNN',
      short: 'GRU',
      input: '60-step sequence (6 channels)',
      val_accuracy: 0.9403,
      test_accuracy: 0.9370,
      test_macro_f1: 0.9362,
      train_time_s: 529.6,
      color: '#0ea5e9',
    },
    lstm: {
      name: 'LSTM',
      short: 'LSTM',
      input: '60-step sequence (6 channels)',
      val_accuracy: 0.9293,
      test_accuracy: 0.9240,
      test_macro_f1: 0.9226,
      train_time_s: 436.2,
      color: '#3b82f6',
    },
  },
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
    total_time_s: 50.1,
    tests_passed: 54,
    tests_total: 54,
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
