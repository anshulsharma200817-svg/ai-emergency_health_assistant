#!/usr/bin/env python3
"""Data Generation Script.

Generates all synthetic datasets for the AI Emergency Health Detection System:
  1. Tabular data    → 100,000 records × 15 features  (data/raw/tabular_raw.csv)
  2. Temporal data   → 20,000 sequences × 60 steps × 6 features
  3. Signal data     → 20,000 ECG-like waveforms × 500 samples

Usage:
    python generate_data.py
    python generate_data.py --quick          # 1K / 400 / 400 samples (smoke test)
    python generate_data.py --tabular-only
    python generate_data.py --temporal-only
    python generate_data.py --signals-only
"""

import os
import sys
import time
import argparse
import json

import numpy as np
import pandas as pd
import yaml

# ── Allow running from project root ──────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.tabular_generator  import TabularDataGenerator
from src.data.temporal_generator import TemporalDataGenerator
from src.data.signal_generator   import SignalDataGenerator
from src.data.validator          import DataValidator
from src.utils.logger            import setup_logger

logger = setup_logger("generate_data")


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dirs(config: dict) -> None:
    for key in ["data_raw", "data_interim", "data_processed"]:
        os.makedirs(config["paths"][key], exist_ok=True)


def generate_tabular(config: dict, quick: bool = False) -> None:
    """Generate and save tabular dataset."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: Generating Tabular Dataset")
    logger.info("=" * 60)

    gen = TabularDataGenerator(config)
    n_per_class = 250 if quick else None  # None → uses config (25K per class)

    t0 = time.time()
    df = gen.generate_dataset(num_samples_per_class=n_per_class)
    elapsed = round(time.time() - t0, 1)

    out_path = os.path.join(config["paths"]["data_raw"], "tabular_raw.csv")
    df.to_csv(out_path, index=False)
    logger.info(f"Saved tabular data → {out_path}  [{elapsed}s]")

    # Validate
    validator = DataValidator(config)
    report = validator.generate_quality_report(df)
    logger.info(f"  Records      : {report['num_records']:,}")
    logger.info(f"  Schema valid : {report['schema_valid']}")
    logger.info(f"  Duplicates   : {report['duplicate_count']}")
    logger.info(f"  Class dist   : {report['class_distribution']}")
    logger.info(f"  Missing vals : {sum(report['missing_values'].values())}")

    report_path = os.path.join(config["paths"]["data_raw"], "tabular_quality_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"  Quality report → {report_path}")


def generate_temporal(config: dict, quick: bool = False) -> None:
    """Generate and save temporal sequence dataset."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Generating Temporal Sequence Dataset")
    logger.info("=" * 60)

    gen = TemporalDataGenerator(config)
    n = 400 if quick else None  # None → uses config (20K)

    t0 = time.time()
    X, y, ids = gen.generate_batch(num_trajectories=n)
    elapsed = round(time.time() - t0, 1)

    raw_dir = config["paths"]["data_raw"]
    np.save(os.path.join(raw_dir, "temporal_X.npy"),   X)
    np.save(os.path.join(raw_dir, "temporal_y.npy"),   y)
    np.save(os.path.join(raw_dir, "temporal_ids.npy"), ids)

    label_counts = dict(zip(*np.unique(y, return_counts=True)))
    logger.info(f"Saved temporal data → {raw_dir}/temporal_*.npy  [{elapsed}s]")
    logger.info(f"  X shape      : {X.shape}")
    logger.info(f"  y shape      : {y.shape}")
    logger.info(f"  Class dist   : {label_counts}")


def generate_signals(config: dict, quick: bool = False) -> None:
    """Generate and save signal dataset."""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Generating ECG-like Signal Dataset")
    logger.info("=" * 60)

    gen = SignalDataGenerator(config)
    n = 400 if quick else None  # None → uses config (20K)

    t0 = time.time()
    X, y = gen.generate_signal_batch(num_samples=n)
    elapsed = round(time.time() - t0, 1)

    raw_dir = config["paths"]["data_raw"]
    np.save(os.path.join(raw_dir, "signal_X.npy"), X)
    np.save(os.path.join(raw_dir, "signal_y.npy"), y)

    label_counts = dict(zip(*np.unique(y, return_counts=True)))
    logger.info(f"Saved signal data → {raw_dir}/signal_*.npy  [{elapsed}s]")
    logger.info(f"  X shape      : {X.shape}")
    logger.info(f"  y shape      : {y.shape}")
    logger.info(f"  Class dist   : {label_counts}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic datasets for Emergency Health AI")
    parser.add_argument("--quick",          action="store_true", help="Quick smoke test (small sizes)")
    parser.add_argument("--tabular-only",   action="store_true", help="Only generate tabular data")
    parser.add_argument("--temporal-only",  action="store_true", help="Only generate temporal data")
    parser.add_argument("--signals-only",   action="store_true", help="Only generate signal data")
    parser.add_argument("--config",         default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_dirs(config)

    run_all = not (args.tabular_only or args.temporal_only or args.signals_only)

    t_start = time.time()
    logger.info("=" * 60)
    logger.info("AI Emergency Health Detection — Data Generation")
    logger.info(f"Quick mode: {args.quick}")
    logger.info("=" * 60)

    if run_all or args.tabular_only:
        generate_tabular(config, quick=args.quick)

    if run_all or args.temporal_only:
        generate_temporal(config, quick=args.quick)

    if run_all or args.signals_only:
        generate_signals(config, quick=args.quick)

    total = round(time.time() - t_start, 1)
    logger.info("\n" + "=" * 60)
    logger.info(f"Data generation complete in {total}s")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
