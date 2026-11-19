#!/usr/bin/env python3
"""End-to-End Training Pipeline.

Orchestrates the full ML/DL workflow:
  generate → validate → preprocess → split → train → evaluate → report

Usage:
    python run_pipeline.py                  # Full run (reads data/raw or generates if missing)
    python run_pipeline.py --quick          # Quick smoke test (small data, 2 epochs; writes to */quick)
    python run_pipeline.py --classical-only # Skip deep learning
    python run_pipeline.py --deep-only      # Skip classical ML
    python run_pipeline.py --skip-generate  # Use existing data/raw files
"""

import os
import sys
import time
import json
import argparse

import numpy as np
import pandas as pd
import yaml

# ── Allow running from project root ──────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.tabular_generator   import TabularDataGenerator
from src.data.temporal_generator  import TemporalDataGenerator
from src.data.signal_generator    import SignalDataGenerator
from src.data.validator           import DataValidator
from src.preprocessing.pipeline   import PreprocessingPipeline
from src.preprocessing.splitters  import split_tabular_data, split_temporal_data
from src.training.train_classical import ClassicalTrainer
from src.training.train_deep      import DeepLearningTrainer
from src.utils.logger             import setup_logger

logger = setup_logger("run_pipeline")


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dirs(config: dict) -> None:
    for key in ["data_raw", "data_interim", "data_processed", "models", "experiments"]:
        p = config["paths"][key]
        os.makedirs(p, exist_ok=True)
    # Subdirs for model artifacts
    os.makedirs(os.path.join(config["paths"]["models"], "classical"), exist_ok=True)
    os.makedirs(os.path.join(config["paths"]["models"], "deep"),      exist_ok=True)
    os.makedirs(os.path.join(config["paths"]["experiments"], "results"), exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_or_generate_tabular(config: dict, quick: bool, skip_generate: bool = False) -> pd.DataFrame:
    raw_path = os.path.join(config["paths"]["data_raw"], "tabular_raw.csv")
    if os.path.exists(raw_path):
        logger.info(f"Loading existing tabular data from {raw_path}")
        return pd.read_csv(raw_path)
    if skip_generate:
        raise FileNotFoundError(
            f"--skip-generate was given but {raw_path} does not exist. "
            "Run `python3 generate_data.py` first, or omit --skip-generate."
        )
    logger.info("Tabular data not found — generating...")
    gen = TabularDataGenerator(config)
    n = 250 if quick else None
    df = gen.generate_dataset(num_samples_per_class=n)
    df.to_csv(raw_path, index=False)
    return df


def load_or_generate_temporal(config: dict, quick: bool, skip_generate: bool = False):
    raw_dir = config["paths"]["data_raw"]
    X_path   = os.path.join(raw_dir, "temporal_X.npy")
    y_path   = os.path.join(raw_dir, "temporal_y.npy")
    ids_path = os.path.join(raw_dir, "temporal_ids.npy")

    if os.path.exists(X_path):
        logger.info(f"Loading existing temporal data from {raw_dir}/temporal_*.npy")
        return (np.load(X_path), np.load(y_path), np.load(ids_path))
    if skip_generate:
        raise FileNotFoundError(
            f"--skip-generate was given but {X_path} does not exist. "
            "Run `python3 generate_data.py` first, or omit --skip-generate."
        )
    logger.info("Temporal data not found — generating...")
    gen = TemporalDataGenerator(config)
    n = 400 if quick else None
    X, y, ids = gen.generate_batch(num_trajectories=n)
    np.save(X_path,   X)
    np.save(y_path,   y)
    np.save(ids_path, ids)
    return X, y, ids


def load_or_generate_signals(config: dict, quick: bool, skip_generate: bool = False):
    raw_dir = config["paths"]["data_raw"]
    X_path  = os.path.join(raw_dir, "signal_X.npy")
    y_path  = os.path.join(raw_dir, "signal_y.npy")

    if os.path.exists(X_path):
        logger.info(f"Loading existing signal data from {raw_dir}/signal_*.npy")
        return np.load(X_path), np.load(y_path)
    if skip_generate:
        raise FileNotFoundError(
            f"--skip-generate was given but {X_path} does not exist. "
            "Run `python3 generate_data.py` first, or omit --skip-generate."
        )
    logger.info("Signal data not found — generating...")
    gen = SignalDataGenerator(config)
    n = 400 if quick else None
    X, y = gen.generate_signal_batch(num_samples=n)
    np.save(X_path, X)
    np.save(y_path, y)
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def use_quick_paths(config: dict) -> dict:
    """Redirect a quick run to ``*/quick`` subdirectories.

    ``--quick`` is a smoke test: it trains on a few hundred rows with two epochs.
    Writing that into ``data/raw``, ``models`` or ``experiments/results`` used to
    overwrite the published production artifacts with toy ones. Keeping every
    quick output in its own subtree makes the mode non-destructive.
    """
    paths = config.setdefault("paths", {})
    for key in ("data_raw", "data_interim", "data_processed",
                "models", "experiments", "reports"):
        if key in paths:
            paths[key] = os.path.join(paths[key], "quick")
    logger.info("Quick mode: writing to */quick paths (production artifacts untouched)")
    return config


def enforce_data_quality(report: dict, allow_issues: bool = False) -> None:
    """Abort on data problems that would silently corrupt every metric.

    The validator already computes these counts, but the pipeline used to log
    them and carry on — so a corrupt file produced a confidently wrong model.
    Duplicates matter in particular: they can straddle the train/test split and
    inflate held-out accuracy.
    """
    problems: list[str] = []

    if not report.get("schema_valid"):
        problems.append(f"schema invalid: {report.get('schema_errors')}")

    duplicates = int(report.get("duplicate_count", 0) or 0)
    if duplicates:
        problems.append(
            f"{duplicates:,} duplicate rows (duplicates can cross the train/test "
            "split and inflate held-out accuracy)"
        )

    missing = {k: v for k, v in (report.get("missing_values") or {}).items() if v}
    if missing:
        problems.append(f"missing values: {missing}")

    out_of_bounds = {
        k: v.get("out_of_bounds_count", 0)
        for k, v in (report.get("range_validation") or {}).items()
        if v.get("out_of_bounds_count")
    }
    if out_of_bounds:
        problems.append(f"values outside the configured ranges: {out_of_bounds}")

    if not problems:
        logger.info("  Data quality gate passed (schema, duplicates, nulls, ranges)")
        return

    for problem in problems:
        logger.error(f"  DATA CHECK FAILED: {problem}")

    if allow_issues:
        logger.warning("  --allow-data-issues given: continuing with invalid data — results will be unreliable.")
        return

    logger.error(
        "Refusing to train on invalid data. Fix the source file, or pass "
        "--allow-data-issues to override."
    )
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Full training pipeline")
    parser.add_argument("--quick",           action="store_true", help="Quick smoke test")
    only = parser.add_mutually_exclusive_group()
    only.add_argument("--classical-only",  action="store_true", help="Skip deep learning")
    only.add_argument("--deep-only",       action="store_true", help="Skip classical ML")
    parser.add_argument("--skip-generate",   action="store_true", help="Use existing data/raw files; fail if missing")
    parser.add_argument("--allow-data-issues", action="store_true",
                        help="Continue even when the data quality gate fails (unreliable results)")
    parser.add_argument("--config",          default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except FileNotFoundError:
        parser.error(f"config file not found: {args.config}")
    except Exception as exc:  # malformed YAML, permissions, …
        parser.error(f"could not read config {args.config}: {exc}")
    if args.quick:
        use_quick_paths(config)
    ensure_dirs(config)

    # Patch epoch count for quick mode
    if args.quick:
        for arch in ["cnn_1d", "rnn", "lstm"]:
            if arch in config.get("models", {}).get("deep", {}):
                config["models"]["deep"][arch]["epochs"] = 2

    t_pipeline_start = time.time()

    logger.info("\n" + "=" * 65)
    logger.info(" AI Emergency Health Detection — Full Training Pipeline")
    logger.info(f" Quick mode: {args.quick}")
    logger.info("=" * 65)

    # ── STEP 1: Load / Generate Data ──────────────────────────────────
    logger.info("\n[STEP 1] Loading / Generating Datasets...")
    try:
        df_tabular                = load_or_generate_tabular(config, args.quick, args.skip_generate)
        X_temporal, y_temporal, ids_temporal = load_or_generate_temporal(config, args.quick, args.skip_generate)
        X_signal,   y_signal                  = load_or_generate_signals(config, args.quick, args.skip_generate)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)

    # ── STEP 2: Validate Tabular Data ─────────────────────────────────
    logger.info("\n[STEP 2] Validating Tabular Data...")
    validator = DataValidator(config)
    report = validator.generate_quality_report(df_tabular)
    logger.info(f"  Records : {report['num_records']:,}  |  Schema valid: {report['schema_valid']}")
    logger.info(f"  Classes : {report['class_distribution']}")
    enforce_data_quality(report, allow_issues=args.allow_data_issues)

    # ── STEP 3: Split Tabular Data ────────────────────────────────────
    logger.info("\n[STEP 3] Splitting Datasets...")
    prep_cfg = config.get("preprocessing", {})
    df_train, df_val, df_test = split_tabular_data(
        df_tabular,
        target_col  = "risk_class",
        train_size  = prep_cfg.get("train_split", 0.70),
        val_size    = prep_cfg.get("val_split",   0.15),
        test_size   = prep_cfg.get("test_split",  0.15),
        seed        = config.get("random_seed", 42),
    )

    # Split temporal sequences
    (X_t_train, y_t_train), (X_t_val, y_t_val), (X_t_test, y_t_test) = split_temporal_data(
        X_temporal, y_temporal, ids_temporal,
        seed=config.get("random_seed", 42),
    )

    # Split signal data
    from sklearn.model_selection import train_test_split
    seed = config.get("random_seed", 42)
    idx_all = np.arange(len(X_signal))
    idx_train, idx_temp = train_test_split(idx_all, train_size=0.70, stratify=y_signal, random_state=seed)
    idx_val,   idx_test = train_test_split(idx_temp, train_size=0.50, stratify=y_signal[idx_temp], random_state=seed)
    X_s_train, y_s_train = X_signal[idx_train], y_signal[idx_train]
    X_s_val,   y_s_val   = X_signal[idx_val],   y_signal[idx_val]
    X_s_test,  y_s_test  = X_signal[idx_test],  y_signal[idx_test]
    logger.info(f"  Signal split → Train:{len(X_s_train)} Val:{len(X_s_val)} Test:{len(X_s_test)}")

    # ── STEP 4: Preprocess Tabular Data ───────────────────────────────
    logger.info("\n[STEP 4] Preprocessing Tabular Data...")
    pipeline = PreprocessingPipeline(config)
    X_tab_train, y_tab_train = pipeline.fit_transform(df_train)
    X_tab_val,   y_tab_val   = pipeline.transform(df_val)
    X_tab_test,  y_tab_test  = pipeline.transform(df_test)
    logger.info(f"  Feature dim: {X_tab_train.shape[1]}")

    # Save fitted pipeline
    pipeline_path = os.path.join(config["paths"]["models"], "preprocessing_pipeline.pkl")
    pipeline.save(pipeline_path)
    logger.info(f"  Pipeline saved → {pipeline_path}")

    # ── STEP 5: Save Processed Arrays ─────────────────────────────────
    proc_dir = config["paths"]["data_processed"]
    np.save(os.path.join(proc_dir, "X_tab_train.npy"), X_tab_train)
    np.save(os.path.join(proc_dir, "y_tab_train.npy"), y_tab_train)
    np.save(os.path.join(proc_dir, "X_tab_val.npy"),   X_tab_val)
    np.save(os.path.join(proc_dir, "y_tab_val.npy"),   y_tab_val)
    np.save(os.path.join(proc_dir, "X_tab_test.npy"),  X_tab_test)
    np.save(os.path.join(proc_dir, "y_tab_test.npy"),  y_tab_test)
    logger.info(f"  Processed arrays saved → {proc_dir}/")

    classical_results = {}
    deep_results      = {}

    # ── STEP 6: Train Classical Models ────────────────────────────────
    if not args.deep_only:
        logger.info("\n[STEP 6] Training Classical ML Models...")
        classical_trainer = ClassicalTrainer(config)
        classical_results = classical_trainer.train_all(
            X_tab_train, y_tab_train,
            X_tab_val,   y_tab_val,
            X_tab_test,  y_tab_test,
        )

    # ── STEP 7: Train Deep Learning Models ────────────────────────────
    if not args.classical_only:
        try:
            import torch
            logger.info("\n[STEP 7] Training Deep Learning Models...")
            deep_trainer = DeepLearningTrainer(config)
            deep_results = deep_trainer.train_all(
                X_t_train, y_t_train, X_t_val, y_t_val, X_t_test, y_t_test,
                X_s_train, y_s_train, X_s_val, y_s_val, X_s_test, y_s_test,
            )
        except ImportError:
            logger.warning("PyTorch not available — skipping deep learning training.")

    # ── STEP 8: Save Pipeline Summary ─────────────────────────────────
    logger.info("\n[STEP 8] Saving Pipeline Summary...")
    total_time = round(time.time() - t_pipeline_start, 1)

    summary = {
        "dataset_sizes": {
            "tabular_total":    len(df_tabular),
            "temporal_total":   len(X_temporal),
            "signal_total":     len(X_signal),
        },
        "splits": {
            "tabular":  {"train": len(df_train),   "val": len(df_val),   "test": len(df_test)},
            "temporal": {"train": len(X_t_train),  "val": len(X_t_val),  "test": len(X_t_test)},
            "signal":   {"train": len(X_s_train),  "val": len(X_s_val),  "test": len(X_s_test)},
        },
        "feature_dim":   int(X_tab_train.shape[1]),
        "total_time_s":  total_time,
        "quick_mode":    args.quick,
    }

    summary_path = os.path.join(config["paths"]["experiments"], "results", "pipeline_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ── STEP 9: Final Report ───────────────────────────────────────────
    logger.info("\n" + "=" * 65)
    logger.info(" PIPELINE COMPLETE")
    logger.info(f" Total time: {total_time}s")
    logger.info("=" * 65)
    logger.info(f"  Tabular rows  : {len(df_tabular):,}")
    logger.info(f"  Temporal seqs : {len(X_temporal):,}")
    logger.info(f"  Signal samples: {len(X_signal):,}")
    logger.info(f"  Feature dim   : {X_tab_train.shape[1]}")
    logger.info(f"\n  Artifacts saved to:")
    logger.info(f"    Models      → {config['paths']['models']}/")
    logger.info(f"    Results     → {config['paths']['experiments']}/results/")
    logger.info(f"    Processed   → {config['paths']['data_processed']}/")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
