"""Export the fitted sklearn artifacts into portable JSON for the web app.

The Vercel deployment runs the model in a Next.js API route (TypeScript), so it
must not depend on Python, joblib pickles, or scikit-learn at runtime. This
script extracts the exact parameters needed to reproduce preprocessing and the
multinomial Logistic Regression decision function:

    StandardScaler.mean_ / scale_        -> scaler_mean / scaler_scale
    SimpleImputer.statistics_            -> imputer_statistics
    LogisticRegression.coef_ / intercept_-> coef / intercept

It also records the observed min/max of each vital in the training set, which
the API uses to warn when a request is an extrapolation.

Run after training (models/ is git-ignored, the JSON output is committed):

    python3 scripts/export_web_model_params.py

The held-out accuracy check runs **before** anything is written: a failed check
leaves the committed parameters untouched and exits non-zero.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import joblib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PIPELINE_PATH = ROOT / "models" / "preprocessing_pipeline.pkl"
MODEL_PATH = ROOT / "models" / "classical" / "logistic_regression.joblib"
OUTPUT_PATH = ROOT / "web" / "lib" / "model-params.json"
SUPPORT_PATH = ROOT / "web" / "lib" / "training-support.json"
RAW_TABULAR = ROOT / "data" / "raw" / "tabular_raw.csv"

MIN_ACCEPTABLE_ACCURACY = 0.965
MIN_TEST_ROWS = 8000
SUPPORT_FIELDS = [
    "age", "heart_rate", "systolic_bp", "diastolic_bp",
    "spo2", "temperature", "respiratory_rate",
]


def write_atomic(path: Path, text: str) -> None:
    """Write to a sibling temp file, then replace, so a crash cannot corrupt it."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def build_params() -> dict:
    pipeline = joblib.load(PIPELINE_PATH)
    model = joblib.load(MODEL_PATH)
    return {
        "model": "LogisticRegression (multinomial, lbfgs)",
        "features": list(pipeline.numeric_cols),
        "imputer_statistics": [float(x) for x in pipeline.imputer.statistics_],
        "scaler_mean": [float(x) for x in pipeline.scaler.mean_],
        "scaler_scale": [float(x) for x in pipeline.scaler.scale_],
        "coef": [[float(x) for x in row] for row in model.coef_],
        "intercept": [float(x) for x in model.intercept_],
        "classes": [int(c) for c in model.classes_],
    }


def verify(params: dict, allow_small: bool = False) -> float | None:
    """Score the exported parameters on the held-out split (already scaled)."""
    x_path = ROOT / "data" / "processed" / "X_tab_test.npy"
    y_path = ROOT / "data" / "processed" / "y_tab_test.npy"
    if not (x_path.exists() and y_path.exists()):
        print("Note: no processed test split found; skipping accuracy check.")
        return None

    X = np.load(x_path)
    y = np.load(y_path)
    print(f"Held-out rows: {len(y):,}")

    if len(y) < MIN_TEST_ROWS and not allow_small:
        print(
            f"ERROR: only {len(y):,} held-out rows — this looks like a --quick run. "
            "Refusing to overwrite the committed parameters. Retrain at full scale:\n"
            "  python3 run_pipeline.py --classical-only --skip-generate\n"
            "or pass --allow-small if you really mean it.",
            file=sys.stderr,
        )
        return 0.0

    logits = X @ np.array(params["coef"]).T + np.array(params["intercept"])
    accuracy = float((logits.argmax(axis=1) == y).mean())
    print(f"Held-out accuracy from exported params: {accuracy:.4f}")

    if accuracy < MIN_ACCEPTABLE_ACCURACY:
        print(
            f"ERROR: accuracy {accuracy:.4f} is below {MIN_ACCEPTABLE_ACCURACY:.3f}. "
            "models/ has probably been overwritten by a test or smoke run. Nothing "
            "was written; retrain with:\n"
            "  python3 run_pipeline.py --classical-only --skip-generate\n"
            "then re-run this script.",
            file=sys.stderr,
        )
    return accuracy


def build_support() -> dict | None:
    """Observed min/max of each vital in the synthetic training set."""
    if not RAW_TABULAR.exists():
        print(f"Note: {RAW_TABULAR} not found; keeping existing training-support.json")
        return None
    import pandas as pd

    df = pd.read_csv(RAW_TABULAR, usecols=SUPPORT_FIELDS)
    return {f: {"min": float(df[f].min()), "max": float(df[f].max())} for f in SUPPORT_FIELDS}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Export the served model parameters")
    parser.add_argument(
        "--allow-small", action="store_true",
        help="Permit exporting from a --quick (small) held-out split",
    )
    args = parser.parse_args()

    params = build_params()

    accuracy = verify(params, allow_small=args.allow_small)
    if accuracy is not None and accuracy < MIN_ACCEPTABLE_ACCURACY:
        sys.exit(1)

    write_atomic(OUTPUT_PATH, json.dumps(params, indent=2) + "\n")
    print(f"Wrote {OUTPUT_PATH} ({len(params['features'])} features)")

    support = build_support()
    if support is not None:
        write_atomic(SUPPORT_PATH, json.dumps(support, indent=2) + "\n")
        print(f"Wrote {SUPPORT_PATH} ({len(support)} fields)")


if __name__ == "__main__":
    main()
