"""Export the fitted sklearn artifacts into a portable JSON file for the web app.

The Vercel deployment runs the model in a Next.js API route (TypeScript), so it
must not depend on Python, joblib pickles, or scikit-learn at runtime. This
script extracts the exact parameters needed to reproduce preprocessing and the
multinomial Logistic Regression decision function:

    StandardScaler.mean_ / scale_        -> scaler_mean / scaler_scale
    SimpleImputer.statistics_            -> imputer_statistics
    LogisticRegression.coef_ / intercept_-> coef / intercept

Run after training (models/ is git-ignored, the JSON output is committed):

    python3 scripts/export_web_model_params.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import joblib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PIPELINE_PATH = ROOT / "models" / "preprocessing_pipeline.pkl"
MODEL_PATH = ROOT / "models" / "classical" / "logistic_regression.joblib"
OUTPUT_PATH = ROOT / "web" / "lib" / "model-params.json"


def main() -> None:
    pipeline = joblib.load(PIPELINE_PATH)
    model = joblib.load(MODEL_PATH)

    params = {
        "model": "LogisticRegression (multinomial, lbfgs)",
        "features": list(pipeline.numeric_cols),
        "imputer_statistics": [float(x) for x in pipeline.imputer.statistics_],
        "scaler_mean": [float(x) for x in pipeline.scaler.mean_],
        "scaler_scale": [float(x) for x in pipeline.scaler.scale_],
        "coef": [[float(x) for x in row] for row in model.coef_],
        "intercept": [float(x) for x in model.intercept_],
        "classes": [int(c) for c in model.classes_],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(params, indent=2) + "\n")
    print(f"Wrote {OUTPUT_PATH} ({len(params['features'])} features)")

    # Sanity check: the exported parameters must reproduce the real model.
    # This catches the case where models/classical/*.joblib was overwritten by
    # a test run (200-sample smoke model) before the export happened.
    x_path = ROOT / "data" / "processed" / "X_tab_test.npy"
    y_path = ROOT / "data" / "processed" / "y_tab_test.npy"
    if not (x_path.exists() and y_path.exists()):
        print("Note: no processed test split found; skipping accuracy check.")
        return

    X = np.load(x_path)
    y = np.load(y_path)
    logits = X @ np.array(params["coef"]).T + np.array(params["intercept"])
    accuracy = float((logits.argmax(axis=1) == y).mean())
    print(f"Held-out accuracy from exported params: {accuracy:.4f}")

    if accuracy < 0.95:
        print(
            "WARNING: accuracy is far below the expected ~0.97 — models/ may have "
            "been overwritten by a test run. Retrain with:\n"
            "  python3 run_pipeline.py --classical-only --skip-generate\n"
            "then re-run this script.",
            file=sys.stderr,
        )
        sys.exit(1)



if __name__ == "__main__":
    main()
