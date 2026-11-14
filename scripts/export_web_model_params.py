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


if __name__ == "__main__":
    main()
