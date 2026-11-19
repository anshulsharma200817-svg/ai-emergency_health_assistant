#!/usr/bin/env python3
"""Export display-ready training results for the web app.

The home screen visualises the actual training outcomes: per-model test
accuracy, the served model's confusion matrix and per-class scores, and the
deep-learning validation curves. Those artifacts live under ``experiments/``
and ``models/``, which are git-ignored, so this script copies the numbers the
UI needs into ``web/lib/training-results.json`` (committed).

Run after a full pipeline (or after ``--classical-only`` / ``--deep-only``):

    python3 scripts/export_web_training_results.py

Every value in the exported file comes from the fitted artifacts; nothing is
typed by hand. That is what keeps the published charts honest after a retrain.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "experiments" / "results"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
OUT_PATH = ROOT / "web" / "lib" / "training-results.json"

CLASSES = ["NORMAL", "MODERATE", "HIGH", "CRITICAL"]


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def served_model_report() -> Dict[str, Any]:
    """Re-evaluate the served Logistic Regression on the held-out test split."""
    X_test = np.load(PROCESSED_DIR / "X_tab_test.npy")
    y_test = np.load(PROCESSED_DIR / "y_tab_test.npy")

    model_path = MODELS_DIR / "classical" / "logistic_regression.joblib"
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    labels = list(range(len(CLASSES)))
    matrix = confusion_matrix(y_test, y_pred, labels=labels).tolist()
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=labels, zero_division=0
    )

    per_class = {
        name: {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }
        for i, name in enumerate(CLASSES)
    }

    return {
        "model": "logistic_regression",
        "test_accuracy": round(float((y_pred == y_test).mean()), 4),
        "test_macro_f1": round(float(f1.mean()), 4),
        "confusion_matrix": matrix,
        "classes": CLASSES,
        "per_class": per_class,
        "test_records": int(len(y_test)),
    }


def main() -> None:
    classical = load_json(RESULTS_DIR / "classical_results.json")
    deep = load_json(RESULTS_DIR / "deep_results.json")

    classical_view = {
        name: {
            "val_accuracy": entry.get("val_accuracy"),
            "val_macro_f1": entry.get("val_macro_f1"),
            "test_accuracy": entry.get("test_accuracy"),
            "test_macro_f1": entry.get("test_macro_f1"),
            "train_time_s": entry.get("train_time_s"),
        }
        for name, entry in classical.items()
    }

    deep_view = {
        name: {
            "val_accuracy": entry.get("val_accuracy"),
            "val_macro_f1": entry.get("val_macro_f1"),
            "test_accuracy": entry.get("test_accuracy"),
            "test_macro_f1": entry.get("test_macro_f1"),
            "train_time_s": entry.get("total_time_s"),
        }
        for name, entry in deep.items()
    }

    histories = {
        name: entry["history"]
        for name, entry in deep.items()
        if isinstance(entry.get("history"), dict)
    }

    payload = {
        "served": served_model_report(),
        "classical": classical_view,
        "deep": deep_view,
        "histories": histories,
        "provenance": {
            "classical": "experiments/results/classical_results.json",
            "deep": "experiments/results/deep_results.json",
            "served": "models/classical/logistic_regression.joblib on data/processed/X_tab_test.npy",
        },
    }

    tmp = OUT_PATH.with_suffix(OUT_PATH.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, OUT_PATH)
    print(f"Wrote {OUT_PATH.relative_to(ROOT)}")
    print(
        f"  served LR: accuracy={payload['served']['test_accuracy']:.4f} "
        f"on {payload['served']['test_records']:,} test records"
    )
    for name, entry in {**classical_view, **deep_view}.items():
        print(f"  {name:<22} test_accuracy={entry['test_accuracy']}")


if __name__ == "__main__":
    main()
