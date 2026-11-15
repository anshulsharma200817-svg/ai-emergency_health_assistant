"""Comprehensive test suite for the AI Emergency Health Detection System.

Covers:
  - Data generators (tabular, temporal, signal)
  - Preprocessing pipeline (feature engineering, scaling, label encoding)
  - Data splits (stratified, no-leakage)
  - Classical model training and prediction
  - Inference predictor
  - Alert engine
  - Data validator
"""

import os
import sys
import json
import tempfile
import pytest
import numpy as np
import pandas as pd

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def config():
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def small_tabular_df(config):
    from src.data.tabular_generator import TabularDataGenerator
    gen = TabularDataGenerator(config)
    return gen.generate_dataset(num_samples_per_class=50)


@pytest.fixture(scope="session")
def fitted_pipeline(config, small_tabular_df):
    from src.preprocessing.pipeline import PreprocessingPipeline
    pipe = PreprocessingPipeline(config)
    pipe.fit(small_tabular_df)
    return pipe


# ─────────────────────────────────────────────────────────────────────────────
# 1. Tabular Generator Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTabularGenerator:

    def test_shape(self, small_tabular_df):
        assert small_tabular_df.shape[0] == 200   # 50 × 4 classes
        assert "risk_class" in small_tabular_df.columns

    def test_class_balance(self, small_tabular_df):
        counts = small_tabular_df["risk_class"].value_counts()
        assert set(counts.index) == {"NORMAL", "MODERATE", "HIGH", "CRITICAL"}
        # Each class should be close to 50 (some borderline/outlier mixing)
        for c in counts:
            assert 40 <= c <= 60, f"Class count {c} out of expected range"

    def test_no_nulls_in_key_cols(self, small_tabular_df):
        key_cols = ["heart_rate", "systolic_bp", "diastolic_bp", "spo2",
                    "temperature", "respiratory_rate"]
        assert small_tabular_df[key_cols].isnull().sum().sum() == 0

    def test_physiological_ranges(self, small_tabular_df):
        assert small_tabular_df["heart_rate"].between(25, 240).all()
        assert small_tabular_df["systolic_bp"].between(45, 255).all()
        assert small_tabular_df["spo2"].between(55, 100).all()
        assert small_tabular_df["temperature"].between(30.5, 44.5).all()

    def test_diastolic_less_than_systolic(self, small_tabular_df):
        assert (small_tabular_df["diastolic_bp"] < small_tabular_df["systolic_bp"]).all()

    def test_binary_symptoms(self, small_tabular_df):
        for col in ["chest_pain", "breathing_difficulty", "dizziness",
                    "unconsciousness", "bleeding_injury"]:
            assert set(small_tabular_df[col].unique()).issubset({0, 1})

    def test_unique_case_ids(self, small_tabular_df):
        assert small_tabular_df["case_id"].nunique() == len(small_tabular_df)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Temporal Generator Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTemporalGenerator:

    @pytest.fixture(scope="class")
    def temporal_batch(self, config):
        from src.data.temporal_generator import TemporalDataGenerator
        gen = TemporalDataGenerator(config)
        return gen.generate_batch(num_trajectories=60)

    def test_shapes(self, temporal_batch, config):
        X, y, ids = temporal_batch
        seq_len = config.get("sequence_length", 60)
        assert X.shape == (60, seq_len, 6)
        assert y.shape == (60,)
        assert ids.shape == (60,)

    def test_label_range(self, temporal_batch):
        _, y, _ = temporal_batch
        assert set(np.unique(y)).issubset({0, 1, 2, 3})

    def test_physiological_values(self, temporal_batch):
        X, _, _ = temporal_batch
        # heart_rate channel
        assert X[:, :, 0].max() < 250
        assert X[:, :, 0].min() > 20
        # spo2 channel
        assert X[:, :, 3].max() <= 100
        assert X[:, :, 3].min() >= 50

    def test_diastolic_less_than_systolic(self, temporal_batch):
        X, _, _ = temporal_batch
        # sbp=channel 1, dbp=channel 2
        assert (X[:, :, 2] < X[:, :, 1]).all()

    def test_dtype(self, temporal_batch):
        X, y, _ = temporal_batch
        assert X.dtype == np.float32
        assert y.dtype == np.int64


# ─────────────────────────────────────────────────────────────────────────────
# 3. Signal Generator Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSignalGenerator:

    @pytest.fixture(scope="class")
    def signal_batch(self, config):
        from src.data.signal_generator import SignalDataGenerator
        gen = SignalDataGenerator(config)
        return gen.generate_signal_batch(num_samples=40)

    def test_shape(self, signal_batch, config):
        X, y = signal_batch
        sig_len = config.get("signal_length", 500)
        assert X.shape == (40, 1, sig_len)
        assert y.shape == (40,)

    def test_label_range(self, signal_batch):
        _, y = signal_batch
        assert set(np.unique(y)).issubset({0, 1, 2, 3})

    def test_balanced_classes(self, signal_batch):
        _, y = signal_batch
        counts = np.bincount(y)
        assert len(counts) == 4
        for c in counts:
            assert c == 10, f"Expected 10 per class, got {c}"

    def test_dtype(self, signal_batch):
        X, y = signal_batch
        assert X.dtype == np.float32
        assert y.dtype == np.int64

    def test_single_signal(self, config):
        from src.data.signal_generator import SignalDataGenerator
        gen = SignalDataGenerator(config)
        sig, cat = gen.generate_single_signal(500, "TACHY_PATTERN")
        assert sig.shape == (500,)
        assert cat == "TACHY_PATTERN"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Feature Engineering Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureEngineering:

    def test_derived_columns_added(self, small_tabular_df, config):
        from src.preprocessing.feature_engineering import FeatureEngineer
        fe = FeatureEngineer(config)
        out = fe.transform(small_tabular_df)
        for col in ["pulse_pressure", "mean_arterial_pressure", "shock_index",
                    "age_group", "spo2_severity", "tachycardia_flag",
                    "hypotension_flag", "fever_flag", "tachypnea_flag"]:
            assert col in out.columns, f"Missing derived column: {col}"

    def test_pulse_pressure_formula(self, small_tabular_df, config):
        from src.preprocessing.feature_engineering import FeatureEngineer
        fe = FeatureEngineer(config)
        out = fe.transform(small_tabular_df)
        expected = small_tabular_df["systolic_bp"] - small_tabular_df["diastolic_bp"]
        np.testing.assert_allclose(out["pulse_pressure"].values, expected.values, rtol=1e-5)

    def test_binary_flags_are_binary(self, small_tabular_df, config):
        from src.preprocessing.feature_engineering import FeatureEngineer
        fe = FeatureEngineer(config)
        out = fe.transform(small_tabular_df)
        for flag in ["tachycardia_flag", "hypotension_flag", "fever_flag", "tachypnea_flag"]:
            assert set(out[flag].unique()).issubset({0, 1})


# ─────────────────────────────────────────────────────────────────────────────
# 5. Preprocessing Pipeline Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocessingPipeline:

    def test_fit_transform_shape(self, small_tabular_df, config):
        from src.preprocessing.pipeline import PreprocessingPipeline
        pipe = PreprocessingPipeline(config)
        X, y = pipe.fit_transform(small_tabular_df)
        assert X.ndim == 2
        assert X.shape[0] == len(small_tabular_df)
        assert y.shape[0] == len(small_tabular_df)

    def test_labels_integer_encoded(self, small_tabular_df, config):
        from src.preprocessing.pipeline import PreprocessingPipeline
        pipe = PreprocessingPipeline(config)
        _, y = pipe.fit_transform(small_tabular_df)
        assert set(np.unique(y)).issubset({0, 1, 2, 3})

    def test_no_nan_in_X(self, small_tabular_df, config):
        from src.preprocessing.pipeline import PreprocessingPipeline
        pipe = PreprocessingPipeline(config)
        X, _ = pipe.fit_transform(small_tabular_df)
        assert not np.isnan(X).any()

    def test_requires_fit_before_transform(self, small_tabular_df, config):
        from src.preprocessing.pipeline import PreprocessingPipeline
        pipe = PreprocessingPipeline(config)
        with pytest.raises(RuntimeError):
            pipe.transform(small_tabular_df)

    def test_save_and_load(self, fitted_pipeline, tmp_path):
        path = str(tmp_path / "pipeline.pkl")
        fitted_pipeline.save(path)
        from src.preprocessing.pipeline import PreprocessingPipeline
        loaded = PreprocessingPipeline.load(path)
        assert loaded.is_fitted
        assert loaded.num_features == fitted_pipeline.num_features

    def test_decode_labels(self):
        from src.preprocessing.pipeline import PreprocessingPipeline
        y = np.array([0, 1, 2, 3])
        decoded = PreprocessingPipeline.decode_labels(y)
        assert decoded == ["NORMAL", "MODERATE", "HIGH", "CRITICAL"]


# ─────────────────────────────────────────────────────────────────────────────
# 6. Data Splitter Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDataSplitters:

    def test_tabular_split_sizes(self, small_tabular_df):
        from src.preprocessing.splitters import split_tabular_data
        train, val, test = split_tabular_data(small_tabular_df, seed=42)
        total = len(small_tabular_df)
        assert len(train) == pytest.approx(total * 0.70, abs=5)
        assert len(val)   == pytest.approx(total * 0.15, abs=5)
        assert len(test)  == pytest.approx(total * 0.15, abs=5)

    def test_tabular_split_no_overlap(self, small_tabular_df):
        from src.preprocessing.splitters import split_tabular_data
        train, val, test = split_tabular_data(small_tabular_df, seed=42)
        train_ids = set(train["case_id"])
        val_ids   = set(val["case_id"])
        test_ids  = set(test["case_id"])
        assert len(train_ids & val_ids) == 0
        assert len(train_ids & test_ids) == 0
        assert len(val_ids & test_ids) == 0

    def test_temporal_split_no_leakage(self, config):
        from src.data.temporal_generator import TemporalDataGenerator
        from src.preprocessing.splitters import split_temporal_data
        gen = TemporalDataGenerator(config)
        X, y, ids = gen.generate_batch(num_trajectories=60)
        (Xtr, ytr), (Xv, yv), (Xt, yt) = split_temporal_data(X, y, ids)
        # No ID leakage
        assert len(Xtr) + len(Xv) + len(Xt) == len(X)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Classical Model Training Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestClassicalTraining:

    @pytest.fixture(scope="class")
    def trained_rf(self, config, small_tabular_df, tmp_path_factory):
        from src.preprocessing.pipeline import PreprocessingPipeline
        from src.preprocessing.splitters import split_tabular_data
        from src.training.train_classical import ClassicalTrainer

        train_df, val_df, test_df = split_tabular_data(small_tabular_df)
        pipe = PreprocessingPipeline(config)
        X_tr, y_tr = pipe.fit_transform(train_df)
        X_v,  y_v  = pipe.transform(val_df)
        X_t,  y_t  = pipe.transform(test_df)

        # Point artifacts at a temp dir: ClassicalTrainer writes to
        # paths.models/classical and paths.experiments/results, and tests must
        # never overwrite the real trained models or experiment metrics.
        test_config = {
            **config,
            "paths": {
                **config.get("paths", {}),
                "models": str(tmp_path_factory.mktemp("models")),
                "experiments": str(tmp_path_factory.mktemp("experiments")),
            },
        }

        trainer = ClassicalTrainer(test_config)
        return trainer.train_all(X_tr, y_tr, X_v, y_v, X_t, y_t)

    def test_all_models_present(self, trained_rf):
        assert set(trained_rf.keys()) == {"knn", "logistic_regression",
                                           "decision_tree", "random_forest"}

    def test_metrics_in_range(self, trained_rf):
        for name, result in trained_rf.items():
            acc = result["val_metrics"]["accuracy"]
            assert 0.0 <= acc <= 1.0, f"{name} val accuracy {acc} out of range"

    def test_better_than_random(self, trained_rf):
        """Models should beat random baseline (25% for 4 classes)."""
        for name, result in trained_rf.items():
            acc = result["val_metrics"]["accuracy"]
            assert acc > 0.30, f"{name} val accuracy {acc:.3f} not better than random"


# ─────────────────────────────────────────────────────────────────────────────
# 8. Inference Predictor Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestInferencePredictor:

    @pytest.fixture(scope="class")
    def predictor(self, config, small_tabular_df, fitted_pipeline):
        from src.preprocessing.splitters import split_tabular_data
        from src.training.train_classical import ClassicalTrainer

        train_df, _, _ = split_tabular_data(small_tabular_df)
        X_tr, y_tr = fitted_pipeline.fit_transform(train_df)

        from src.models.classical.random_forest import RandomForestModel
        rf_cfg = config.get("models", {}).get("classical", {}).get("random_forest", {})
        rf = RandomForestModel(rf_cfg)
        rf.fit(X_tr, y_tr)

        from src.inference.predictor import HealthRiskPredictor
        return HealthRiskPredictor(rf.model, fitted_pipeline, config, model_name="random_forest")

    def test_predict_record_keys(self, predictor, small_tabular_df):
        record = small_tabular_df.iloc[0].to_dict()
        result = predictor.predict_record(record)
        assert "predicted_risk_class" in result
        assert "probabilities" in result
        assert "confidence" in result

    def test_predicted_class_valid(self, predictor, small_tabular_df):
        record = small_tabular_df.iloc[0].to_dict()
        result = predictor.predict_record(record)
        assert result["predicted_risk_class"] in {"NORMAL", "MODERATE", "HIGH", "CRITICAL"}

    def test_probabilities_sum_to_one(self, predictor, small_tabular_df):
        record = small_tabular_df.iloc[0].to_dict()
        result = predictor.predict_record(record)
        prob_sum = sum(result["probabilities"].values())
        assert abs(prob_sum - 1.0) < 0.01

    def test_batch_prediction(self, predictor, small_tabular_df):
        records = small_tabular_df.head(10).to_dict(orient="records")
        results = predictor.predict_batch(records)
        assert len(results) == 10
        for r in results:
            assert "predicted_risk_class" in r

    def test_unconsciousness_constraint_single(self, predictor, small_tabular_df):
        """`unconsciousness = 1` must never resolve to NORMAL or MODERATE.

        That flag never appears in the low-risk classes of the training data, so
        the serving layer restricts the output distribution to HIGH/CRITICAL.
        """
        normal = small_tabular_df[small_tabular_df["risk_class"] == "NORMAL"]
        record = (normal if len(normal) else small_tabular_df).iloc[0].to_dict()
        record["unconsciousness"] = 1

        result = predictor.predict_record(record)

        assert result["applied_constraint"] == "unconsciousness"
        assert result["predicted_risk_class"] in {"HIGH", "CRITICAL"}
        assert result["probabilities"]["NORMAL"] == 0
        assert result["probabilities"]["MODERATE"] == 0
        assert abs(sum(result["probabilities"].values()) - 1.0) < 0.01
        assert abs(result["confidence"] - max(result["probabilities"].values())) < 0.001

    def test_unconsciousness_constraint_batch(self, predictor, small_tabular_df):
        records = small_tabular_df.head(5).to_dict(orient="records")
        for record in records:
            record["unconsciousness"] = 1
        results = predictor.predict_batch(records)
        assert len(results) == 5
        for r in results:
            assert r["applied_constraint"] == "unconsciousness"
            assert r["predicted_risk_class"] in {"HIGH", "CRITICAL"}
            assert r["probabilities"]["NORMAL"] == 0
            assert r["probabilities"]["MODERATE"] == 0

    def test_no_constraint_without_flag(self, predictor, small_tabular_df):
        record = small_tabular_df.iloc[0].to_dict()
        record["unconsciousness"] = 0
        result = predictor.predict_record(record)
        assert result["applied_constraint"] is None


# ─────────────────────────────────────────────────────────────────────────────
# 9. Alert Engine Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAlertEngine:

    @pytest.fixture
    def engine(self, config):
        from src.inference.alert_engine import SimulatedAlertEngine
        return SimulatedAlertEngine(config)

    @pytest.mark.parametrize("risk_class,expected_priority", [
        ("CRITICAL", "P0"),
        ("HIGH",     "P1"),
        ("MODERATE", "P2"),
        ("NORMAL",   "P3"),
    ])
    def test_priority_mapping(self, engine, risk_class, expected_priority):
        pred = {"case_id": "TEST_001", "predicted_risk_class": risk_class, "probabilities": {}}
        alert = engine.generate_alert(pred)
        assert alert["priority"] == expected_priority

    def test_alert_has_disclaimer(self, engine):
        pred = {"case_id": "TEST_001", "predicted_risk_class": "HIGH", "probabilities": {}}
        alert = engine.generate_alert(pred)
        assert "disclaimer" in alert
        assert "EDUCATIONAL" in alert["disclaimer"].upper()

    @pytest.mark.parametrize("bogus", ["BOGUS", "", "critical", "Normal", None])
    def test_unknown_class_fails_closed(self, engine, bogus):
        """An unrecognised class must raise, never silently become P3/NORMAL."""
        pred = {"case_id": "TEST_001", "predicted_risk_class": bogus, "probabilities": {}}
        with pytest.raises(ValueError):
            engine.generate_alert(pred)

    def test_missing_class_fails_closed(self, engine):
        with pytest.raises(ValueError):
            engine.generate_alert({"case_id": "TEST_001", "probabilities": {}})


# ─────────────────────────────────────────────────────────────────────────────
# 10. Data Validator Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDataValidator:

    def test_valid_dataset_passes(self, small_tabular_df, config):
        from src.data.validator import DataValidator
        v = DataValidator(config)
        is_valid, errors = v.validate_schema(small_tabular_df)
        assert is_valid, f"Validation errors: {errors}"

    def test_quality_report_structure(self, small_tabular_df, config):
        from src.data.validator import DataValidator
        v = DataValidator(config)
        report = v.generate_quality_report(small_tabular_df)
        for key in ["num_records", "schema_valid", "class_distribution",
                    "missing_values", "duplicate_count"]:
            assert key in report

    def test_no_duplicates_generated(self, small_tabular_df, config):
        from src.data.validator import DataValidator
        v = DataValidator(config)
        dup_count = v.check_duplicates(small_tabular_df)
        assert dup_count == 0

    def test_range_report_no_violations(self, small_tabular_df, config):
        from src.data.validator import DataValidator
        v = DataValidator(config)
        range_report = v.check_physiological_ranges(small_tabular_df)
        for feature, info in range_report.items():
            assert info["out_of_bounds_count"] == 0, (
                f"Feature '{feature}' has {info['out_of_bounds_count']} out-of-bounds values"
            )


# ─────────────────────────────────────────────────────────────────────────────
# 11. Pipeline data-quality gate
# ─────────────────────────────────────────────────────────────────────────────

class TestDataQualityGate:
    """The pipeline must refuse to train on data the validator rejects."""

    @staticmethod
    def _report(**overrides):
        report = {
            "schema_valid": True,
            "schema_errors": [],
            "duplicate_count": 0,
            "missing_values": {"age": 0, "heart_rate": 0},
            "range_validation": {
                "age": {"out_of_bounds_count": 0},
                "heart_rate": {"out_of_bounds_count": 0},
            },
        }
        report.update(overrides)
        return report

    def test_clean_report_passes(self):
        from run_pipeline import enforce_data_quality
        enforce_data_quality(self._report())  # must not raise

    @pytest.mark.parametrize("overrides", [
        {"schema_valid": False, "schema_errors": ["missing column: spo2"]},
        {"duplicate_count": 3},
        {"missing_values": {"age": 2, "heart_rate": 0}},
        {"range_validation": {"age": {"out_of_bounds_count": 1}}},
    ])
    def test_invalid_data_exits(self, overrides):
        from run_pipeline import enforce_data_quality
        with pytest.raises(SystemExit) as exc:
            enforce_data_quality(self._report(**overrides))
        assert exc.value.code == 1

    def test_override_continues(self):
        from run_pipeline import enforce_data_quality
        enforce_data_quality(self._report(duplicate_count=5), allow_issues=True)
