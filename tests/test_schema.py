"""Unit tests for Data Schema definition and derived feature engineering."""

import pandas as pd
import pytest

from src.data.schema import (
    RiskClass,
    RAW_FEATURES,
    DERIVED_FEATURES,
    TARGET_RISK_CLASSES,
    HealthRecordSchema,
)
from src.preprocessing.feature_engineering import FeatureEngineer


def test_target_risk_classes():
    """Verify target risk classes match specification."""
    expected = ["NORMAL", "MODERATE", "HIGH", "CRITICAL"]
    assert TARGET_RISK_CLASSES == expected, f"Expected {expected}, got {TARGET_RISK_CLASSES}"


def test_raw_features_list():
    """Verify raw features list contains required features."""
    required = [
        "case_id", "age", "sex", "heart_rate", "systolic_bp", "diastolic_bp",
        "spo2", "temperature", "respiratory_rate", "chest_pain",
        "breathing_difficulty", "dizziness", "unconsciousness", "bleeding_injury"
    ]
    for req in required:
        assert req in RAW_FEATURES, f"Missing required raw feature: {req}"


def test_derived_features_list():
    """Verify derived features list contains specified calculated features."""
    required = [
        "pulse_pressure", "mean_arterial_pressure", "shock_index", "age_group",
        "spo2_severity", "tachycardia_flag", "hypotension_flag", "fever_flag", "tachypnea_flag"
    ]
    for req in required:
        assert req in DERIVED_FEATURES, f"Missing required derived feature: {req}"


def test_health_record_schema_validation():
    """Test HealthRecordSchema dataclass validation."""
    valid_record = HealthRecordSchema(
        case_id="CASE_000001",
        age=45.0,
        sex=1,
        heart_rate=75.0,
        systolic_bp=120.0,
        diastolic_bp=80.0,
        spo2=98.0,
        temperature=36.8,
        respiratory_rate=16.0,
        chest_pain=0,
        breathing_difficulty=0,
        dizziness=0,
        unconsciousness=0,
        bleeding_injury=0,
        risk_class=RiskClass.NORMAL
    )
    is_valid, errors = valid_record.validate_schema()
    assert is_valid is True, f"Expected valid record, got errors: {errors}"


def test_derived_feature_computation():
    """Test FeatureEngineer derived features calculation logic."""
    df_raw = pd.DataFrame([{
        "case_id": "CASE_001",
        "age": 50,
        "sex": 1,
        "heart_rate": 110,
        "systolic_bp": 85,
        "diastolic_bp": 55,
        "spo2": 85,
        "temperature": 39.0,
        "respiratory_rate": 24,
        "chest_pain": 1,
        "breathing_difficulty": 1,
        "dizziness": 1,
        "unconsciousness": 0,
        "bleeding_injury": 0,
        "risk_class": "HIGH"
    }])

    fe = FeatureEngineer()
    df_engineered = fe.transform(df_raw)

    # Assert derived columns exist
    for derived in DERIVED_FEATURES:
        assert derived in df_engineered.columns, f"Derived feature '{derived}' not in engineered DataFrame"

    row = df_engineered.iloc[0]
    assert row["pulse_pressure"] == 30.0  # 85 - 55
    assert pytest.approx(row["mean_arterial_pressure"], 0.01) == 65.0  # 55 + 30/3
    assert pytest.approx(row["shock_index"], 0.01) == 1.2941  # 110 / 85
    assert row["tachycardia_flag"] == 1  # HR 110 > 100
    assert row["hypotension_flag"] == 1  # SBP 85 < 90
    assert row["fever_flag"] == 1  # Temp 39.0 > 38.0
    assert row["tachypnea_flag"] == 1  # RR 24 > 20
