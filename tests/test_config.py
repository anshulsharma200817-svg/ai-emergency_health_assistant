"""Unit tests for config.yaml parsing and parameters validation."""

import pytest
from src.utils.config_loader import load_config


def test_load_config_values():
    """Verify config.yaml contains all required fields and correct specifications."""
    config = load_config("config.yaml")

    assert config.get("random_seed") == 42
    assert config.get("sequence_length") == 60

    target = config.get("target")
    assert target == ["NORMAL", "MODERATE", "HIGH", "CRITICAL"]

    planned = config.get("planned_dataset_sizes", {})
    assert planned.get("tabular") == 100000
    assert planned.get("temporal") == 20000
    assert planned.get("signals") == 20000


def test_config_tabular_schema_keys():
    """Verify config tabular_schema definitions."""
    config = load_config("config.yaml")
    schema_cfg = config.get("tabular_schema", {})

    raw = schema_cfg.get("raw_features", [])
    derived = schema_cfg.get("derived_features", [])

    assert "heart_rate" in raw
    assert "pulse_pressure" in derived
    assert schema_cfg.get("target_column") == "risk_class"
