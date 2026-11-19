"""Unit tests verifying clean imports of all repository modules."""


def test_imports_all_modules():
    """Import core subpackages to ensure zero syntax or missing dependency errors."""
    import src.utils
    import src.data
    import src.preprocessing
    import src.models.classical
    import src.models.deep
    import src.training
    import src.evaluation
    import src.explainability
    import src.inference

    assert src.utils is not None
    assert src.data is not None
    assert src.preprocessing is not None
    assert src.models.classical is not None
    assert src.models.deep is not None
    assert src.training is not None
    assert src.evaluation is not None
    assert src.explainability is not None
    assert src.inference is not None


def test_type_annotations_resolve():
    """Every annotation must be importable (catches missing typing names)."""
    from typing import get_type_hints
    from src.data.validator import DataValidator

    assert get_type_hints(DataValidator.check_duplicates)


def test_generator_instantiation():
    """Verify data generator classes instantiate with config."""
    from src.utils.config_loader import load_config
    from src.data import TabularDataGenerator, SignalDataGenerator, TemporalDataGenerator

    config = load_config("config.yaml")

    tab_gen = TabularDataGenerator(config)
    sig_gen = SignalDataGenerator(config)
    temp_gen = TemporalDataGenerator(config)

    assert tab_gen is not None
    assert sig_gen is not None
    assert temp_gen is not None
