"""Data schema definitions for the AI-Based Emergency Health Detection System.

Educational Simulation Scope.
Defines raw features, derived features, target risk classes, and validation ranges.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple, Any, Optional


class RiskClass(str, Enum):
    """Target risk classification levels for simulation."""
    NORMAL = "NORMAL"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# List of target classes
TARGET_RISK_CLASSES: List[str] = [e.value for e in RiskClass]

# Raw features specification
RAW_FEATURES: List[str] = [
    "case_id",
    "age",
    "sex",
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "spo2",
    "temperature",
    "respiratory_rate",
    "chest_pain",
    "breathing_difficulty",
    "dizziness",
    "unconsciousness",
    "bleeding_injury",
]

# Derived features specification
DERIVED_FEATURES: List[str] = [
    "pulse_pressure",
    "mean_arterial_pressure",
    "shock_index",
    "age_group",
    "spo2_severity",
    "tachycardia_flag",
    "hypotension_flag",
    "fever_flag",
    "tachypnea_flag",
]

# All features combined (excluding target)
ALL_INPUT_FEATURES: List[str] = RAW_FEATURES + DERIVED_FEATURES

# Physiological valid numeric ranges for range validation
PHYSIOLOGICAL_RANGES: Dict[str, Tuple[float, float]] = {
    "age": (0.0, 120.0),
    "sex": (0.0, 1.0),  # 0: Female, 1: Male
    "heart_rate": (20.0, 250.0),
    "systolic_bp": (40.0, 260.0),
    "diastolic_bp": (20.0, 160.0),
    "spo2": (50.0, 100.0),
    "temperature": (30.0, 45.0),
    "respiratory_rate": (4.0, 60.0),
    "chest_pain": (0.0, 1.0),
    "breathing_difficulty": (0.0, 1.0),
    "dizziness": (0.0, 1.0),
    "unconsciousness": (0.0, 1.0),
    "bleeding_injury": (0.0, 1.0),
}


@dataclass
class HealthRecordSchema:
    """Schema definition and container for a single tabular health simulation record."""
    case_id: str
    age: float
    sex: int
    heart_rate: float
    systolic_bp: float
    diastolic_bp: float
    spo2: float
    temperature: float
    respiratory_rate: float
    chest_pain: int
    breathing_difficulty: int
    dizziness: int
    unconsciousness: int
    bleeding_injury: int
    risk_class: RiskClass

    # Derived fields (Optional until computed)
    pulse_pressure: Optional[float] = None
    mean_arterial_pressure: Optional[float] = None
    shock_index: Optional[float] = None
    age_group: Optional[str] = None
    spo2_severity: Optional[str] = None
    tachycardia_flag: Optional[int] = None
    hypotension_flag: Optional[int] = None
    fever_flag: Optional[int] = None
    tachypnea_flag: Optional[int] = None

    def validate_schema(self) -> Tuple[bool, List[str]]:
        """Validate record against physiological boundary schema rules.

        Returns:
            Tuple of (is_valid, list_of_error_messages)
        """
        errors = []
        if self.risk_class.value not in TARGET_RISK_CLASSES:
            errors.append(f"Invalid risk_class: {self.risk_class}")

        for param, (min_v, max_v) in PHYSIOLOGICAL_RANGES.items():
            val = getattr(self, param, None)
            if val is not None and not (min_v <= float(val) <= max_v):
                errors.append(f"Parameter '{param}' value {val} outside bounds [{min_v}, {max_v}]")

        return len(errors) == 0, errors
