"""Synthetic Tabular Health Data Generator.

Educational Software Simulation.
Generates scenario-based synthetic physiological records with realistic
physiological distributions, correlated features, borderline cases,
and controlled outliers for each risk class.
"""

from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np

from src.data.schema import RiskClass, RAW_FEATURES, TARGET_RISK_CLASSES
from src.utils.seed import set_seed
from src.utils.logger import setup_logger

logger = setup_logger("tabular_generator")


# ──────────────────────────────────────────────────────────────────────────────
# Physiological scenario templates per risk class
# Each dict contains (mean, std) pairs for continuous features
# ──────────────────────────────────────────────────────────────────────────────
SCENARIO_TEMPLATES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "NORMAL": {
        "heart_rate":       (72.0,  8.0),
        "systolic_bp":      (120.0, 10.0),
        "diastolic_bp":     (78.0,  7.0),
        "spo2":             (98.5,  0.8),
        "temperature":      (36.7,  0.3),
        "respiratory_rate": (15.0,  2.0),
    },
    "MODERATE": {
        "heart_rate":       (95.0,  10.0),
        "systolic_bp":      (105.0, 12.0),
        "diastolic_bp":     (68.0,  8.0),
        "spo2":             (95.0,  1.5),
        "temperature":      (37.8,  0.5),
        "respiratory_rate": (20.0,  3.0),
    },
    "HIGH": {
        "heart_rate":       (115.0, 12.0),
        "systolic_bp":      (90.0,  12.0),
        "diastolic_bp":     (58.0,  8.0),
        "spo2":             (91.0,  2.0),
        "temperature":      (38.8,  0.6),
        "respiratory_rate": (26.0,  4.0),
    },
    "CRITICAL": {
        "heart_rate":       (135.0, 18.0),
        "systolic_bp":      (72.0,  14.0),
        "diastolic_bp":     (44.0,  10.0),
        "spo2":             (83.0,  4.0),
        "temperature":      (39.6,  0.7),
        "respiratory_rate": (34.0,  5.0),
    },
}

# Symptom probability tables per class [chest_pain, breathing_difficulty, dizziness, unconsciousness, bleeding_injury]
SYMPTOM_PROBS: Dict[str, List[float]] = {
    "NORMAL":   [0.03, 0.02, 0.04, 0.00, 0.02],
    "MODERATE": [0.20, 0.25, 0.20, 0.00, 0.10],
    "HIGH":     [0.50, 0.55, 0.45, 0.05, 0.20],
    "CRITICAL": [0.80, 0.85, 0.70, 0.35, 0.30],
}

# Physiological hard clamps (absolute limits)
PHYS_CLAMPS: Dict[str, Tuple[float, float]] = {
    "heart_rate":       (25.0,  240.0),
    "systolic_bp":      (45.0,  255.0),
    "diastolic_bp":     (25.0,  155.0),
    "spo2":             (55.0,  100.0),
    "temperature":      (30.5,  44.5),
    "respiratory_rate": (5.0,   58.0),
}


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, value))


class TabularDataGenerator:
    """Scenario-based synthetic tabular health data generator.

    Generates realistic emergency health records with:
    - Class-specific physiological distributions
    - Feature correlations (shock index, hypoxia patterns)
    - 5% borderline samples near class boundaries
    - 2% controlled outliers within physiological limits
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize generator with configuration.

        Args:
            config: Configuration dictionary loaded from config.yaml.
        """
        self.config = config
        self.seed = config.get("random_seed", 42)
        set_seed(self.seed)
        self.rng = np.random.default_rng(self.seed)
        logger.info(f"Initialized TabularDataGenerator with seed={self.seed}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sample_vitals(self, risk_class: str, borderline: bool = False) -> Dict[str, float]:
        """Sample physiological vitals for a given risk class.

        Args:
            risk_class: Target risk class string.
            borderline: If True, blend towards adjacent class boundary.

        Returns:
            Dict of vital names to float values.
        """
        template = SCENARIO_TEMPLATES[risk_class]
        vitals: Dict[str, float] = {}

        for feature, (mean, std) in template.items():
            if borderline:
                # Shift mean towards boundary (±1.5σ in direction of adjacent class)
                boundary_shift = std * 1.5
                direction = 1 if risk_class in ["NORMAL", "MODERATE"] else -1
                mean = mean + direction * boundary_shift
            val = float(self.rng.normal(mean, std))
            lo, hi = PHYS_CLAMPS[feature]
            vitals[feature] = _clamp(val, lo, hi)

        return vitals

    def _sample_vitals_outlier(self, risk_class: str) -> Dict[str, float]:
        """Sample vitals with controlled extreme values (still within physiological limits).

        Args:
            risk_class: Target risk class string.

        Returns:
            Dict of vital names to float values with one feature at extremes.
        """
        vitals = self._sample_vitals(risk_class)
        # Pick a random feature and push it toward physical extreme
        feat = self.rng.choice(list(vitals.keys()))
        lo, hi = PHYS_CLAMPS[feat]
        if self.rng.random() < 0.5:
            vitals[feat] = float(lo + self.rng.uniform(0, (hi - lo) * 0.12))
        else:
            vitals[feat] = float(hi - self.rng.uniform(0, (hi - lo) * 0.12))
        return vitals

    def _apply_correlations(self, vitals: Dict[str, float], risk_class: str) -> Dict[str, float]:
        """Apply physiological correlations between vitals.

        HIGH/CRITICAL: enforce shock pattern (HR up, SBP down).
        NORMAL: keep diastolic close to systolic * 0.65 (realistic PP).

        Args:
            vitals: Dict of current vital values.
            risk_class: Risk class string.

        Returns:
            Adjusted vitals dict.
        """
        v = vitals.copy()

        if risk_class in ["HIGH", "CRITICAL"]:
            # Shock correlation: raise HR if SBP is very low
            if v["systolic_bp"] < 80:
                v["heart_rate"] = _clamp(v["heart_rate"] + self.rng.uniform(10, 25),
                                         *PHYS_CLAMPS["heart_rate"])
            # Lower SpO2 if RR is very high
            if v["respiratory_rate"] > 30:
                v["spo2"] = _clamp(v["spo2"] - self.rng.uniform(2, 6),
                                   *PHYS_CLAMPS["spo2"])

        elif risk_class == "NORMAL":
            # Keep diastolic physiologically realistic relative to systolic
            target_dbp = v["systolic_bp"] * 0.64 + self.rng.normal(0, 3)
            v["diastolic_bp"] = _clamp(target_dbp, *PHYS_CLAMPS["diastolic_bp"])

        # Ensure diastolic always < systolic
        if v["diastolic_bp"] >= v["systolic_bp"]:
            v["diastolic_bp"] = _clamp(v["systolic_bp"] - self.rng.uniform(8, 20),
                                       *PHYS_CLAMPS["diastolic_bp"])
        return v

    def _sample_symptoms(self, risk_class: str) -> Dict[str, int]:
        """Sample binary symptom flags based on class-specific probabilities.

        Args:
            risk_class: Risk class string.

        Returns:
            Dict of symptom name to 0/1 integer.
        """
        probs = SYMPTOM_PROBS[risk_class]
        symptom_names = ["chest_pain", "breathing_difficulty", "dizziness",
                         "unconsciousness", "bleeding_injury"]
        return {s: int(self.rng.random() < p) for s, p in zip(symptom_names, probs)}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_case(self, risk_class: str, case_id: str,
                      sample_type: str = "normal") -> Dict[str, Any]:
        """Generate a single scenario-based synthetic health record.

        Args:
            risk_class: Target risk category ('NORMAL', 'MODERATE', 'HIGH', 'CRITICAL').
            case_id: Unique identifier string for the generated case.
            sample_type: One of 'normal', 'borderline', 'outlier'.

        Returns:
            Dict containing raw physiological features and assigned risk class.
        """
        # Age and sex (independent of risk class — realistic demographic spread)
        age = float(self.rng.integers(18, 90))
        sex = int(self.rng.integers(0, 2))

        # Sample vitals based on type
        if sample_type == "borderline":
            vitals = self._sample_vitals(risk_class, borderline=True)
        elif sample_type == "outlier":
            vitals = self._sample_vitals_outlier(risk_class)
        else:
            vitals = self._sample_vitals(risk_class)

        # Apply inter-feature correlations
        vitals = self._apply_correlations(vitals, risk_class)

        # Sample symptoms
        symptoms = self._sample_symptoms(risk_class)

        case: Dict[str, Any] = {
            "case_id":            case_id,
            "age":                age,
            "sex":                sex,
            "heart_rate":         round(vitals["heart_rate"], 1),
            "systolic_bp":        round(vitals["systolic_bp"], 1),
            "diastolic_bp":       round(vitals["diastolic_bp"], 1),
            "spo2":               round(vitals["spo2"], 1),
            "temperature":        round(vitals["temperature"], 2),
            "respiratory_rate":   round(vitals["respiratory_rate"], 1),
            "chest_pain":         symptoms["chest_pain"],
            "breathing_difficulty": symptoms["breathing_difficulty"],
            "dizziness":          symptoms["dizziness"],
            "unconsciousness":    symptoms["unconsciousness"],
            "bleeding_injury":    symptoms["bleeding_injury"],
            "risk_class":         risk_class,
        }
        return case

    def generate_dataset(self, num_samples_per_class: Optional[int] = None) -> pd.DataFrame:
        """Generate full synthetic tabular dataset.

        Composition per class:
        - 93% normal scenario samples
        - 5%  borderline samples (near class boundaries)
        - 2%  outlier samples (extreme but physiologically valid)

        Args:
            num_samples_per_class: Override for samples per class. Defaults to
                config planned_dataset_sizes.tabular / 4.

        Returns:
            pandas DataFrame containing generated raw tabular cases.
        """
        planned_total = self.config.get("planned_dataset_sizes", {}).get("tabular", 100000)
        if num_samples_per_class is None:
            num_samples_per_class = planned_total // len(TARGET_RISK_CLASSES)

        records: List[Dict[str, Any]] = []
        count = 0

        for r_class in TARGET_RISK_CLASSES:
            n_borderline = max(1, int(num_samples_per_class * 0.05))
            n_outlier    = max(1, int(num_samples_per_class * 0.02))
            n_normal     = num_samples_per_class - n_borderline - n_outlier

            type_counts = (
                [("normal", n_normal),
                 ("borderline", n_borderline),
                 ("outlier", n_outlier)]
            )

            for sample_type, n in type_counts:
                for _ in range(n):
                    count += 1
                    case_id = f"CASE_{count:07d}"
                    records.append(self.generate_case(r_class, case_id, sample_type))

        # Shuffle to mix classes
        self.rng.shuffle(records)

        df = pd.DataFrame(records)
        # Reset case IDs after shuffle for sequential ordering
        df["case_id"] = [f"CASE_{i+1:07d}" for i in range(len(df))]

        logger.info(
            f"Generated synthetic tabular dataset: {df.shape[0]} records × {df.shape[1]} cols. "
            f"Class distribution:\n{df['risk_class'].value_counts().to_dict()}"
        )
        return df
