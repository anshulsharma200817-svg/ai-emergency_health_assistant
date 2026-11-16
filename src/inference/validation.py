"""Input validation for the Python inference path.

This mirrors the rules in ``web/lib/ranges.ts`` and ``web/lib/inference.ts`` so
the Python predictor and the deployed TypeScript API accept and reject exactly
the same records:

* **absent** (``None`` / ``""`` / missing key) → healthy-adult default;
* **present but malformed** → :class:`VitalsValidationError` (the HTTP route
  turns this into a 400, never a confident ``NORMAL``);
* **present and in range** → used as given.

``temperature`` is Celsius. ``temperature_unit: "F"`` converts first, matching
the API. Keep ``VITALS_RANGES`` in sync with ``web/lib/ranges.ts``.

Educational software simulation only — NOT a medical device.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

# Accepted ranges, keyed by vitals field. Mirrors web/lib/ranges.ts.
VITALS_RANGES: Dict[str, Tuple[float, float]] = {
    "age": (0, 120),
    "heart_rate": (20, 300),
    "systolic_bp": (40, 300),
    "diastolic_bp": (20, 200),
    "spo2": (50, 100),
    "temperature": (25, 45),
    "respiratory_rate": (4, 80),
    "sex": (0, 1),
    "chest_pain": (0, 1),
    "breathing_difficulty": (0, 1),
    "dizziness": (0, 1),
    "unconsciousness": (0, 1),
    "bleeding_injury": (0, 1),
}

NUMERIC_FIELDS: List[str] = list(VITALS_RANGES)
FLAG_FIELDS = (
    "chest_pain",
    "breathing_difficulty",
    "dizziness",
    "unconsciousness",
    "bleeding_injury",
)

# Healthy-adult baseline, identical to DEFAULT_VITALS in web/lib/inference.ts.
DEFAULT_VITALS: Dict[str, float] = {
    "age": 35,
    "sex": 1,
    "heart_rate": 75,
    "systolic_bp": 120,
    "diastolic_bp": 80,
    "spo2": 98,
    "temperature": 37.0,
    "respiratory_rate": 16,
    "chest_pain": 0,
    "breathing_difficulty": 0,
    "dizziness": 0,
    "unconsciousness": 0,
    "bleeding_injury": 0,
}

# Strict decimal parser: "72", " 72 ", "37.5", "1e2" are fine; "80abc", "0x50",
# "Infinity", "NaN", booleans, arrays and objects are not.
_DECIMAL_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


class VitalsValidationError(ValueError):
    """Raised when a vitals record contains malformed or out-of-range values."""

    def __init__(self, issues: List[Dict[str, Any]]):
        self.issues = issues
        super().__init__("; ".join(issue["message"] for issue in issues))


def _describe(raw: Any) -> str:
    if isinstance(raw, str):
        return repr(raw)
    return repr(raw)


def _to_finite_number(raw: Any) -> Optional[float]:
    # bool is a subclass of int in Python; the TS API does not treat it as a number.
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        value = float(raw)
        return value if math.isfinite(value) else None
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not _DECIMAL_RE.match(text):
        return None
    value = float(text)
    return value if math.isfinite(value) else None


def _to_flag(raw: Any) -> Optional[int]:
    if isinstance(raw, bool):
        return 1 if raw else 0
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        if raw == 0:
            return 0
        if raw == 1:
            return 1
        return None
    if isinstance(raw, str):
        text = raw.strip()
        if text in ("0", "1"):
            return int(text)
    return None


def _to_sex(raw: Any) -> Optional[int]:
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        if raw == 0:
            return 0
        if raw == 1:
            return 1
        return None
    if isinstance(raw, str):
        text = raw.strip()
        if text in ("0", "1"):
            return int(text)
    return None


def coerce_vitals(
    payload: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Coerce a raw payload into a complete vitals record.

    Returns ``(vitals, issues)``. ``issues`` is non-empty when a field was
    present but malformed; callers that want fail-closed behaviour should use
    :func:`normalize_vitals` instead.
    """
    source: Dict[str, Any] = payload or {}
    out: Dict[str, float] = dict(DEFAULT_VITALS)
    issues: List[Dict[str, Any]] = []
    temperature_supplied = False

    for field in NUMERIC_FIELDS:
        raw = source.get(field)
        if raw is None or raw == "":
            continue

        if field in FLAG_FIELDS:
            flag = _to_flag(raw)
            if flag is None:
                issues.append({
                    "field": field,
                    "value": math.nan,
                    "min": 0,
                    "max": 1,
                    "message": f"{field} must be 0, 1, true or false (received {_describe(raw)})",
                })
            else:
                out[field] = flag
            continue

        if field == "sex":
            sex = _to_sex(raw)
            if sex is None:
                issues.append({
                    "field": field,
                    "value": math.nan,
                    "min": 0,
                    "max": 1,
                    "message": f"sex must be 0 (female) or 1 (male) (received {_describe(raw)})",
                })
            else:
                out[field] = sex
            continue

        value = _to_finite_number(raw)
        if value is None:
            minimum, maximum = VITALS_RANGES[field]
            issues.append({
                "field": field,
                "value": math.nan,
                "min": minimum,
                "max": maximum,
                "message": f"{field} must be a finite number (received {_describe(raw)})",
            })
            continue

        out[field] = value
        if field == "temperature":
            temperature_supplied = True

    unit = str(
        source.get("temperature_unit", source.get("temperatureUnit", ""))
    ).strip().lower()

    if temperature_supplied and (
        unit == "f" or unit.startswith("fahrenheit") or unit == "°f"
    ):
        out["temperature"] = (out["temperature"] - 32) * 5 / 9

    return out, issues


def cross_field_error(vitals: Dict[str, float]) -> Optional[Dict[str, Any]]:
    """Reject only a strictly inverted blood pressure (`diastolic > systolic`)."""
    systolic = vitals.get("systolic_bp")
    diastolic = vitals.get("diastolic_bp")
    if systolic is None or diastolic is None:
        return None
    if not math.isfinite(systolic) or not math.isfinite(diastolic):
        return None
    if diastolic > systolic:
        return {
            "field": "diastolic_bp",
            "message": (
                f"Diastolic BP ({diastolic}) cannot be higher than "
                f"systolic BP ({systolic})"
            ),
        }
    return None


def validate_vitals(vitals: Dict[str, float]) -> List[Dict[str, Any]]:
    """Check a normalized record against the accepted ranges."""
    issues: List[Dict[str, Any]] = []

    for field in NUMERIC_FIELDS:
        if field in FLAG_FIELDS:
            continue
        minimum, maximum = VITALS_RANGES[field]
        value = vitals.get(field)
        if value is None or not math.isfinite(value) or not (minimum <= value <= maximum):
            if field == "temperature" and value is not None and 90 <= value <= 115:
                message = (
                    "temperature looks like Fahrenheit; send Celsius or add "
                    '"temperature_unit": "F"'
                )
            else:
                message = f"{field} must be between {minimum} and {maximum} (received {value})"
            issues.append({
                "field": field,
                "value": value,
                "min": minimum,
                "max": maximum,
                "message": message,
            })

    for flag in FLAG_FIELDS:
        value = vitals.get(flag)
        if value in (0, 1):
            continue
        issues.append({
            "field": flag,
            "value": value,
            "min": 0,
            "max": 1,
            "message": f"{flag} must be 0 or 1 (received {value})",
        })

    cross = cross_field_error(vitals)
    if cross is not None:
        field = cross["field"]
        issues.append({
            "field": field,
            "value": vitals.get(field),
            "min": VITALS_RANGES[field][0],
            "max": vitals.get("systolic_bp"),
            "message": cross["message"],
        })

    return issues


def normalize_vitals(payload: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """Coerce and validate a payload, raising on malformed or out-of-range values."""
    vitals, issues = coerce_vitals(payload)
    issues.extend(validate_vitals(vitals))
    if issues:
        raise VitalsValidationError(issues)
    return vitals
