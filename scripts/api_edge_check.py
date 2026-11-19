#!/usr/bin/env python3
"""Exhaustive edge-case sweep of the inference API.

Run against a local production server (or the live deployment):

    cd web && npm run build && npm run start -- -p 3216 &
    python3 scripts/api_edge_check.py --base http://localhost:3216

Checks, for every case:
  * the HTTP status is the one we expect (200 valid / 400 rejected),
  * response invariants hold (probabilities sum to 1, confidence == max
    probability, priority/colour match the class, disclaimer present),
  * the extrapolation warning matches the generated training-support table.

Any 500, any silent acceptance of garbage, or any invariant break is a FAIL.
The support bounds are read from `web/lib/training-support.json`, so this
script can never drift from the values the app actually uses.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUPPORT_PATH = ROOT / "web" / "lib" / "training-support.json"
RESULTS_DIR = ROOT / "experiments" / "results"

# Fallbacks for a fresh clone (results/ is git-ignored). When local results
# exist they are the source of truth, so this sweep cannot keep passing after a
# retrain while /api/metrics still serves the old numbers.
FALLBACK_CLASSICAL = {
    "knn": 0.9586,
    "logistic_regression": 0.9703,
    "decision_tree": 0.9661,
    "random_forest": 0.9805,
}
FALLBACK_DEEP = {"cnn_1d": 0.9987, "rnn": 0.9553, "lstm": 0.9503}


def expected_metrics(filename: str, fallback: dict[str, float]) -> dict[str, float]:
    path = RESULTS_DIR / filename
    if not path.exists():
        return fallback
    with open(path, encoding="utf-8") as handle:
        results = json.load(handle)
    return {
        key: round(float(value["test_accuracy"]), 4)
        for key, value in results.items()
        if isinstance(value, dict) and "test_accuracy" in value
    }

RANGES = {
    "age": (0, 120),
    "sex": (0, 1),
    "heart_rate": (20, 300),
    "systolic_bp": (40, 300),
    "diastolic_bp": (20, 200),
    "spo2": (50, 100),
    "temperature": (25, 45),
    "respiratory_rate": (4, 80),
}
FLAGS = ["chest_pain", "breathing_difficulty", "dizziness",
         "unconsciousness", "bleeding_injury"]
LABELS = ["NORMAL", "MODERATE", "HIGH", "CRITICAL"]
PRIORITY = {"NORMAL": "P3", "MODERATE": "P2", "HIGH": "P1", "CRITICAL": "P0"}
COLOR = {"NORMAL": "#22c55e", "MODERATE": "#f59e0b",
         "HIGH": "#f97316", "CRITICAL": "#ef4444"}
DISCLAIMER = "EDUCATIONAL SIMULATION ONLY — NOT FOR CLINICAL USE"
DEFAULT = dict(age=35, sex=1, heart_rate=75, systolic_bp=120, diastolic_bp=80,
               spo2=98, temperature=37.0, respiratory_rate=16, chest_pain=0,
               breathing_difficulty=0, dizziness=0, unconsciousness=0,
               bleeding_injury=0)

BASE = ""
SUPPORT: dict[str, dict[str, float]] = {}
failures: list[str] = []
passes = 0


def post(payload=None, raw=None, method="POST"):
    data = raw.encode() if raw is not None else json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE + "/api/predict", data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode()
        try:
            return exc.code, json.loads(body)
        except json.JSONDecodeError:
            return exc.code, body


def fail(name: str, problems: list[str]) -> None:
    failures.append(f"{name:44s} {problems}")


def check(name, payload, expect, expect_fields=None):
    """expect: 200 or 400. expect_fields: fields that must appear in `issues`."""
    global passes
    status, body = post(payload)
    problems: list[str] = []

    if status != expect:
        problems.append(f"status {status} != {expect}: {str(body)[:160]}")

    if status == 200:
        cls = body.get("predicted_risk_class")
        if cls not in LABELS:
            problems.append("bad class")
        probs = body.get("probabilities") or {}
        if set(probs) != set(LABELS):
            problems.append(f"probability keys {list(probs)}")
        total = sum(probs.values())
        if abs(total - 1.0) > 5e-4:
            problems.append(f"probabilities sum to {total}")
        if abs(body.get("confidence", -1) - max(probs.values())) > 2e-4:
            problems.append(f"confidence {body.get('confidence')} != max prob {max(probs.values())}")
        if body.get("priority") != PRIORITY.get(cls):
            problems.append("priority mismatch")
        if body.get("alert_color") != COLOR.get(cls):
            problems.append("colour mismatch")
        if body.get("disclaimer") != DISCLAIMER:
            problems.append("disclaimer mismatch")

        warning = body.get("distribution_warning")
        outside = body.get("outside_training_range") or []
        if warning is None and outside:
            problems.append("outside_training_range without warning")
        if warning and not outside:
            problems.append("warning without outside_training_range")
        for item in outside:
            field = item["field"]
            support = SUPPORT.get(field)
            if support and (support["min"], support["max"]) != (
                item["observed_min"], item["observed_max"]
            ):
                problems.append(
                    f"support bounds for {field} are {item['observed_min']}-"
                    f"{item['observed_max']} not {support['min']}-{support['max']}"
                )
            if support and support["min"] <= item["value"] <= support["max"]:
                problems.append(f"{field}={item['value']} flagged inside support")

    if status == 400 and expect_fields is not None:
        issues = body.get("issues") if isinstance(body, dict) else None
        got = [i.get("field") for i in issues] if isinstance(issues, list) else []
        for field in expect_fields:
            if field not in got:
                problems.append(f"expected issue for {field}, got {got}")

    if problems:
        fail(name, problems)
    else:
        passes += 1


def with_(field, value):
    payload = dict(DEFAULT)
    payload[field] = value
    return payload


def check_body(name, raw, expect):
    global passes
    status, body = post(raw=raw)
    if status != expect:
        fail(name, [f"status {status} != {expect}: {str(body)[:120]}"])
    else:
        passes += 1
        print(f"    ok  {name:40s} -> {status}")


def check_constraint(name, payload):
    """`unconsciousness = 1` must never resolve to NORMAL or MODERATE."""
    global passes
    status, body = post(payload)
    problems: list[str] = []
    if status != 200:
        problems.append(f"status {status}: {str(body)[:160]}")
    else:
        cls = body.get("predicted_risk_class")
        probs = body.get("probabilities") or {}
        if cls not in {"HIGH", "CRITICAL"}:
            problems.append(f"class {cls} on unconsciousness=1")
        if probs.get("NORMAL") != 0:
            problems.append(f"NORMAL probability {probs.get('NORMAL')}")
        if probs.get("MODERATE") != 0:
            problems.append(f"MODERATE probability {probs.get('MODERATE')}")
        if abs(sum(probs.values()) - 1.0) > 5e-4:
            problems.append(f"probabilities sum to {sum(probs.values())}")
        if body.get("applied_constraint") != "unconsciousness":
            problems.append(f"applied_constraint {body.get('applied_constraint')!r}")
        if abs(body.get("confidence", -1) - max(probs.values())) > 2e-4:
            problems.append("confidence != max probability")
    if problems:
        fail(name, problems)
    else:
        passes += 1
        print(f"    ok  {name:40s} -> {body.get('predicted_risk_class')}")


def check_constraint_free(name, payload):
    """Payloads without the flag must report `applied_constraint: null`."""
    global passes
    status, body = post(payload)
    if status != 200:
        fail(name, [f"status {status}: {str(body)[:160]}"])
    elif body.get("applied_constraint") is not None:
        fail(name, [f"applied_constraint {body.get('applied_constraint')!r}"])
    elif "applied_constraint" not in body:
        fail(name, ["applied_constraint field missing"])
    else:
        passes += 1
        print(f"    ok  {name:40s} -> {body.get('predicted_risk_class')}")


def main() -> int:
    global BASE, SUPPORT, passes
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://localhost:3216",
                        help="Base URL of the running server")
    args = parser.parse_args()
    BASE = args.base.rstrip("/")
    SUPPORT = json.loads(SUPPORT_PATH.read_text())

    print(f"TARGET {BASE}\n")

    # ── 1. Boundary sweep for every numeric field ────────────────────────────
    print("[1] boundary values per field (min-1, min, max, max+1, far outside)")
    for field, (lo, hi) in RANGES.items():
        if field == "sex":
            continue
        step = 0.5 if field in ("spo2", "temperature") else 1
        for label, value, expect in [
            ("below min", lo - step, 400),
            ("exactly min", lo, 200),
            ("exactly max", hi, 200),
            ("above max", hi + step, 400),
            ("far below", -1e6, 400),
            ("far above", 1e6, 400),
        ]:
            payload = with_(field, value)
            # Keep the blood-pressure pair consistent so this case isolates the
            # field under test instead of tripping the cross-field rule.
            if field == "systolic_bp":
                payload["diastolic_bp"] = 20
            elif field == "diastolic_bp":
                payload["systolic_bp"] = 300
            check(f"{field} {label} ({value})", payload, expect,
                  [field] if expect == 400 else None)

    # ── 2. Non-numeric / type-confused values ────────────────────────────────
    print("[2] type confusion (strings, nulls, booleans, arrays, objects)")
    cases = [
        ("string number '80'", {"heart_rate": "80"}, 200),
        ("string float '37.2'", {"temperature": "37.2"}, 200),
        ("numeric string with spaces ' 80 '", {"heart_rate": " 80 "}, 200),
        ("plain garbage 'abc'", {"heart_rate": "abc"}, 400),
        ("partial number '80abc'", {"heart_rate": "80abc"}, 400),
        ("empty string", {"heart_rate": ""}, 200),
        ("null", {"heart_rate": None}, 200),
        ("boolean true (numeric field)", {"heart_rate": True}, 400),
        ("array", {"heart_rate": [80]}, 400),
        ("object", {"heart_rate": {"v": 80}}, 400),
        ("exponent '1e2'", {"heart_rate": "1e2"}, 200),
        ("hex '0x50'", {"heart_rate": "0x50"}, 400),
        ("Infinity string", {"heart_rate": "Infinity"}, 400),
        ("NaN string", {"heart_rate": "NaN"}, 400),
        ("1e309 overflows", {"heart_rate": "1e309"}, 400),
        ("negative zero", {"chest_pain": -0.0}, 200),
        ("flag true", {"chest_pain": True}, 200),
        ("flag false", {"chest_pain": False}, 200),
        ("flag string '1'", {"chest_pain": "1"}, 200),
        ("flag 2", {"chest_pain": 2}, 400),
        ("flag -1", {"chest_pain": -1}, 400),
        ("flag 0.5", {"chest_pain": 0.5}, 400),
        ("sex 0.5", {"sex": 0.5}, 400),
        ("sex 2", {"sex": 2}, 400),
        ("sex true", {"sex": True}, 400),
        ("age 37.5 (fractional years)", {"age": 37.5}, 200),
        ("age '1e1'", {"age": "1e1"}, 200),
    ]
    for name, patch, expect in cases:
        payload = dict(DEFAULT)
        payload.update(patch)
        check(name, payload, expect, list(patch) if expect == 400 else None)

    # Boolean flags must be exactly equivalent to 0/1.
    for field in FLAGS:
        for value, numeric in ((True, 1), (False, 0)):
            with_flag = dict(DEFAULT)
            with_flag[field] = value
            with_number = dict(DEFAULT)
            with_number[field] = numeric
            check(f"{field}={value} accepted", with_flag, 200)
            if post(with_flag)[1].get("probabilities") != post(with_number)[1].get("probabilities"):
                fail(f"{field}={value} vs {numeric}", ["different probabilities"])

    # ── 3. Temperature units ─────────────────────────────────────────────────
    print("[3] Fahrenheit handling")
    for name, patch, expect in [
        ("98.6F + unit F", {"temperature": 98.6, "temperature_unit": "F"}, 200),
        ("98.6F + unit fahrenheit", {"temperature": 98.6, "temperature_unit": "fahrenheit"}, 200),
        ("98.6F + unit Fahrenheit", {"temperature": 98.6, "temperature_unit": "Fahrenheit"}, 200),
        ("98.6F + unit °F", {"temperature": 98.6, "temperature_unit": "°F"}, 200),
        ("98.6F + unit C", {"temperature": 98.6, "temperature_unit": "C"}, 400),
        ("98.6F no unit", {"temperature": 98.6}, 400),
        ("99F -> 37.2C", {"temperature": 99.0, "temperature_unit": "F"}, 200),
        ("212F -> 100C", {"temperature": 212.0, "temperature_unit": "F"}, 400),
        ("10F -> -12.2C", {"temperature": 10.0, "temperature_unit": "F"}, 400),
        ("temperatureUnit camelCase", {"temperature": 98.6, "temperatureUnit": "F"}, 200),
        ("unit as number", {"temperature": 98.6, "temperature_unit": 1}, 400),
    ]:
        payload = dict(DEFAULT)
        payload.update(patch)
        check(name, payload, expect)

    # ── 4. Body-level / protocol cases ───────────────────────────────────────
    print("[4] bodies, methods, protocol")
    for name, raw, expect in [
        ("empty object", "{}", 200),
        ("empty raw body", "", 400),
        ("whitespace body", "   ", 400),
        ("not json", "nope", 400),
        ("json array", "[1,2,3]", 400),
        ("json null", "null", 400),
        ("json number", "42", 400),
        ("json string", '"hello"', 400),
        ("nested object junk", '{"age": {"nested": 1}}', 400),
        ("unknown fields only", '{"foo": 1, "bar": 2}', 200),
        ("__proto__ payload", '{"__proto__": {"age": 1}, "age": 40}', 200),
        ("duplicate keys", '{"age": 40, "age": 90}', 200),
    ]:
        check_body(name, raw, expect)

    request = urllib.request.Request(BASE + "/api/predict", method="OPTIONS")
    with urllib.request.urlopen(request, timeout=30) as response:
        ok = (
            response.status == 204
            and response.headers.get("access-control-allow-origin") == "*"
            and "POST" in (response.headers.get("access-control-allow-methods") or "")
        )
    print(f"    {'ok ' if ok else 'FAIL'} OPTIONS 204 + CORS headers")
    if ok:
        passes += 1
    else:
        fail("OPTIONS/CORS", ["missing headers"])

    with urllib.request.urlopen(BASE + "/api/predict", timeout=30) as response:
        health = json.load(response)
    sample = health.get("sample_prediction", {})
    defaults = post({})[1]
    if sample.get("predicted_risk_class") != defaults.get("predicted_risk_class"):
        fail("GET sample_prediction", ["disagrees with POST {}"])
    else:
        passes += 1
        print("    ok  GET sample_prediction == POST {}")

    declared = health.get("accepted_ranges", {})
    for field, (lo, hi) in RANGES.items():
        rng = declared.get(field) or {}
        if (rng.get("min"), rng.get("max")) != (lo, hi):
            fail(f"GET accepted_ranges[{field}]",
                 [f"{rng.get('min')}-{rng.get('max')} != {lo}-{hi}"])
        else:
            passes += 1

    # ── 5. Extrapolation warning boundaries ──────────────────────────────────
    print("[5] training-support warning boundaries")
    for field, support in SUPPORT.items():
        accepted_lo, accepted_hi = RANGES[field]
        for label, value, should_warn in [
            ("below support", support["min"] - 0.5, True),
            ("at support min", support["min"], False),
            ("at support max", support["max"], False),
            ("above support", support["max"] + 0.5, True),
        ]:
            # Skip probes the accepted-range layer rejects first (e.g. SpO2
            # support already ends at the accepted maximum of 100).
            if not (accepted_lo <= value <= accepted_hi):
                continue
            payload = dict(DEFAULT)
            payload[field] = value
            # Keep the pressure pair valid **and** inside its own support, so
            # this case isolates the field under test.
            if field == "systolic_bp":
                payload["diastolic_bp"] = max(25.0, min(80.0, value - 5))
            elif field == "diastolic_bp":
                payload["systolic_bp"] = max(45.0, value + 20)
            status, body = post(payload)
            warned = bool(body.get("distribution_warning")) if status == 200 else None
            if status != 200:
                fail(f"{field} {label} ({value})", [f"status {status}"])
            elif warned != should_warn:
                fail(f"{field} {label} ({value})",
                     [f"warning={warned} expected {should_warn}"])
            else:
                passes += 1

    # ── 5b. Cross-field blood pressure ───────────────────────────────────────
    print("[5b] cross-field blood pressure")
    for name, sbp, dbp, expect in [
        ("dbp > sbp (inverted)", 70, 120, 400),
        ("dbp == sbp (pulse 0)", 120, 120, 200),
        ("dbp just below sbp", 120, 119, 200),
        ("inverted at low pressure", 45, 46, 400),
        ("inverted, both in range", 150, 180, 400),
    ]:
        payload = dict(DEFAULT)
        payload["systolic_bp"] = sbp
        payload["diastolic_bp"] = dbp
        check(f"{name} ({sbp}/{dbp})", payload, expect,
              ["diastolic_bp"] if expect == 400 else None)

    # ── 5c. Data-derived red-flag constraint ─────────────────────────────────
    print("[5c] unconsciousness constraint")
    for name, patch in [
        ("healthy vitals + unconsciousness", {"unconsciousness": 1}),
        ("unconsciousness as string \"1\"", {"unconsciousness": "1"}),
        ("unconsciousness as boolean true", {"unconsciousness": True}),
        ("all five symptoms", {flag: 1 for flag in FLAGS}),
        ("critical vitals + unconsciousness",
         dict(age=67, heart_rate=142, systolic_bp=72, diastolic_bp=44, spo2=84,
              temperature=39.4, respiratory_rate=34, unconsciousness=1)),
        ("minimum vitals + unconsciousness",
         dict(age=0, heart_rate=20, systolic_bp=40, diastolic_bp=20, spo2=50,
              temperature=25.0, respiratory_rate=4, unconsciousness=1)),
    ]:
        payload = dict(DEFAULT)
        payload.update(patch)
        check_constraint(name, payload)

    check_constraint_free("default vitals carry no constraint", dict(DEFAULT))

    # ── 6. Determinism + concurrency ─────────────────────────────────────────
    print("[6] determinism and concurrency")
    payload = dict(DEFAULT)
    payload.update(age=67, heart_rate=142, systolic_bp=72, diastolic_bp=44,
                   spo2=84, temperature=39.4, respiratory_rate=34, chest_pain=1,
                   breathing_difficulty=1, dizziness=1)
    runs = [post(payload)[1] for _ in range(5)]
    if all(run == runs[0] for run in runs):
        print("    ok  5 sequential requests identical")
        passes += 1
    else:
        fail("sequential determinism", ["responses differ"])

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(lambda _: post(payload)[1], range(48)))
    if all(result == results[0] for result in results):
        print("    ok  48 concurrent requests identical")
        passes += 1
    else:
        fail("concurrent determinism", ["responses differ"])

    # ── 7. Metrics endpoint ──────────────────────────────────────────────────
    print("[7] /api/metrics")
    with urllib.request.urlopen(BASE + "/api/metrics", timeout=30) as response:
        metrics = json.load(response)
    classical = {k: v["test_accuracy"] for k, v in metrics["models"].items()}
    deep = {k: v["test_accuracy"] for k, v in metrics["deep_models"].items()}
    expected_classical = expected_metrics("classical_results.json", FALLBACK_CLASSICAL)
    expected_deep = expected_metrics("deep_results.json", FALLBACK_DEEP)
    for key, expected in expected_classical.items():
        if abs(classical.get(key, -1) - expected) > 1e-4:
            fail(f"metrics classical {key}", [f"{classical.get(key)} != {expected}"])
        else:
            passes += 1
    for key, expected in expected_deep.items():
        if abs(deep.get(key, -1) - expected) > 1e-4:
            fail(f"metrics deep {key}", [f"{deep.get(key)} != {expected}"])
        else:
            passes += 1
    print(f"    ok  classical {classical}")
    print(f"    ok  deep {deep}")

    print(f"\n{'=' * 78}")
    print(f"PASSED {passes} checks | FAILURES {len(failures)}")
    for failure in failures:
        print("  FAIL", failure)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
