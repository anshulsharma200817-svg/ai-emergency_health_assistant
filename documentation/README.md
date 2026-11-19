# Documentation

| Document | What it covers | Best for |
|---|---|---|
| [`PROJECT_REPORT.md`](PROJECT_REPORT.md) | Submission-ready report: abstract, problem, objectives, design, results, testing, limitations | Project evaluation and viva |
| [`../README.md`](../README.md) | Project overview, technologies, features, setup, screenshots, team | First look at the repository |
| [`../SYSTEMDESIGN.md`](../SYSTEMDESIGN.md) | Full technical design: architecture, data, models, API contract, verification, deployment | Understanding or auditing the internals |
| [`../BUILD_STATUS.md`](../BUILD_STATUS.md) | Current build state, measured model results, bugs found and fixed | Checking what is actually done |
| [`../architecture.md`](../architecture.md) | One-page visual of the training and serving planes | Quick orientation |

## Suggested reading order

1. `README.md` §1–§10 — problem, objectives, technologies, setup and screenshots.
2. `PROJECT_REPORT.md` — the same material in report form, with the full results tables.
3. `SYSTEMDESIGN.md` §1–§9 — how the data flows and why each model was or was not served.
4. `SYSTEMDESIGN.md` §10–§16 — API contract, parity verification and deployment.
5. `BUILD_STATUS.md` §3 — the defects found during verification and how each was fixed.

## Reproducing the published numbers

```bash
python3 generate_data.py
python3 run_pipeline.py --classical-only --skip-generate
python3 run_pipeline.py --deep-only --skip-generate        # optional, needs PyTorch
python3 -m pytest tests -q                                 # 81 tests
python3 scripts/export_web_model_params.py                 # refresh served model JSON
python3 scripts/export_web_training_results.py             # refresh home-screen charts
```

Every figure quoted in the report is produced by these commands from seed 42.

## Screenshots

Captured from the running production build; see [`../screenshots/`](../screenshots).
