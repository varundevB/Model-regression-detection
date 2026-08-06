# Model Regression Detection System

A CI-style evaluation system for detecting quality regressions in an LLM-powered customer-support email classifier.

The project treats prompts and model behavior as versioned software artifacts. Instead of changing a prompt and hoping for the best, it runs a frozen evaluation dataset, records metrics, applies quality gates, and compares new candidates against an approved baseline.

## Why I built this

LLM features can appear to work during manual testing while quietly getting worse after a prompt or model change.

This project provides a repeatable way to answer one practical question:

> Did this change improve the model, or did it introduce a regression?

The example feature classifies customer-support emails into four categories:

- `billing`
- `technical`
- `account`
- `general`

It also produces a faithful one-sentence summary of each email.

## How it works

```text
Versioned prompt + frozen golden dataset
                    ↓
            LLM evaluation run
                    ↓
       Validated JSON evaluation report
                    ↓
          Absolute quality gates
                    ↓
    Baseline-versus-candidate comparison
                    ↓
                PASS or FAIL
```

The system records:

- expected and actual categories;
- category accuracy;
- provider failures;
- response latency;
- input and output token usage;
- prompt and model versions;
- expected and actual summaries;
- errors for individual test cases.

## Baseline experiment

The first baseline used `support-classifier-v1` against the frozen 10-case golden dataset.

| Metric | v1 baseline | v2 candidate |
|---|---:|---:|
| Successful cases | 10 / 10 | 10 / 10 |
| Provider failures | 0 | 0 |
| Category accuracy | 90% | 100% |
| Average latency | 1740.64 ms | 1853.99 ms |

The v1 model incorrectly classified `billing-003`, a question about where to submit receipts required for a refund, as `general`.

Prompt v2 clarified that refund receipts, payment documents, and questions about where or how to submit them belong to the `billing` category.

The v2 candidate corrected that classification without changing the frozen dataset.

Its average latency increased by approximately 6.5%, which remained within the configured 25% allowance. It passed both the absolute regression gate and the baseline comparison.

## Key design decisions

### Typed contracts

Pydantic validates the project’s important inputs and outputs:

- classification requests;
- structured LLM responses;
- prompt configuration;
- golden datasets;
- individual evaluation results;
- evaluation summaries and reports;
- regression policies;
- quality-gate results.

This prevents malformed model responses or inconsistent evaluation data from silently entering the results.

### Versioned prompts and datasets

Prompts live in `prompts/`, and evaluation data lives in `data/`.

The original v1 prompt remains unchanged while prompt v2 is evaluated against the same frozen dataset. This makes the comparison meaningful and preserves the history of the experiment.

### Provider boundary

The evaluation engine works through a provider interface instead of depending directly on one model vendor.

The production provider uses OpenAI structured outputs. Tests use fake providers, so the evaluation logic can be verified without making paid API calls.

### Two kinds of quality gates

Absolute gates enforce minimum acceptable standards:

- category accuracy must meet a configured minimum;
- provider failures must remain below a configured maximum;
- average latency must remain below a configured maximum.

Baseline comparison detects relative regressions:

- candidate accuracy cannot drop below the approved baseline;
- candidate provider failures cannot increase;
- candidate latency cannot exceed the allowed increase ratio.

A candidate must be evaluated with the same dataset version and model as the baseline. Prompt versions are allowed to differ because comparing prompt changes is one of the main purposes of the project.

### Local reports and safe CI

Real evaluation runs create JSON reports in `reports/`.

Those reports remain local and are excluded from Git. Automated tests use fake providers and make no OpenAI API calls.

GitHub Actions runs the no-cost test suite whenever code is pushed or a pull request targets `main`.

## Project structure

```text
.github/workflows/        GitHub Actions test workflow
config/                   Versioned regression policies
data/                     Frozen golden evaluation dataset
prompts/                  Versioned LLM prompts
scripts/                  Evaluation and validation commands
src/model_regression_detection/
  baseline.py             Baseline comparison logic
  baseline_models.py      Baseline comparison contracts
  classifier.py           Classification feature boundary
  dataset_models.py       Golden dataset contracts
  evaluation_models.py    Evaluation result and report contracts
  evaluation_service.py   Evaluation and report orchestration
  evaluator.py            Case, dataset, and summary evaluation
  gmail_parser.py         Gmail message parsing
  models.py               Core classification and prompt models
  openai_provider.py      OpenAI structured-output provider
  regression.py           Absolute regression gates
  reporting.py            Report construction and persistence
tests/                    No-cost automated tests
```

## Local setup

Python 3.11 or newer is required.

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the project and development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Create a local environment file:

```bash
cp .env.example .env
```

Add your OpenAI configuration to `.env`:

```text
OPENAI_API_KEY=your-key-here
OPENAI_MODEL=gpt-4o-mini
```

Never commit `.env`, OAuth credentials, authentication tokens, or generated reports.

## Running the tests

Run the complete no-cost test suite:

```bash
python -m pytest -v
```

The tests cover:

- input and output validation;
- prompt loading;
- Gmail message parsing;
- provider contracts;
- dataset validation;
- evaluation logic;
- metric summaries;
- report generation;
- absolute regression gates;
- baseline comparison;
- versioned policy loading.

## Validating the golden dataset

```bash
python scripts/validate_golden_dataset.py
```

## Running an evaluation

Run the default evaluation with prompt v1:

```bash
python scripts/run_evaluation.py
```

Run an experiment with prompt v2:

```bash
python scripts/run_evaluation.py \
  --prompt prompts/support_classifier_v2.yaml
```

Real evaluations use the configured OpenAI model and may incur API charges.

## Checking an absolute regression gate

Check a saved report against the absolute regression policy:

```bash
python scripts/check_evaluation_report.py \
  reports/<run-id>.json
```

The command exits with:

- `0` when every quality gate passes;
- `1` when at least one quality gate fails.

These exit codes make the command suitable for CI systems.

## Comparing two evaluation reports

Compare a candidate report against an approved baseline:

```bash
python scripts/compare_evaluation_reports.py \
  reports/<baseline-run-id>.json \
  reports/<candidate-run-id>.json
```

The comparison evaluates accuracy, provider failures, and average latency.

For the v1-to-v2 experiment, the comparison passed:

```text
Baseline comparison: PASS
- category_accuracy: PASS
- failed_cases: PASS
- average_latency_ms: PASS
```

## Continuous integration

GitHub Actions runs the complete no-cost test suite for pushes and pull requests.

The workflow intentionally does not run live OpenAI evaluations. This keeps API keys out of CI, prevents unexpected costs, and makes automated tests deterministic.

## Data and privacy

The golden dataset contains dummy customer-support emails created for this project.

The project excludes the following local or sensitive files from Git:

- API keys and `.env`;
- Gmail OAuth credentials;
- authentication tokens;
- imported Gmail data;
- generated evaluation reports;
- Python virtual environments.

## Current limitations

- The golden dataset currently contains only 10 cases.
- Summary quality is recorded but is not yet scored semantically.
- Live model responses and latency can vary between runs.
- CI verifies the code and configuration but does not make live OpenAI calls.
- The current evaluation focuses on one classification use case.

## Docker

The project can run inside a Python 3.12 Linux container. The image runs as a non-root user and excludes local environments, credentials, OAuth tokens, generated reports, and private imported data.

Build the image:

```bash
docker build -t model-regression-detection:dev .
```

Run the automated test suite:

```bash
docker run --rm model-regression-detection:dev
```

The automated tests use fake providers and do not make OpenAI API calls.

Validate the golden dataset:

```bash
docker run --rm \
  model-regression-detection:dev \
  python scripts/validate_golden_dataset.py
```

Run a real evaluation:

```bash
docker run --rm \
  --env-file .env \
  -v "$(pwd)/reports:/app/reports" \
  model-regression-detection:dev \
  python scripts/run_evaluation.py \
  --dataset data/golden_dataset_v1.json \
  --prompt prompts/support_classifier_v2.yaml \
  --policy config/regression_policy_v1.yaml \
  --output-directory reports
```

The `.env` file is supplied only at runtime and is not copied into the image. The bind-mounted `reports/` directory preserves generated reports on the host after the container exits.


## What I learned

Building this project involved more than calling an LLM API. The important work was creating reliable boundaries around the model:

- defining strict input and output contracts;
- building and reviewing a golden dataset;
- separating deterministic tests from paid model calls;
- recording enough metadata to reproduce experiments;
- treating prompts as versioned code;
- detecting both absolute failures and relative regressions;
