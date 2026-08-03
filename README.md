# Model Regression Detection System

CI-style evaluation for catching quality regressions in LLM-powered features.

## Current milestone

Part 1 establishes the feature under test:

- typed customer-email input and structured classification output;
- four supported categories: `billing`, `technical`, `account`, and `general`;
- versioned YAML prompt configuration;
- a provider boundary that keeps the classifier independent of one LLM vendor;
- contract tests that run without making paid API calls.

## Local setup

Requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
```

Copy `.env.example` to `.env` before running provider-backed classifications.
Never commit the `.env` file.

## Project structure

```text
prompts/       Versioned production prompts
src/           Feature and configuration code
tests/         Contract and prompt-validation tests
```

