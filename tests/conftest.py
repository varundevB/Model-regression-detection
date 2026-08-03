from pathlib import Path

import pytest

from model_regression_detection.models import PromptConfig
from model_regression_detection.prompts import load_prompt_config


@pytest.fixture
def prompt_config() -> PromptConfig:
    project_root = Path(__file__).parents[1]
    return load_prompt_config(project_root / "prompts" / "support_classifier_v1.yaml")
