from pathlib import Path

import yaml
from pydantic import ValidationError

from .models import PromptConfig


class PromptConfigError(ValueError):
    """Raised when a prompt file cannot be loaded or validated."""


def load_prompt_config(path: str | Path) -> PromptConfig:
    prompt_path = Path(path)
    if not prompt_path.is_file():
        raise PromptConfigError(f"Prompt file does not exist: {prompt_path}")

    try:
        raw_config = yaml.safe_load(prompt_path.read_text(encoding="utf-8"))
        return PromptConfig.model_validate(raw_config)
    except (yaml.YAMLError, ValidationError, OSError) as exc:
        raise PromptConfigError(f"Invalid prompt file {prompt_path}: {exc}") from exc
