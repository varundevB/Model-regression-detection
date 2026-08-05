from pathlib import Path

import yaml

from .baseline_models import BaselineComparisonPolicy


def load_baseline_comparison_policy(
    path: str | Path,
) -> BaselineComparisonPolicy:
    policy_path = Path(path)

    raw_policy = yaml.safe_load(
        policy_path.read_text(encoding="utf-8")
    )

    if not isinstance(raw_policy, dict):
        raise ValueError(
            "baseline comparison policy must be "
            "a YAML object"
        )

    return BaselineComparisonPolicy.model_validate(
        raw_policy
    )