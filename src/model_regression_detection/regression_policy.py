from pathlib import Path

import yaml

from .regression_models import RegressionPolicy


def load_regression_policy(
    path: str | Path,
) -> RegressionPolicy:
    policy_path = Path(path)

    raw_policy = yaml.safe_load(
        policy_path.read_text(encoding="utf-8")
    )

    if not isinstance(raw_policy, dict):
        raise ValueError(
            "regression policy must be a YAML object"
        )

    return RegressionPolicy.model_validate(raw_policy)