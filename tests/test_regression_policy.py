from pathlib import Path

import pytest

from model_regression_detection.regression_policy import (
    load_regression_policy,
)


def test_v1_regression_policy_is_valid() -> None:
    project_root = Path(__file__).resolve().parents[1]

    policy = load_regression_policy(
        project_root
        / "config"
        / "regression_policy_v1.yaml"
    )

    assert policy.minimum_category_accuracy == 0.9
    assert policy.maximum_failed_cases == 0
    assert (
        policy.maximum_average_latency_ms
        == 3_000.0
    )


def test_policy_loader_rejects_non_object_yaml(
    tmp_path: Path,
) -> None:
    policy_path = tmp_path / "invalid_policy.yaml"
    policy_path.write_text(
        "- first item\n- second item\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=(
            "regression policy must be "
            "a YAML object"
        ),
    ):
        load_regression_policy(policy_path)
