from pathlib import Path

import pytest

from model_regression_detection.baseline_policy import (
    load_baseline_comparison_policy,
)


def test_v1_baseline_comparison_policy_is_valid() -> None:
    project_root = Path(__file__).resolve().parents[1]

    policy = load_baseline_comparison_policy(
        project_root
        / "config"
        / "baseline_comparison_policy_v1.yaml"
    )

    assert policy.maximum_category_accuracy_drop == 0.0
    assert policy.maximum_failed_case_increase == 0
    assert (
        policy.maximum_average_latency_increase_ratio
        == 0.25
    )


def test_baseline_policy_loader_rejects_non_object_yaml(
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
            "baseline comparison policy must be "
            "a YAML object"
        ),
    ):
        load_baseline_comparison_policy(policy_path)