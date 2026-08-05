import pytest
from pydantic import ValidationError

from model_regression_detection.baseline_models import (
    BaselineComparisonCheck,
    BaselineComparisonPolicy,
    BaselineComparisonResult,
)


def test_baseline_policy_accepts_valid_thresholds() -> None:
    policy = BaselineComparisonPolicy(
        maximum_category_accuracy_drop=0.0,
        maximum_failed_case_increase=0,
        maximum_average_latency_increase_ratio=0.25,
    )

    assert policy.maximum_category_accuracy_drop == 0.0
    assert policy.maximum_failed_case_increase == 0
    assert (
        policy.maximum_average_latency_increase_ratio
        == 0.25
    )


def test_baseline_policy_rejects_negative_latency_ratio() -> None:
    with pytest.raises(ValidationError):
        BaselineComparisonPolicy(
            maximum_category_accuracy_drop=0.0,
            maximum_failed_case_increase=0,
            maximum_average_latency_increase_ratio=-0.1,
        )


def test_comparison_rejects_inconsistent_overall_result() -> None:
    checks = [
        BaselineComparisonCheck(
            metric_name="category_accuracy",
            passed=True,
            baseline_value=0.9,
            candidate_value=0.9,
            allowed_value=0.9,
            message="Category accuracy did not decrease.",
        ),
        BaselineComparisonCheck(
            metric_name="failed_cases",
            passed=False,
            baseline_value=0.0,
            candidate_value=1.0,
            allowed_value=0.0,
            message="Provider failures increased.",
        ),
    ]

    with pytest.raises(
        ValidationError,
        match=(
            "passed must match the combined "
            "check results"
        ),
    ):
        BaselineComparisonResult(
            passed=True,
            checks=checks,
        )