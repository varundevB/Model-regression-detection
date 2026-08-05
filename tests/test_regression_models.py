import pytest
from pydantic import ValidationError

from model_regression_detection.regression_models import (
    RegressionCheck,
    RegressionGateResult,
    RegressionPolicy,
)


def test_regression_policy_accepts_valid_thresholds() -> None:
    policy = RegressionPolicy(
        minimum_category_accuracy=0.9,
        maximum_failed_cases=0,
        maximum_average_latency_ms=3_000.0,
    )

    assert policy.minimum_category_accuracy == 0.9
    assert policy.maximum_failed_cases == 0
    assert policy.maximum_average_latency_ms == 3_000.0


def test_regression_policy_rejects_invalid_accuracy() -> None:
    with pytest.raises(ValidationError):
        RegressionPolicy(
            minimum_category_accuracy=1.1,
            maximum_failed_cases=0,
            maximum_average_latency_ms=3_000.0,
        )


def test_gate_rejects_inconsistent_overall_result() -> None:
    checks = [
        RegressionCheck(
            metric_name="category_accuracy",
            passed=True,
            actual_value=0.9,
            threshold_value=0.9,
            message="Category accuracy meets the threshold.",
        ),
        RegressionCheck(
            metric_name="failed_cases",
            passed=False,
            actual_value=1.0,
            threshold_value=0.0,
            message="Failed cases exceed the threshold.",
        ),
    ]

    with pytest.raises(
        ValidationError,
        match=(
            "passed must match the combined "
            "check results"
        ),
    ):
        RegressionGateResult(
            passed=True,
            checks=checks,
        )