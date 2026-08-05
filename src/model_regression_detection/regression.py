from .evaluation_models import EvaluationSummary
from .regression_models import (
    RegressionCheck,
    RegressionGateResult,
    RegressionPolicy,
)


def evaluate_regression_gate(
    summary: EvaluationSummary,
    policy: RegressionPolicy,
) -> RegressionGateResult:
    accuracy_passed = (
        summary.category_accuracy
        >= policy.minimum_category_accuracy
    )

    failures_passed = (
        summary.failed_cases
        <= policy.maximum_failed_cases
    )

    latency_passed = (
        summary.average_latency_ms
        <= policy.maximum_average_latency_ms
    )

    checks = [
        RegressionCheck(
            metric_name="category_accuracy",
            passed=accuracy_passed,
            actual_value=summary.category_accuracy,
            threshold_value=(
                policy.minimum_category_accuracy
            ),
            message=(
                "Category accuracy meets the threshold."
                if accuracy_passed
                else
                "Category accuracy is below the threshold."
            ),
        ),
        RegressionCheck(
            metric_name="failed_cases",
            passed=failures_passed,
            actual_value=float(summary.failed_cases),
            threshold_value=float(
                policy.maximum_failed_cases
            ),
            message=(
                "Failed cases meet the threshold."
                if failures_passed
                else
                "Failed cases exceed the threshold."
            ),
        ),
        RegressionCheck(
            metric_name="average_latency_ms",
            passed=latency_passed,
            actual_value=summary.average_latency_ms,
            threshold_value=(
                policy.maximum_average_latency_ms
            ),
            message=(
                "Average latency meets the threshold."
                if latency_passed
                else
                "Average latency exceeds the threshold."
            ),
        ),
    ]

    return RegressionGateResult(
        passed=all(
            check.passed
            for check in checks
        ),
        checks=checks,
    )