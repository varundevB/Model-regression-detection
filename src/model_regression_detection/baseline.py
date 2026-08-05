from .baseline_models import (
    BaselineComparisonCheck,
    BaselineComparisonPolicy,
    BaselineComparisonResult,
)
from .evaluation_models import EvaluationRunReport


def compare_against_baseline(
    baseline: EvaluationRunReport,
    candidate: EvaluationRunReport,
    policy: BaselineComparisonPolicy,
) -> BaselineComparisonResult:
    if baseline.dataset_version != candidate.dataset_version:
        raise ValueError(
            "baseline and candidate must use the same "
            "dataset version"
        )

    if baseline.model_name != candidate.model_name:
        raise ValueError(
            "baseline and candidate must use the same "
            "model name"
        )

    if (
        baseline.summary.total_cases
        != candidate.summary.total_cases
    ):
        raise ValueError(
            "baseline and candidate must contain the same "
            "number of cases"
        )

    accuracy_threshold = (
        baseline.summary.category_accuracy
        - policy.maximum_category_accuracy_drop
    )
    accuracy_passed = (
        candidate.summary.category_accuracy
        >= accuracy_threshold
    )

    failure_threshold = (
        baseline.summary.failed_cases
        + policy.maximum_failed_case_increase
    )
    failures_passed = (
        candidate.summary.failed_cases
        <= failure_threshold
    )

    latency_threshold = (
        baseline.summary.average_latency_ms
        * (
            1
            + policy.maximum_average_latency_increase_ratio
        )
    )
    latency_passed = (
        candidate.summary.average_latency_ms
        <= latency_threshold
    )

    checks = [
        BaselineComparisonCheck(
            metric_name="category_accuracy",
            passed=accuracy_passed,
            baseline_value=baseline.summary.category_accuracy,
            candidate_value=candidate.summary.category_accuracy,
            allowed_value=accuracy_threshold,
            message=(
                "Category accuracy did not regress."
                if accuracy_passed
                else
                "Category accuracy regressed beyond "
                "the allowed drop."
            ),
        ),
        BaselineComparisonCheck(
            metric_name="failed_cases",
            passed=failures_passed,
            baseline_value=float(
                baseline.summary.failed_cases
            ),
            candidate_value=float(
                candidate.summary.failed_cases
            ),
            allowed_value=float(failure_threshold),
            message=(
                "Provider failures did not increase."
                if failures_passed
                else
                "Provider failures increased beyond "
                "the allowed limit."
            ),
        ),
        BaselineComparisonCheck(
            metric_name="average_latency_ms",
            passed=latency_passed,
            baseline_value=(
                baseline.summary.average_latency_ms
            ),
            candidate_value=(
                candidate.summary.average_latency_ms
            ),
            allowed_value=latency_threshold,
            message=(
                "Average latency stayed within the "
                "allowed increase."
                if latency_passed
                else
                "Average latency increased beyond the "
                "allowed limit."
            ),
        ),
    ]

    return BaselineComparisonResult(
        passed=all(
            check.passed
            for check in checks
        ),
        checks=checks,
    )