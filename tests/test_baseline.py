from datetime import datetime, timezone

import pytest

from model_regression_detection.baseline import (
    compare_against_baseline,
)
from model_regression_detection.baseline_models import (
    BaselineComparisonPolicy,
)
from model_regression_detection.evaluation_models import (
    EvaluationCaseResult,
    EvaluationRunReport,
    EvaluationSummary,
)
from model_regression_detection.models import EmailCategory


def create_report(
    *,
    run_id: str,
    dataset_version: str = "golden-dataset-v1",
    prompt_version: str = "support-classifier-v1",
    model_name: str = "fake-model",
    category_accuracy: float = 1.0,
    failed_cases: int = 0,
    average_latency_ms: float = 100.0,
) -> EvaluationRunReport:
    has_failed = failed_cases == 1
    category_match = category_accuracy == 1.0 and not has_failed

    result = EvaluationCaseResult(
        test_case_id="billing-001",
        expected_category=EmailCategory.BILLING,
        actual_category=(
            None
            if has_failed
            else (
                EmailCategory.BILLING
                if category_match
                else EmailCategory.GENERAL
            )
        ),
        category_match=category_match,
        expected_summary=(
            "The customer reports a billing problem."
        ),
        actual_summary=(
            None
            if has_failed
            else "The customer reports a billing problem."
        ),
        latency_ms=average_latency_ms,
        prompt_version=prompt_version,
        model_name=model_name,
        input_tokens=100,
        output_tokens=20,
        error=(
            "TimeoutError: provider timed out"
            if has_failed
            else None
        ),
    )

    summary = EvaluationSummary(
        total_cases=1,
        successful_cases=1 - failed_cases,
        failed_cases=failed_cases,
        category_matches=1 if category_match else 0,
        category_accuracy=category_accuracy,
        average_latency_ms=average_latency_ms,
        total_input_tokens=100,
        total_output_tokens=20,
        prompt_version=prompt_version,
        model_name=model_name,
    )

    return EvaluationRunReport(
        run_id=run_id,
        created_at=datetime(
            2026,
            8,
            5,
            tzinfo=timezone.utc,
        ),
        dataset_version=dataset_version,
        prompt_version=prompt_version,
        model_name=model_name,
        results=[result],
        summary=summary,
    )


def create_policy() -> BaselineComparisonPolicy:
    return BaselineComparisonPolicy(
        maximum_category_accuracy_drop=0.0,
        maximum_failed_case_increase=0,
        maximum_average_latency_increase_ratio=0.25,
    )


def test_comparison_accepts_improved_prompt() -> None:
    baseline = create_report(run_id="baseline-run")
    candidate = create_report(
        run_id="candidate-run",
        prompt_version="support-classifier-v2",
        average_latency_ms=125.0,
    )

    result = compare_against_baseline(
        baseline=baseline,
        candidate=candidate,
        policy=create_policy(),
    )

    assert result.passed is True
    assert all(check.passed for check in result.checks)


def test_comparison_detects_metric_regressions() -> None:
    baseline = create_report(run_id="baseline-run")
    candidate = create_report(
        run_id="candidate-run",
        category_accuracy=0.0,
        failed_cases=1,
        average_latency_ms=126.0,
    )

    result = compare_against_baseline(
        baseline=baseline,
        candidate=candidate,
        policy=create_policy(),
    )

    checks = {
        check.metric_name: check
        for check in result.checks
    }

    assert result.passed is False
    assert checks["category_accuracy"].passed is False
    assert checks["failed_cases"].passed is False
    assert checks["average_latency_ms"].passed is False


def test_comparison_rejects_different_dataset_versions() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "baseline and candidate must use the same "
            "dataset version"
        ),
    ):
        compare_against_baseline(
            baseline=create_report(run_id="baseline-run"),
            candidate=create_report(
                run_id="candidate-run",
                dataset_version="golden-dataset-v2",
            ),
            policy=create_policy(),
        )


def test_comparison_rejects_different_model_names() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "baseline and candidate must use the same "
            "model name"
        ),
    ):
        compare_against_baseline(
            baseline=create_report(run_id="baseline-run"),
            candidate=create_report(
                run_id="candidate-run",
                model_name="another-model",
            ),
            policy=create_policy(),
        )