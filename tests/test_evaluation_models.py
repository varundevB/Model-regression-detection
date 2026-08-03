import pytest
from pydantic import ValidationError

from model_regression_detection.evaluation_models import (
    EvaluationCaseResult,
    EvaluationSummary,
)
from model_regression_detection.models import EmailCategory


def test_successful_evaluation_result() -> None:
    result = EvaluationCaseResult(
        test_case_id="billing-001",
        expected_category=EmailCategory.BILLING,
        actual_category=EmailCategory.BILLING,
        category_match=True,
        expected_summary=(
            "The customer reports a duplicate charge."
        ),
        actual_summary=(
            "The customer reports a duplicate charge."
        ),
        latency_ms=250.5,
        prompt_version="support-classifier-v1",
        model_name="fake-model",
        input_tokens=100,
        output_tokens=20,
    )

    assert result.test_case_id == "billing-001"
    assert result.category_match is True
    assert result.actual_category is EmailCategory.BILLING
    assert result.error is None
    assert result.input_tokens == 100
    assert result.output_tokens == 20


def test_failed_evaluation_result() -> None:
    result = EvaluationCaseResult(
        test_case_id="technical-001",
        expected_category=EmailCategory.TECHNICAL,
        actual_category=None,
        category_match=False,
        expected_summary=(
            "The customer cannot upload a file."
        ),
        actual_summary=None,
        latency_ms=500.0,
        prompt_version="support-classifier-v1",
        model_name="fake-model",
        error="TimeoutError: provider timed out",
    )

    assert result.test_case_id == "technical-001"
    assert result.category_match is False
    assert result.actual_category is None
    assert result.actual_summary is None
    assert result.error is not None
    assert "TimeoutError" in result.error


def test_evaluation_result_rejects_negative_latency() -> None:
    with pytest.raises(ValidationError):
        EvaluationCaseResult(
            test_case_id="billing-001",
            expected_category=EmailCategory.BILLING,
            actual_category=EmailCategory.BILLING,
            category_match=True,
            expected_summary=(
                "The customer reports a duplicate charge."
            ),
            actual_summary=(
                "The customer reports a duplicate charge."
            ),
            latency_ms=-1.0,
            prompt_version="support-classifier-v1",
            model_name="fake-model",
            input_tokens=100,
            output_tokens=20,
        )


def test_evaluation_summary_accepts_valid_metrics() -> None:
    summary = EvaluationSummary(
        total_cases=10,
        successful_cases=9,
        failed_cases=1,
        category_matches=8,
        category_accuracy=0.8,
        average_latency_ms=250.5,
        total_input_tokens=1_000,
        total_output_tokens=200,
        prompt_version="support-classifier-v1",
        model_name="fake-model",
    )

    assert summary.total_cases == 10
    assert summary.successful_cases == 9
    assert summary.failed_cases == 1
    assert summary.category_matches == 8
    assert summary.category_accuracy == 0.8
    assert summary.average_latency_ms == 250.5
    assert summary.total_input_tokens == 1_000
    assert summary.total_output_tokens == 200


def test_evaluation_summary_rejects_inconsistent_counts() -> None:
    with pytest.raises(
        ValidationError,
        match=(
            "successful_cases and failed_cases "
            "must equal total_cases"
        ),
    ):
        EvaluationSummary(
            total_cases=10,
            successful_cases=8,
            failed_cases=1,
            category_matches=7,
            category_accuracy=0.7,
            average_latency_ms=200.0,
            total_input_tokens=900,
            total_output_tokens=180,
            prompt_version="support-classifier-v1",
            model_name="fake-model",
        )


def test_evaluation_summary_rejects_too_many_matches() -> None:
    with pytest.raises(
        ValidationError,
        match=(
            "category_matches cannot exceed "
            "successful_cases"
        ),
    ):
        EvaluationSummary(
            total_cases=10,
            successful_cases=8,
            failed_cases=2,
            category_matches=9,
            category_accuracy=0.9,
            average_latency_ms=200.0,
            total_input_tokens=900,
            total_output_tokens=180,
            prompt_version="support-classifier-v1",
            model_name="fake-model",
        )