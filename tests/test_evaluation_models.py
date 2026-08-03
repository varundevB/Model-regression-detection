import pytest
from pydantic import ValidationError

from model_regression_detection.evaluation_models import (
    EvaluationCaseResult,
)
from model_regression_detection.models import EmailCategory


def test_successful_evaluation_result() -> None:
    result = EvaluationCaseResult(
        test_case_id="billing-001",
        expected_category=EmailCategory.BILLING,
        actual_category=EmailCategory.BILLING,
        category_match=True,
        expected_summary=(
            "The customer reports rejected payment methods."
        ),
        actual_summary=(
            "The customer cannot update their payment details."
        ),
        latency_ms=425.5,
        prompt_version="support-classifier-v1",
        model_name="gpt-4o-mini",
        input_tokens=120,
        output_tokens=20,
    )

    assert result.category_match is True
    assert result.error is None


def test_failed_evaluation_result() -> None:
    result = EvaluationCaseResult(
        test_case_id="technical-001",
        expected_category=EmailCategory.TECHNICAL,
        actual_category=None,
        category_match=False,
        expected_summary=(
            "The customer reports a file-upload error."
        ),
        actual_summary=None,
        latency_ms=30_000,
        prompt_version="support-classifier-v1",
        model_name="gpt-4o-mini",
        error="Request timed out",
    )

    assert result.actual_category is None
    assert result.error == "Request timed out"


def test_evaluation_result_rejects_negative_latency() -> None:
    with pytest.raises(ValidationError):
        EvaluationCaseResult(
            test_case_id="general-001",
            expected_category=EmailCategory.GENERAL,
            actual_category=EmailCategory.GENERAL,
            category_match=True,
            expected_summary="The customer asks about student benefits.",
            actual_summary="The customer asks about student benefits.",
            latency_ms=-1,
            prompt_version="support-classifier-v1",
            model_name="gpt-4o-mini",
        )