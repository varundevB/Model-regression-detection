import pytest
from pydantic import ValidationError

from model_regression_detection.models import (
    ClassificationRequest,
    ClassificationResult,
    EmailCategory,
)


def test_request_rejects_blank_email() -> None:
    with pytest.raises(ValidationError):
        ClassificationRequest(email_text="   ")


def test_result_rejects_unknown_category() -> None:
    with pytest.raises(ValidationError):
        ClassificationResult(category="sales", summary="The customer asks for a demo.")


def test_result_accepts_contract() -> None:
    result = ClassificationResult(
        category=EmailCategory.BILLING,
        summary="The customer requests a refund for a duplicate charge.",
    )

    assert result.category is EmailCategory.BILLING


def test_request_rejects_unexpected_fields() -> None:
    with pytest.raises(ValidationError):
        ClassificationRequest(
            email_text="Please help me access my account.",
            customer_id="customer-123",
        )


def test_result_rejects_multiple_sentences() -> None:
    with pytest.raises(ValidationError):
        ClassificationResult(
            category=EmailCategory.TECHNICAL,
            summary="The application crashes. The customer needs assistance.",
        )