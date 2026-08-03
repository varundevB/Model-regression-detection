import pytest
from pydantic import ValidationError

from model_regression_detection.dataset_models import (
    GoldenDataset,
    GoldenTestCase,
)
from model_regression_detection.models import (
    ClassificationResult,
    EmailCategory,
)


def create_billing_case() -> GoldenTestCase:
    return GoldenTestCase(
        id="billing-001",
        subject="Charged twice for my subscription",
        input=(
            "I upgraded yesterday and noticed two "
            "identical charges on my card."
        ),
        expected_output=ClassificationResult(
            category=EmailCategory.BILLING,
            summary=(
                "The customer reports duplicate "
                "subscription charges."
            ),
        ),
        expected_difficulty="easy",
        notes=(
            "Clear billing case involving a "
            "duplicate charge."
        ),
    )


def test_golden_dataset_accepts_valid_case() -> None:
    dataset = GoldenDataset(
        version_id="golden-dataset-v1",
        created_at="2026-07-31T00:00:00Z",
        test_cases=[create_billing_case()],
    )

    assert len(dataset.test_cases) == 1
    assert dataset.test_cases[0].id == "billing-001"


def test_golden_dataset_rejects_duplicate_ids() -> None:
    test_case = create_billing_case()

    with pytest.raises(ValidationError):
        GoldenDataset(
            version_id="golden-dataset-v1",
            created_at="2026-07-31T00:00:00Z",
            test_cases=[
                test_case,
                test_case,
            ],
        )