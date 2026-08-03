from datetime import datetime, timezone

from model_regression_detection.dataset_models import (
    GoldenDataset,
    GoldenTestCase,
)
from model_regression_detection.evaluation_models import (
    ProviderClassificationResult,
    EvaluationCaseResult,
)
from model_regression_detection.evaluator import EvaluationRunner
from model_regression_detection.models import (
    ClassificationRequest,
    ClassificationResult,
    EmailCategory,
    PromptConfig,
)
import pytest

class SuccessfulFakeProvider:
    model_name = "fake-model"

    def __init__(self) -> None:
        self.received_request: ClassificationRequest | None = None

    def classify_with_metadata(
        self,
        request: ClassificationRequest,
        prompt: PromptConfig,
    ) -> ProviderClassificationResult:
        self.received_request = request

        return ProviderClassificationResult(
            classification=ClassificationResult(
                category=EmailCategory.BILLING,
                summary=(
                    "The customer cannot update their "
                    "payment details."
                ),
            ),
            model_name=self.model_name,
            input_tokens=100,
            output_tokens=20,
        )


class FailingFakeProvider:
    model_name = "fake-model"

    def classify_with_metadata(
        self,
        request: ClassificationRequest,
        prompt: PromptConfig,
    ) -> ProviderClassificationResult:
        raise TimeoutError("provider timed out")


def create_billing_test_case() -> GoldenTestCase:
    return GoldenTestCase(
        id="billing-001",
        subject="Transaction error",
        input="All my payment methods are being rejected.",
        expected_output=ClassificationResult(
            category=EmailCategory.BILLING,
            summary=(
                "The customer cannot update their "
                "payment details."
            ),
        ),
        expected_difficulty="easy",
        notes="A clear billing request.",
    )


def test_evaluator_records_successful_result(
    prompt_config: PromptConfig,
) -> None:
    provider = SuccessfulFakeProvider()
    runner = EvaluationRunner(
        provider=provider,
        prompt=prompt_config,
    )

    result = runner.evaluate_case(
        create_billing_test_case()
    )

    assert result.category_match is True
    assert result.actual_category is EmailCategory.BILLING
    assert result.model_name == "fake-model"
    assert result.input_tokens == 100
    assert result.output_tokens == 20
    assert result.error is None

    assert provider.received_request is not None
    assert "Subject: Transaction error" in (
        provider.received_request.email_text
    )


def test_evaluator_records_provider_failure(
    prompt_config: PromptConfig,
) -> None:
    runner = EvaluationRunner(
        provider=FailingFakeProvider(),
        prompt=prompt_config,
    )

    result = runner.evaluate_case(
        create_billing_test_case()
    )

    assert result.category_match is False
    assert result.actual_category is None
    assert result.actual_summary is None
    assert result.model_name == "fake-model"
    assert result.error is not None
    assert "TimeoutError" in result.error
    assert "provider timed out" in result.error


def test_evaluator_processes_complete_dataset(
    prompt_config: PromptConfig,
) -> None:
    dataset = GoldenDataset(
        version_id="golden-dataset-v1",
        created_at=datetime(
            2026,
            7,
            31,
            tzinfo=timezone.utc,
        ),
        test_cases=[
            create_billing_test_case(),
            GoldenTestCase(
                id="billing-002",
                subject="Duplicate charge",
                input=(
                    "I was charged twice for my "
                    "subscription."
                ),
                expected_output=ClassificationResult(
                    category=EmailCategory.BILLING,
                    summary=(
                        "The customer reports a duplicate "
                        "subscription charge."
                    ),
                ),
                expected_difficulty="easy",
                notes="Another clear billing request.",
            ),
        ],
    )

    runner = EvaluationRunner(
        provider=SuccessfulFakeProvider(),
        prompt=prompt_config,
    )

    results = runner.evaluate_dataset(dataset)

    assert len(results) == 2

    assert [
        result.test_case_id
        for result in results
    ] == [
        "billing-001",
        "billing-002",
    ]

    assert all(
        result.category_match
        for result in results
    )
def test_summarize_results_calculates_metrics(
    prompt_config: PromptConfig,
) -> None:
    runner = EvaluationRunner(
        provider=SuccessfulFakeProvider(),
        prompt=prompt_config,
    )

    results = [
        EvaluationCaseResult(
            test_case_id="billing-001",
            expected_category=EmailCategory.BILLING,
            actual_category=EmailCategory.BILLING,
            category_match=True,
            expected_summary=(
                "The customer reports a billing problem."
            ),
            actual_summary=(
                "The customer reports a billing problem."
            ),
            latency_ms=100.0,
            prompt_version="support-classifier-v1",
            model_name="fake-model",
            input_tokens=100,
            output_tokens=20,
        ),
        EvaluationCaseResult(
            test_case_id="technical-001",
            expected_category=EmailCategory.TECHNICAL,
            actual_category=None,
            category_match=False,
            expected_summary=(
                "The customer reports an upload problem."
            ),
            actual_summary=None,
            latency_ms=300.0,
            prompt_version="support-classifier-v1",
            model_name="fake-model",
            error="TimeoutError: provider timed out",
        ),
    ]

    summary = runner.summarize_results(results)

    assert summary.total_cases == 2
    assert summary.successful_cases == 1
    assert summary.failed_cases == 1
    assert summary.category_matches == 1
    assert summary.category_accuracy == 0.5
    assert summary.average_latency_ms == 200.0
    assert summary.total_input_tokens == 100
    assert summary.total_output_tokens == 20
    assert (
        summary.prompt_version
        == "support-classifier-v1"
    )
    assert summary.model_name == "fake-model"


def test_summarize_results_rejects_empty_results(
    prompt_config: PromptConfig,
) -> None:
    runner = EvaluationRunner(
        provider=SuccessfulFakeProvider(),
        prompt=prompt_config,
    )

    with pytest.raises(
        ValueError,
        match="results must contain at least one case",
    ):
        runner.summarize_results([])