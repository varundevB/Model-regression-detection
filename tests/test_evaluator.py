from model_regression_detection.dataset_models import GoldenTestCase
from model_regression_detection.evaluation_models import (
    ProviderClassificationResult,
)
from model_regression_detection.evaluator import EvaluationRunner
from model_regression_detection.models import (
    ClassificationRequest,
    ClassificationResult,
    EmailCategory,
    PromptConfig,
)


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
                summary="The customer cannot update their payment details.",
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
            summary="The customer cannot update their payment details.",
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