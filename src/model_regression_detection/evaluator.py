from time import perf_counter
from typing import Protocol

from .dataset_models import GoldenDataset, GoldenTestCase
from .evaluation_models import (
    EvaluationCaseResult,
    ProviderClassificationResult,
)
from .models import (
    ClassificationRequest,
    PromptConfig,
)


class MetadataClassificationProvider(Protocol):
    @property
    def model_name(self) -> str:
        ...

    def classify_with_metadata(
        self,
        request: ClassificationRequest,
        prompt: PromptConfig,
    ) -> ProviderClassificationResult:
        ...


class EvaluationRunner:
    def __init__(
        self,
        provider: MetadataClassificationProvider,
        prompt: PromptConfig,
    ) -> None:
        self._provider = provider
        self._prompt = prompt

    def evaluate_dataset(
            self,
            dataset: GoldenDataset,
    ) -> list[EvaluationCaseResult]:
        return [
            self.evaluate_case(test_case)
            for test_case in dataset.test_cases
        ]

    def evaluate_case(
        self,
        test_case: GoldenTestCase,
    ) -> EvaluationCaseResult:
        email_text = (
            f"Subject: {test_case.subject}\n\n"
            f"{test_case.input}"
        )

        request = ClassificationRequest(
            email_text=email_text
        )

        started_at = perf_counter()

        try:
            provider_result = (
                self._provider.classify_with_metadata(
                    request=request,
                    prompt=self._prompt,
                )
            )

            latency_ms = (
                perf_counter() - started_at
            ) * 1000

            actual = provider_result.classification
            expected = test_case.expected_output

            return EvaluationCaseResult(
                test_case_id=test_case.id,
                expected_category=expected.category,
                actual_category=actual.category,
                category_match=(
                    actual.category == expected.category
                ),
                expected_summary=expected.summary,
                actual_summary=actual.summary,
                latency_ms=latency_ms,
                prompt_version=self._prompt.version_id,
                model_name=provider_result.model_name,
                input_tokens=provider_result.input_tokens,
                output_tokens=provider_result.output_tokens,
            )

        except Exception as error:
            latency_ms = (
                perf_counter() - started_at
            ) * 1000

            return EvaluationCaseResult(
                test_case_id=test_case.id,
                expected_category=(
                    test_case.expected_output.category
                ),
                actual_category=None,
                category_match=False,
                expected_summary=(
                    test_case.expected_output.summary
                ),
                actual_summary=None,
                latency_ms=latency_ms,
                prompt_version=self._prompt.version_id,
                model_name=self._provider.model_name,
                error=(
                    f"{type(error).__name__}: {error}"
                ),
            )