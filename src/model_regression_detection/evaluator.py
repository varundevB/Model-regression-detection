from time import perf_counter
from typing import Protocol

from .dataset_models import GoldenDataset, GoldenTestCase
from .evaluation_models import (
    EvaluationCaseResult,
    EvaluationSummary,
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

    def summarize_results(
                self,
                results: list[EvaluationCaseResult],
        ) -> EvaluationSummary:
            if not results:
                raise ValueError(
                    "results must contain at least one case"
                )

            prompt_versions = {
                result.prompt_version
                for result in results
            }
            model_names = {
                result.model_name
                for result in results
            }

            if len(prompt_versions) != 1:
                raise ValueError(
                    "results must share one prompt version"
                )

            if len(model_names) != 1:
                raise ValueError(
                    "results must share one model name"
                )

            total_cases = len(results)

            failed_cases = sum(
                result.error is not None
                for result in results
            )

            successful_cases = total_cases - failed_cases

            category_matches = sum(
                result.category_match
                for result in results
            )

            average_latency_ms = (
                    sum(
                        result.latency_ms
                        for result in results
                    )
                    / total_cases
            )

            total_input_tokens = sum(
                result.input_tokens or 0
                for result in results
            )

            total_output_tokens = sum(
                result.output_tokens or 0
                for result in results
            )

            return EvaluationSummary(
                total_cases=total_cases,
                successful_cases=successful_cases,
                failed_cases=failed_cases,
                category_matches=category_matches,
                category_accuracy=(
                        category_matches / total_cases
                ),
                average_latency_ms=average_latency_ms,
                total_input_tokens=total_input_tokens,
                total_output_tokens=total_output_tokens,
                prompt_version=next(iter(prompt_versions)),
                model_name=next(iter(model_names)),
            )