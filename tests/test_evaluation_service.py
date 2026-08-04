from datetime import datetime, timezone
from pathlib import Path

from model_regression_detection.dataset_models import (
    GoldenDataset,
    GoldenTestCase,
)
from model_regression_detection.evaluation_models import (
    ProviderClassificationResult,
)
from model_regression_detection.evaluation_service import (
    run_evaluation,
)
from model_regression_detection.models import (
    ClassificationRequest,
    ClassificationResult,
    EmailCategory,
    PromptConfig,
)


class FakeProvider:
    model_name = "fake-model"

    def __init__(self) -> None:
        self.call_count = 0

    def classify_with_metadata(
        self,
        request: ClassificationRequest,
        prompt: PromptConfig,
    ) -> ProviderClassificationResult:
        self.call_count += 1

        return ProviderClassificationResult(
            classification=ClassificationResult(
                category=EmailCategory.BILLING,
                summary=(
                    "The customer reports a duplicate "
                    "subscription charge."
                ),
            ),
            model_name=self.model_name,
            input_tokens=100,
            output_tokens=20,
        )


def test_run_evaluation_creates_complete_report(
    tmp_path: Path,
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
            GoldenTestCase(
                id="billing-001",
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
                notes="A clear billing request.",
            )
        ],
    )

    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        dataset.model_dump_json(indent=2),
        encoding="utf-8",
    )

    project_root = Path(__file__).resolve().parents[1]
    prompt_path = (
        project_root
        / "prompts"
        / "support_classifier_v1.yaml"
    )

    provider = FakeProvider()

    report, report_path = run_evaluation(
        provider=provider,
        dataset_path=dataset_path,
        prompt_path=prompt_path,
        output_directory=tmp_path / "reports",
    )

    assert provider.call_count == 1
    assert report_path.exists()
    assert report.dataset_version == "golden-dataset-v1"
    assert report.prompt_version == (
        "support-classifier-v1"
    )
    assert report.model_name == "fake-model"
    assert report.summary.total_cases == 1
    assert report.summary.successful_cases == 1
    assert report.summary.failed_cases == 0
    assert report.summary.category_accuracy == 1.0
    assert len(report.results) == 1
