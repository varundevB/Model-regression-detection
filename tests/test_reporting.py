import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from model_regression_detection.dataset_models import (
    GoldenDataset,
    GoldenTestCase,
)
from model_regression_detection.evaluation_models import (
    EvaluationCaseResult,
    EvaluationSummary,
)
from model_regression_detection.models import (
    ClassificationResult,
    EmailCategory,
)
from model_regression_detection.reporting import (
    build_evaluation_report,
    write_evaluation_report,
)


def create_dataset() -> GoldenDataset:
    return GoldenDataset(
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


def create_result() -> EvaluationCaseResult:
    return EvaluationCaseResult(
        test_case_id="billing-001",
        expected_category=EmailCategory.BILLING,
        actual_category=EmailCategory.BILLING,
        category_match=True,
        expected_summary=(
            "The customer reports a duplicate "
            "subscription charge."
        ),
        actual_summary=(
            "The customer reports a duplicate "
            "subscription charge."
        ),
        latency_ms=100.0,
        prompt_version="support-classifier-v1",
        model_name="fake-model",
        input_tokens=100,
        output_tokens=20,
    )


def create_summary() -> EvaluationSummary:
    return EvaluationSummary(
        total_cases=1,
        successful_cases=1,
        failed_cases=0,
        category_matches=1,
        category_accuracy=1.0,
        average_latency_ms=100.0,
        total_input_tokens=100,
        total_output_tokens=20,
        prompt_version="support-classifier-v1",
        model_name="fake-model",
    )


def test_report_is_written_as_json(
    tmp_path: Path,
) -> None:
    created_at = datetime(
        2026,
        8,
        4,
        12,
        0,
        tzinfo=timezone.utc,
    )

    report = build_evaluation_report(
        dataset=create_dataset(),
        results=[create_result()],
        summary=create_summary(),
        created_at=created_at,
    )

    report_path = write_evaluation_report(
        report=report,
        output_directory=tmp_path / "reports",
    )

    assert report_path.exists()
    assert report_path.name == (
        "run-20260804T120000000000Z.json"
    )

    payload = json.loads(
        report_path.read_text(encoding="utf-8")
    )

    assert payload["dataset_version"] == (
        "golden-dataset-v1"
    )
    assert payload["prompt_version"] == (
        "support-classifier-v1"
    )
    assert payload["model_name"] == "fake-model"
    assert payload["summary"]["total_cases"] == 1
    assert payload["summary"]["category_accuracy"] == 1.0
    assert len(payload["results"]) == 1
    assert payload["results"][0]["test_case_id"] == (
        "billing-001"
    )


def test_report_rejects_timestamp_without_timezone() -> None:
    timestamp_without_timezone = datetime(
        2026,
        8,
        4,
        12,
        0,
    )

    with pytest.raises(
        ValueError,
        match=(
            "created_at must include "
            "timezone information"
        ),
    ):
        build_evaluation_report(
            dataset=create_dataset(),
            results=[create_result()],
            summary=create_summary(),
            created_at=timestamp_without_timezone,
        )