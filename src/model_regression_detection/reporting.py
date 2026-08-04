from datetime import datetime, timezone
from pathlib import Path

from .dataset_models import GoldenDataset
from .evaluation_models import (
    EvaluationCaseResult,
    EvaluationRunReport,
    EvaluationSummary,
)


def build_evaluation_report(
    dataset: GoldenDataset,
    results: list[EvaluationCaseResult],
    summary: EvaluationSummary,
    created_at: datetime | None = None,
) -> EvaluationRunReport:
    timestamp = created_at or datetime.now(timezone.utc)

    if timestamp.tzinfo is None:
        raise ValueError(
            "created_at must include timezone information"
        )

    run_id = (
        "run-"
        + timestamp.strftime("%Y%m%dT%H%M%S%fZ")
    )

    return EvaluationRunReport(
        run_id=run_id,
        created_at=timestamp,
        dataset_version=dataset.version_id,
        prompt_version=summary.prompt_version,
        model_name=summary.model_name,
        results=results,
        summary=summary,
    )


def write_evaluation_report(
    report: EvaluationRunReport,
    output_directory: Path,
) -> Path:
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = output_directory / (
        f"{report.run_id}.json"
    )

    report_path.write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8",
    )

    return report_path