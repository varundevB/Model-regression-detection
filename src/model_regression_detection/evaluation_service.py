from pathlib import Path

from .dataset_models import GoldenDataset
from .evaluation_models import EvaluationRunReport
from .evaluator import (
    EvaluationRunner,
    MetadataClassificationProvider,
)
from .prompts import load_prompt_config
from .reporting import (
    build_evaluation_report,
    write_evaluation_report,
)


def load_golden_dataset(
    path: str | Path,
) -> GoldenDataset:
    dataset_path = Path(path)

    return GoldenDataset.model_validate_json(
        dataset_path.read_text(encoding="utf-8")
    )


def run_evaluation(
    provider: MetadataClassificationProvider,
    dataset_path: str | Path,
    prompt_path: str | Path,
    output_directory: str | Path,
) -> tuple[EvaluationRunReport, Path]:
    dataset = load_golden_dataset(dataset_path)
    prompt = load_prompt_config(prompt_path)

    runner = EvaluationRunner(
        provider=provider,
        prompt=prompt,
    )

    results = runner.evaluate_dataset(dataset)
    summary = runner.summarize_results(results)

    report = build_evaluation_report(
        dataset=dataset,
        results=results,
        summary=summary,
    )

    report_path = write_evaluation_report(
        report=report,
        output_directory=Path(output_directory),
    )

    return report, report_path