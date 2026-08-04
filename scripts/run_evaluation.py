from pathlib import Path

from model_regression_detection.evaluation_service import (
    run_evaluation,
)
from model_regression_detection.openai_provider import (
    OpenAIClassificationProvider,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "golden_dataset_v1.json"
)

PROMPT_PATH = (
    PROJECT_ROOT
    / "prompts"
    / "support_classifier_v1.yaml"
)

REPORT_DIRECTORY = PROJECT_ROOT / "reports"


def main() -> None:
    provider = OpenAIClassificationProvider()

    report, report_path = run_evaluation(
        provider=provider,
        dataset_path=DATASET_PATH,
        prompt_path=PROMPT_PATH,
        output_directory=REPORT_DIRECTORY,
    )

    summary = report.summary

    print()
    print("Evaluation completed")
    print(f"Run ID: {report.run_id}")
    print(f"Dataset: {report.dataset_version}")
    print(f"Prompt: {report.prompt_version}")
    print(f"Model: {report.model_name}")
    print(f"Total cases: {summary.total_cases}")
    print(
        f"Successful cases: "
        f"{summary.successful_cases}"
    )
    print(f"Failed cases: {summary.failed_cases}")
    print(
        f"Category accuracy: "
        f"{summary.category_accuracy:.2%}"
    )
    print(
        f"Average latency: "
        f"{summary.average_latency_ms:.2f} ms"
    )
    print(
        f"Input tokens: "
        f"{summary.total_input_tokens}"
    )
    print(
        f"Output tokens: "
        f"{summary.total_output_tokens}"
    )
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
