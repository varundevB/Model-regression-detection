import argparse
from pathlib import Path

from model_regression_detection.evaluation_service import (
    run_evaluation,
)
from model_regression_detection.openai_provider import (
    OpenAIClassificationProvider,
)
from model_regression_detection.regression import (
    evaluate_regression_gate,
)
from model_regression_detection.regression_policy import (
    load_regression_policy,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "golden_dataset_v1.json"
)

DEFAULT_PROMPT_PATH = (
    PROJECT_ROOT
    / "prompts"
    / "support_classifier_v1.yaml"
)

DEFAULT_POLICY_PATH = (
    PROJECT_ROOT
    / "config"
    / "regression_policy_v1.yaml"
)

DEFAULT_REPORT_DIRECTORY = PROJECT_ROOT / "reports"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete golden dataset evaluation."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the golden dataset JSON file.",
    )

    parser.add_argument(
        "--prompt",
        type=Path,
        default=DEFAULT_PROMPT_PATH,
        help="Path to the prompt YAML file.",
    )

    parser.add_argument(
        "--policy",
        type=Path,
        default=DEFAULT_POLICY_PATH,
        help="Path to the absolute regression policy.",
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_REPORT_DIRECTORY,
        help="Directory where the JSON report is written.",
    )

    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    provider = OpenAIClassificationProvider()

    report, report_path = run_evaluation(
        provider=provider,
        dataset_path=arguments.dataset,
        prompt_path=arguments.prompt,
        output_directory=arguments.output_directory,
    )

    summary = report.summary
    policy = load_regression_policy(arguments.policy)

    gate = evaluate_regression_gate(
        summary=summary,
        policy=policy,
    )

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

    print()
    print(
        "Regression gate: "
        + ("PASS" if gate.passed else "FAIL")
    )

    for check in gate.checks:
        status = "PASS" if check.passed else "FAIL"

        print(
            f"- {check.metric_name}: {status} "
            f"(actual={check.actual_value}, "
            f"threshold={check.threshold_value})"
        )

    return 0 if gate.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())