import argparse
from pathlib import Path

from model_regression_detection.evaluation_models import (
    EvaluationRunReport,
)
from model_regression_detection.regression import (
    evaluate_regression_gate,
)
from model_regression_detection.regression_policy import (
    load_regression_policy,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_POLICY_PATH = (
    PROJECT_ROOT
    / "config"
    / "regression_policy_v1.yaml"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check an evaluation report against "
            "a regression policy."
        )
    )

    parser.add_argument(
        "report_path",
        type=Path,
        help="Path to an evaluation JSON report.",
    )

    parser.add_argument(
        "--policy",
        type=Path,
        default=DEFAULT_POLICY_PATH,
        help="Path to the regression policy YAML file.",
    )

    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()

    report = EvaluationRunReport.model_validate_json(
        arguments.report_path.read_text(
            encoding="utf-8"
        )
    )

    policy = load_regression_policy(
        arguments.policy
    )

    gate = evaluate_regression_gate(
        summary=report.summary,
        policy=policy,
    )

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
