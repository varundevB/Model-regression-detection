import argparse
from pathlib import Path

from model_regression_detection.baseline import (
    compare_against_baseline,
)
from model_regression_detection.baseline_policy import (
    load_baseline_comparison_policy,
)
from model_regression_detection.evaluation_models import (
    EvaluationRunReport,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_POLICY_PATH = (
    PROJECT_ROOT
    / "config"
    / "baseline_comparison_policy_v1.yaml"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a candidate evaluation report "
            "against an approved baseline report."
        )
    )

    parser.add_argument(
        "baseline_report_path",
        type=Path,
        help="Path to the approved baseline JSON report.",
    )

    parser.add_argument(
        "candidate_report_path",
        type=Path,
        help="Path to the candidate JSON report.",
    )

    parser.add_argument(
        "--policy",
        type=Path,
        default=DEFAULT_POLICY_PATH,
        help="Path to the baseline comparison policy.",
    )

    return parser.parse_args()


def load_report(path: Path) -> EvaluationRunReport:
    return EvaluationRunReport.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def main() -> int:
    arguments = parse_arguments()

    baseline = load_report(
        arguments.baseline_report_path
    )
    candidate = load_report(
        arguments.candidate_report_path
    )
    policy = load_baseline_comparison_policy(
        arguments.policy
    )

    comparison = compare_against_baseline(
        baseline=baseline,
        candidate=candidate,
        policy=policy,
    )

    print(
        "Baseline comparison: "
        + ("PASS" if comparison.passed else "FAIL")
    )

    for check in comparison.checks:
        status = "PASS" if check.passed else "FAIL"

        print(
            f"- {check.metric_name}: {status} "
            f"(baseline={check.baseline_value}, "
            f"candidate={check.candidate_value}, "
            f"allowed={check.allowed_value})"
        )

    return 0 if comparison.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())