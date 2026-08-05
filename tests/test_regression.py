from model_regression_detection.evaluation_models import (
    EvaluationSummary,
)
from model_regression_detection.regression import (
    evaluate_regression_gate,
)
from model_regression_detection.regression_models import (
    RegressionPolicy,
)


def create_policy() -> RegressionPolicy:
    return RegressionPolicy(
        minimum_category_accuracy=0.9,
        maximum_failed_cases=0,
        maximum_average_latency_ms=3_000.0,
    )


def test_regression_gate_passes_at_thresholds() -> None:
    summary = EvaluationSummary(
        total_cases=10,
        successful_cases=10,
        failed_cases=0,
        category_matches=9,
        category_accuracy=0.9,
        average_latency_ms=3_000.0,
        total_input_tokens=3_500,
        total_output_tokens=230,
        prompt_version="support-classifier-v1",
        model_name="gpt-4o-mini",
    )

    gate = evaluate_regression_gate(
        summary=summary,
        policy=create_policy(),
    )

    assert gate.passed is True
    assert len(gate.checks) == 3
    assert all(
        check.passed
        for check in gate.checks
    )


def test_regression_gate_reports_all_failures() -> None:
    summary = EvaluationSummary(
        total_cases=10,
        successful_cases=9,
        failed_cases=1,
        category_matches=7,
        category_accuracy=0.7,
        average_latency_ms=4_000.0,
        total_input_tokens=3_500,
        total_output_tokens=230,
        prompt_version="support-classifier-v1",
        model_name="gpt-4o-mini",
    )

    gate = evaluate_regression_gate(
        summary=summary,
        policy=create_policy(),
    )

    checks_by_name = {
        check.metric_name: check
        for check in gate.checks
    }

    assert gate.passed is False

    assert (
        checks_by_name["category_accuracy"].passed
        is False
    )
    assert (
        checks_by_name["failed_cases"].passed
        is False
    )
    assert (
        checks_by_name["average_latency_ms"].passed
        is False
    )

    assert (
        checks_by_name["category_accuracy"].actual_value
        == 0.7
    )
    assert (
        checks_by_name["failed_cases"].actual_value
        == 1.0
    )
    assert (
        checks_by_name[
            "average_latency_ms"
        ].actual_value
        == 4_000.0
    )