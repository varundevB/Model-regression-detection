from pathlib import Path

from model_regression_detection.prompts import load_prompt_config


def test_v1_prompt_is_valid() -> None:
    project_root = Path(__file__).parents[1]
    prompt = load_prompt_config(
        project_root / "prompts" / "support_classifier_v1.yaml"
    )

    assert prompt.version_id == "support-classifier-v1"
    assert len(prompt.few_shot_examples) == 4
