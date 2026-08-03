from pathlib import Path

from model_regression_detection.classifier import EmailClassifier
from model_regression_detection.openai_provider import (
    OpenAIClassificationProvider,
)
from model_regression_detection.prompts import load_prompt_config


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    prompt = load_prompt_config(
        project_root / "prompts" / "support_classifier_v1.yaml"
    )

    provider = OpenAIClassificationProvider()
    classifier = EmailClassifier(
        provider=provider,
        prompt=prompt,
    )

    result = classifier.classify(
        "I was charged twice for my subscription this month. "
        "Can you refund the duplicate payment?"
    )

    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()