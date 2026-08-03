from model_regression_detection.classifier import EmailClassifier
from model_regression_detection.models import (
    ClassificationRequest,
    ClassificationResult,
    EmailCategory,
    PromptConfig,
)


class FakeProvider:
    def classify(
        self,
        request: ClassificationRequest,
        prompt: PromptConfig,
    ) -> ClassificationResult:
        assert "charged" in request.email_text
        assert prompt.version_id == "support-classifier-v1"
        return ClassificationResult(
            category=EmailCategory.BILLING,
            summary="The customer reports an unexpected charge.",
        )


def test_classifier_exposes_stable_feature_contract(prompt_config: PromptConfig) -> None:
    classifier = EmailClassifier(provider=FakeProvider(), prompt=prompt_config)

    result = classifier.classify("Why was I charged yesterday?")

    assert classifier.prompt_version == "support-classifier-v1"
    assert result.category is EmailCategory.BILLING
