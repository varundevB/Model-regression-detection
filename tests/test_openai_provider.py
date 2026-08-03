from types import SimpleNamespace
from typing import Any

from model_regression_detection.models import (
    ClassificationRequest,
    ClassificationResult,
    EmailCategory,
    PromptConfig,
)
from model_regression_detection.openai_provider import (
    OpenAIClassificationProvider,
)


class FakeResponses:
    def __init__(self) -> None:
        self.last_request: dict[str, Any] | None = None

    def parse(self, **kwargs: Any) -> SimpleNamespace:
        self.last_request = kwargs

        return SimpleNamespace(
            output_parsed=ClassificationResult(
                category=EmailCategory.BILLING,
                summary="The customer asks about an unexpected charge.",
            )
        )


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.responses = FakeResponses()


def test_provider_returns_structured_classification(
    prompt_config: PromptConfig,
) -> None:
    fake_client = FakeOpenAIClient()
    provider = OpenAIClassificationProvider(
        model="test-model",
        client=fake_client,  # type: ignore[arg-type]
    )

    result = provider.classify(
        request=ClassificationRequest(
            email_text="Why was I charged again?"
        ),
        prompt=prompt_config,
    )

    assert result.category is EmailCategory.BILLING
    assert fake_client.responses.last_request is not None
    assert fake_client.responses.last_request["model"] == "test-model"

    messages = fake_client.responses.last_request["input"]
    assert messages[-1]["content"] == "Why was I charged again?"