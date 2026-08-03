from typing import Protocol

from .models import ClassificationRequest, ClassificationResult, PromptConfig


class ClassificationProvider(Protocol):
    def classify(
        self,
        request: ClassificationRequest,
        prompt: PromptConfig,
    ) -> ClassificationResult: ...


class EmailClassifier:
    """Stable feature boundary used by both the application and eval runner."""

    def __init__(self, provider: ClassificationProvider, prompt: PromptConfig) -> None:
        self._provider = provider
        self._prompt = prompt

    @property
    def prompt_version(self) -> str:
        return self._prompt.version_id

    def classify(self, email_text: str) -> ClassificationResult:
        request = ClassificationRequest(email_text=email_text)
        return self._provider.classify(request=request, prompt=self._prompt)
