import os

from dotenv import load_dotenv
from openai import OpenAI

from .evaluation_models import ProviderClassificationResult
from .models import (
    ClassificationRequest,
    ClassificationResult,
    PromptConfig,
)


class OpenAIClassificationProvider:
    def __init__(
        self,
        model: str | None = None,
        client: OpenAI | None = None,
    ) -> None:
        load_dotenv()

        self._model = model or os.getenv(
            "OPENAI_MODEL",
            "gpt-4o-mini",
        )
        self._client = client or OpenAI()
    @property
    def model_name(self) -> str:
        return self._model

    def classify(
        self,
        request: ClassificationRequest,
        prompt: PromptConfig,
    ) -> ClassificationResult:
        provider_result = self.classify_with_metadata(
            request=request,
            prompt=prompt,
        )

        return provider_result.classification

    def classify_with_metadata(
        self,
        request: ClassificationRequest,
        prompt: PromptConfig,
    ) -> ProviderClassificationResult:
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": prompt.system_prompt,
            }
        ]

        for example in prompt.few_shot_examples:
            messages.extend(
                [
                    {
                        "role": "user",
                        "content": example.input,
                    },
                    {
                        "role": "assistant",
                        "content": example.model_dump_json(
                            include={
                                "category",
                                "summary",
                            }
                        ),
                    },
                ]
            )

        messages.append(
            {
                "role": "user",
                "content": request.email_text,
            }
        )

        response = self._client.responses.parse(
            model=self._model,
            input=messages,
            text_format=ClassificationResult,
        )

        if response.output_parsed is None:
            raise RuntimeError(
                "OpenAI returned no parsed classification"
            )

        usage = getattr(
            response,
            "usage",
            None,
        )

        input_tokens = (
            usage.input_tokens
            if usage is not None
            else None
        )

        output_tokens = (
            usage.output_tokens
            if usage is not None
            else None
        )

        return ProviderClassificationResult(
            classification=response.output_parsed,
            model_name=self._model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )