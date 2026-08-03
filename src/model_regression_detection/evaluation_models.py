from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from .models import (
    ClassificationResult,
    EmailCategory,
)


class EvaluationCaseResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    test_case_id: str
    expected_category: EmailCategory
    actual_category: EmailCategory | None = None
    category_match: bool

    expected_summary: str
    actual_summary: str | None = None

    latency_ms: float = Field(ge=0)

    prompt_version: str
    model_name: str

    input_tokens: int | None = Field(
        default=None,
        ge=0,
    )
    output_tokens: int | None = Field(
        default=None,
        ge=0,
    )

    error: str | None = None


class ProviderClassificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    classification: ClassificationResult
    model_name: str

    input_tokens: int | None = Field(
        default=None,
        ge=0,
    )
    output_tokens: int | None = Field(
        default=None,
        ge=0,
    )