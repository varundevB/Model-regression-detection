from datetime import datetime
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
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
class EvaluationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_cases: int = Field(gt=0)
    successful_cases: int = Field(ge=0)
    failed_cases: int = Field(ge=0)
    category_matches: int = Field(ge=0)
    category_accuracy: float = Field(
        ge=0.0,
        le=1.0,
    )
    average_latency_ms: float = Field(ge=0.0)
    total_input_tokens: int = Field(ge=0)
    total_output_tokens: int = Field(ge=0)
    prompt_version: str = Field(min_length=1)
    model_name: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_counts(self) -> "EvaluationSummary":
        if (
            self.successful_cases + self.failed_cases
            != self.total_cases
        ):
            raise ValueError(
                "successful_cases and failed_cases "
                "must equal total_cases"
            )

        if self.category_matches > self.successful_cases:
            raise ValueError(
                "category_matches cannot exceed "
                "successful_cases"
            )

        return self
class EvaluationRunReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    created_at: datetime
    dataset_version: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    results: list[EvaluationCaseResult] = Field(
        min_length=1
    )
    summary: EvaluationSummary

    @model_validator(mode="after")
    def validate_report(self) -> "EvaluationRunReport":
        if self.summary.total_cases != len(self.results):
            raise ValueError(
                "summary total_cases must match "
                "the number of results"
            )

        if self.summary.prompt_version != self.prompt_version:
            raise ValueError(
                "summary prompt_version must match "
                "report prompt_version"
            )

        if self.summary.model_name != self.model_name:
            raise ValueError(
                "summary model_name must match "
                "report model_name"
            )

        if any(
            result.prompt_version != self.prompt_version
            for result in self.results
        ):
            raise ValueError(
                "all results must match "
                "report prompt_version"
            )

        if any(
            result.model_name != self.model_name
            for result in self.results
        ):
            raise ValueError(
                "all results must match "
                "report model_name"
            )

        return self