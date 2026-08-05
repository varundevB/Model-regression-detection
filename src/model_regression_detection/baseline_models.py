from pydantic import BaseModel, ConfigDict, Field, model_validator


class BaselineComparisonPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    maximum_category_accuracy_drop: float = Field(
        ge=0.0,
        le=1.0,
    )
    maximum_failed_case_increase: int = Field(ge=0)
    maximum_average_latency_increase_ratio: float = Field(
        ge=0.0,
    )


class BaselineComparisonCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric_name: str = Field(min_length=1)
    passed: bool
    baseline_value: float
    candidate_value: float
    allowed_value: float
    message: str = Field(min_length=1)


class BaselineComparisonResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool
    checks: list[BaselineComparisonCheck] = Field(
        min_length=1,
    )

    @model_validator(mode="after")
    def validate_overall_result(
        self,
    ) -> "BaselineComparisonResult":
        expected_passed = all(
            check.passed
            for check in self.checks
        )

        if self.passed != expected_passed:
            raise ValueError(
                "passed must match the combined "
                "check results"
            )

        return self