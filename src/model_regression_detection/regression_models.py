from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


class RegressionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    minimum_category_accuracy: float = Field(
        ge=0.0,
        le=1.0,
    )
    maximum_failed_cases: int = Field(ge=0)
    maximum_average_latency_ms: float = Field(gt=0.0)


class RegressionCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric_name: str = Field(min_length=1)
    passed: bool
    actual_value: float
    threshold_value: float
    message: str = Field(min_length=1)


class RegressionGateResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool
    checks: list[RegressionCheck] = Field(
        min_length=1
    )

    @model_validator(mode="after")
    def validate_overall_result(
        self,
    ) -> "RegressionGateResult":
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