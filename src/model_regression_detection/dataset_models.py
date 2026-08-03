from datetime import datetime
from enum import StrEnum

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from .models import ClassificationResult


class ExpectedDifficulty(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class GoldenTestCase(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    id: str = Field(
        pattern=(
            r"^(billing|technical|account|general)-\d{3}$"
        )
    )
    subject: str = Field(
        min_length=1,
        max_length=200,
    )
    input: str = Field(
        min_length=1,
        max_length=20_000,
    )
    expected_output: ClassificationResult
    expected_difficulty: ExpectedDifficulty
    notes: str = Field(
        min_length=1,
        max_length=500,
    )


class GoldenDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version_id: str = Field(
        pattern=r"^golden-dataset-v\d+$"
    )
    created_at: datetime
    test_cases: list[GoldenTestCase] = Field(
        min_length=1
    )

    @field_validator("test_cases")
    @classmethod
    def require_unique_ids(
        cls,
        test_cases: list[GoldenTestCase],
    ) -> list[GoldenTestCase]:
        ids = [test_case.id for test_case in test_cases]

        if len(ids) != len(set(ids)):
            raise ValueError(
                "test case IDs must be unique"
            )

        return test_cases