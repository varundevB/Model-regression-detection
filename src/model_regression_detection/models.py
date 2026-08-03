from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EmailCategory(StrEnum):
    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    GENERAL = "general"


class ClassificationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    email_text: str = Field(min_length=1, max_length=20_000)

    @field_validator("email_text")
    @classmethod
    def reject_blank_email(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("email_text must not be blank")
        return value


class ClassificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    category: EmailCategory
    summary: str = Field(min_length=1, max_length=500)

    @field_validator("summary")
    @classmethod
    def require_one_sentence_summary(cls, value: str) -> str:
        terminal_marks = sum(value.count(mark) for mark in ".!?")
        if terminal_marks > 1:
            raise ValueError("summary must contain at most one sentence")
        return value


class FewShotExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: str = Field(min_length=1)
    category: EmailCategory
    summary: str = Field(min_length=1)


class PromptConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version_id: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    created_at: datetime
    description: str = Field(min_length=1)
    system_prompt: str = Field(min_length=1)
    few_shot_examples: list[FewShotExample] = Field(default_factory=list)
