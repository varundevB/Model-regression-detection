"""Model regression detection system."""

from .classifier import EmailClassifier
from .models import (
    ClassificationRequest,
    ClassificationResult,
    EmailCategory,
    PromptConfig,
)

__all__ = [
    "ClassificationRequest",
    "ClassificationResult",
    "EmailCategory",
    "EmailClassifier",
    "PromptConfig",
]
