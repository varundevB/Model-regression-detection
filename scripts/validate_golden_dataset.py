from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from model_regression_detection.dataset_models import (
    GoldenDataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "golden_dataset_v1.json"
)


def main() -> None:
    try:
        dataset = GoldenDataset.model_validate_json(
            DATASET_PATH.read_text(encoding="utf-8")
        )
    except ValidationError as error:
        print("Golden dataset validation failed:")
        print(error)
        raise SystemExit(1) from error

    category_counts = Counter(
        test_case.expected_output.category.value
        for test_case in dataset.test_cases
    )

    difficulty_counts = Counter(
        test_case.expected_difficulty.value
        for test_case in dataset.test_cases
    )

    print("Golden dataset is valid.")
    print(f"Version: {dataset.version_id}")
    print(f"Total cases: {len(dataset.test_cases)}")

    print("Cases by category:")
    for category, count in sorted(
        category_counts.items()
    ):
        print(f"- {category}: {count}")

    print("Cases by difficulty:")
    for difficulty, count in sorted(
        difficulty_counts.items()
    ):
        print(f"- {difficulty}: {count}")


if __name__ == "__main__":
    main()