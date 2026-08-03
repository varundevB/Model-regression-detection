import hashlib
import json
from pathlib import Path
from typing import Any

from googleapiclient.discovery import build

from gmail_auth_test import get_credentials
from model_regression_detection.gmail_parser import (
    extract_plain_text,
)


GMAIL_QUERY = "label:golden-dataset-candidate"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "imported"
    / "gmail_candidates.json"
)


def get_header(
    payload: dict[str, Any],
    header_name: str,
) -> str:
    headers = payload.get("headers", [])

    for header in headers:
        name = header.get("name", "")

        if name.lower() == header_name.lower():
            return header.get("value", "")

    return ""


def create_candidate_id(message_id: str) -> str:
    digest = hashlib.sha256(
        message_id.encode("utf-8")
    ).hexdigest()

    return f"gmail-{digest[:12]}"


def main() -> None:
    credentials = get_credentials()

    service = build(
        "gmail",
        "v1",
        credentials=credentials,
    )

    response = (
        service.users()
        .messages()
        .list(
            userId="me",
            q=GMAIL_QUERY,
            maxResults=100,
        )
        .execute()
    )

    message_refs = response.get("messages", [])
    candidates = []

    for message_ref in message_refs:
        message = (
            service.users()
            .messages()
            .get(
                userId="me",
                id=message_ref["id"],
                format="full",
            )
            .execute()
        )

        payload = message.get("payload", {})
        subject = get_header(payload, "Subject")
        body = extract_plain_text(payload)

        candidates.append(
            {
                "candidate_id": create_candidate_id(
                    message_ref["id"]
                ),
                "subject": subject,
                "body": body,
                "review_status": "pending",
            }
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            candidates,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        f"Exported {len(candidates)} candidates "
        f"to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()