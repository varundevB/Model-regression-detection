from googleapiclient.discovery import build

from gmail_auth_test import get_credentials


GMAIL_QUERY = "label:golden-dataset-candidate"


def get_subject(message: dict) -> str:
    headers = message.get("payload", {}).get("headers", [])

    for header in headers:
        if header.get("name", "").lower() == "subject":
            return header.get("value", "(no subject)")

    return "(no subject)"


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

    print(f"Found {len(message_refs)} candidate emails:")

    for index, message_ref in enumerate(message_refs, start=1):
        message = (
            service.users()
            .messages()
            .get(
                userId="me",
                id=message_ref["id"],
                format="metadata",
                metadataHeaders=["Subject"],
            )
            .execute()
        )

        subject = get_subject(message)
        print(f"{index}. {subject}")


if __name__ == "__main__":
    main()