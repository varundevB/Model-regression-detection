from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CREDENTIALS_PATH = PROJECT_ROOT / "credentials.json"
TOKEN_PATH = PROJECT_ROOT / "token.json"


def get_credentials() -> Credentials:
    credentials = None

    if TOKEN_PATH.exists():
        credentials = Credentials.from_authorized_user_file(
            TOKEN_PATH,
            SCOPES,
        )

    if not credentials or not credentials.valid:
        if (
            credentials
            and credentials.expired
            and credentials.refresh_token
        ):
            credentials.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_PATH,
                SCOPES,
            )
            credentials = flow.run_local_server(port=0)

        TOKEN_PATH.write_text(
            credentials.to_json(),
            encoding="utf-8",
        )

    return credentials


def main() -> None:
    credentials = get_credentials()

    service = build(
        "gmail",
        "v1",
        credentials=credentials,
    )

    response = (
        service.users()
        .labels()
        .list(userId="me")
        .execute()
    )

    labels = response.get("labels", [])

    print("Gmail labels:")

    for label in labels:
        print(f"- {label['name']}")


if __name__ == "__main__":
    main()