import base64

from model_regression_detection.gmail_parser import (
    decode_base64url,
    extract_plain_text,
)


def encode_base64url(text: str) -> str:
    encoded = base64.urlsafe_b64encode(
        text.encode("utf-8")
    ).decode("ascii")

    return encoded.rstrip("=")


def test_decode_base64url() -> None:
    encoded = encode_base64url("Hello from Gmail.")

    result = decode_base64url(encoded)

    assert result == "Hello from Gmail."


def test_extract_plain_text_from_nested_payload() -> None:
    payload = {
        "mimeType": "multipart/alternative",
        "body": {},
        "parts": [
            {
                "mimeType": "text/plain",
                "body": {
                    "data": encode_base64url(
                        "I cannot upload my file."
                    )
                },
            },
            {
                "mimeType": "text/html",
                "body": {
                    "data": encode_base64url(
                        "<p>I cannot upload my file.</p>"
                    )
                },
            },
        ],
    }

    result = extract_plain_text(payload)

    assert result == "I cannot upload my file."