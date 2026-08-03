import base64
from typing import Any


def decode_base64url(data: str) -> str:
    padding = "=" * (-len(data) % 4)
    padded_data = data + padding

    decoded_bytes = base64.urlsafe_b64decode(padded_data)

    return decoded_bytes.decode(
        "utf-8",
        errors="replace",
    )


def extract_plain_text(payload: dict[str, Any]) -> str:
    mime_type = payload.get("mimeType", "")
    body_data = payload.get("body", {}).get("data")

    if mime_type == "text/plain" and body_data:
        return decode_base64url(body_data).strip()

    for part in payload.get("parts", []):
        extracted_text = extract_plain_text(part)

        if extracted_text:
            return extracted_text

    return ""