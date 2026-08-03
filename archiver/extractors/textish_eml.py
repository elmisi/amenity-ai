from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


def extract_eml_text(path: Path, *, max_chars: int) -> Optional[str]:
    """Extract headers, body and attachment names from an RFC 822 email.

    The headers carry most of what the analyzer needs — sender organisation,
    subject, and a Date whose year feeds the year hint. The body prefers the
    text/plain part and falls back to text/html stripped of tags. Attachment
    names are listed because they often name the actual document ("fattura
    marzo.pdf") better than the email text does.
    """
    import email
    from email import policy

    try:
        msg = email.message_from_bytes(path.read_bytes(), policy=policy.default)
    except Exception:
        return None

    parts: list[str] = []
    for header in ("From", "To", "Cc", "Subject", "Date"):
        value = msg.get(header)
        if value:
            parts.append(f"{header}: {str(value).strip()}")

    body_text = ""
    try:
        body = msg.get_body(preferencelist=("plain", "html"))
        if body is not None:
            content = body.get_content()
            if isinstance(content, str):
                if body.get_content_type() == "text/html":
                    content = re.sub(
                        r"<(script|style)[^>]*>.*?</\1>", " ", content,
                        flags=re.DOTALL | re.IGNORECASE,
                    )
                    content = re.sub(r"<[^>]+>", " ", content)
                body_text = " ".join(content.split()).strip()
    except Exception:
        body_text = ""

    attachments: list[str] = []
    try:
        for att in msg.iter_attachments():
            name = att.get_filename()
            if name:
                attachments.append(name)
    except Exception:
        pass

    if body_text:
        parts.append("")
        parts.append(body_text)
    if attachments:
        parts.append("Attachments: " + ", ".join(attachments))

    if not parts:
        return None
    out = "\n".join(parts).strip()
    return out[:max_chars] if out else None
