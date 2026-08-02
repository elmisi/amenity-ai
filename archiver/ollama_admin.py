"""Administrative operations: model download and capability probing.

The download happens ON THE MACHINE HOSTING OLLAMA, not on the one running
amenity-ai: point the endpoint at a remote host and the gigabytes land
there.

Cancellation is cooperative and follows the convention already used by
scan, classify and move: a should_cancel callback polled between one chunk
of the stream and the next.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Optional
from urllib.request import Request, urlopen

from .capabilities import interpret_probe

_PNG_1X1 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA"
    "60e6kgAAAABJRU5ErkJggg=="
)


@dataclass(frozen=True)
class PullProgress:
    status: str = ""
    completed: int = 0
    total: int = 0

    @property
    def fraction(self) -> float:
        if self.total <= 0:
            return 0.0
        return min(1.0, self.completed / self.total)


def pull_model(
    *,
    base_url: str,
    model: str,
    on_progress: Optional[Callable[[PullProgress], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    opener: Optional[Callable[..., Any]] = None,
    timeout_s: float = 3600.0,
) -> Optional[str]:
    """Pull a model. Returns None on success, the error message otherwise."""
    send = opener or urlopen
    request = Request(
        base_url.rstrip("/") + "/api/pull",
        data=json.dumps({"model": model, "stream": True}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with send(request, timeout=timeout_s) as resp:
            for raw in resp:
                if should_cancel is not None and should_cancel():
                    return "cancelled"
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    continue
                if not isinstance(data, dict):
                    continue
                error = data.get("error")
                if error:
                    return str(error)
                if on_progress is not None:
                    on_progress(
                        PullProgress(
                            status=str(data.get("status") or ""),
                            completed=int(data.get("completed") or 0),
                            total=int(data.get("total") or 0),
                        )
                    )
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def probe_vision(
    *,
    base_url: str,
    model: str,
    opener: Optional[Callable[..., Any]] = None,
    timeout_s: float = 30.0,
) -> Optional[bool]:
    """Ask the model to look at a 1x1 PNG. None means inconclusive."""
    send = opener or urlopen
    payload = {
        "model": model,
        "max_tokens": 8,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "ok?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{_PNG_1X1}"},
                    },
                ],
            }
        ],
    }
    request = Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with send(request, timeout=timeout_s) as resp:
            status = getattr(resp, "status", 200) or 200
            body = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        status = getattr(exc, "code", 0) or 0
        try:
            body = exc.read().decode("utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            body = str(exc)
    return interpret_probe(status=status, body=body)
