"""Model capabilities and parameter-size parsing.

Three rungs of decreasing trust:
  declared  - stated by the provider (Ollama >= 0.31, in /api/tags)
  heuristic - guessed from the name, for OpenAI-compatible providers
  probed    - confirmed by a real request, only inside the doctor

The "probed" rung exists because the heuristic has real false negatives: a
model served over the OpenAI API can accept images without saying so in its
name.
"""
from __future__ import annotations

import re
from typing import Optional

CAP_COMPLETION = "completion"
CAP_VISION = "vision"
CAP_EMBEDDING = "embedding"

SOURCE_DECLARED = "declared"
SOURCE_HEURISTIC = "heuristic"
SOURCE_PROBED = "probed"

# A number counts as a size only when followed by "b" at the end of a token:
# in "qwen3.6-27b" the 3.6 is the version and must be ignored.
_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*b(?![a-z0-9])", re.IGNORECASE)

_VISION_TOKENS = (
    "llava",
    "moondream",
    "minicpm-v",
    "bakllava",
    "pixtral",
    "internvl",
    "vision",
    "-vl-",
    "-vl:",
    "vl-",
    "qwen2vl",
    "qwen2.5vl",
)

_IMAGE_ERROR_TOKENS = (
    "image",
    "multimodal",
    "vision",
    "image_url",
)


def parse_parameter_size(*texts: str) -> Optional[float]:
    """First size found while walking the arguments in the order given."""
    for text in texts:
        if not text:
            continue
        match = _SIZE_RE.search(text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    return None


def guess_capabilities(*, model_id: str, root: str = "") -> frozenset[str]:
    """Capabilities guessed from the name, for providers that declare none."""
    haystack = f"{model_id} {root}".lower()
    caps = {CAP_COMPLETION}
    if any(token in haystack for token in _VISION_TOKENS):
        caps.add(CAP_VISION)
    return frozenset(caps)


def interpret_probe(*, status: int, body: str) -> Optional[bool]:
    """True = vision confirmed, False = text-only confirmed, None = inconclusive.

    The probe may contradict the heuristic, but it cannot fake a certainty it
    does not have: a 500 or a timeout says nothing about the model.
    """
    if status == 200:
        return True
    if status in (400, 415, 422):
        lowered = body.lower()
        if any(token in lowered for token in _IMAGE_ERROR_TOKENS):
            return False
    return None
